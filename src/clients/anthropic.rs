//! Anthropic LM client.
//!
//! Port of `rlm/clients/anthropic.py`. Uses the Anthropic Messages API directly
//! via HTTP rather than a Python SDK.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::clients::{LmClient, DEFAULT_TIMEOUT};
use crate::errors::{Result, RlmError};
use crate::types::{ModelUsageSummary, UsageSummary};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(Debug, Default)]
struct ModelCounters {
    call_count: u64,
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Default)]
struct UsageState {
    counters: HashMap<String, ModelCounters>,
    last_prompt_tokens: u64,
    last_completion_tokens: u64,
}

/// Request body for the Anthropic Messages API.
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u64,
    messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

/// Response from the Anthropic Messages API (subset).
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

/// Anthropic Messages API client.
pub struct AnthropicClient {
    http: Client,
    model_name: String,
    api_key: String,
    max_tokens: u64,
    usage: Mutex<UsageState>,
}

impl AnthropicClient {
    /// Create from a JSON config object.
    ///
    /// Expected fields: `model_name`, `api_key`, `max_tokens` (optional, default 32768).
    pub fn from_json(config: &serde_json::Value) -> Result<Self> {
        let model_name = config
            .get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or("claude-3-5-sonnet-20241022")
            .to_string();

        let api_key = config
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| {
                RlmError::ConfigError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
                        .to_string(),
                )
            })?;

        let max_tokens = config
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(32768);

        let timeout = config
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT);

        let http = Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(10)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .map_err(|e| RlmError::ClientError(e.to_string()))?;

        Ok(Self {
            http,
            model_name,
            api_key,
            max_tokens,
            usage: Mutex::new(UsageState::default()),
        })
    }

    /// Separate system message from regular messages (Anthropic requires this).
    fn prepare_messages(
        &self,
        prompt: &serde_json::Value,
    ) -> Result<(Vec<serde_json::Value>, Option<String>)> {
        if let Some(s) = prompt.as_str() {
            return Ok((
                vec![serde_json::json!({"role": "user", "content": s})],
                None,
            ));
        }

        if let Some(arr) = prompt.as_array() {
            let mut messages = Vec::new();
            let mut system = None;

            for msg in arr {
                let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
                if role == "system" {
                    system = msg
                        .get("content")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                } else {
                    messages.push(msg.clone());
                }
            }

            return Ok((messages, system));
        }

        Err(RlmError::ClientError(
            "Invalid prompt type for Anthropic client".to_string(),
        ))
    }

    async fn call_api(
        &self,
        messages: Vec<serde_json::Value>,
        system: Option<String>,
    ) -> Result<String> {
        let body = MessagesRequest {
            model: self.model_name.clone(),
            max_tokens: self.max_tokens,
            messages,
            system,
        };

        let resp = self
            .http
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("anthropic-beta", "prompt-caching-2024-07-31")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(RlmError::ClientError(format!(
                "Anthropic API returned {status}: {body_text}"
            )));
        }

        let response: MessagesResponse = resp.json().await?;

        // Track usage
        {
            let mut state = self.usage.lock().unwrap();
            let counters = state.counters.entry(self.model_name.clone()).or_default();
            counters.call_count += 1;
            counters.input_tokens += response.usage.input_tokens;
            counters.output_tokens += response.usage.output_tokens;
            state.last_prompt_tokens = response.usage.input_tokens;
            state.last_completion_tokens = response.usage.output_tokens;
        }

        let text = response
            .content
            .first()
            .and_then(|b| b.text.as_deref())
            .unwrap_or("")
            .to_string();

        Ok(text)
    }
}

#[async_trait]
impl LmClient for AnthropicClient {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&self, prompt: serde_json::Value) -> Result<String> {
        let (messages, system) = self.prepare_messages(&prompt)?;
        crate::utils::async_utils::block_on_future(self.call_api(messages, system))
    }

    async fn acompletion(&self, prompt: serde_json::Value) -> Result<String> {
        let (messages, system) = self.prepare_messages(&prompt)?;
        self.call_api(messages, system).await
    }

    fn get_usage_summary(&self) -> UsageSummary {
        let state = self.usage.lock().unwrap();
        let model_usage_summaries = state
            .counters
            .iter()
            .map(|(model, c)| {
                (
                    model.clone(),
                    ModelUsageSummary {
                        total_calls: c.call_count,
                        total_input_tokens: c.input_tokens,
                        total_output_tokens: c.output_tokens,
                        total_cost: None,
                    },
                )
            })
            .collect();
        UsageSummary {
            model_usage_summaries,
        }
    }

    fn get_last_usage(&self) -> ModelUsageSummary {
        let state = self.usage.lock().unwrap();
        ModelUsageSummary {
            total_calls: 1,
            total_input_tokens: state.last_prompt_tokens,
            total_output_tokens: state.last_completion_tokens,
            total_cost: None,
        }
    }
}
