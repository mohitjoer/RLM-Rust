//! OpenAI-compatible LM client.
//!
//! Port of `rlm/clients/openai.py`. Works with OpenAI, vLLM, OpenRouter,
//! Vercel, and any other OpenAI API-compatible endpoint.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::clients::{LmClient, DEFAULT_TIMEOUT};
use crate::errors::{Result, RlmError};
use crate::types::{ModelUsageSummary, UsageSummary};

/// Default OpenAI API base URL.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Internal per-model usage counters.
#[derive(Debug, Default, Clone)]
struct ModelCounters {
    call_count: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_cost: f64,
}

/// OpenAI chat completion request body.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(flatten)]
    sampling_args: HashMap<String, serde_json::Value>,
}

/// OpenAI chat completion response (subset of fields we need).
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    /// OpenRouter returns cost here.
    #[serde(default)]
    cost: Option<f64>,
    /// Some providers put extra cost info here.
    #[serde(default)]
    model_extra: Option<serde_json::Value>,
}

/// OpenAI-compatible LM client using raw HTTP.
pub struct OpenAiClient {
    http: Client,
    model_name: String,
    base_url: String,
    api_key: String,
    sampling_args: HashMap<String, serde_json::Value>,
    /// Mutable usage tracking behind a Mutex for interior mutability.
    usage: Mutex<UsageState>,
}

#[derive(Debug, Default)]
struct UsageState {
    counters: HashMap<String, ModelCounters>,
    last_prompt_tokens: u64,
    last_completion_tokens: u64,
    last_cost: Option<f64>,
}

impl OpenAiClient {
    /// Create from a JSON config object (mirrors Python kwargs).
    ///
    /// Expected fields: `model_name`, `api_key` (optional — falls back to env),
    /// `base_url` (optional), `sampling_args` (optional).
    pub fn from_json(config: &serde_json::Value) -> Result<Self> {
        let model_name = config
            .get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt-4o")
            .to_string();

        let base_url = config
            .get("base_url")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_BASE_URL)
            .to_string();

        // API key: explicit > env var
        let api_key = config
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| Self::env_api_key(&base_url))
            .unwrap_or_default();

        let timeout = config
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT);

        let sampling_args: HashMap<String, serde_json::Value> = config
            .get("sampling_args")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .map_err(|e| RlmError::ClientError(e.to_string()))?;

        Ok(Self {
            http,
            model_name,
            base_url,
            api_key,
            sampling_args,
            usage: Mutex::new(UsageState::default()),
        })
    }

    /// Look up the API key from environment variables based on the base URL.
    fn env_api_key(base_url: &str) -> Option<String> {
        let key = match base_url {
            "https://openrouter.ai/api/v1" => "OPENROUTER_API_KEY",
            "https://ai-gateway.vercel.sh/v1" => "AI_GATEWAY_API_KEY",
            _ => "OPENAI_API_KEY",
        };
        std::env::var(key).ok()
    }

    /// Normalize sampling args (rename max_tokens → max_completion_tokens, etc.).
    fn normalize_sampling_args(&self) -> HashMap<String, serde_json::Value> {
        let mut args = self.sampling_args.clone();
        if let Some(max_tokens) = args.remove("max_tokens") {
            args.insert("max_completion_tokens".to_string(), max_tokens);
        }
        args.remove("extra_body");
        args.retain(|_, v| !v.is_null());
        args
    }

    /// Prepare messages from a prompt value (string or message list).
    fn prepare_messages(&self, prompt: &serde_json::Value) -> Result<Vec<serde_json::Value>> {
        if let Some(s) = prompt.as_str() {
            Ok(vec![serde_json::json!({"role": "user", "content": s})])
        } else if let Some(arr) = prompt.as_array() {
            Ok(arr.clone())
        } else {
            Err(RlmError::ClientError(format!(
                "Invalid prompt type: expected string or array, got {}",
                prompt
            )))
        }
    }

    /// Make the HTTP call and return the response text.
    async fn call_api(&self, messages: Vec<serde_json::Value>) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = ChatCompletionRequest {
            model: self.model_name.clone(),
            messages,
            sampling_args: self.normalize_sampling_args(),
        };

        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(RlmError::ClientError(format!(
                "API returned {status}: {body_text}"
            )));
        }

        let response: ChatCompletionResponse = resp.json().await?;

        // Track usage
        if let Some(usage) = &response.usage {
            self.track_usage(usage);
        }

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    /// Update internal usage counters from an API response.
    fn track_usage(&self, usage: &Usage) {
        let mut state = self.usage.lock().unwrap();

        // Extract cost (OpenRouter style)
        let cost = usage.cost.or_else(|| {
            usage.model_extra.as_ref().and_then(|extra| {
                extra.get("cost").and_then(|c| c.as_f64()).or_else(|| {
                    extra
                        .get("cost_details")
                        .and_then(|cd| cd.get("upstream_inference_cost"))
                        .and_then(|c| c.as_f64())
                })
            })
        });

        // Update counters
        let counters = state.counters.entry(self.model_name.clone()).or_default();
        counters.call_count += 1;
        counters.input_tokens += usage.prompt_tokens;
        counters.output_tokens += usage.completion_tokens;
        if let Some(c) = cost {
            if c > 0.0 {
                counters.total_cost += c;
            }
        }

        // Update last-call tracking (separate from counters borrow)
        state.last_prompt_tokens = usage.prompt_tokens;
        state.last_completion_tokens = usage.completion_tokens;
        state.last_cost = cost.filter(|&c| c > 0.0);
    }
}

#[async_trait]
impl LmClient for OpenAiClient {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&self, prompt: serde_json::Value) -> Result<String> {
        let messages = self.prepare_messages(&prompt)?;
        crate::utils::async_utils::block_on_future(self.call_api(messages))
    }

    async fn acompletion(&self, prompt: serde_json::Value) -> Result<String> {
        let messages = self.prepare_messages(&prompt)?;
        self.call_api(messages).await
    }

    fn get_usage_summary(&self) -> UsageSummary {
        let state = self.usage.lock().unwrap();
        let model_usage_summaries = state
            .counters
            .iter()
            .map(|(model, counters)| {
                (
                    model.clone(),
                    ModelUsageSummary {
                        total_calls: counters.call_count,
                        total_input_tokens: counters.input_tokens,
                        total_output_tokens: counters.output_tokens,
                        total_cost: if counters.total_cost > 0.0 {
                            Some(counters.total_cost)
                        } else {
                            None
                        },
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
            total_cost: state.last_cost,
        }
    }
}
