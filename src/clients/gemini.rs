//! Google Gemini LM client.
//!
//! Port of `rlm/clients/gemini.py`. Uses the Gemini `generateContent` REST API
//! directly via HTTP.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::clients::{LmClient, DEFAULT_TIMEOUT};
use crate::errors::{Result, RlmError};
use crate::types::{ModelUsageSummary, UsageSummary};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

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

// ─── Gemini API request/response types ──────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<GeminiContent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<u64>,
    candidates_token_count: Option<u64>,
}

/// Google Gemini API client.
pub struct GeminiClient {
    http: Client,
    model_name: String,
    api_key: String,
    usage: Mutex<UsageState>,
}

impl GeminiClient {
    /// Create from a JSON config object.
    ///
    /// Expected fields: `model_name` (optional, default "gemini-2.5-flash"),
    /// `api_key` (optional — falls back to GEMINI_API_KEY env).
    pub fn from_json(config: &serde_json::Value) -> Result<Self> {
        let model_name = config
            .get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or("gemini-2.5-flash")
            .to_string();

        let api_key = config
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| {
                RlmError::ConfigError(
                    "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
                        .to_string(),
                )
            })?;

        let timeout = config
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT);

        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .map_err(|e| RlmError::ClientError(e.to_string()))?;

        Ok(Self {
            http,
            model_name,
            api_key,
            usage: Mutex::new(UsageState::default()),
        })
    }

    /// Convert OpenAI-style messages to Gemini format, extracting system instruction.
    fn prepare_contents(
        &self,
        prompt: &serde_json::Value,
    ) -> Result<(Vec<GeminiContent>, Option<GeminiContent>)> {
        if let Some(s) = prompt.as_str() {
            return Ok((
                vec![GeminiContent {
                    role: "user".to_string(),
                    parts: vec![GeminiPart {
                        text: s.to_string(),
                    }],
                }],
                None,
            ));
        }

        if let Some(arr) = prompt.as_array() {
            let mut contents = Vec::new();
            let mut system_instruction = None;

            for msg in arr {
                let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = msg
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                match role {
                    "system" => {
                        system_instruction = Some(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart { text: content }],
                        });
                    }
                    "assistant" => {
                        // Gemini uses "model" instead of "assistant"
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts: vec![GeminiPart { text: content }],
                        });
                    }
                    _ => {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart { text: content }],
                        });
                    }
                }
            }

            return Ok((contents, system_instruction));
        }

        Err(RlmError::ClientError(
            "Invalid prompt type for Gemini client".to_string(),
        ))
    }

    async fn call_api(
        &self,
        contents: Vec<GeminiContent>,
        system_instruction: Option<GeminiContent>,
    ) -> Result<String> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            GEMINI_API_BASE, self.model_name, self.api_key
        );

        let body = GenerateContentRequest {
            contents,
            system_instruction,
        };

        let resp = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(RlmError::ClientError(format!(
                "Gemini API returned {status}: {body_text}"
            )));
        }

        let response: GenerateContentResponse = resp.json().await?;

        // Track usage
        if let Some(usage) = &response.usage_metadata {
            let mut state = self.usage.lock().unwrap();
            let counters = state.counters.entry(self.model_name.clone()).or_default();
            let input = usage.prompt_token_count.unwrap_or(0);
            let output = usage.candidates_token_count.unwrap_or(0);
            counters.call_count += 1;
            counters.input_tokens += input;
            counters.output_tokens += output;
            state.last_prompt_tokens = input;
            state.last_completion_tokens = output;
        }

        // Extract text from first candidate
        let text = response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.content.as_ref())
            .and_then(|c| c.parts.first())
            .map(|p| p.text.clone())
            .unwrap_or_default();

        Ok(text)
    }
}

#[async_trait]
impl LmClient for GeminiClient {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn completion(&self, prompt: serde_json::Value) -> Result<String> {
        let (contents, system) = self.prepare_contents(&prompt)?;
        crate::utils::async_utils::block_on_future(self.call_api(contents, system))
    }

    async fn acompletion(&self, prompt: serde_json::Value) -> Result<String> {
        let (contents, system) = self.prepare_contents(&prompt)?;
        self.call_api(contents, system).await
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
