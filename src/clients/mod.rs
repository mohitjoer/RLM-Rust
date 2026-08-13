//! LM Client trait and implementations.
//!
//! Port of `rlm/clients/`. Defines the `LmClient` trait (analogous to `BaseLM`)
//! and provides implementations for OpenAI-compatible, Anthropic, and Gemini APIs.

pub mod anthropic;
pub mod gemini;
pub mod openai;

use async_trait::async_trait;

use crate::errors::Result;
use crate::types::{ClientBackend, ModelUsageSummary, UsageSummary};

/// Default timeout for LM API calls (in seconds).
pub const DEFAULT_TIMEOUT: u64 = 300;

/// Trait for all language model clients.
///
/// Port of Python `BaseLM` abstract class. All clients must provide sync and
/// async completion, plus usage tracking.
#[async_trait]
pub trait LmClient: Send + Sync {
    /// The model name this client is configured for.
    fn model_name(&self) -> &str;

    /// Synchronous chat completion.
    ///
    /// `prompt` is either a plain string (wrapped as a user message) or a list
    /// of `{role, content}` message objects (JSON).
    fn completion(&self, prompt: serde_json::Value) -> Result<String>;

    /// Asynchronous chat completion.
    async fn acompletion(&self, prompt: serde_json::Value) -> Result<String>;

    /// Get cumulative usage summary for all calls made through this client.
    fn get_usage_summary(&self) -> UsageSummary;

    /// Get usage for the most recent call only.
    fn get_last_usage(&self) -> ModelUsageSummary;
}

/// Factory: create an `LmClient` from a backend name and kwargs.
///
/// Port of `rlm/clients/__init__.py::get_client()`.
pub fn get_client(backend: ClientBackend, kwargs: &serde_json::Value) -> Result<Box<dyn LmClient>> {
    match backend {
        ClientBackend::OpenAi
        | ClientBackend::Vllm
        | ClientBackend::OpenRouter
        | ClientBackend::Vercel => {
            let mut config = kwargs.clone();
            // Set default base_url for known providers
            if config.get("base_url").is_none() {
                match backend {
                    ClientBackend::OpenRouter => {
                        config["base_url"] = serde_json::json!("https://openrouter.ai/api/v1");
                    }
                    ClientBackend::Vercel => {
                        config["base_url"] = serde_json::json!("https://ai-gateway.vercel.sh/v1");
                    }
                    ClientBackend::Vllm => {
                        return Err(crate::errors::RlmError::ConfigError(
                            "base_url is required for vLLM backend".to_string(),
                        ));
                    }
                    _ => {}
                }
            }
            Ok(Box::new(openai::OpenAiClient::from_json(&config)?))
        }
        ClientBackend::Anthropic => Ok(Box::new(anthropic::AnthropicClient::from_json(kwargs)?)),
        ClientBackend::Gemini => Ok(Box::new(gemini::GeminiClient::from_json(kwargs)?)),
        ClientBackend::AzureOpenAi => {
            // Azure uses the same OpenAI-compatible API with a different base_url
            Ok(Box::new(openai::OpenAiClient::from_json(kwargs)?))
        }
        ClientBackend::Portkey => {
            // Portkey also uses OpenAI-compatible API
            Ok(Box::new(openai::OpenAiClient::from_json(kwargs)?))
        }
    }
}
