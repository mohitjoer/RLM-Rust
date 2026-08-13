//! Core data types for the RLM crate.
//!
//! Direct port of `rlm/core/types.py`. All types derive `Serialize` / `Deserialize`
//! for JSON round-tripping, replacing the manual `to_dict()` / `from_dict()` pattern.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─── Backend / Environment enums ────────────────────────────────────────────

/// Supported LLM client backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientBackend {
    OpenAi,
    Portkey,
    OpenRouter,
    Vercel,
    Vllm,
    Anthropic,
    AzureOpenAi,
    Gemini,
}

impl std::fmt::Display for ClientBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAi => write!(f, "openai"),
            Self::Portkey => write!(f, "portkey"),
            Self::OpenRouter => write!(f, "openrouter"),
            Self::Vercel => write!(f, "vercel"),
            Self::Vllm => write!(f, "vllm"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::AzureOpenAi => write!(f, "azure_openai"),
            Self::Gemini => write!(f, "gemini"),
        }
    }
}

impl std::str::FromStr for ClientBackend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "openai" => Ok(Self::OpenAi),
            "portkey" => Ok(Self::Portkey),
            "openrouter" => Ok(Self::OpenRouter),
            "vercel" => Ok(Self::Vercel),
            "vllm" => Ok(Self::Vllm),
            "anthropic" => Ok(Self::Anthropic),
            "azure_openai" => Ok(Self::AzureOpenAi),
            "gemini" => Ok(Self::Gemini),
            other => Err(format!("Unknown backend: {other}")),
        }
    }
}

/// Supported REPL environment types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentType {
    Local,
    Ipython,
    Docker,
    Modal,
    Prime,
    Daytona,
    E2b,
}

impl std::fmt::Display for EnvironmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Ipython => write!(f, "ipython"),
            Self::Docker => write!(f, "docker"),
            Self::Modal => write!(f, "modal"),
            Self::Prime => write!(f, "prime"),
            Self::Daytona => write!(f, "daytona"),
            Self::E2b => write!(f, "e2b"),
        }
    }
}

// ─── Prompt type (string or structured messages) ────────────────────────────

/// A prompt can be a plain string or a list of chat messages (dicts).
///
/// This mirrors the Python `str | dict[str, Any]` union used everywhere.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// A single string prompt.
    Text(String),
    /// A list of chat messages, each with at least `role` and `content`.
    Messages(Vec<ChatMessage>),
}

impl Prompt {
    /// Convert to message list form (wrapping a bare string as a user message).
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        match self {
            Self::Text(s) => vec![ChatMessage {
                role: "user".to_string(),
                content: s.clone(),
                name: None,
            }],
            Self::Messages(msgs) => msgs.clone(),
        }
    }

    /// Total character length of all content.
    pub fn total_length(&self) -> usize {
        match self {
            Self::Text(s) => s.len(),
            Self::Messages(msgs) => msgs.iter().map(|m| m.content.len()).sum(),
        }
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<&String> for Prompt {
    fn from(s: &String) -> Self {
        Self::Text(s.clone())
    }
}

impl From<Vec<ChatMessage>> for Prompt {
    fn from(msgs: Vec<ChatMessage>) -> Self {
        Self::Messages(msgs)
    }
}

/// A single chat message in the OpenAI-style format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Optional `name` field (used by some APIs for function calling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            name: None,
        }
    }
}

// ─── LM Cost Tracking ──────────────────────────────────────────────────────

/// Token and cost tracking for a single model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsageSummary {
    pub total_calls: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    /// Cost in USD, if available from provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cost: Option<f64>,
}

/// Aggregated usage across all models used in a completion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub model_usage_summaries: HashMap<String, ModelUsageSummary>,
}

impl UsageSummary {
    /// Aggregate cost across all models. Returns `None` if no cost data available.
    pub fn total_cost(&self) -> Option<f64> {
        let costs: Vec<f64> = self
            .model_usage_summaries
            .values()
            .filter_map(|s| s.total_cost)
            .collect();
        if costs.is_empty() {
            None
        } else {
            Some(costs.iter().sum())
        }
    }

    /// Aggregate input tokens across all models.
    pub fn total_input_tokens(&self) -> u64 {
        self.model_usage_summaries
            .values()
            .map(|s| s.total_input_tokens)
            .sum()
    }

    /// Aggregate output tokens across all models.
    pub fn total_output_tokens(&self) -> u64 {
        self.model_usage_summaries
            .values()
            .map(|s| s.total_output_tokens)
            .sum()
    }

    /// Merge another summary into this one (mutating).
    pub fn merge(&mut self, other: &UsageSummary) {
        for (model, summary) in &other.model_usage_summaries {
            self.model_usage_summaries
                .insert(model.clone(), summary.clone());
        }
    }
}

// ─── REPL and RLM Iteration types ──────────────────────────────────────────

/// Record of a single LLM call made from within the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmChatCompletion {
    pub root_model: String,
    pub prompt: Prompt,
    pub response: String,
    pub usage_summary: UsageSummary,
    pub execution_time: f64,
    /// Full trajectory (run_metadata + iterations) when logger captures it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Set when this single call failed (e.g. in a batch); response is empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Result of executing one code block in the REPL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplResult {
    pub stdout: String,
    pub stderr: String,
    /// Variables in the REPL namespace after execution (JSON-serialized values).
    pub locals: HashMap<String, serde_json::Value>,
    pub execution_time: f64,
    /// LLM calls made during this code block execution.
    #[serde(default)]
    pub rlm_calls: Vec<RlmChatCompletion>,
    /// If the code set `answer["ready"] = True`, this holds `answer["content"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
}

impl std::fmt::Display for ReplResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReplResult(stdout={}, stderr={}, locals={} keys, execution_time={:.3}, rlm_calls={})",
            truncate(&self.stdout, 50),
            truncate(&self.stderr, 50),
            self.locals.len(),
            self.execution_time,
            self.rlm_calls.len(),
        )
    }
}

/// A code block and the result of executing it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub code: String,
    pub result: ReplResult,
}

/// One iteration (turn) of the RLM agentic loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmIteration {
    pub prompt: Prompt,
    pub response: String,
    pub code_blocks: Vec<CodeBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration_time: Option<f64>,
}

// ─── RLM Metadata ───────────────────────────────────────────────────────────

/// Metadata about the RLM configuration (logged at start of each run).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmMetadata {
    pub root_model: String,
    pub max_depth: u32,
    pub max_iterations: u32,
    pub backend: String,
    pub backend_kwargs: HashMap<String, serde_json::Value>,
    pub environment_type: String,
    pub environment_kwargs: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_backends: Option<Vec<String>>,
}

// ─── Query Metadata ─────────────────────────────────────────────────────────

/// Computed statistics about the incoming prompt/context.
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    pub context_lengths: Vec<usize>,
    pub context_total_length: usize,
    pub context_type: String,
}

impl QueryMetadata {
    /// Compute metadata from a prompt.
    pub fn from_prompt(prompt: &Prompt) -> Self {
        match prompt {
            Prompt::Text(s) => Self {
                context_lengths: vec![s.len()],
                context_total_length: s.len(),
                context_type: "str".to_string(),
            },
            Prompt::Messages(msgs) => {
                let lengths: Vec<usize> = msgs.iter().map(|m| m.content.len()).collect();
                let total: usize = lengths.iter().sum();
                Self {
                    context_lengths: lengths,
                    context_total_length: total,
                    context_type: "list".to_string(),
                }
            }
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Truncate a string for display purposes.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len])
    }
}
