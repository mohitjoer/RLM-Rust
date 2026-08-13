//! Custom error types for RLM execution limits and cancellation.

use thiserror::Error;

/// Top-level error type for the RLM crate.
#[derive(Debug, Error)]
pub enum RlmError {
    /// Budget (USD) exceeded during execution.
    #[error("Budget exceeded: spent ${spent:.6} of ${budget:.6} budget")]
    BudgetExceeded {
        spent: f64,
        budget: f64,
        partial_answer: Option<String>,
    },

    /// Wall-clock timeout exceeded during execution.
    #[error("Timeout exceeded: {elapsed:.1}s of {timeout:.1}s limit")]
    TimeoutExceeded {
        elapsed: f64,
        timeout: f64,
        partial_answer: Option<String>,
    },

    /// Total token limit (input + output) exceeded.
    #[error("Token limit exceeded: {tokens_used} of {token_limit} tokens")]
    TokenLimitExceeded {
        tokens_used: u64,
        token_limit: u64,
        partial_answer: Option<String>,
    },

    /// Too many consecutive REPL errors.
    #[error("Error threshold exceeded: {error_count} consecutive errors (limit: {threshold})")]
    ErrorThresholdExceeded {
        error_count: u32,
        threshold: u32,
        last_error: Option<String>,
        partial_answer: Option<String>,
    },

    /// User cancelled execution (e.g. Ctrl+C).
    #[error("Execution cancelled by user")]
    Cancelled { partial_answer: Option<String> },

    /// An LM client API call failed.
    #[error("LM client error: {0}")]
    ClientError(String),

    /// HTTP / network error from reqwest.
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization / deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// I/O error (file, socket, subprocess).
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid configuration or argument.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Environment (REPL) execution error.
    #[error("Environment error: {0}")]
    EnvironmentError(String),

    /// Socket communication protocol error.
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, RlmError>;
