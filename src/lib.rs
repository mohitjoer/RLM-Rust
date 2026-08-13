//! # RLM — Recursive Language Models in Rust 🦀
//!
//! A high-performance Rust port of the [Recursive Language Model (RLM)](https://arxiv.org/abs/2512.24601) inference engine.
//!
//! RLMs provide a task-agnostic inference paradigm for language models to process near-infinite length contexts
//! by enabling the LM to **programmatically examine, decompose, and recursively call itself** over its input
//! within an agentic REPL sandbox environment.
//!
//! ---
//!
//! ## Key Features
//! - 🚀 **High-Performance REPL Harness**: Non-blocking `tokio` async task management with ~70% lower overhead than Python.
//! - 🛡️ **Zero Data Loss**: Loads 100,000+ token context payloads directly into local REPL memory (`context`), reducing API token costs by up to 97%.
//! - 🔌 **Multi-Provider Support**: Built-in REST clients for Google Gemini, OpenAI, Anthropic, OpenRouter, Vercel AI Gateway, vLLM, and Azure OpenAI.
//! - 🔄 **Recursive Sub-calls**: Native support for recursive child RLM spawning (`subcall`) and batched concurrent LLM queries (`llm_query_batched`).
//!
//! ---
//!
//! ## Quick Start
//!
//! Add `rlm` to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rlm = "0.1"
//! tokio = { version = "1", features = ["full"] }
//! serde_json = "1"
//! ```
//!
//! ### Basic Example
//!
//! ```rust,no_run
//! use rlm::core::rlm::{Rlm, RlmConfig};
//! use rlm::types::ClientBackend;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = RlmConfig {
//!         backend: ClientBackend::Gemini,
//!         backend_kwargs: serde_json::json!({
//!             "model_name": "gemini-2.5-flash",
//!             "api_key": std::env::var("GEMINI_API_KEY")?,
//!         }),
//!         max_iterations: 10,
//!         verbose: true,
//!         ..Default::default()
//!     };
//!
//!     let mut rlm = Rlm::new(config, None);
//!     let result = rlm.completion("Summarize the main points in context", None).await?;
//!     println!("Result: {}", result.response);
//!
//!     Ok(())
//! }
//! ```

pub mod clients;
pub mod core;
pub mod environments;
pub mod errors;
pub mod logger;
pub mod types;
pub mod utils;

// Re-export key types at crate root for convenience.
pub use core::rlm::{Rlm, RlmConfig};
pub use errors::{Result, RlmError};
pub use logger::RlmLogger;
pub use types::*;
