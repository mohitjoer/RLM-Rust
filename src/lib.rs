//! # RLM — Recursive Language Models in Rust 🦀
//!
//! A high-performance, **fully Rust-based** inference engine for [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601).
//!
//! RLMs provide a task-agnostic inference paradigm for language models to process near-infinite length contexts
//! by enabling the LM to **programmatically examine, decompose, and recursively call itself** over its input
//! within a sandboxed scripting REPL environment.
//!
//! ---
//!
//! ## Key Features
//! - 🚀 **Pure Rust** — No Python dependency. Uses the embedded [Rhai](https://rhai.rs) scripting engine.
//! - 📉 **97% Token Reduction** — Loads context into local REPL memory, sending only instructions to the LLM API.
//! - ⚡ **HTTP/2 Optimized** — `tcp_nodelay`, connection pooling, and keep-alive for minimal API latency.
//! - 🔌 **Multi-Provider** — Gemini, OpenAI, Anthropic, OpenRouter, Vercel AI Gateway, vLLM, Azure OpenAI.
//! - 🔄 **Recursive Sub-calls** — Native `subcall`, `llm_query_batched` for agentic decomposition.
//!
//! ---
//!
//! ## Quick Start
//!
//! Add `rlm` to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rlm = "0.2"
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
