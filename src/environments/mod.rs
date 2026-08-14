//! Environment trait and factory.
//!
//! Provides both a pure-Rust Rhai REPL (default) and optional Python subprocess REPL.

#[cfg(feature = "python-repl")]
pub mod local_repl;

#[cfg(feature = "rhai-repl")]
pub mod rhai_repl;

use async_trait::async_trait;

use crate::errors::Result;
use crate::types::ReplResult;

/// Base trait for all REPL-like environments.
#[async_trait]
pub trait Environment: Send + Sync {
    /// Execute a code string in the environment and return the result.
    async fn execute_code(&mut self, code: &str) -> Result<ReplResult>;

    /// Clean up environment resources.
    fn cleanup(&mut self);
}

/// Create an environment from its type name and configuration.
///
/// With the default `rhai-repl` feature, uses the pure-Rust Rhai scripting engine.
/// With the `python-repl` feature, falls back to the Python subprocess REPL.
#[allow(clippy::needless_return)]
pub fn get_environment(
    env_type: crate::types::EnvironmentType,
    lm_handler_address: &str,
    context_payload: &serde_json::Value,
    depth: u32,
) -> Result<Box<dyn Environment>> {
    match env_type {
        crate::types::EnvironmentType::Local => {
            #[cfg(feature = "rhai-repl")]
            {
                let repl = rhai_repl::RhaiRepl::new(lm_handler_address, context_payload, depth)?;
                return Ok(Box::new(repl));
            }

            #[cfg(feature = "python-repl")]
            {
                let repl = local_repl::LocalRepl::new(lm_handler_address, context_payload, depth)?;
                return Ok(Box::new(repl));
            }

            #[cfg(not(any(feature = "rhai-repl", feature = "python-repl")))]
            {
                Err(crate::errors::RlmError::ConfigError(
                    "No REPL backend enabled. Enable either `rhai-repl` (default) or `python-repl` feature.".to_string(),
                ))
            }
        }
        other => Err(crate::errors::RlmError::ConfigError(format!(
            "Environment type '{other}' is not yet supported. Currently supported: local"
        ))),
    }
}
