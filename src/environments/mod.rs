//! Environment trait and factory.
//!
//! Port of `rlm/environments/base_env.py` and `rlm/environments/__init__.py`.

pub mod local_repl;

use async_trait::async_trait;

use crate::errors::Result;
use crate::types::ReplResult;

/// Base trait for all REPL-like environments.
///
/// Port of Python `BaseEnv` abstract class.
#[async_trait]
pub trait Environment: Send + Sync {
    /// Execute a code string in the environment and return the result.
    async fn execute_code(&mut self, code: &str) -> Result<ReplResult>;

    /// Clean up environment resources.
    fn cleanup(&mut self);
}

/// Create an environment from its type name and configuration.
pub fn get_environment(
    env_type: crate::types::EnvironmentType,
    lm_handler_address: &str,
    context_payload: &serde_json::Value,
    depth: u32,
) -> Result<Box<dyn Environment>> {
    match env_type {
        crate::types::EnvironmentType::Local => {
            let repl = local_repl::LocalRepl::new(lm_handler_address, context_payload, depth)?;
            Ok(Box::new(repl))
        }
        other => Err(crate::errors::RlmError::ConfigError(format!(
            "Environment type '{other}' is not yet supported in the Rust port. \
             Currently supported: local"
        ))),
    }
}
