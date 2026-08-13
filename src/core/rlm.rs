//! RLM — Recursive Language Model orchestrator.
//!
//! Port of `rlm/core/rlm.py`. This is the main entry point for querying an RLM.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::clients;
use crate::core::lm_handler::LmHandler;
use crate::environments::{self, Environment};
use crate::errors::{Result, RlmError};
use crate::logger::RlmLogger;
use crate::types::*;
use crate::utils::parsing::{find_code_blocks, format_iteration};
use crate::utils::prompts::{
    build_rlm_system_prompt, build_user_prompt, DEFAULT_MAX_ITERATIONS, RLM_SYSTEM_PROMPT,
};
use crate::utils::rlm_utils::filter_sensitive_keys;
use crate::utils::token_utils::{count_tokens, get_context_limit};

/// Configuration for constructing an [`Rlm`] instance.
///
/// Uses the builder pattern as a Rust-idiomatic replacement for the Python
/// constructor's 30+ keyword arguments.
pub struct RlmConfig {
    pub backend: ClientBackend,
    pub backend_kwargs: serde_json::Value,
    pub environment: EnvironmentType,
    pub environment_kwargs: serde_json::Value,
    pub depth: u32,
    pub max_depth: u32,
    pub max_iterations: u32,
    pub max_budget: Option<f64>,
    pub max_timeout: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_errors: Option<u32>,
    pub custom_system_prompt: Option<String>,
    pub other_backends: Option<Vec<ClientBackend>>,
    pub other_backend_kwargs: Option<Vec<serde_json::Value>>,
    pub verbose: bool,
    pub orchestrator: bool,
    pub compaction: bool,
    pub compaction_threshold_pct: f64,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            backend: ClientBackend::OpenAi,
            backend_kwargs: serde_json::json!({}),
            environment: EnvironmentType::Local,
            environment_kwargs: serde_json::json!({}),
            depth: 0,
            max_depth: 1,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_budget: None,
            max_timeout: None,
            max_tokens: None,
            max_errors: None,
            custom_system_prompt: None,
            other_backends: None,
            other_backend_kwargs: None,
            verbose: false,
            orchestrator: true,
            compaction: false,
            compaction_threshold_pct: 0.85,
        }
    }
}

/// Recursive Language Model orchestrator.
///
/// Each `completion()` call spawns its own environment and LM handler, which
/// are cleaned up when the call completes.
pub struct Rlm {
    config: RlmConfig,
    logger: Option<RlmLogger>,

    // Tracking (cumulative across all calls)
    cumulative_cost: f64,
    consecutive_errors: u32,
    last_error: Option<String>,
    best_partial_answer: Option<String>,
    completion_start_time: Option<Instant>,
}

impl Rlm {
    /// Create a new RLM with the given configuration.
    pub fn new(config: RlmConfig, logger: Option<RlmLogger>) -> Self {
        Self {
            config,
            logger,
            cumulative_cost: 0.0,
            consecutive_errors: 0,
            last_error: None,
            best_partial_answer: None,
            completion_start_time: None,
        }
    }

    /// Create with default configuration, only specifying the backend kwargs.
    pub fn with_backend(backend: ClientBackend, backend_kwargs: serde_json::Value) -> Self {
        Self::new(
            RlmConfig {
                backend,
                backend_kwargs,
                ..Default::default()
            },
            None,
        )
    }

    /// Main entry point — run an RLM completion.
    ///
    /// Spawns its own environment and LM handler for the duration of this call.
    pub async fn completion(
        &mut self,
        prompt: impl Into<Prompt>,
        root_prompt: Option<&str>,
    ) -> Result<RlmChatCompletion> {
        let prompt: Prompt = prompt.into();
        let time_start = Instant::now();
        self.completion_start_time = Some(time_start);

        // Reset tracking state
        self.consecutive_errors = 0;
        self.last_error = None;
        self.best_partial_answer = None;

        // If at max depth, fall back to plain LLM call
        if self.config.depth >= self.config.max_depth {
            return self.fallback_answer(&prompt).await;
        }

        if let Some(ref mut logger) = self.logger {
            logger.clear_iterations();
        }

        // Log metadata
        self.log_metadata();

        // Create client and handler
        let client = clients::get_client(self.config.backend, &self.config.backend_kwargs)?;

        let other_client = if let (Some(backends), Some(kwargs)) = (
            &self.config.other_backends,
            &self.config.other_backend_kwargs,
        ) {
            if !backends.is_empty() {
                Some(clients::get_client(backends[0], &kwargs[0])?)
            } else {
                None
            }
        } else {
            None
        };

        let mut handler = LmHandler::new(client, other_client, 16);

        // Register all other clients by model name
        if let (Some(backends), Some(kwargs)) = (
            &self.config.other_backends,
            &self.config.other_backend_kwargs,
        ) {
            for (backend, kw) in backends.iter().zip(kwargs.iter()) {
                let add_client = clients::get_client(*backend, kw)?;
                let model_name = add_client.model_name().to_string();
                handler.register_client(model_name, Arc::new(add_client));
            }
        }

        let address = handler.start().await?;

        // Create environment
        let prompt_json = serde_json::to_value(&prompt)?;
        let mut environment = environments::get_environment(
            self.config.environment,
            &address,
            &prompt_json,
            self.config.depth + 1,
        )?;

        // Build initial message history
        let query_metadata = QueryMetadata::from_prompt(&prompt);
        let mut message_history = build_rlm_system_prompt(
            self.config
                .custom_system_prompt
                .as_deref()
                .unwrap_or(RLM_SYSTEM_PROMPT),
            &query_metadata,
            None, // TODO: custom tools formatting
            root_prompt,
            self.config.orchestrator,
        );

        // Main agentic loop
        let result = self
            .run_loop(
                &mut message_history,
                &handler,
                environment.as_mut(),
                root_prompt,
                time_start,
                &prompt,
            )
            .await;

        // Cleanup
        handler.stop();
        environment.cleanup();

        result
    }

    /// The main iterative loop.
    async fn run_loop(
        &mut self,
        message_history: &mut Vec<serde_json::Value>,
        handler: &LmHandler,
        environment: &mut dyn Environment,
        root_prompt: Option<&str>,
        time_start: Instant,
        original_prompt: &Prompt,
    ) -> Result<RlmChatCompletion> {
        for i in 0..self.config.max_iterations {
            // Check timeout before each iteration
            self.check_timeout(i, time_start)?;

            // Check compaction
            if self.config.compaction {
                let model_name = self.model_name();
                let max_tokens = get_context_limit(&model_name);
                let current_tokens = count_tokens(message_history, &model_name);
                let threshold_tokens =
                    (self.config.compaction_threshold_pct * max_tokens as f64) as u64;

                if current_tokens >= threshold_tokens {
                    *message_history = self.compact_history(handler, message_history)?;
                }
            }

            // Build per-turn user prompt
            message_history.push(build_user_prompt(
                root_prompt,
                i,
                1, // context_count (TODO: persistence)
                0, // history_count
                self.config.max_iterations,
            ));

            // Run one iteration
            let iteration = self
                .completion_turn(message_history, handler, environment)
                .await?;

            // Check limits
            self.check_iteration_limits(&iteration, i, handler)?;

            // Check for final answer
            let mut final_answer = None;
            for block in &iteration.code_blocks {
                if block.result.final_answer.is_some() {
                    final_answer = block.result.final_answer.clone();
                    break;
                }
            }

            // Store best partial answer
            if !iteration.response.trim().is_empty() {
                self.best_partial_answer = Some(iteration.response.clone());
            }

            // Log iteration
            if let Some(ref mut logger) = self.logger {
                logger.log_iteration(&iteration);
            }

            if self.config.verbose {
                eprintln!(
                    "[RLM] Iteration {}/{} completed ({})",
                    i + 1,
                    self.config.max_iterations,
                    if final_answer.is_some() {
                        "FINAL ANSWER"
                    } else {
                        "continuing"
                    }
                );
            }

            if let Some(answer) = final_answer {
                let elapsed = time_start.elapsed().as_secs_f64();
                let usage = handler.get_usage_summary();

                return Ok(RlmChatCompletion {
                    root_model: self.model_name(),
                    prompt: original_prompt.clone(),
                    response: answer,
                    usage_summary: usage,
                    execution_time: elapsed,
                    metadata: self.logger.as_ref().and_then(|l| l.get_trajectory()),
                    error: None,
                });
            }

            // Format iteration for next prompt
            let new_messages = format_iteration(&iteration);
            message_history.extend(new_messages);
        }

        // Exhausted iterations — generate default answer
        let elapsed = time_start.elapsed().as_secs_f64();
        let final_answer = self.default_answer(message_history, handler)?;
        let usage = handler.get_usage_summary();

        Ok(RlmChatCompletion {
            root_model: self.model_name(),
            prompt: original_prompt.clone(),
            response: final_answer,
            usage_summary: usage,
            execution_time: elapsed,
            metadata: self.logger.as_ref().and_then(|l| l.get_trajectory()),
            error: None,
        })
    }

    /// Single iteration: LLM call → parse code → execute in REPL.
    async fn completion_turn(
        &self,
        message_history: &[serde_json::Value],
        handler: &LmHandler,
        environment: &mut dyn Environment,
    ) -> Result<RlmIteration> {
        let iter_start = Instant::now();

        // Call the LLM
        let prompt_value = serde_json::Value::Array(message_history.to_vec());
        let response = handler.completion(prompt_value, None)?;

        // Parse code blocks from response
        let code_block_strs = find_code_blocks(&response);
        let mut code_blocks = Vec::new();

        for code_str in code_block_strs {
            let result = environment.execute_code(&code_str).await?;
            code_blocks.push(CodeBlock {
                code: code_str,
                result,
            });
        }

        let iteration_time = iter_start.elapsed().as_secs_f64();

        Ok(RlmIteration {
            prompt: Prompt::Messages(Vec::new()), // Omit the full prompt for memory
            response,
            code_blocks,
            final_answer: None,
            iteration_time: Some(iteration_time),
        })
    }

    /// Check timeout limit.
    fn check_timeout(&self, _iteration: u32, time_start: Instant) -> Result<()> {
        if let Some(max_timeout) = self.config.max_timeout {
            let elapsed = time_start.elapsed().as_secs_f64();
            if elapsed > max_timeout {
                return Err(RlmError::TimeoutExceeded {
                    elapsed,
                    timeout: max_timeout,
                    partial_answer: self.best_partial_answer.clone(),
                });
            }
        }
        Ok(())
    }

    /// Check error tracking, budget, and token limits after an iteration.
    fn check_iteration_limits(
        &mut self,
        iteration: &RlmIteration,
        _iteration_num: u32,
        handler: &LmHandler,
    ) -> Result<()> {
        // Track consecutive errors
        let mut had_error = false;
        for block in &iteration.code_blocks {
            if !block.result.stderr.is_empty() {
                had_error = true;
                self.last_error = Some(block.result.stderr.clone());
                break;
            }
        }

        if had_error {
            self.consecutive_errors += 1;
        } else {
            self.consecutive_errors = 0;
        }

        // Check error threshold
        if let Some(max_errors) = self.config.max_errors {
            if self.consecutive_errors >= max_errors {
                return Err(RlmError::ErrorThresholdExceeded {
                    error_count: self.consecutive_errors,
                    threshold: max_errors,
                    last_error: self.last_error.clone(),
                    partial_answer: self.best_partial_answer.clone(),
                });
            }
        }

        // Check budget
        if let Some(max_budget) = self.config.max_budget {
            let usage = handler.get_usage_summary();
            let cost = usage.total_cost().unwrap_or(0.0);
            self.cumulative_cost = cost;
            if self.cumulative_cost > max_budget {
                return Err(RlmError::BudgetExceeded {
                    spent: self.cumulative_cost,
                    budget: max_budget,
                    partial_answer: self.best_partial_answer.clone(),
                });
            }
        }

        // Check token limit
        if let Some(max_tokens) = self.config.max_tokens {
            let usage = handler.get_usage_summary();
            let total = usage.total_input_tokens() + usage.total_output_tokens();
            if total > max_tokens {
                return Err(RlmError::TokenLimitExceeded {
                    tokens_used: total,
                    token_limit: max_tokens,
                    partial_answer: self.best_partial_answer.clone(),
                });
            }
        }

        Ok(())
    }

    /// Default answer when iterations are exhausted.
    fn default_answer(
        &mut self,
        message_history: &[serde_json::Value],
        handler: &LmHandler,
    ) -> Result<String> {
        let mut prompt = message_history.to_vec();
        prompt.push(serde_json::json!({
            "role": "assistant",
            "content": "Please provide a final answer to the user's question based on the information provided."
        }));

        let response = handler.completion(serde_json::Value::Array(prompt), None)?;

        if let Some(ref mut logger) = self.logger {
            logger.log_iteration(&RlmIteration {
                prompt: Prompt::Messages(Vec::new()),
                response: response.clone(),
                code_blocks: Vec::new(),
                final_answer: Some(response.clone()),
                iteration_time: None,
            });
        }

        Ok(response)
    }

    /// Fallback: plain LLM call at max depth (no REPL).
    pub async fn fallback_answer(&self, prompt: &Prompt) -> Result<RlmChatCompletion> {
        let client = clients::get_client(self.config.backend, &self.config.backend_kwargs)?;

        let start = Instant::now();
        let prompt_value = serde_json::to_value(prompt)?;
        let response = client.acompletion(prompt_value).await?;
        let elapsed = start.elapsed().as_secs_f64();

        Ok(RlmChatCompletion {
            root_model: self.model_name(),
            prompt: prompt.clone(),
            response,
            usage_summary: client.get_usage_summary(),
            execution_time: elapsed,
            metadata: None,
            error: None,
        })
    }

    /// Compact message history when context is too large.
    fn compact_history(
        &self,
        handler: &LmHandler,
        message_history: &[serde_json::Value],
    ) -> Result<Vec<serde_json::Value>> {
        let mut summary_prompt = message_history.to_vec();
        summary_prompt.push(serde_json::json!({
            "role": "user",
            "content": "Summarize your progress so far. Include:\n\
                1. Which steps/sub-tasks you have completed and which remain.\n\
                2. Any concrete intermediate results (numbers, values, variable names) \
                   you computed — preserve these exactly.\n\
                3. What your next action should be.\n\
                Be concise (1–3 paragraphs) but preserve all key results and your \
                current position in the task."
        }));

        let summary = handler.completion(serde_json::Value::Array(summary_prompt), None)?;

        // Keep system + metadata, then append summary + continue instruction
        let mut new_history = message_history[..2.min(message_history.len())].to_vec();
        new_history.push(serde_json::json!({
            "role": "assistant",
            "content": summary,
        }));
        new_history.push(serde_json::json!({
            "role": "user",
            "content": "Your conversation has been compacted. \
                Continue from the above summary. Do NOT repeat work you have already \
                completed. Use SHOW_VARS() to check which REPL variables exist, \
                and check `history` for full context. Your next action:"
        }));

        Ok(new_history)
    }

    /// Log run metadata.
    fn log_metadata(&mut self) {
        let model_name = self.model_name();
        let metadata = RlmMetadata {
            root_model: model_name,
            max_depth: self.config.max_depth,
            max_iterations: self.config.max_iterations,
            backend: self.config.backend.to_string(),
            backend_kwargs: self
                .config
                .backend_kwargs
                .as_object()
                .map(|obj| {
                    let map: HashMap<String, serde_json::Value> =
                        obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    filter_sensitive_keys(&map).into_iter().collect()
                })
                .unwrap_or_default(),
            environment_type: self.config.environment.to_string(),
            environment_kwargs: HashMap::new(),
            other_backends: self
                .config
                .other_backends
                .as_ref()
                .map(|bs| bs.iter().map(|b| b.to_string()).collect()),
        };

        if let Some(ref mut logger) = self.logger {
            logger.log_metadata(&metadata);
        }
    }

    /// Extract model name from backend kwargs.
    fn model_name(&self) -> String {
        self.config
            .backend_kwargs
            .get("model_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Perform a recursive subcall, spawning a child RLM if depth allows.
    ///
    /// Port of Python `_subcall`. If depth + 1 >= max_depth, falls back to plain LM completion.
    pub async fn subcall(
        &mut self,
        prompt: impl Into<Prompt>,
        model: Option<&str>,
    ) -> Result<RlmChatCompletion> {
        let prompt: Prompt = prompt.into();
        let next_depth = self.config.depth + 1;

        if next_depth >= self.config.max_depth {
            let mut child_kwargs = self.config.backend_kwargs.clone();
            if let Some(m) = model {
                child_kwargs["model_name"] = serde_json::json!(m);
            }
            let client = clients::get_client(self.config.backend, &child_kwargs)?;
            let start = Instant::now();
            let prompt_value = serde_json::to_value(&prompt)?;
            let response = client.acompletion(prompt_value).await?;
            let elapsed = start.elapsed().as_secs_f64();
            let usage = client.get_usage_summary();

            return Ok(RlmChatCompletion {
                root_model: model
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| client.model_name().to_string()),
                prompt,
                response,
                usage_summary: usage,
                execution_time: elapsed,
                metadata: None,
                error: None,
            });
        }

        let mut child_config = RlmConfig {
            backend: self.config.backend,
            backend_kwargs: self.config.backend_kwargs.clone(),
            environment: self.config.environment,
            environment_kwargs: self.config.environment_kwargs.clone(),
            depth: next_depth,
            max_depth: self.config.max_depth,
            max_iterations: self.config.max_iterations,
            max_budget: self
                .config
                .max_budget
                .map(|b| (b - self.cumulative_cost).max(0.0)),
            max_timeout: self.config.max_timeout,
            max_tokens: self.config.max_tokens,
            max_errors: self.config.max_errors,
            custom_system_prompt: self.config.custom_system_prompt.clone(),
            other_backends: self.config.other_backends.clone(),
            other_backend_kwargs: self.config.other_backend_kwargs.clone(),
            verbose: self.config.verbose,
            orchestrator: self.config.orchestrator,
            compaction: self.config.compaction,
            compaction_threshold_pct: self.config.compaction_threshold_pct,
        };

        if let Some(m) = model {
            child_config.backend_kwargs["model_name"] = serde_json::json!(m);
        }

        let mut child_rlm = Rlm::new(child_config, None);
        child_rlm.completion(prompt, None).await
    }
}
