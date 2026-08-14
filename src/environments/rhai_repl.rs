//! Pure-Rust REPL environment using the Rhai embedded scripting engine.
//!
//! Replaces the Python subprocess with a fully in-process scripting engine.
//! The LLM generates Rhai code (JavaScript-like syntax) instead of Python.
//!
//! Rhai is a pure-Rust, safe, sandboxed scripting language with:
//! - No filesystem/network access by default
//! - Sub-microsecond execution for simple scripts
//! - Built-in string, array, and object map operations
//! - Custom function registration for `llm_query`, `SHOW_VARS`, etc.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use rhai::{Dynamic, Engine, EvalAltResult, Map, Scope};

use crate::environments::Environment;
use crate::errors::Result;
use crate::types::ReplResult;

/// Captured stdout from `print()` calls inside Rhai scripts.
type StdoutBuf = Arc<Mutex<String>>;

/// Shared answer object: `#{ content: "", ready: false }`.
type AnswerState = Arc<Mutex<Map>>;

/// Shared variable registry for `SHOW_VARS()`.
type VarRegistry = Arc<Mutex<HashMap<String, String>>>;

/// Address of the LM handler TCP server for `llm_query` calls.
type LmAddress = Arc<String>;

/// Pure-Rust REPL environment backed by the Rhai scripting engine.
pub struct RhaiRepl {
    engine: Engine,
    scope: Scope<'static>,
    stdout_buf: StdoutBuf,
    answer_state: AnswerState,
    var_registry: VarRegistry,
    #[allow(dead_code)]
    lm_address: LmAddress,
    #[allow(dead_code)]
    depth: u32,
}

impl RhaiRepl {
    /// Create a new Rhai REPL and initialize it with context and LM handler access.
    pub fn new(
        lm_handler_address: &str,
        context_payload: &serde_json::Value,
        depth: u32,
    ) -> Result<Self> {
        let stdout_buf: StdoutBuf = Arc::new(Mutex::new(String::new()));
        let answer_state: AnswerState = Arc::new(Mutex::new(Map::new()));
        let var_registry: VarRegistry = Arc::new(Mutex::new(HashMap::new()));
        let lm_address: LmAddress = Arc::new(lm_handler_address.to_string());

        // Initialize answer state
        {
            let mut ans = answer_state.lock().unwrap();
            ans.insert("content".into(), Dynamic::from("".to_string()));
            ans.insert("ready".into(), Dynamic::from(false));
        }

        let mut engine = Engine::new();

        // Safety: limit script execution
        engine.set_max_operations(10_000_000); // 10M ops max
        engine.set_max_string_size(50_000_000); // 50MB max string

        // Register custom print function that captures to stdout buffer
        let print_buf = stdout_buf.clone();
        engine.on_print(move |s| {
            let mut buf = print_buf.lock().unwrap();
            buf.push_str(s);
            buf.push('\n');
        });

        // Register debug handler similarly
        let debug_buf = stdout_buf.clone();
        engine.on_debug(move |s, _src, _pos| {
            let mut buf = debug_buf.lock().unwrap();
            buf.push_str(s);
            buf.push('\n');
        });

        // Register `get_answer` — returns the current answer map
        let ans_get = answer_state.clone();
        engine.register_fn("get_answer", move || -> Map {
            ans_get.lock().unwrap().clone()
        });

        // Register `set_answer_content` — sets answer["content"]
        let ans_set_content = answer_state.clone();
        engine.register_fn("set_answer_content", move |val: Dynamic| {
            let mut ans = ans_set_content.lock().unwrap();
            ans.insert("content".into(), val);
        });

        // Register `set_answer_ready` — sets answer["ready"]
        let ans_set_ready = answer_state.clone();
        engine.register_fn("set_answer_ready", move |val: bool| {
            let mut ans = ans_set_ready.lock().unwrap();
            ans.insert("ready".into(), Dynamic::from(val));
        });

        // Register `submit_answer(content)` — convenience: sets both content and ready=true
        let ans_submit = answer_state.clone();
        engine.register_fn("submit_answer", move |val: Dynamic| {
            let mut ans = ans_submit.lock().unwrap();
            ans.insert("content".into(), val);
            ans.insert("ready".into(), Dynamic::from(true));
        });

        // Register `SHOW_VARS` — lists all variables in scope
        let vars_reg = var_registry.clone();
        engine.register_fn("SHOW_VARS", move || -> String {
            let vars = vars_reg.lock().unwrap();
            if vars.is_empty() {
                return "No user variables".to_string();
            }
            let entries: Vec<String> = vars.iter().map(|(k, v)| format!("{k}: {v}")).collect();
            format!("{{{}}}", entries.join(", "))
        });

        // Register `llm_query(prompt)` — synchronous LLM sub-call via TCP
        let addr_single = lm_address.clone();
        let depth_single = depth;
        engine.register_fn(
            "llm_query",
            move |prompt: String| -> std::result::Result<String, Box<EvalAltResult>> {
                llm_query_sync(&addr_single, &prompt, None, depth_single as i32)
                    .map_err(|e| Box::new(EvalAltResult::from(e.to_string())))
            },
        );

        // Register `llm_query_model(prompt, model)` — single call with model override
        let addr_model = lm_address.clone();
        let depth_model = depth;
        engine.register_fn(
            "llm_query_model",
            move |prompt: String,
                  model: String|
                  -> std::result::Result<String, Box<EvalAltResult>> {
                llm_query_sync(&addr_model, &prompt, Some(&model), depth_model as i32)
                    .map_err(|e| Box::new(EvalAltResult::from(e.to_string())))
            },
        );

        // Register `llm_query_batched(prompts)` — batched concurrent LLM sub-calls
        let addr_batch = lm_address.clone();
        let depth_batch = depth;
        engine.register_fn(
            "llm_query_batched",
            move |prompts: rhai::Array| -> std::result::Result<rhai::Array, Box<EvalAltResult>> {
                let prompt_strings: Vec<String> = prompts
                    .iter()
                    .map(|p| p.clone().into_string().unwrap_or_default())
                    .collect();
                let results =
                    llm_query_batched_sync(&addr_batch, &prompt_strings, None, depth_batch as i32)
                        .map_err(|e| Box::new(EvalAltResult::from(e.to_string())))?;
                Ok(results.into_iter().map(Dynamic::from).collect())
            },
        );

        // Register convenience aliases
        let addr_rlm = lm_address.clone();
        let depth_rlm = depth;
        engine.register_fn(
            "rlm_query",
            move |prompt: String| -> std::result::Result<String, Box<EvalAltResult>> {
                llm_query_sync(&addr_rlm, &prompt, None, depth_rlm as i32)
                    .map_err(|e| Box::new(EvalAltResult::from(e.to_string())))
            },
        );

        // Register string utility functions the LLM commonly uses
        engine.register_fn("len", |s: &str| -> i64 { s.len() as i64 });
        engine.register_fn("str", |val: Dynamic| -> String { val.to_string() });
        engine.register_fn(
            "int",
            |s: &str| -> std::result::Result<i64, Box<EvalAltResult>> {
                s.trim().parse::<i64>().map_err(|e| {
                    Box::new(EvalAltResult::from(format!(
                        "Cannot parse '{s}' as integer: {e}"
                    )))
                })
            },
        );
        engine.register_fn(
            "float",
            |s: &str| -> std::result::Result<f64, Box<EvalAltResult>> {
                s.trim().parse::<f64>().map_err(|e| {
                    Box::new(EvalAltResult::from(format!(
                        "Cannot parse '{s}' as float: {e}"
                    )))
                })
            },
        );

        // Initialize scope with context
        let mut scope = Scope::new();
        let context_str = match context_payload {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        scope.push("context", context_str);

        Ok(Self {
            engine,
            scope,
            stdout_buf,
            answer_state,
            var_registry,
            lm_address,
            depth,
        })
    }
}

#[async_trait]
impl Environment for RhaiRepl {
    async fn execute_code(&mut self, code: &str) -> Result<ReplResult> {
        let start = Instant::now();

        // Clear stdout buffer
        {
            let mut buf = self.stdout_buf.lock().unwrap();
            buf.clear();
        }

        // Compile and execute
        let stderr = match self.engine.compile_with_scope(&self.scope, code) {
            Ok(ast) => match self.engine.run_ast_with_scope(&mut self.scope, &ast) {
                Ok(_) => String::new(),
                Err(e) => format!("Runtime error: {e}"),
            },
            Err(e) => format!("Compilation error: {e}"),
        };

        let execution_time = start.elapsed().as_secs_f64();

        // Capture stdout
        let stdout = {
            let buf = self.stdout_buf.lock().unwrap();
            buf.clone()
        };

        // Update var registry from scope
        {
            let mut vars = self.var_registry.lock().unwrap();
            vars.clear();
            for (name, _is_const, value) in self.scope.iter() {
                if name == "context" || name.starts_with('_') {
                    continue;
                }
                vars.insert(name.to_string(), value.type_name().to_string());
            }
        }

        // Collect locals from scope
        let mut locals = HashMap::new();
        for (name, _is_const, value) in self.scope.iter() {
            if name.starts_with('_') {
                continue;
            }
            let json_val = dynamic_to_json(&value);
            locals.insert(name.to_string(), json_val);
        }

        // Check if answer is ready
        let final_answer = {
            let ans = self.answer_state.lock().unwrap();
            let ready = ans
                .get("ready")
                .and_then(|v| v.as_bool().ok())
                .unwrap_or(false);
            if ready {
                let content = ans
                    .get("content")
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                Some(content)
            } else {
                None
            }
        };

        Ok(ReplResult {
            stdout,
            stderr,
            locals,
            execution_time,
            rlm_calls: Vec::new(),
            final_answer,
        })
    }

    fn cleanup(&mut self) {
        // No subprocess to kill — Rhai engine is in-process.
        // Just clear the scope for potential reuse.
        self.scope.clear();
    }
}

// ─── LLM Query Helpers (sync TCP calls from within Rhai) ────────────────────

/// Perform a synchronous single LLM query over TCP to the LmHandler.
fn llm_query_sync(
    address: &str,
    prompt: &str,
    model: Option<&str>,
    depth: i32,
) -> std::result::Result<String, String> {
    crate::utils::async_utils::block_on_future(async {
        let req = serde_json::json!({
            "prompt": prompt,
            "depth": depth,
            "model": model,
        });

        let resp = crate::core::comms::socket_request(address, &req, 300)
            .await
            .map_err(|e| format!("LLM query failed: {e}"))?;

        if let Some(err) = resp.get("error").and_then(|e| e.as_str()) {
            return Err(format!("LLM error: {err}"));
        }

        Ok(resp
            .get("chat_completion")
            .and_then(|cc| cc.get("response"))
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string())
    })
}

/// Perform a synchronous batched LLM query over TCP.
fn llm_query_batched_sync(
    address: &str,
    prompts: &[String],
    model: Option<&str>,
    depth: i32,
) -> std::result::Result<Vec<String>, String> {
    crate::utils::async_utils::block_on_future(async {
        let prompt_values: Vec<serde_json::Value> = prompts
            .iter()
            .map(|p| serde_json::Value::String(p.clone()))
            .collect();

        let req = serde_json::json!({
            "prompts": prompt_values,
            "depth": depth,
            "model": model,
        });

        let resp = crate::core::comms::socket_request(address, &req, 300)
            .await
            .map_err(|e| format!("Batched LLM query failed: {e}"))?;

        if let Some(err) = resp.get("error").and_then(|e| e.as_str()) {
            return Err(format!("LLM error: {err}"));
        }

        let completions = resp
            .get("chat_completions")
            .and_then(|ccs| ccs.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|cc| {
                        cc.get("response")
                            .and_then(|r| r.as_str())
                            .unwrap_or("")
                            .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(completions)
    })
}

// ─── Rhai Dynamic → serde_json::Value conversion ───────────────────────────

/// Convert a Rhai `Dynamic` value to a `serde_json::Value` for serialization.
fn dynamic_to_json(value: &Dynamic) -> serde_json::Value {
    if value.is_string() {
        serde_json::Value::String(value.clone().into_string().unwrap_or_default())
    } else if value.is_int() {
        serde_json::Value::Number(serde_json::Number::from(value.as_int().unwrap_or(0)))
    } else if value.is_float() {
        if let Some(n) = serde_json::Number::from_f64(value.as_float().unwrap_or(0.0)) {
            serde_json::Value::Number(n)
        } else {
            serde_json::Value::Null
        }
    } else if value.is_bool() {
        serde_json::Value::Bool(value.as_bool().unwrap_or(false))
    } else if value.is_unit() {
        serde_json::Value::Null
    } else if value.is_array() {
        let arr = value
            .clone()
            .into_typed_array::<Dynamic>()
            .unwrap_or_default();
        serde_json::Value::Array(arr.iter().map(dynamic_to_json).collect())
    } else if value.is_map() {
        let map = value.clone().cast::<Map>();
        let obj: serde_json::Map<String, serde_json::Value> = map
            .iter()
            .map(|(k, v)| (k.to_string(), dynamic_to_json(v)))
            .collect();
        serde_json::Value::Object(obj)
    } else {
        // Fallback: render as string representation
        serde_json::Value::String(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhai_repl_basic_execution() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let context = serde_json::Value::String("Hello World".to_string());
            let mut repl = RhaiRepl::new("127.0.0.1:0", &context, 1).unwrap();

            let result = repl
                .execute_code(
                    r#"
                let lines = context.split("\n");
                print("Lines: " + lines.len());
            "#,
                )
                .await
                .unwrap();

            assert!(result.stdout.contains("Lines: 1"));
            assert!(result.stderr.is_empty());
        });
    }

    #[test]
    fn test_rhai_repl_answer_submission() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let context = serde_json::Value::String("SECRET_NUMBER=42".to_string());
            let mut repl = RhaiRepl::new("127.0.0.1:0", &context, 1).unwrap();

            let result = repl
                .execute_code(
                    r#"
                let lines = context.split("\n");
                for line in lines {
                    if line.contains("SECRET_NUMBER=") {
                        let val = line.split("=");
                        submit_answer(val[1]);
                    }
                }
            "#,
                )
                .await
                .unwrap();

            assert!(result.final_answer.is_some());
            assert_eq!(result.final_answer.unwrap(), "42");
        });
    }

    #[test]
    fn test_rhai_repl_error_handling() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let context = serde_json::Value::String("test".to_string());
            let mut repl = RhaiRepl::new("127.0.0.1:0", &context, 1).unwrap();

            let result = repl.execute_code("let x = undefined_var;").await.unwrap();
            assert!(!result.stderr.is_empty());
        });
    }

    #[test]
    fn test_rhai_repl_scope_persistence() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let context = serde_json::Value::String("test data".to_string());
            let mut repl = RhaiRepl::new("127.0.0.1:0", &context, 1).unwrap();

            // First execution: set a variable
            repl.execute_code("let my_var = 42;").await.unwrap();

            // Second execution: use the variable
            let result = repl.execute_code("print(my_var);").await.unwrap();
            assert!(result.stdout.contains("42"));
        });
    }

    #[test]
    fn test_rhai_repl_show_vars() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let context = serde_json::Value::String("data".to_string());
            let mut repl = RhaiRepl::new("127.0.0.1:0", &context, 1).unwrap();

            repl.execute_code("let x = 1; let y = \"hello\";")
                .await
                .unwrap();
            let result = repl.execute_code("print(SHOW_VARS());").await.unwrap();
            assert!(result.stdout.contains("x"));
            assert!(result.stdout.contains("y"));
        });
    }
}
