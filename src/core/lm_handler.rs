//! LMHandler — routes LLM requests from the RLM process and environment subprocesses.
//!
//! Port of `rlm/core/lm_handler.py`.
//!
//! Uses a tokio TCP server. Protocol: 4-byte length prefix + JSON payload.

use std::collections::HashMap;
use std::sync::Arc;

use std::sync::Mutex;
use tokio::net::TcpListener;

use crate::clients::LmClient;
use crate::core::comms::{socket_recv, socket_send, LmRequest, LmResponse};
use crate::errors::Result;
use crate::types::{RlmChatCompletion, UsageSummary};

/// Type alias for client routing registry.
pub type ClientMap = HashMap<String, Arc<Box<dyn LmClient>>>;

/// Handles all LM calls from the RLM main process and environment subprocesses.
///
/// Uses a tokio TCP server for concurrent requests.
pub struct LmHandler {
    /// The primary (root) LM client.
    pub default_client: Arc<Box<dyn LmClient>>,
    /// Client used for depth=1 sub-calls (if provided).
    pub other_backend_client: Option<Arc<Box<dyn LmClient>>>,
    /// Named clients for model-specific routing.
    pub clients: Arc<Mutex<ClientMap>>,
    /// Handle to the background server task.
    server_handle: Option<tokio::task::JoinHandle<()>>,
    /// The actual bound address.
    pub host: String,
    pub port: u16,
    /// Max concurrent batched calls.
    pub batch_max_concurrent: usize,
}

impl LmHandler {
    /// Create a new handler with a primary client.
    pub fn new(
        client: Box<dyn LmClient>,
        other_backend_client: Option<Box<dyn LmClient>>,
        batch_max_concurrent: usize,
    ) -> Self {
        let model_name = client.model_name().to_string();
        let client = Arc::new(client);
        let clients = Arc::new(Mutex::new(HashMap::new()));

        // Register default client
        clients.lock().unwrap().insert(model_name, client.clone());

        Self {
            default_client: client,
            other_backend_client: other_backend_client.map(Arc::new),
            clients,
            server_handle: None,
            host: "127.0.0.1".to_string(),
            port: 0,
            batch_max_concurrent,
        }
    }

    /// Register a named client for model-specific routing.
    pub fn register_client(&self, model_name: String, client: Arc<Box<dyn LmClient>>) {
        self.clients.lock().unwrap().insert(model_name, client);
    }

    /// Get client by model name or depth, or return default.
    pub fn get_client(&self, model: Option<&str>, depth: i32) -> Arc<Box<dyn LmClient>> {
        // If model is specified and registered, use that
        if let Some(model_name) = model {
            let clients = self.clients.lock().unwrap();
            if let Some(client) = clients.get(model_name) {
                return client.clone();
            }
        }

        // Route based on depth
        if depth == 1 {
            if let Some(ref other) = self.other_backend_client {
                return other.clone();
            }
        }

        self.default_client.clone()
    }

    /// Start the TCP server in the background. Returns the bound address.
    pub async fn start(&mut self) -> Result<String> {
        let listener = TcpListener::bind(format!("{}:0", self.host)).await?;
        let addr = listener.local_addr()?;
        self.port = addr.port();

        let address = format!("{}:{}", self.host, self.port);

        let default_client = self.default_client.clone();
        let other_backend_client = self.other_backend_client.clone();
        let clients = self.clients.clone();

        let handle = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let dc = default_client.clone();
                let obc = other_backend_client.clone();
                let cls = clients.clone();

                tokio::spawn(async move {
                    let request_data = match socket_recv(&mut stream).await {
                        Ok(Some(data)) => data,
                        _ => return,
                    };

                    let request: LmRequest = match serde_json::from_value(request_data) {
                        Ok(r) => r,
                        Err(e) => {
                            let resp = LmResponse::error(format!("Invalid request: {e}"));
                            let _ = socket_send(&mut stream, &serde_json::to_value(&resp).unwrap())
                                .await;
                            return;
                        }
                    };

                    // Resolve client
                    let client = {
                        if let Some(model_name) = &request.model {
                            let locked = cls.lock().unwrap();
                            if let Some(c) = locked.get(model_name) {
                                c.clone()
                            } else if request.depth == 1 {
                                obc.as_ref().map(|c| c.clone()).unwrap_or(dc.clone())
                            } else {
                                dc.clone()
                            }
                        } else if request.depth == 1 {
                            obc.as_ref().map(|c| c.clone()).unwrap_or(dc.clone())
                        } else {
                            dc.clone()
                        }
                    };

                    let response = if request.is_batched() {
                        handle_batched(&request, &**client).await
                    } else if request.prompt.is_some() {
                        handle_single(&request, &**client).await
                    } else {
                        LmResponse::error("Missing 'prompt' or 'prompts'")
                    };

                    let resp_value = serde_json::to_value(&response).unwrap_or_default();
                    let _ = socket_send(&mut stream, &resp_value).await;
                });
            }
        });

        self.server_handle = Some(handle);
        Ok(address)
    }

    /// Stop the TCP server.
    pub fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }

    /// Direct completion call (for main process use — bypasses socket).
    pub fn completion(&self, prompt: serde_json::Value, model: Option<&str>) -> Result<String> {
        let client = self.get_client(model, 0);
        client.completion(prompt)
    }

    /// Get the aggregate usage summary for all clients.
    pub fn get_usage_summary(&self) -> UsageSummary {
        let mut merged = UsageSummary::default();
        merged.merge(&self.default_client.get_usage_summary());

        if let Some(ref other) = self.other_backend_client {
            merged.merge(&other.get_usage_summary());
        }

        // Merge registered clients' usage
        let clients = self.clients.lock().unwrap();
        for client in clients.values() {
            merged.merge(&client.get_usage_summary());
        }

        merged
    }

    /// Get the bound address as "host:port".
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl Drop for LmHandler {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Handle a single prompt request.
async fn handle_single(request: &LmRequest, client: &dyn LmClient) -> LmResponse {
    let prompt = match &request.prompt {
        Some(p) => p.clone(),
        None => return LmResponse::error("Missing 'prompt'"),
    };

    let start = std::time::Instant::now();
    match client.acompletion(prompt.clone()).await {
        Ok(content) => {
            let elapsed = start.elapsed().as_secs_f64();
            let model_usage = client.get_last_usage();
            let root_model = request
                .model
                .clone()
                .unwrap_or_else(|| client.model_name().to_string());
            let usage = UsageSummary {
                model_usage_summaries: [(root_model.clone(), model_usage)].into_iter().collect(),
            };
            let prompt_text = prompt
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| prompt.to_string());
            LmResponse::success(RlmChatCompletion {
                root_model,
                prompt: crate::types::Prompt::Text(prompt_text),
                response: content,
                usage_summary: usage,
                execution_time: elapsed,
                metadata: None,
                error: None,
            })
        }
        Err(e) => LmResponse::error(e.to_string()),
    }
}

/// Handle a batched request concurrently.
async fn handle_batched(request: &LmRequest, client: &dyn LmClient) -> LmResponse {
    let prompts = match &request.prompts {
        Some(p) => p.clone(),
        None => return LmResponse::error("Missing 'prompts'"),
    };

    let start = std::time::Instant::now();
    let futures: Vec<_> = prompts
        .iter()
        .map(|p| client.acompletion(p.clone()))
        .collect();

    let results = futures::future::join_all(futures).await;
    let total_elapsed = start.elapsed().as_secs_f64();
    let per_prompt_time = if prompts.is_empty() {
        0.0
    } else {
        total_elapsed / prompts.len() as f64
    };

    let mut completions = Vec::with_capacity(prompts.len());
    for (prompt, res) in prompts.into_iter().zip(results) {
        let root_model = request
            .model
            .clone()
            .unwrap_or_else(|| client.model_name().to_string());

        let prompt_text = prompt
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| prompt.to_string());

        match res {
            Ok(content) => {
                let model_usage = client.get_last_usage();
                let usage = UsageSummary {
                    model_usage_summaries: [(root_model.clone(), model_usage)]
                        .into_iter()
                        .collect(),
                };
                completions.push(RlmChatCompletion {
                    root_model,
                    prompt: crate::types::Prompt::Text(prompt_text),
                    response: content,
                    usage_summary: usage,
                    execution_time: per_prompt_time,
                    metadata: None,
                    error: None,
                });
            }
            Err(e) => {
                completions.push(RlmChatCompletion {
                    root_model,
                    prompt: crate::types::Prompt::Text(prompt_text),
                    response: String::new(),
                    usage_summary: UsageSummary::default(),
                    execution_time: 0.0,
                    metadata: None,
                    error: Some(format!("llm() call failed - {e}")),
                });
            }
        }
    }

    LmResponse::batched_success(completions)
}
