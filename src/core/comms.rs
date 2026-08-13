//! Socket communication protocol and message types.
//!
//! Port of `rlm/core/comms_utils.py`.
//!
//! Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::errors::{Result, RlmError};
use crate::types::RlmChatCompletion;

// ─── Message types ──────────────────────────────────────────────────────────

/// Request message sent to the LM Handler.
///
/// Supports both single prompt (`prompt`) and batched prompts (`prompts`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub depth: i32,
}

impl LmRequest {
    /// Create a single-prompt request.
    pub fn single(prompt: serde_json::Value, model: Option<String>, depth: i32) -> Self {
        Self {
            prompt: Some(prompt),
            prompts: None,
            model,
            depth,
        }
    }

    /// Create a batched request.
    pub fn batched(prompts: Vec<serde_json::Value>, model: Option<String>, depth: i32) -> Self {
        Self {
            prompt: None,
            prompts: Some(prompts),
            model,
            depth,
        }
    }

    /// Check if this is a batched request.
    pub fn is_batched(&self) -> bool {
        self.prompts.as_ref().is_some_and(|p| !p.is_empty())
    }
}

/// Response message from the LM Handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_completion: Option<RlmChatCompletion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_completions: Option<Vec<RlmChatCompletion>>,
}

impl LmResponse {
    /// Check if response was successful.
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }

    /// Create a successful single response.
    pub fn success(completion: RlmChatCompletion) -> Self {
        Self {
            error: None,
            chat_completion: Some(completion),
            chat_completions: None,
        }
    }

    /// Create a successful batched response.
    pub fn batched_success(completions: Vec<RlmChatCompletion>) -> Self {
        Self {
            error: None,
            chat_completion: None,
            chat_completions: Some(completions),
        }
    }

    /// Create an error response.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            error: Some(msg.into()),
            chat_completion: None,
            chat_completions: None,
        }
    }
}

// ─── Socket protocol helpers ────────────────────────────────────────────────

/// Send a length-prefixed JSON message over an async TCP stream.
///
/// Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
pub async fn socket_send(stream: &mut TcpStream, data: &serde_json::Value) -> Result<()> {
    let payload = serde_json::to_vec(data)?;
    let len_bytes = (payload.len() as u32).to_be_bytes();
    stream.write_all(&len_bytes).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    Ok(())
}

/// Receive a length-prefixed JSON message from an async TCP stream.
///
/// Returns `None` if the connection was closed cleanly (no length prefix received).
pub async fn socket_recv(stream: &mut TcpStream) -> Result<Option<serde_json::Value>> {
    // Read 4-byte length prefix
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }

    let length = u32::from_be_bytes(len_buf) as usize;

    // Read the payload
    let mut payload = vec![0u8; length];
    stream.read_exact(&mut payload).await?;

    let value: serde_json::Value = serde_json::from_slice(&payload)?;
    Ok(Some(value))
}

/// Synchronous version: send request and receive response over a new connection.
pub async fn socket_request(
    addr: &str,
    data: &serde_json::Value,
    timeout_secs: u64,
) -> Result<serde_json::Value> {
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        TcpStream::connect(addr),
    )
    .await
    .map_err(|_| RlmError::ProtocolError(format!("Connection timed out to {addr}")))??;

    socket_send(&mut stream, data).await?;

    socket_recv(&mut stream)
        .await?
        .ok_or_else(|| RlmError::ProtocolError("Connection closed before response".into()))
}

/// Send an LM request and return a typed response.
pub async fn send_lm_request(addr: &str, request: &LmRequest, timeout_secs: u64) -> LmResponse {
    let data = match serde_json::to_value(request) {
        Ok(v) => v,
        Err(e) => return LmResponse::error(format!("Serialization failed: {e}")),
    };

    match socket_request(addr, &data, timeout_secs).await {
        Ok(resp_data) => serde_json::from_value(resp_data)
            .unwrap_or_else(|e| LmResponse::error(format!("Response deserialization failed: {e}"))),
        Err(e) => LmResponse::error(format!("Request failed: {e}")),
    }
}
