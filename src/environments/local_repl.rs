//! Local REPL environment using a Python subprocess.
//!
//! This is the Rust equivalent of `rlm/environments/local_repl.py`. Instead of
//! using Python's `exec()` directly (which we can't do from Rust), we spawn a
//! persistent Python subprocess that receives code over stdin and returns
//! `ReplResult` JSON over stdout.
//!
//! The subprocess runs a small Python harness that:
//! 1. Receives code as a JSON `{"code": "..."}` message (length-prefixed)
//! 2. Executes it with `exec()` in a sandboxed namespace
//! 3. Returns `{"stdout": ..., "stderr": ..., "locals": ..., "final_answer": ...}` as JSON

use std::collections::HashMap;
use std::io::Write;
use std::process::{Child, Command, Stdio};

use async_trait::async_trait;

use crate::environments::Environment;
use crate::errors::{Result, RlmError};
use crate::types::ReplResult;

/// The Python harness script injected into the subprocess.
///
/// This script sets up a persistent REPL loop that:
/// - Reads length-prefixed JSON commands from stdin
/// - Executes code in a sandboxed globals dict
/// - Returns results as length-prefixed JSON on stdout
const PYTHON_HARNESS: &str = r#"
import sys, json, io, struct, traceback, time

# Namespace for code execution
_globals = {"__builtins__": __builtins__}
_answer = {"content": "", "ready": False}
_globals["answer"] = _answer

# LM handler address for llm_query calls
_lm_handler_address = None
_depth = 1

def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)

def _setup_lm_handler(address, depth):
    global _lm_handler_address, _depth
    _lm_handler_address = address
    _depth = depth

    import socket as _socket

    def _socket_send(sock, data):
        payload = json.dumps(data).encode("utf-8")
        sock.sendall(struct.pack(">I", len(payload)) + payload)

    def _socket_recv(sock):
        raw_len = _recv_exact(sock, 4)
        if not raw_len:
            return {}
        length = struct.unpack(">I", raw_len)[0]
        data = _recv_exact(sock, length)
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def llm_query(prompt, model=None):
        """Single LLM completion via the handler."""
        host, port = _lm_handler_address.split(":")
        port = int(port)
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(300)
            s.connect((host, port))
            req = {"prompt": prompt, "depth": _depth}
            if model:
                req["model"] = model
            _socket_send(s, req)
            resp = _socket_recv(s)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        cc = resp.get("chat_completion", {})
        return cc.get("response", "")

    def llm_query_batched(prompts, model=None):
        """Batched LLM completion via the handler."""
        host, port = _lm_handler_address.split(":")
        port = int(port)
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(300)
            s.connect((host, port))
            req = {"prompts": prompts, "depth": _depth}
            if model:
                req["model"] = model
            _socket_send(s, req)
            resp = _socket_recv(s)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        ccs = resp.get("chat_completions", [])
        return [cc.get("response", "") for cc in ccs]

    def rlm_query(prompt, model=None):
        return llm_query(prompt, model)

    def rlm_query_batched(prompts, model=None):
        return llm_query_batched(prompts, model)

    def SHOW_VARS():
        skip = {"__builtins__", "llm_query", "llm_query_batched", "rlm_query",
                "rlm_query_batched", "SHOW_VARS", "answer"}
        vars_list = {k: type(v).__name__ for k, v in _globals.items() if k not in skip and not k.startswith("_")}
        return str(vars_list)

    _globals["llm_query"] = llm_query
    _globals["llm_query_batched"] = llm_query_batched
    _globals["rlm_query"] = rlm_query
    _globals["rlm_query_batched"] = rlm_query_batched
    _globals["SHOW_VARS"] = SHOW_VARS

_protocol_out = sys.stdout.buffer
_protocol_in = sys.stdin.buffer

def _read_msg():
    raw = _read_exact(_protocol_in, 4)
    if not raw or len(raw) < 4:
        return None
    length = struct.unpack(">I", raw)[0]
    data = _read_exact(_protocol_in, length)
    if not data:
        return None
    return json.loads(data.decode("utf-8"))

def _write_msg(data):
    payload = json.dumps(data, default=str).encode("utf-8")
    _protocol_out.write(struct.pack(">I", len(payload)))
    _protocol_out.write(payload)
    _protocol_out.flush()

# Main loop
while True:
    msg = _read_msg()
    if msg is None:
        break

    cmd = msg.get("cmd")

    if cmd == "setup":
        address = msg.get("address")
        depth = msg.get("depth", 1)
        context = msg.get("context")
        _setup_lm_handler(address, depth)
        if context is not None:
            _globals["context"] = context
        _write_msg({"status": "ok"})

    elif cmd == "exec":
        code = msg.get("code", "")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        t0 = time.perf_counter()
        final_answer = None

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        try:
            exec(code, _globals)
        except Exception:
            traceback.print_exc(file=stderr_buf)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        elapsed = time.perf_counter() - t0

        # Check if answer is ready
        ans = _globals.get("answer", {})
        if isinstance(ans, dict) and ans.get("ready"):
            final_answer = str(ans.get("content", ""))

        # Collect simple locals
        simple_locals = {}
        for k, v in _globals.items():
            if k.startswith("_") or k in ("__builtins__",):
                continue
            if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None))):
                try:
                    json.dumps(v)
                    simple_locals[k] = v
                except (TypeError, ValueError):
                    simple_locals[k] = repr(v)
            elif callable(v):
                simple_locals[k] = f"<function {k}>"
            else:
                simple_locals[k] = repr(v)

        _write_msg({
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "locals": simple_locals,
            "execution_time": elapsed,
            "final_answer": final_answer,
        })

    elif cmd == "quit":
        _write_msg({"status": "bye"})
        break
"#;

/// Local REPL environment backed by a persistent Python subprocess.
pub struct LocalRepl {
    child: Option<Child>,
    address: String,
    depth: u32,
}

impl LocalRepl {
    /// Spawn a new Python subprocess and set up the REPL.
    pub fn new(
        lm_handler_address: &str,
        context_payload: &serde_json::Value,
        depth: u32,
    ) -> Result<Self> {
        let child = Command::new("python3")
            .arg("-u") // Unbuffered
            .arg("-c")
            .arg(PYTHON_HARNESS)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                RlmError::EnvironmentError(format!(
                    "Failed to spawn Python subprocess: {e}. \
                     Make sure python3 is available on PATH."
                ))
            })?;

        let mut repl = Self {
            child: Some(child),
            address: lm_handler_address.to_string(),
            depth,
        };

        // Send setup command
        repl.send_command(&serde_json::json!({
            "cmd": "setup",
            "address": lm_handler_address,
            "depth": depth,
            "context": context_payload,
        }))?;

        // Read setup response
        let resp = repl.read_response()?;
        if resp.get("status").and_then(|v| v.as_str()) != Some("ok") {
            return Err(RlmError::EnvironmentError(
                "Python REPL setup failed".to_string(),
            ));
        }

        Ok(repl)
    }

    /// Get the LM handler address configured for this REPL.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Get the recursion depth of this REPL environment.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Send a length-prefixed JSON message to the subprocess stdin.
    fn send_command(&mut self, data: &serde_json::Value) -> Result<()> {
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| RlmError::EnvironmentError("REPL process not running".into()))?;

        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| RlmError::EnvironmentError("REPL stdin not available".into()))?;

        let payload = serde_json::to_vec(data)?;
        let len_bytes = (payload.len() as u32).to_be_bytes();
        stdin.write_all(&len_bytes)?;
        stdin.write_all(&payload)?;
        stdin.flush()?;

        Ok(())
    }

    /// Read a length-prefixed JSON response from the subprocess stdout.
    fn read_response(&mut self) -> Result<serde_json::Value> {
        use std::io::Read;

        let child = self
            .child
            .as_mut()
            .ok_or_else(|| RlmError::EnvironmentError("REPL process not running".into()))?;

        let stdout = child
            .stdout
            .as_mut()
            .ok_or_else(|| RlmError::EnvironmentError("REPL stdout not available".into()))?;

        // Read 4-byte length prefix
        let mut len_buf = [0u8; 4];
        stdout
            .read_exact(&mut len_buf)
            .map_err(|e| RlmError::EnvironmentError(format!("Failed to read from REPL: {e}")))?;

        let length = u32::from_be_bytes(len_buf) as usize;

        // Read the payload
        let mut payload = vec![0u8; length];
        stdout
            .read_exact(&mut payload)
            .map_err(|e| RlmError::EnvironmentError(format!("Failed to read REPL payload: {e}")))?;

        let value: serde_json::Value = serde_json::from_slice(&payload)?;
        Ok(value)
    }
}

#[async_trait]
impl Environment for LocalRepl {
    async fn execute_code(&mut self, code: &str) -> Result<ReplResult> {
        self.send_command(&serde_json::json!({
            "cmd": "exec",
            "code": code,
        }))?;

        let resp = self.read_response()?;

        let stdout = resp
            .get("stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let stderr = resp
            .get("stderr")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let locals: HashMap<String, serde_json::Value> = resp
            .get("locals")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let execution_time = resp
            .get("execution_time")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let final_answer = resp
            .get("final_answer")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

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
        let _ = self.send_command(&serde_json::json!({"cmd": "quit"}));
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for LocalRepl {
    fn drop(&mut self) {
        self.cleanup();
    }
}
