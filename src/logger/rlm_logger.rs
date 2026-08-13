//! Logger for RLM iterations.
//!
//! Port of `rlm/logger/rlm_logger.py`.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Local;
use uuid::Uuid;

use crate::types::{RlmIteration, RlmMetadata};

/// Captures trajectory (run metadata + iterations) for each completion.
///
/// - `log_dir = None`: trajectory available via `get_trajectory()` only (in-memory).
/// - `log_dir = Some(path)`: same capture plus appends to a JSONL file per run.
pub struct RlmLogger {
    save_to_disk: bool,
    log_file_path: Option<PathBuf>,

    run_metadata: Option<serde_json::Value>,
    iterations: Vec<serde_json::Value>,
    iteration_count: u32,
    metadata_logged: bool,
}

impl RlmLogger {
    /// Create a new logger.
    ///
    /// If `log_dir` is provided, a JSONL file is created for this run.
    pub fn new(log_dir: Option<&str>, file_name: Option<&str>) -> Self {
        let file_name = file_name.unwrap_or("rlm");
        let (save_to_disk, log_file_path) = if let Some(dir) = log_dir {
            fs::create_dir_all(dir).ok();
            let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
            let run_id = &Uuid::new_v4().to_string()[..8];
            let path = Path::new(dir).join(format!("{file_name}_{timestamp}_{run_id}.jsonl"));
            (true, Some(path))
        } else {
            (false, None)
        };

        Self {
            save_to_disk,
            log_file_path,
            run_metadata: None,
            iterations: Vec::new(),
            iteration_count: 0,
            metadata_logged: false,
        }
    }

    /// Create an in-memory-only logger (no disk writes).
    pub fn in_memory() -> Self {
        Self::new(None, None)
    }

    /// Capture run metadata (and optionally write to file).
    pub fn log_metadata(&mut self, metadata: &RlmMetadata) {
        if self.metadata_logged {
            return;
        }

        let metadata_value = serde_json::to_value(metadata).unwrap_or(serde_json::Value::Null);
        self.run_metadata = Some(metadata_value.clone());
        self.metadata_logged = true;

        if self.save_to_disk {
            let mut entry = serde_json::json!({
                "type": "metadata",
                "timestamp": Local::now().to_rfc3339(),
            });
            if let serde_json::Value::Object(map) = metadata_value {
                for (k, v) in map {
                    entry[k] = v;
                }
            }
            self.append_to_file(&entry);
        }
    }

    /// Capture one iteration (and optionally append to file).
    pub fn log_iteration(&mut self, iteration: &RlmIteration) {
        self.iteration_count += 1;

        let iter_value = serde_json::to_value(iteration).unwrap_or(serde_json::Value::Null);

        let mut entry = serde_json::json!({
            "type": "iteration",
            "iteration": self.iteration_count,
            "timestamp": Local::now().to_rfc3339(),
        });
        if let serde_json::Value::Object(map) = iter_value {
            for (k, v) in map {
                entry[k] = v;
            }
        }

        self.iterations.push(entry.clone());

        if self.save_to_disk {
            self.append_to_file(&entry);
        }
    }

    /// Reset iterations for the next completion (trajectory is per completion).
    pub fn clear_iterations(&mut self) {
        self.iterations.clear();
        self.iteration_count = 0;
    }

    /// Return captured trajectory (run_metadata + iterations), or `None`.
    pub fn get_trajectory(&self) -> Option<serde_json::Value> {
        self.run_metadata.as_ref().map(|metadata| {
            serde_json::json!({
                "run_metadata": metadata,
                "iterations": self.iterations,
            })
        })
    }

    /// Current iteration count.
    pub fn iteration_count(&self) -> u32 {
        self.iteration_count
    }

    /// Append a JSON entry to the log file (one line).
    fn append_to_file(&self, entry: &serde_json::Value) {
        if let Some(path) = &self.log_file_path {
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = serde_json::to_writer(&mut file, entry);
                let _ = file.write_all(b"\n");
            }
        }
    }
}
