//! Code block extraction and iteration formatting.
//!
//! Port of `rlm/utils/parsing.py`.

use regex::Regex;
use std::sync::LazyLock;

use crate::types::{ReplResult, RlmIteration};

/// Compiled regex for extracting ` ```repl ` fenced code blocks.
static CODE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```repl\s*\n(.*?)\n```").unwrap());

/// Find all REPL code blocks in text wrapped in ` ```repl ` fences.
///
/// Returns a list of the code content (without the fences).
pub fn find_code_blocks(text: &str) -> Vec<String> {
    CODE_BLOCK_RE
        .captures_iter(text)
        .map(|cap| cap[1].trim().to_string())
        .collect()
}

/// Maximum characters per code-block output before truncation.
const MAX_CHARACTER_LENGTH: usize = 20_000;

/// Format an RLM iteration for appending to message history.
///
/// Each iteration produces exactly two messages: one assistant turn containing
/// the model's response, followed by a single user message that concatenates
/// the outputs of all executed code blocks.
pub fn format_iteration(iteration: &RlmIteration) -> Vec<serde_json::Value> {
    let mut messages = vec![serde_json::json!({
        "role": "assistant",
        "content": iteration.response,
    })];

    let multi = iteration.code_blocks.len() > 1;
    let mut parts: Vec<String> = Vec::new();

    for (i, code_block) in iteration.code_blocks.iter().enumerate() {
        let mut result = format_execution_result(&code_block.result);
        if result.len() > MAX_CHARACTER_LENGTH {
            let overflow = result.len() - MAX_CHARACTER_LENGTH;
            result.truncate(MAX_CHARACTER_LENGTH);
            result.push_str(&format!("... + [{overflow} chars...]"));
        }
        let header = if multi {
            format!("REPL output (block {}):", i + 1)
        } else {
            "REPL output:".to_string()
        };
        parts.push(format!("{header}\n{result}"));
    }

    if !parts.is_empty() {
        messages.push(serde_json::json!({
            "role": "user",
            "content": parts.join("\n\n"),
        }));
    }

    messages
}

/// Format a single REPL execution result as a string for display.
pub fn format_execution_result(result: &ReplResult) -> String {
    let mut parts: Vec<String> = Vec::new();

    if !result.stdout.is_empty() {
        parts.push(format!("\n{}", result.stdout));
    }

    if !result.stderr.is_empty() {
        parts.push(format!("\n{}", result.stderr));
    }

    // Show variable names (excluding internal ones)
    let important_vars: Vec<&String> = result
        .locals
        .keys()
        .filter(|k| {
            !k.starts_with('_') && !matches!(k.as_str(), "__builtins__" | "__name__" | "__doc__")
        })
        .collect();

    if !important_vars.is_empty() {
        let var_names: Vec<&str> = important_vars.iter().map(|s| s.as_str()).collect();
        parts.push(format!("REPL variables: {var_names:?}\n"));
    }

    if parts.is_empty() {
        "No output".to_string()
    } else {
        parts.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_code_blocks_single() {
        let text = "Some text\n```repl\nprint('hello')\n```\nmore text";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks, vec!["print('hello')"]);
    }

    #[test]
    fn test_find_code_blocks_multiple() {
        let text = "```repl\nx = 1\n```\nText\n```repl\ny = 2\n```";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks, vec!["x = 1", "y = 2"]);
    }

    #[test]
    fn test_find_code_blocks_none() {
        let text = "No code blocks here\n```python\nprint('hi')\n```";
        let blocks = find_code_blocks(text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_find_code_blocks_multiline() {
        let text = "```repl\nimport math\nx = math.sqrt(2)\nprint(x)\n```";
        let blocks = find_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("import math"));
        assert!(blocks[0].contains("print(x)"));
    }
}
