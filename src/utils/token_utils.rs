//! Token counting and model context limit lookup.
//!
//! Port of `rlm/utils/token_utils.py`.
//!
//! Uses character-based estimation (~4 chars per token) since we don't have
//! tiktoken in Rust. This matches the Python fallback behaviour.

use std::sync::LazyLock;

/// Default context limit when model is unknown (tokens).
pub const DEFAULT_CONTEXT_LIMIT: u64 = 128_000;

/// Characters per token when tokenizer is unavailable (conservative estimate).
const CHARS_PER_TOKEN_ESTIMATE: u64 = 4;

/// Model context limits (max input context in tokens).
///
/// Match: key contained in model_name (longest matching key wins).
static MODEL_CONTEXT_LIMITS: LazyLock<Vec<(&'static str, u64)>> = LazyLock::new(|| {
    vec![
        // OpenAI
        ("gpt-5-nano", 272_000),
        ("gpt-5", 272_000),
        ("gpt-4o-mini", 128_000),
        ("gpt-4o-2024", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-4-turbo-preview", 128_000),
        ("gpt-4-turbo", 128_000),
        ("gpt-4-32k", 32_768),
        ("gpt-4", 8_192),
        ("gpt-3.5-turbo-16k", 16_385),
        ("gpt-3.5-turbo", 16_385),
        ("o1-mini", 128_000),
        ("o1-preview", 128_000),
        ("o1", 200_000),
        // Anthropic
        ("claude-3-5-sonnet", 200_000),
        ("claude-3-5-haiku", 200_000),
        ("claude-3-opus", 200_000),
        ("claude-3-sonnet", 200_000),
        ("claude-3-haiku", 200_000),
        ("claude-2.1", 200_000),
        ("claude-2", 100_000),
        // Gemini
        ("gemini-2.5-flash", 1_000_000),
        ("gemini-2.5-pro", 1_000_000),
        ("gemini-2.0-flash", 1_000_000),
        ("gemini-1.5-pro", 1_000_000),
        ("gemini-1.5-flash", 1_000_000),
        ("gemini-1.0-pro", 30_720),
        // Qwen
        ("qwen3-max", 256_000),
        ("qwen3-72b", 128_000),
        ("qwen3-32b", 128_000),
        ("qwen3-8b", 32_768),
        ("qwen3", 128_000),
        // Kimi
        ("kimi-k2.5", 262_000),
        ("kimi-k2-0905", 256_000),
        ("kimi-k2-thinking", 256_000),
        ("kimi-k2", 128_000),
        ("kimi", 128_000),
        // GLM
        ("glm-4.6", 200_000),
        ("glm-4-9b", 1_000_000),
        ("glm-4", 128_000),
        ("glm", 128_000),
    ]
});

/// Return max context size in tokens for a model.
///
/// Matches when the dict key is contained in `model_name`
/// (e.g. `"gpt-4o"` matches `"@openai/gpt-4o"`). Longest matching key wins.
/// Falls back to [`DEFAULT_CONTEXT_LIMIT`] for unknown models.
pub fn get_context_limit(model_name: &str) -> u64 {
    if model_name.is_empty() || model_name == "unknown" {
        return DEFAULT_CONTEXT_LIMIT;
    }

    // Look for exact match first
    for &(key, limit) in MODEL_CONTEXT_LIMITS.iter() {
        if key == model_name {
            return limit;
        }
    }

    // Substring match — longest key wins
    let mut best_len = 0;
    let mut best_limit = DEFAULT_CONTEXT_LIMIT;
    for &(key, limit) in MODEL_CONTEXT_LIMITS.iter() {
        if model_name.contains(key) && key.len() > best_len {
            best_len = key.len();
            best_limit = limit;
        }
    }

    best_limit
}

/// Count tokens in a list of chat messages.
///
/// Uses character length / `CHARS_PER_TOKEN_ESTIMATE` as a rough estimation.
pub fn count_tokens(messages: &[serde_json::Value], _model_name: &str) -> u64 {
    if messages.is_empty() {
        return 0;
    }

    let total_chars: u64 = messages
        .iter()
        .map(|m| {
            let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            content.len() as u64
        })
        .sum();

    total_chars.div_ceil(CHARS_PER_TOKEN_ESTIMATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_model() {
        assert_eq!(get_context_limit("gpt-4o"), 128_000);
    }

    #[test]
    fn test_substring_match() {
        assert_eq!(get_context_limit("@openai/gpt-4o"), 128_000);
    }

    #[test]
    fn test_longest_match_wins() {
        // "gpt-4o-mini" should match before "gpt-4o"
        assert_eq!(get_context_limit("gpt-4o-mini"), 128_000);
        // "gpt-4-turbo" should match before "gpt-4"
        assert_eq!(get_context_limit("gpt-4-turbo"), 128_000);
    }

    #[test]
    fn test_unknown_model() {
        assert_eq!(
            get_context_limit("some-random-model"),
            DEFAULT_CONTEXT_LIMIT
        );
    }

    #[test]
    fn test_empty_model() {
        assert_eq!(get_context_limit(""), DEFAULT_CONTEXT_LIMIT);
    }

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(count_tokens(&[], "gpt-4o"), 0);
    }

    #[test]
    fn test_count_tokens_basic() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello world"})];
        // "hello world" = 11 chars -> ceil(11/4) = 3 tokens
        assert_eq!(count_tokens(&msgs, "gpt-4o"), 3);
    }
}
