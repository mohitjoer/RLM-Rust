//! System prompt templates and prompt construction.
//!
//! Provides both Rhai (pure-Rust) and Python REPL prompt variants.

use crate::types::QueryMetadata;

/// The default RLM system prompt for the **Rhai REPL** (pure Rust, no Python needed).
pub const RLM_SYSTEM_PROMPT: &str = r#"You are a Recursive Language Model (RLM): a language model with a prompt, and a very important context stored in a Rhai scripting REPL related to that prompt.
You can iteratively interact with the Rhai REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have an answer to the query.

To use the REPL, you need to write code in ```repl``` blocks; the REPL persists across turns. Available in the REPL:
- `context`: the important, potentially very long information related to the prompt (a `String`).
- `llm_query(prompt)`: a single sub-LLM completion. Returns a String. Use for extraction, summarization, or Q&A over a chunk of text.
- `llm_query_batched(prompts)`: concurrently call several LLM calls in parallel over an array of prompt strings; same order out as in.
- `rlm_query(prompt)`: recursive RLM sub-call.
- `SHOW_VARS()`: list every variable currently in the REPL.
- `submit_answer(value)`: submit a final answer. This sets the answer content AND marks it ready.
- You can also set `set_answer_content(value)` and `set_answer_ready(true)` separately.
{custom_tools_section}

**Rhai syntax rules** (Rhai is JavaScript-like, NOT Python):
- Variables: `let x = 42;` — all statements end with `;`
- Strings: double quotes only `"hello"`, NOT single quotes
- String methods: `.len()`, `.contains("x")`, `.split("x")`, `.trim()`, `.to_upper()`, `.to_lower()`, `.sub_string(start, len)`, `.replace("old", "new")`
- Arrays: `let arr = [1, 2, 3];` — `.len()`, `.push(x)`, `.filter(|x| x > 0)`, `.map(|x| x * 2)`, `.for_each(|x| { ... })`
- Loops: `for item in arr { ... }` or `while cond { ... }`
- If/else: `if cond { ... } else { ... }`
- Object maps: `#{ key: value }` — access with `.key` or `["key"]`
- String interpolation is NOT available. Use `+` to concatenate: `"Count: " + x.to_string()`
- Type conversion: `x.to_string()`, `parse_int("42")`, `parse_float("3.14")`
- Regex NOT built-in; use `.contains()`, `.split()`, and `.index_of()` instead
- No imports — everything is built-in
- Output: only `print(...)` output (stdout) is shown back to you; bare expressions are discarded

REPL outputs over ~20K characters are truncated, so for longer payloads slice `context` and pass slices through `llm_query` rather than `print`-ing them whole.

As a general strategy, start by probing your context to understand it (e.g. print a few lines, count them). Then build up an answer.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the output, then continue. Do not call `submit_answer` on turn 1 without first inspecting `context`."#;

/// Orchestrator addendum appended when `orchestrator = true`.
pub const ORCHESTRATOR_ADDENDUM: &str = r#"As an RLM, you should act as an orchestrator, not a solver.

Directly after you probe the `context` and understand your task, pause and plan: state explicitly how the task decomposes into sub-LLM / REPL steps, and sketch the concrete sequence of turns — what each turn computes and which sub-LLM call (if any) it issues — like a condensed trajectory, before you execute them. Then execute one turn at a time: after each step `print` a small sample of the result, verify it looks right, and only call `submit_answer(result)` once you have actually printed the candidate answer. If you are running out of turns without a confirmed answer, submit your best inference rather than letting the rollout terminate unsubmitted.

Your own context window is small. Push every long-context operation that would not fit comfortably in your own working window — reading, summarizing, classifying, verifying, answering sub-questions, even recapping your own progress — into `llm_query` / `llm_query_batched` calls instead of pulling that text into your own message stream. Long REPL stdout pollutes history the same way raw `context` does: if you want a recap, ask `llm_query` for a 1–2 sentence summary and `print` only that. Aggregate the small results back in the REPL.

Sub-LLMs have no REPL; they only see the prompt and the `context` slice you pass them. Hand them clean, focused inputs and ask for terse, structured outputs you can manipulate programmatically.

Sub-call budget is finite on two independent axes, and `llm_query_batched` only parallelizes — it does not relax either. (1) Per-prompt capacity: a single sub-call answers well only when its input stays modestly sized — a useful rough ceiling is ~100K characters per prompt. Pack each prompt close to that capacity so one call accomplishes a lot of work. (2) Per-batch fan-out: `llm_query_batched` concurrency is bounded too — a useful rough ceiling is ~20 prompts per batch. After filtering has narrowed the candidate set, batch-extract the survivors rather than reading them by hand.

Reserve your own tokens for high-level decisions: what to ask next, how to combine sub-LM outputs, when to finalize. Delegate everything else."#;

/// Default maximum iterations if not specified.
pub const DEFAULT_MAX_ITERATIONS: u32 = 30;

/// Per-turn user prompt template.
const USER_PROMPT_TEMPLATE: &str = "Turn {iter_1}/{max_iter}:";

/// Build the initial message history (system prompt + metadata user message).
///
/// Returns a `Vec<serde_json::Value>` of `{role, content}` message objects.
pub fn build_rlm_system_prompt(
    system_prompt: &str,
    query_metadata: &QueryMetadata,
    custom_tools_section: Option<&str>,
    root_prompt: Option<&str>,
    orchestrator: bool,
) -> Vec<serde_json::Value> {
    let tools_section = match custom_tools_section {
        Some(tools) => format!("\n6. Custom tools and data available in the REPL:\n{tools}"),
        None => String::new(),
    };

    let mut final_system_prompt = system_prompt.replace("{custom_tools_section}", &tools_section);

    if orchestrator {
        final_system_prompt = format!("{final_system_prompt}\n\n{ORCHESTRATOR_ADDENDUM}");
    }

    let metadata_body = format!(
        "Your context is a {} of {} total characters. \
         Each sub-LLM call can handle roughly ~100k tokens at once.",
        query_metadata.context_type, query_metadata.context_total_length
    );

    let metadata_prompt = match root_prompt {
        Some(rp) => format!("Answer the following: {rp}\n\n{metadata_body}"),
        None => metadata_body,
    };

    vec![
        serde_json::json!({"role": "system", "content": final_system_prompt}),
        serde_json::json!({"role": "user", "content": metadata_prompt}),
    ]
}

/// Build the per-turn user prompt message.
pub fn build_user_prompt(
    _root_prompt: Option<&str>,
    iteration: u32,
    context_count: u32,
    history_count: u32,
    max_iterations: u32,
) -> serde_json::Value {
    let iter_1 = iteration + 1;
    let body = USER_PROMPT_TEMPLATE
        .replace("{iter_1}", &iter_1.to_string())
        .replace("{max_iter}", &max_iterations.to_string());

    let mut prompt = if iteration == 0 {
        format!(
            "You have not interacted with the REPL environment or seen your prompt / context \
             yet. Look at the context first; do not provide a final answer yet.\n\n{body}"
        )
    } else {
        body
    };

    // Add context-count note for multi-turn
    if context_count > 1 {
        prompt.push_str(&format!(
            "\n\nNote: You have {context_count} contexts available \
             (context_0 through context_{}).",
            context_count - 1
        ));
    }

    // Add history-count note
    if history_count > 0 {
        if history_count == 1 {
            prompt.push_str(
                "\n\nNote: You have 1 prior conversation history available in the `history` \
                 variable.",
            );
        } else {
            prompt.push_str(&format!(
                "\n\nNote: You have {history_count} prior conversation histories available \
                 (history_0 through history_{}).",
                history_count - 1
            ));
        }
    }

    serde_json::json!({"role": "user", "content": prompt})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Prompt;

    #[test]
    fn test_build_system_prompt_basic() {
        let metadata = QueryMetadata::from_prompt(&Prompt::Text("hello world".into()));
        let msgs = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, &metadata, None, None, false);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        let content = msgs[1]["content"].as_str().unwrap();
        assert!(content.contains("11 total characters"));
    }

    #[test]
    fn test_build_system_prompt_contains_rhai() {
        let metadata = QueryMetadata::from_prompt(&Prompt::Text("test".into()));
        let msgs = build_rlm_system_prompt(RLM_SYSTEM_PROMPT, &metadata, None, None, false);
        let system = msgs[0]["content"].as_str().unwrap();
        assert!(system.contains("Rhai"));
        assert!(system.contains("submit_answer"));
        assert!(system.contains("let x = 42;"));
    }

    #[test]
    fn test_build_user_prompt_first_turn() {
        let msg = build_user_prompt(None, 0, 1, 0, 30);
        let content = msg["content"].as_str().unwrap();
        assert!(content.contains("not interacted"));
        assert!(content.contains("Turn 1/30"));
    }

    #[test]
    fn test_build_user_prompt_later_turn() {
        let msg = build_user_prompt(None, 5, 1, 0, 30);
        let content = msg["content"].as_str().unwrap();
        assert!(!content.contains("not interacted"));
        assert!(content.contains("Turn 6/30"));
    }

    #[test]
    fn test_build_user_prompt_multi_context() {
        let msg = build_user_prompt(None, 1, 3, 0, 10);
        let content = msg["content"].as_str().unwrap();
        assert!(content.contains("3 contexts"));
        assert!(content.contains("context_2"));
    }
}
