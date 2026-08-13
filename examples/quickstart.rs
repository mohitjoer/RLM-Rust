//! Quickstart example demonstrating RLM in Rust.

use rlm::core::rlm::{Rlm, RlmConfig};
use rlm::logger::RlmLogger;
use rlm::types::ClientBackend;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file if present
    dotenvy::dotenv().ok();

    // Automatically select provider based on available API keys
    let (backend, model_name, api_key) = if let Ok(key) = std::env::var("GEMINI_API_KEY") {
        (ClientBackend::Gemini, "gemini-2.5-flash", key)
    } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        (ClientBackend::OpenAi, "gpt-4o-mini", key)
    } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        (ClientBackend::Anthropic, "claude-3-5-haiku-20241022", key)
    } else {
        println!("============================================================");
        println!("               RLM Rust Engine Quickstart                   ");
        println!("============================================================");
        println!("Please set an API key environment variable to run the demo:");
        println!("  export GEMINI_API_KEY=\"your_gemini_key\"");
        println!("  or");
        println!("  export OPENAI_API_KEY=\"your_openai_key\"");
        println!("  or");
        println!("  export ANTHROPIC_API_KEY=\"your_anthropic_key\"");
        println!("============================================================");
        return Ok(());
    };

    println!(
        "Starting RLM completion query using provider {:?} ({model_name})...",
        backend
    );

    // Generate a haystack of text with a hidden secret number
    let secret_number: u64 = 428_571_932;
    let mut filler_lines: Vec<String> = (0..1_000)
        .map(|i| format!("line {i}: padded log content in context payload"))
        .collect();
    filler_lines.insert(500, format!("SECRET_NUMBER={secret_number}"));
    let haystack = filler_lines.join("\n");

    let config = RlmConfig {
        backend,
        backend_kwargs: serde_json::json!({
            "model_name": model_name,
            "api_key": api_key,
        }),
        max_iterations: 10,
        verbose: true,
        ..Default::default()
    };

    let logger = RlmLogger::new(Some("./logs"), Some("quickstart"));
    let mut rlm = Rlm::new(config, Some(logger));

    let prompt = format!(
        "The context contains 1000 lines of text with a single line \
         matching SECRET_NUMBER=<digits>.\n\
         Find the key in context, set answer['content'] to the extracted digits, and set answer['ready'] = True.\n\n{haystack}"
    );

    let result = rlm.completion(&prompt, None).await?;

    println!("------------------------------------------------------------");
    println!("Model Output   : {}", result.response.trim());
    println!("Actual Number  : {secret_number}");
    println!(
        "Correct        : {}",
        result.response.contains(&secret_number.to_string())
    );
    println!("------------------------------------------------------------");

    Ok(())
}
