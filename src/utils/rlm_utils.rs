//! Small utility functions.
//!
//! Port of `rlm/utils/rlm_utils.py`.

use std::collections::HashMap;

/// Filter out sensitive keys (API keys) from a kwargs map.
///
/// Removes any key whose lowercase form contains both `"api"` and `"key"`.
pub fn filter_sensitive_keys(
    kwargs: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    kwargs
        .iter()
        .filter(|(k, _)| {
            let lower = k.to_lowercase();
            !(lower.contains("api") && lower.contains("key"))
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_sensitive_keys() {
        let mut map = HashMap::new();
        map.insert("api_key".to_string(), serde_json::json!("sk-secret"));
        map.insert("model_name".to_string(), serde_json::json!("gpt-4o"));
        map.insert(
            "OPENAI_API_KEY".to_string(),
            serde_json::json!("sk-secret2"),
        );

        let filtered = filter_sensitive_keys(&map);
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("model_name"));
        assert!(!filtered.contains_key("api_key"));
        assert!(!filtered.contains_key("OPENAI_API_KEY"));
    }
}
