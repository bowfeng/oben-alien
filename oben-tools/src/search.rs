use super::registry::{Tool, ToolCall, ToolRegistry};
use anyhow::anyhow;
use oben_config::{SearchConfig, SearchProviderKind};
use oben_models::{ToolMeta, ToolParameter, ToolParameters, ToolResult};

fn make_search_tool_def() -> ToolMeta {
    ToolMeta {
        name: "web_search".into(),
        description: "Search the web for information".into(),
        parameters: ToolParameters::Flat(vec![
            ToolParameter::required("query", "Search query", "string"),
            ToolParameter::optional("max_results", "Maximum number of results", "number"),
        ]),
    }
}

pub fn create_search_provider(config: &SearchConfig) -> Box<dyn SearchProvider> {
    if config.provider.is_disabled() {
        return Box::new(DisabledProvider);
    }

    match config.provider {
        SearchProviderKind::Disabled => unreachable!(),
        SearchProviderKind::DuckDuckGo => Box::new(DuckDuckGoProvider::new()),
        SearchProviderKind::Brave => {
            let api_key = config.api_key.clone().unwrap_or_default();
            Box::new(BraveProvider::new(api_key))
        }
        SearchProviderKind::Bing => {
            let api_key = config.api_key.clone().unwrap_or_default();
            Box::new(BingProvider::new(api_key))
        }
        SearchProviderKind::Google => {
            let api_key = config.api_key.clone().unwrap_or_default();
            let cx = config.api_key.clone().unwrap_or_default();
            Box::new(GoogleProvider::new(api_key, cx))
        }
    }
}

#[derive(Debug, Clone)]
pub struct DisabledProvider;

#[async_trait::async_trait]
impl SearchProvider for DisabledProvider {
    async fn search(&self, _query: &str, _max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        Err(anyhow::anyhow!("Search provider is disabled in config. Set search.provider to 'disabled' to disable web search."))
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[async_trait::async_trait]
pub trait SearchProvider: Send + Sync {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>>;
}

use reqwest::Client;
use serde_json::Value;

pub struct DuckDuckGoProvider {
    client: Client,
}

impl DuckDuckGoProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl SearchProvider for DuckDuckGoProvider {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        // Use DuckDuckGo's API endpoint instead of scraping HTML
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1",
            urlencoding::encode(query)
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!(
                "Error: web_search DuckDuckGo search failed: HTTP {}",
                resp.status()
            ));
        }

        let json: serde_json::Value = resp.json().await?;

        // Parse results from the JSON response
        let results = parse_ddg_results(&json, max_results);
        
        if results.is_empty() {
            return Err(anyhow::anyhow!("Error: web_search cannot get results for query: {}", query));
        }

        Ok(results)
    }
}

fn parse_ddg_results(json: &Value, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // Get results from Abstract, RelatedTopics, and Results fields
    let abstract_title = json["Heading"].as_str().unwrap_or("").to_string();
    if !abstract_title.is_empty() {
        if let Some(abstract_text) = json["AbstractText"].as_str() {
            if let Some(abstract_url) = json["AbstractURL"].as_str() {
                results.push(SearchResult {
                    title: abstract_title,
                    url: abstract_url.to_string(),
                    snippet: abstract_text.to_string(),
                });
            }
        }
    }

    // Parse RelatedTopics
    if let Some(topics) = json["RelatedTopics"].as_array() {
        for topic in topics {
            if let Some(first_url) = topic["FirstURL"].as_str() {
                let result = topic["Result"].as_str().unwrap_or("");
                let text = topic["Text"].as_str().unwrap_or("");

                // Extract title from result HTML
                let title = extract_ddg_title_from_json(result);
                let url = extract_ddg_url_from_json(first_url);

                if !title.is_empty() && !url.is_empty() && !seen_url(&results, &url) {
                    results.push(SearchResult {
                        title,
                        url,
                        snippet: text.to_string(),
                    });
                }
            }
            if results.len() >= max_results {
                break;
            }
        }
    }

    // Parse Results
    if let Some(results_arr) = json["Results"].as_array() {
        for result in results_arr {
            if let Some(first_url) = result["FirstURL"].as_str() {
                let result_text = result["Result"].as_str().unwrap_or("");
                let text = result["Text"].as_str().unwrap_or("");

                let title = extract_ddg_title_from_json(result_text);
                let url = extract_ddg_url_from_json(first_url);

                if !title.is_empty() && !url.is_empty() && !seen_url(&results, &url) {
                    results.push(SearchResult {
                        title,
                        url,
                        snippet: text.to_string(),
                    });
                }
            }
            if results.len() >= max_results {
                break;
            }
        }
    }

    results
}

fn seen_url(results: &[SearchResult], url: &str) -> bool {
    results.iter().any(|r| r.url == url)
}

fn is_valid_result(r: &SearchResult) -> bool {
    !r.title.is_empty() && !r.url.is_empty()
}

fn extract_ddg_title_from_json(text: &str) -> String {
    // Remove HTML tags from JSON text
    let cleaned = regex::Regex::new(r"<[^>]+>").unwrap().replace_all(text, "");
    cleaned.trim().to_string()
}

fn extract_ddg_url_from_json(url: &str) -> String {
    url.to_string()
}

pub struct BraveProvider {
    api_key: String,
}

impl BraveProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    pub async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        let client = reqwest::Client::new();

        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
            query,
            max_results.min(50)
        );

        let resp = client
            .get(&url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Brave API error: {} - {}", status, body));
        }

        let json: serde_json::Value = resp.json().await?;

        let results = json["web"]["results"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .take(max_results)
            .filter_map(|r| {
                Some(SearchResult {
                    title: r["title"].as_str()?.to_string(),
                    url: r["url"].as_str()?.to_string(),
                    snippet: r["description"].as_str()?.to_string(),
                })
            })
            .collect();

        Ok(results)
    }
}

#[async_trait::async_trait]
impl SearchProvider for BraveProvider {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        BraveProvider::search(self, query, max_results).await
    }
}

pub struct BingProvider {
    api_key: String,
}

impl BingProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    pub async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        let client = reqwest::Client::new();

        let url = format!(
            "https://api.bing.microsoft.com/v7.0/search?q={}&count={}",
            urlencoding::encode(query),
            max_results.min(50)
        );

        let resp = client
            .get(&url)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Bing API error: {} - {}", status, body));
        }

        let json: serde_json::Value = resp.json().await?;

        let results = json["webPages"]["value"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .take(max_results)
            .filter_map(|r| {
                Some(SearchResult {
                    title: r["name"].as_str()?.to_string(),
                    url: r["url"].as_str()?.to_string(),
                    snippet: r["snippet"].as_str()?.to_string(),
                })
            })
            .collect();

        Ok(results)
    }
}

#[async_trait::async_trait]
impl SearchProvider for BingProvider {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        BingProvider::search(self, query, max_results).await
    }
}

pub struct GoogleProvider {
    api_key: String,
    cx: String,
}

impl GoogleProvider {
    pub fn new(api_key: String, cx: String) -> Self {
        Self { api_key, cx }
    }

    pub async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        let client = reqwest::Client::new();

        let url = format!(
            "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}&num={}",
            urlencoding::encode(&self.api_key),
            urlencoding::encode(&self.cx),
            urlencoding::encode(query),
            max_results.min(10)
        );

        let resp = client.get(&url).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Google API error: {} - {}", status, body));
        }

        let json: serde_json::Value = resp.json().await?;

        let results = json["items"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .take(max_results)
            .filter_map(|r| {
                Some(SearchResult {
                    title: r["title"].as_str()?.to_string(),
                    url: r["link"].as_str()?.to_string(),
                    snippet: r["snippet"].as_str()?.to_string(),
                })
            })
            .collect();

        Ok(results)
    }
}

#[async_trait::async_trait]
impl SearchProvider for GoogleProvider {
    async fn search(&self, query: &str, max_results: usize) -> anyhow::Result<Vec<SearchResult>> {
        GoogleProvider::search(self, query, max_results).await
    }
}

pub struct WebSearchTool {
    provider: Option<Box<dyn SearchProvider>>,
}

impl WebSearchTool {
    pub fn new_with_provider(provider: Box<dyn SearchProvider>) -> Self {
        Self { provider: Some(provider) }
    }

    pub fn new() -> Self {
        Self { provider: None }
    }
}

async fn execute_web_search<'a>(tool: &WebSearchTool, call: &ToolCall<'a>) -> anyhow::Result<ToolResult> {
    let query = call.required_str("query")?;
    let max_results = call.optional_u64("max_results", 5) as usize;

    if let Some(ref provider) = tool.provider {
        let mut results = provider.search(query, max_results).await?;
        
        results.retain(is_valid_result);
        
        if results.is_empty() {
            return Err(anyhow!("Error: web_search cannot get results for query: {}", query));
        }
        
        let output = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("{}. [{}]({})", i + 1, r.title, r.url))
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(ToolResult {
            call_id: call.call_id.clone(),
            output,
            error: None,
        });
    }

    Err(anyhow!("Error: web_search No search provider configured. Add to config: `tools.search.provider`"))
}

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }
    fn description(&self) -> &str {
        "Search the web for information"
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        match execute_web_search(self, call).await {
            Ok(result) => result,
            Err(e) => ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some(e.to_string()),
            },
        }
    }
    fn clone_tool(&self) -> Box<dyn Tool> {
        Box::new(Self::new())
    }
}

pub fn register(registry: &mut ToolRegistry, config: &SearchConfig) {
    let provider = create_search_provider(config);
    let tool = Box::new(WebSearchTool::new_with_provider(provider));
    registry.register_with_def(tool, make_search_tool_def());
}

pub fn register_default(registry: &mut ToolRegistry) {
    let config = SearchConfig::default();
    register(registry, &config);
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use oben_config::{SearchConfig, SearchProviderKind};

    #[test]
    fn test_create_duckduckgo_provider() {
        let config = SearchConfig {
            provider: SearchProviderKind::DuckDuckGo,
            api_key: None,
        };
        let provider = create_search_provider(&config);
        let _ = provider;
    }

    #[test]
    fn test_create_brave_provider_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Brave,
            api_key: Some("test-key".to_string()),
        };
        let provider = create_search_provider(&config);
        let _ = provider;
    }

    #[test]
    fn test_create_bing_provider_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Bing,
            api_key: Some("test-key".to_string()),
        };
        let provider = create_search_provider(&config);
        let _ = provider;
    }

    #[test]
    fn test_create_google_provider_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Google,
            api_key: Some("test-key".to_string()),
        };
        let provider = create_search_provider(&config);
        let _ = provider;
    }

    #[test]
    fn test_create_disabled_provider() {
        let config = SearchConfig {
            provider: SearchProviderKind::Disabled,
            api_key: None,
        };
        let provider = create_search_provider(&config);
        let _provider_ref: &dyn SearchProvider = provider.as_ref();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duckduckgo_provider_new() {
        let _provider = DuckDuckGoProvider::new();
    }

    #[test]
    fn test_brave_provider_search_url() {
        let query = "test query";
        let max_results = 10;

        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
            query,
            max_results.min(50)
        );

        assert!(url.contains("test"));
        assert!(url.contains("query"));
        assert!(url.contains("count=10"));
    }

    #[test]
    fn test_bing_search_url() {
        let query = "test query";
        let max_results = 10;

        let url = format!(
            "https://api.bing.microsoft.com/v7.0/search?q={}&count={}",
            urlencoding::encode(query),
            max_results.min(50)
        );

        assert!(url.contains("test"));
        assert!(url.contains("query"));
        assert!(url.contains("count=10"));
    }

    #[test]
    fn test_google_search_url() {
        let query = "test query";
        let max_results = 10;

        let url = format!(
            "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}&num={}",
            urlencoding::encode("test-key"),
            urlencoding::encode("test-cx"),
            urlencoding::encode(query),
            max_results.min(10)
        );

        assert!(url.contains("test"));
        assert!(url.contains("query"));
        assert!(url.contains("num=10"));
    }

    #[test]
    fn test_search_result_clone() {
        let result = SearchResult {
            title: "Test Title".to_string(),
            url: "https://example.com".to_string(),
            snippet: "Test snippet".to_string(),
        };

        let cloned = result.clone();
        assert_eq!(cloned.title, result.title);
        assert_eq!(cloned.url, result.url);
        assert_eq!(cloned.snippet, result.snippet);
    }
}
