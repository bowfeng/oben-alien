use reqwest::Client;
use scraper::{Html, Selector};

use super::registry::{Tool, ToolCall, ToolRegistry};
use oben_models::{ToolMeta, ToolParameter, ToolParameters, ToolResult};

// ---------------------------------------------------------------------------
// SSRF protection — block private/internal URLs
// ---------------------------------------------------------------------------

/// Check if a URL is safe (not pointing to private/internal networks).
fn is_safe_url(url: &str) -> bool {
    let url = url.trim();

    // Block empty or malformed URLs
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return false;
    }

    // Extract host from URL
    let host = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .and_then(|u| u.split('/').next())
        .unwrap_or("");

    // Remove port if present
    let host = host.split(':').next().unwrap_or("");

    // Block localhost and internal names first
    if host == "localhost"
        || host == "127.0.0.1"
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".corp")
        || host.ends_with(".home")
    {
        return false;
    }

    // Block private IP ranges
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return true;
    }

    // Check for private IP ranges (RFC 1918)
    let first = parts[0].parse::<u8>().unwrap_or(0);
    let second = parts[1].parse::<u8>().unwrap_or(0);

    // 10.0.0.0/8
    if first == 10 {
        return false;
    }

    // 172.16.0.0/12
    if first == 172 && second >= 16 && second <= 31 {
        return false;
    }

    // 192.168.0.0/16
    if first == 192 && second == 168 {
        return false;
    }

    // 127.0.0.0/8 (loopback)
    if first == 127 {
        return false;
    }

    // 169.254.0.0/16 (link-local)
    if first == 169 && second == 254 {
        return false;
    }

    // 0.0.0.0
    if first == 0 {
        return false;
    }

    // Block localhost and internal names
    if host == "localhost"
        || host == "127.0.0.1"
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".corp")
        || host.ends_with(".home")
    {
        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// HTML extraction
// ---------------------------------------------------------------------------

/// Extract readable content from HTML.
fn extract_content(html: &str, max_length: usize) -> String {
    let document = Html::parse_document(html);

    // Try to extract article/primary content first
    if let Ok(article_selector) = Selector::parse("article") {
        if let Some(el) = document.select(&article_selector).next() {
            return extract_text(&el, max_length);
        }
    }

    if let Ok(primary_selector) = Selector::parse("main") {
        if let Some(el) = document.select(&primary_selector).next() {
            return extract_text(&el, max_length);
        }
    }

    // Fallback: extract all text
    extract_all_text(&document, max_length)
}

/// Extract text from an element.
fn extract_text(element: &scraper::ElementRef<'_>, max_length: usize) -> String {
    let text: Vec<&str> = element.text().collect();
    let joined = text.join(" ");

    if joined.len() > max_length {
        format!(
            "{}... ({} chars total)",
            &joined[..max_length.min(joined.len())],
            joined.len()
        )
    } else {
        joined.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Extract all text from a document by walking the DOM tree.
fn extract_all_text(document: &scraper::Html, max_length: usize) -> String {
    use scraper::node::Node;

    let mut text_parts = Vec::new();

    // Walk through all nodes in the document
    for node in document.root_element().descendants() {
        if let Node::Text(text_node) = node.value() {
            let trimmed = text_node.text.trim();
            if !trimmed.is_empty() {
                text_parts.push(trimmed);
            }
        }
    }

    let joined = text_parts.join(" ");

    if joined.len() > max_length {
        format!(
            "{}... ({} chars total)",
            &joined[..max_length.min(joined.len())],
            joined.len()
        )
    } else {
        joined.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Extract title and content from HTML.
fn extract_page(html: &str) -> (String, String) {
    let document = Html::parse_document(html);

    let title = if let Ok(title_selector) = Selector::parse("title") {
        document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_else(|| "(no title)".to_string())
    } else {
        "(no title)".to_string()
    };

    let content = extract_content(html, 10000);

    (title, content)
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

fn make_web_extract_tool_def() -> ToolMeta {
    ToolMeta {
        name: "web_extract".into(),
        description: "Fetches a web page and extracts text or structured data from it".into(),
        parameters: ToolParameters::Flat(vec![
            ToolParameter::required("url", "The URL of the web page to extract", "string"),
            ToolParameter::optional("extract_type", "Type of extraction: `all`, `text`, `links`, etc. (default: all)", "string"),
        ]),
    }
}// ---------------------------------------------------------------------------
// Tool struct
// ---------------------------------------------------------------------------

pub struct WebExtractTool;

/// Extract readable content from a web page URL.
async fn execute_web_extract<'a>(call: &ToolCall<'a>) -> anyhow::Result<ToolResult> {
    let url = call.required_str("url")?;
    let format_type = call.optional_str_with_default("format", "text");

    // SSRF protection
    if !is_safe_url(url) {
        return Ok(ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(
                "Blocked: URL targets a private or internal network address".to_string(),
            ),
        });
    }

    let client = Client::new();
    let response = match client
        .get(url)
        .header("User-Agent", "ObenAgent/1.0 (web extract tool)")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return Ok(ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some(format!("Failed to fetch {}: {}", url, e)),
            });
        }
    };

    let status = response.status();
    if !status.is_success() {
        return Ok(ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(format!("HTTP {} fetching {}", status, url)),
        });
    }

    let html = match response.text().await {
        Ok(h) => h,
        Err(e) => {
            return Ok(ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some(format!("Failed to read response: {}", e)),
            });
        }
    };

    let (title, content) = extract_page(&html);

    // Return error if no readable content found
    if content.trim().is_empty() {
        return Ok(ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(format!("No readable content found in {}", url)),
        });
    }

    Ok(ToolResult {
        call_id: call.call_id.clone(),
        output: if format_type == "markdown" {
            let stripped_html: String = content
                .chars()
                .scan(false, |in_tag, c| match c {
                    '<' => {
                        *in_tag = true;
                        None
                    }
                    '>' => {
                        *in_tag = false;
                        None
                    }
                    _ => Some(if *in_tag { ' ' } else { c }),
                })
                .collect();
            let body_content: String = stripped_html
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            format!("Title: {}\nURL: {}\n\n{}", title, url, body_content)
        } else {
            format!("Title: {}\nURL: {}\n\n{}", title, url, content)
        },
        error: None,
    })
}

#[async_trait::async_trait]
impl Tool for WebExtractTool {
    fn name(&self) -> &str {
        "web_extract"
    }
    fn description(&self) -> &str {
        "Extract readable content from web pages"
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_web_extract(call).await.unwrap_or_else(|e| ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(e.to_string()),
        })
    }
    fn clone_tool(&self) -> Box<dyn Tool> {
        Box::new(Self)
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register this module into the given registry.
/// Called automatically by `discover_builtin_tools`.
pub fn register(registry: &mut ToolRegistry) {
    let tool = Box::new(WebExtractTool);
    registry.register_with_def(tool, make_web_extract_tool_def());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_registry() -> super::super::registry::ToolRegistry {
        let mut registry = super::super::registry::ToolRegistry::new();
        register(&mut registry);
        registry
    }

    #[test]
    fn test_safe_urls() {
        // Public URLs should pass
        assert!(is_safe_url("https://example.com"));
        assert!(is_safe_url("http://google.com/search"));
        assert!(is_safe_url("https://github.com/owner/repo"));

        // Private IPs should be blocked
        assert!(!is_safe_url("http://192.168.1.1"));
        assert!(!is_safe_url("https://10.0.0.1/admin"));
        assert!(!is_safe_url("http://172.16.0.1"));
        assert!(!is_safe_url("http://127.0.0.1:8080"));
        assert!(!is_safe_url("http://localhost:3000"));
        assert!(!is_safe_url("http://169.254.169.254/latest/meta-data/"));
    }

    #[test]
    fn test_extract_content_from_html() {
        let html = r#"
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <article>
                        <h1>Article Title</h1>
                        <p>This is the main content.</p>
                        <p>More content here.</p>
                    </article>
                </body>
            </html>
        "#;

        let (title, content) = extract_page(html);
        assert_eq!(title, "Test Page");
        assert!(content.contains("Article Title"));
        assert!(content.contains("main content"));
    }

    #[test]
    fn test_extract_without_article_tag() {
        let html = r#"
            <html>
                <head><title>Simple Page</title></head>
                <body>
                    <div>Simple content</div>
                </body>
            </html>
        "#;

        let (title, content) = extract_page(html);
        assert_eq!(title, "Simple Page");
        assert!(content.contains("Simple content"));
    }

    #[test]
    fn test_extract_truncates_long_content() {
        let content = "x".repeat(15000);
        let html = format!(
            r#"<html><head><title>Long</title></head><body><div>{}</div></body></html>"#,
            content
        );

        let (title, text) = extract_page(&html);
        assert_eq!(title, "Long");
        assert!(text.contains("..."));
        assert!(text.len() < 20000); // Should be truncated
    }

    #[tokio::test]
    async fn extracts_valid_public_url() {
        let registry = make_registry();
        let result = registry
            .execute(
                "web_extract",
                &json!({
                    "url": "https://example.com",
                    "call_id": "test-1",
                }),
            )
            .await;

        // Should not error
        assert!(result.error.is_none());
        assert!(result.output.contains("Title:"));
    }

    #[tokio::test]
    async fn blocks_private_ip() {
        let registry = make_registry();
        let result = registry
            .execute(
                "web_extract",
                &json!({
                    "url": "http://192.168.1.1/admin",
                    "call_id": "test-2",
                }),
            )
            .await;

        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("Blocked"));
    }

    #[tokio::test]
    async fn blocks_localhost() {
        let registry = make_registry();
        let result = registry
            .execute(
                "web_extract",
                &json!({
                    "url": "http://localhost:3000/api",
                    "call_id": "test-3",
                }),
            )
            .await;

        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("Blocked"));
    }

    #[tokio::test]
    async fn handles_missing_url_arg() {
        let registry = make_registry();
        let result = registry
            .execute(
                "web_extract",
                &json!({
                    "call_id": "test-4",
                }),
            )
            .await;

        assert!(result.error.is_some());
        assert!(result.error.as_ref().unwrap().contains("Missing required argument: 'url'"));
    }

    #[tokio::test]
    async fn formats_markdown() {
        let registry = make_registry();
        let result = registry
            .execute(
                "web_extract",
                &json!({
                    "url": "https://example.com",
                    "format": "markdown",
                    "call_id": "test-5",
                }),
            )
            .await;

        assert!(result.error.is_none());
        // Markdown format should still work
        assert!(result.output.contains("Title:"));
    }
}
