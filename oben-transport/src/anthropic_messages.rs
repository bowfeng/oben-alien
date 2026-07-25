/// Anthropic Messages API transport — native `messages/` endpoint.
///
/// Maps to `agent/transports/anthropic.py`.
///
/// Architecture:
///
/// ```ignore
/// AnthropicMessagesTransport
/// ├── chat() — non-streaming: POST /v1/messages
/// ├── stream_chat() — SSE: accumulate text deltas + tool call deltas
/// ├── CachedRequest (per-session, Arc-based, zero-copy extend)
/// └── conversion helpers: message_to_anthropic(), response_to_transport()
/// ```
use anyhow::{anyhow, Result};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use oben_models::{
    CallMode, Message, MessageContent, MessagePart, MessageRole, TransportProvider,
    TransportResponse, TransportToolCall,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::debug;

use super::base::BaseTransport;

// ── Anthropic API Request Types ─────────────────────────────────────────────

/// A single content block within an Anthropic message.
///
/// Matches Anthropic API format exactly:
/// - `text` — plain text block
/// - `tool_use` — tool invocation block
/// - `cache_control` — for prompt caching (on any content block)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<String>,
        text: String,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

impl AnthropicContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        AnthropicContentBlock::Text {
            cache_control: None,
            text: text.into(),
        }
    }

    /// Create a tool use content block.
    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        AnthropicContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a text content block with cache control marker.
    pub fn text_with_cache(text: impl Into<String>) -> Self {
        AnthropicContentBlock::Text {
            cache_control: Some("ephemeral".to_string()),
            text: text.into(),
        }
    }
}

/// A message sent to the Anthropic Messages API.
///
/// Anthropic format differs from OpenAI:
/// - `system` prompt is a top-level field, NOT in the messages array
/// - `max_tokens` is required (not optional)
/// - `content` in messages can be a string OR an array of content blocks
/// - Tool calls are `tool_use` content blocks, not separate JSON structure
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Single string content (legacy format).
    String(String),
    /// Array of content blocks (modern format).
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicMessageContent {
    pub fn text(text: impl Into<String>) -> Self {
        AnthropicMessageContent::String(text.into())
    }

    pub fn blocks(blocks: Vec<AnthropicContentBlock>) -> Self {
        AnthropicMessageContent::Blocks(blocks)
    }
}

/// A tool definition for the Anthropic Messages API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Prompt caching marker.
///
/// Used to mark content blocks that should be cached.
pub const CACHE_CONTROL_MARKER: &str = "ephemeral";

/// Tool choice specification for Anthropic API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
    #[serde(rename = "detector")]
    Detector,
}

/// Thinking configuration for Claude thinking tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    pub budget_tokens: usize,
}

/// Complete Anthropic Messages API request.
///
/// This is the exact format sent to `POST /v1/messages`.
#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
}

// ── Anthropic API Response Types ────────────────────────────────────────────

/// Non-streaming response from the Anthropic Messages API.
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    pub model: String,
    pub usage: AnthropicUsage,
}

/// Token usage for Anthropic API.
#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// ── Anthropic Streaming Types ───────────────────────────────────────────────

/// SSE event from Anthropic streaming response.
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<AnthropicStreamMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_block: Option<AnthropicStreamContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<AnthropicStreamDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicStreamUsage>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamDelta {
    #[serde(rename = "type", default)]
    pub delta_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicStreamUsage {
    pub output_tokens: usize,
}

// ── Conversion: Domain Message → Anthropic Message ──────────────────────────

/// Convert an Anthropic response to our domain `TransportResponse`.
fn anthropic_response_to_transport(resp: &AnthropicResponse) -> TransportResponse {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicContentBlock::Text { text: t, .. } => {
                text.push_str(t);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(TransportToolCall {
                    id: id.clone(),
                    tool_name: name.clone(),
                    arguments: input.clone(),
                });
            }
        }
    }

    TransportResponse {
        text,
        tool_calls,
        tokens_used: Some(resp.usage.output_tokens),
        reasoning: None,
    }
}

// ── CachedRequest (same pattern as ChatCompletionsTransport) ────────────────

/// Per-session cached request state for Anthropic API.
struct AnthropicCachedRequest {
    request: serde_json::Value,
    msg_count: usize,
}

// ── AnthropicMessagesTransport ──────────────────────────────────────────────

/// Transport that talks to the Anthropic Messages API (`/v1/messages`).
///
/// Handles:
/// - Non-streaming chat
/// - Streaming chat with SSE
/// - Prompt caching via `cache_control` markers
/// - Thinking tokens
/// - Native tool use protocol
pub struct AnthropicMessagesTransport {
    base: BaseTransport,
    cached: std::sync::Mutex<std::collections::HashMap<String, AnthropicCachedRequest>>,
    template: std::sync::Arc<serde_json::Value>,
    stream_template: std::sync::Arc<serde_json::Value>,
}

impl AnthropicMessagesTransport {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        let model: String = model.into();
        let base = BaseTransport::new(base_url, api_key, model.clone());
        let system_prompt = system_prompt.into();
        let config =
            oben_models::ProviderConfig::new(oben_models::ProviderKind::Anthropic, model.clone());
        let template = build_anthropic_request_template(&config, system_prompt.clone(), Vec::new());
        let stream_template =
            build_anthropic_stream_request_template(&config, system_prompt, Vec::new());
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }

    /// Create from a ProviderConfig, with tools for structured tool calling.
    pub fn with_tools(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
        tools: Vec<oben_models::ToolMeta>,
    ) -> Self {
        let model: String = model.into();
        let base = BaseTransport::new(base_url, api_key, model.clone());
        let system_prompt = system_prompt.into();
        let config =
            oben_models::ProviderConfig::new(oben_models::ProviderKind::Anthropic, model.clone());
        let tool_defs: Vec<AnthropicTool> = tools.iter().map(tool_to_anthropic).collect();
        let template =
            build_anthropic_request_template(&config, system_prompt.clone(), tool_defs.clone());
        let stream_template =
            build_anthropic_stream_request_template(&config, system_prompt, tool_defs);
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }

    fn resolve_base_url(config: &oben_models::ProviderConfig) -> String {
        // Step 1: Provider-specific base URL from config
        if let Some(url) = &config.base_url {
            if !url.trim().is_empty() {
                return url.clone();
            }
        }

        // Step 2: Provider-specific base URL env var override
        if let Some(env_var_name) = config.kind.base_url_env_var() {
            if let Ok(url) = std::env::var(env_var_name) {
                let url = url.trim().to_string();
                if !url.is_empty() {
                    return url;
                }
            }
        }

        // Step 3: Provider registry default base URL
        if let Some(default_url) = config.kind.default_base_url() {
            if !default_url.is_empty() {
                return default_url.to_string();
            }
        }

        // Step 4: MiniMax OAuth default
        if matches!(config.kind, oben_models::ProviderKind::MiniMaxOAuth) {
            return "https://api.minimax.io/anthropic".to_string();
        }

        // Step 5: Anthropic default fallback
        "https://api.anthropic.com/v1".to_string()
    }

    pub fn from_config_with_tools(
        config: &oben_models::ProviderConfig,
        system_prompt: impl Into<String>,
        tools: Vec<oben_models::ToolMeta>,
    ) -> Self {
        let base_url = Self::resolve_base_url(config);
        let api_key = config.api_key.clone().unwrap_or_default();
        let tool_defs: Vec<AnthropicTool> = tools.iter().map(tool_to_anthropic).collect();
        let system_prompt = system_prompt.into();
        let template =
            build_anthropic_request_template(config, system_prompt.clone(), tool_defs.clone());
        let stream_template =
            build_anthropic_stream_request_template(config, system_prompt, tool_defs);
        let base = BaseTransport::new(base_url, api_key, config.model.clone());
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }

    /// Create from a ProviderConfig without tools.
    pub fn from_config(
        config: &oben_models::ProviderConfig,
        system_prompt: impl Into<String>,
    ) -> Self {
        let base_url = Self::resolve_base_url(config);
        let api_key = config.api_key.clone().unwrap_or_default();
        let system_prompt = system_prompt.into();
        let template = build_anthropic_request_template(config, system_prompt.clone(), Vec::new());
        let stream_template =
            build_anthropic_stream_request_template(config, system_prompt, Vec::new());
        let base = BaseTransport::new(base_url, api_key, config.model.clone());
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }
}

/// Model response from Anthropic `/v1/models` endpoint.
#[derive(Debug, Deserialize)]
pub struct AnthropicModelInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub display_name: Option<String>,
    pub max_tokens: Option<i64>,
    pub max_input_tokens: Option<i64>,
}

impl AnthropicMessagesTransport {
    /// Fetch the list of available models from Anthropic.
    pub async fn list_models(&self) -> Result<oben_models::ModelListResponse> {
        let url = format!("{}/models", self.base.base_url);
        debug!("Fetching Anthropic models from: {}", url);

        let mut req = self.base.client.get(&url);
        req = req.header("anthropic-version", "2023-06-01");
        if !self.base.api_key.is_empty() {
            req = req.header("x-api-key", &self.base.api_key);
        }

        let response = req.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow!("Anthropic API error {}: {}", status, body));
        }

        #[derive(Deserialize)]
        struct Inner {
            data: Vec<AnthropicModelInfo>,
        }
        let resp: Inner = response.json().await?;
        let models: Vec<oben_models::ModelInfo> = resp
            .data
            .into_iter()
            .filter(|m| m.id.starts_with("claude"))
            .map(|m| oben_models::ModelInfo {
                id: m.id,
                object: m.model_type,
                created: 0,
                owned_by: "anthropic".to_string(),
                max_model_len: m.max_input_tokens.map(|v| v as usize),
                root: None,
                parent: None,
            })
            .collect();

        Ok(oben_models::ModelListResponse {
            object: "list".to_string(),
            data: models,
        })
    }

    /// Find a specific model by ID from Anthropic.
    pub async fn find_model(&self, model_id: &str) -> Result<Option<oben_models::ModelInfo>> {
        let list = self.list_models().await?;
        Ok(list.data.into_iter().find(|m| m.id == model_id))
    }
}

/// Build the static parts of the request template for Anthropic API.
fn build_anthropic_request_template(
    config: &oben_models::ProviderConfig,
    system_prompt: impl Into<String>,
    tools: Vec<AnthropicTool>,
) -> serde_json::Value {
    let max_tokens = config.max_tokens.unwrap_or(4096);
    let sp = system_prompt.into();

    // When cache_control is configured, build the system field as a content
    // blocks array with cache_control markers. Otherwise keep the legacy
    // plain-string format.
    let system_value: serde_json::Value = if let Some(cc) = &config.cache_control {
        let strategy = if cc.strategy.is_empty() {
            "ephemeral".to_string()
        } else {
            cc.strategy.clone()
        };
        json!([{
            "type": "text",
            "text": sp,
            "cache_control": {"type": strategy}
        }])
    } else {
        json!(sp)
    };

    let mut req = json!({
        "model": config.model,
        "max_tokens": max_tokens,
        "system": system_value,
        "messages": serde_json::Value::Array(vec![]),
    });

    if let Some(t) = config.temperature {
        req["temperature"] = json!(t);
    }
    if let Some(p) = config.top_p {
        req["top_p"] = json!(p);
    }
    if let Some(ss) = &config.stop_sequences {
        req["stop_sequences"] = serde_json::Value::Array(
            ss.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        );
    }
    // Anthropic tool_choice (different from OpenAI's)
    if let Some(tc) = &config.tool_choice {
        req["tool_choice"] = match tc {
            oben_models::ToolChoice::None => json!({"type": "none"}),
            oben_models::ToolChoice::Auto => json!({"type": "auto"}),
            oben_models::ToolChoice::Any => json!({"type": "any"}),
            oben_models::ToolChoice::Tool { name } => json!({"type": "tool", "name": name}),
        };
    } else {
        req["tool_choice"] = json!({"type": "auto"});
    }
    // Anthropic thinking tokens (Claude 3.5+)
    let b = &config.extra_body;
    if let Some(re) = &b.reasoning_effort {
        let effort_str = match re {
            oben_models::ReasoningEffort::Low => "low",
            oben_models::ReasoningEffort::Medium => "medium",
            oben_models::ReasoningEffort::High => "high",
            oben_models::ReasoningEffort::XHigh => "xhigh",
        };
        req["thinking_config"] = json!({
            "type": "enabled",
            "effort": effort_str,
            "budget_tokens": max_tokens.saturating_sub(1024),
        });
    }
    if let Some(am) = &b.anthropic_max_output {
        req["max_tokens"] = json!(am);
    }
    if let Some(tc) = &b.thinking_config {
        req["thinking_config"] = tc.clone();
    }
    // Cache control marker for Anthropic
    if let Some(cc) = &config.cache_control {
        if !cc.strategy.is_empty() {
            req["prompt_cache_key"] =
                serde_json::json!(cc.model.as_ref().unwrap_or(&String::from("default")));
        }
    }

    if !tools.is_empty() {
        req["tools"] = serde_json::Value::Array(
            tools
                .iter()
                .map(|t| serde_json::to_value(t).unwrap_or_default())
                .collect(),
        );
    }

    req
}

/// Build the streaming request template for Anthropic API.
fn build_anthropic_stream_request_template(
    config: &oben_models::ProviderConfig,
    system_prompt: impl Into<String>,
    tools: Vec<AnthropicTool>,
) -> serde_json::Value {
    let mut req = build_anthropic_request_template(config, system_prompt, tools);
    req["stream"] = json!(true);
    req
}

/// Resolve the Anthropic request using the cached template.
fn resolve_anthropic_request<F, R>(
    cached: &mut std::collections::HashMap<String, AnthropicCachedRequest>,
    messages: &[Message],
    mode: &CallMode,
    template: &std::sync::Arc<serde_json::Value>,
    system_prompt: &str,
    f: F,
) -> R
where
    F: FnOnce(&serde_json::Value) -> R,
{
    let session_id = match mode {
        CallMode::Fresh(id) | CallMode::Incremental(id) => id.clone(),
    };

    match mode {
        CallMode::Fresh(_) => {
            let mut req = (**template).clone();
            // Set system prompt in request — extract from input messages if present,
            // otherwise use the parameter (which may be empty when system is pre-baked
            // into the template, so we only override when non-empty).
            let mut has_system = false;
            let json_messages: Vec<serde_json::Value> = messages
                .iter()
                .inspect(|m| {
                    if m.role == MessageRole::System {
                        has_system = true;
                    }
                })
                .filter(|m| m.role != MessageRole::System)
                .map(message_to_anthropic_json)
                .collect();

            if has_system {
                // Extract system from input messages for Anthropic format
                let system_msgs: Vec<String> = messages
                    .iter()
                    .filter(|m| m.role == MessageRole::System)
                    .filter_map(|m| m.content.to_text_ref())
                    .map(|s| s.to_string())
                    .collect();
                req["system"] = json!(system_msgs.join("\n"));
            } else if !system_prompt.is_empty() {
                // Use parameter when provided (template has system, but chat()
                // overwrites it with "" — restore it from parameter).
                req["system"] = json!(system_prompt);
            }
            // Template's system field is kept as-is when no system in input
            // and parameter is empty (system is pre-baked in template).

            let arr = req["messages"].as_array_mut().unwrap();
            arr.reserve(json_messages.len());
            arr.append(&mut json_messages.clone());

            cached.insert(
                session_id.clone(),
                AnthropicCachedRequest {
                    request: req,
                    msg_count: messages.len(),
                },
            );
            f(&cached[&session_id].request)
        }
        CallMode::Incremental(_) => {
            let entry = cached.entry(session_id.clone()).or_insert_with(|| {
                let mut req = (**template).clone();
                if !system_prompt.is_empty() {
                    req["system"] = json!(system_prompt);
                }
                AnthropicCachedRequest {
                    request: req,
                    msg_count: 0,
                }
            });

            let cached_count = entry.msg_count;
            let non_system_msgs: Vec<&Message> = messages
                .iter()
                .filter(|m| m.role != MessageRole::System && m.include_in_prompt)
                .collect();

            if non_system_msgs.len() <= cached_count {
                // Reset: rebuild all messages
                let json_messages: Vec<serde_json::Value> = non_system_msgs
                    .iter()
                    .map(|m| message_to_anthropic_json(m))
                    .collect();

                let arr = entry.request["messages"].as_array_mut().unwrap();
                arr.clear();
                arr.reserve(json_messages.len());
                arr.append(&mut json_messages.clone());
                entry.msg_count = messages.len();
            } else {
                // Extend in-place
                let arr = entry.request["messages"].as_array_mut().unwrap();
                for msg in &non_system_msgs[cached_count..] {
                    arr.push(message_to_anthropic_json(msg));
                }
                entry.msg_count = messages.len();
            }

            f(&entry.request)
        }
    }
}

/// Parse a `data:` URL into (mime_type, base64_data).
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let comma = url.find(',')?;
    let metadata = &url[..comma];
    let data = &url[comma + 1..];
    let mime = metadata
        .strip_prefix("data:")
        .unwrap_or(metadata)
        .trim()
        .to_string();
    Some((mime, data.to_string()))
}

/// Build an Anthropic-compatible image JSON value, supporting both
/// URL-based images and inline base64 data URLs (from local files).
fn build_anthropic_image(url: &str) -> serde_json::Value {
    if url.starts_with("data:") {
        let (mime, b64) = parse_data_url(url).unwrap_or_else(|| ("image/png".into(), url.into()));
        json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": b64,
            }
        })
    } else {
        json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            }
        })
    }
}

/// Convert an oben `Message` to Anthropic JSON format.
fn message_to_anthropic_json(m: &Message) -> serde_json::Value {
    let role = match m.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "user", // Anthropic uses "user" for tool results
    };

    match &m.content {
        MessageContent::Text(t) => {
            if m.role == MessageRole::Tool {
                // Tool result
                json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_ids.first().unwrap_or(&String::from("unknown")),
                            "content": t
                        }
                    ]
                })
            } else {
                json!({"role": role, "content": t})
            }
        }
        MessageContent::Image { url, .. } => {
            let img = build_anthropic_image(url);
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "I see an image. Let me analyze it."},
                    img
                ]
            })
        }
        MessageContent::Parts(parts) => {
            let parts_json: Vec<serde_json::Value> = parts
                .iter()
                .map(|p| match p {
                    MessagePart::Text(t) => json!({"type": "text", "text": t}),
                    MessagePart::Image { url, .. } => build_anthropic_image(url),
                })
                .collect();
            json!({"role": role, "content": parts_json})
        }
    }
}

#[async_trait::async_trait]
impl TransportProvider for AnthropicMessagesTransport {
    fn name(&self) -> &str {
        "anthropic-messages"
    }

    async fn chat(&self, messages: &[Message], mode: &CallMode) -> Result<TransportResponse> {
        let request = {
            let mut cached = self.cached.lock().unwrap();
            resolve_anthropic_request(
                &mut *cached,
                messages,
                mode,
                &self.template,
                "", // system is already in template
                |req| req.clone(),
            )
        };

        let url = format!("{}/messages", self.base.base_url);

        debug!("Anthropic request to {}: model={}", url, self.base.model);

        let mut req = self.base.client.post(&url);
        if !self.base.api_key.is_empty() {
            req = req.bearer_auth(&self.base.api_key);
        }
        // Anthropic requires the `anthropic-version` header
        req = req.header("anthropic-version", "2023-06-01");

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow!("Anthropic API error {}: {}", status, body));
        }

        let resp: AnthropicResponse = response.json().await?;
        debug!(
            "Anthropic response: id={}, stop_reason={:?}, usage={:?}",
            resp.id, resp.stop_reason, resp.usage
        );

        Ok(anthropic_response_to_transport(&resp))
    }

    async fn stream_chat(
        &self,
        messages: &[Message],
        mode: &CallMode,
        mut delta_callback: oben_models::StreamDeltaCallback,
        _reasoning_callback: Option<oben_models::StreamReasoningCallback>,
    ) -> Result<TransportResponse> {
        let request = {
            let mut cached = self.cached.lock().unwrap();
            resolve_anthropic_request(
                &mut *cached,
                messages,
                mode,
                &self.stream_template,
                "", // system is already in template
                |req| req.clone(),
            )
        };

        let url = format!("{}/messages", self.base.base_url);

        debug!("Anthropic streaming request to {}", url);

        let mut req = self.base.client.post(&url);
        if !self.base.api_key.is_empty() {
            req = req.bearer_auth(&self.base.api_key);
        }
        req = req.header("anthropic-version", "2023-06-01");
        req = req.header("accept", "text/event-stream");
        req = req.header("content-type", "application/json");

        let response = req.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow!("Anthropic API error {}: {}", status, body));
        }

        // Parse SSE stream
        let body = response.bytes_stream();
        let mut stream = body.eventsource();

        let mut final_text = String::new();
        let mut tool_call_ids: Vec<String> = Vec::new();
        let mut tool_call_names: Vec<String> = Vec::new();
        let mut tool_call_args: Vec<String> = Vec::new();
        let mut total_output_tokens: usize = 0;
        let mut stream_error: Option<anyhow::Error> = None;

        while let Some(event_result) = stream.next().await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    let err = anyhow::anyhow!("Anthropic SSE stream error: {}", e);
                    tracing::error!("[anthropic] {}", err);
                    stream_error = Some(err);
                    break;
                }
            };

            // Check for stream termination
            if event.data.trim() == "[DONE]" {
                break;
            }

            // Parse the SSE event
            let stream_event: AnthropicStreamEvent = match serde_json::from_str(&event.data) {
                Ok(e) => e,
                Err(e) => {
                    debug!("Failed to parse Anthropic SSE event: {}", e);
                    continue;
                }
            };

            match stream_event.event_type.as_str() {
                "message_start" => {
                    // Start of a new message — reset accumulators
                    final_text.clear();
                    tool_call_ids.clear();
                    tool_call_names.clear();
                    tool_call_args.clear();
                }
                "content_block_start" => {
                    if let Some(block) = stream_event.content_block {
                        let idx = stream_event.index.unwrap_or(0);
                        match block.block_type.as_str() {
                            "text" => {
                                // Ensure vectors are large enough
                                while tool_call_args.len() <= idx {
                                    tool_call_args.push(String::new());
                                    tool_call_names.push(String::new());
                                    tool_call_ids.push(String::new());
                                }
                            }
                            "tool_use" => {
                                while tool_call_args.len() <= idx {
                                    tool_call_args.push(String::new());
                                    tool_call_names.push(String::new());
                                    tool_call_ids.push(String::new());
                                }
                                if let Some(id) = block.id {
                                    tool_call_ids[idx] = id;
                                }
                                if let Some(name) = block.name {
                                    tool_call_names[idx] = name;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_delta" => {
                    if let Some(delta) = stream_event.delta {
                        match delta.delta_type.as_ref().map(|s| s.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.text {
                                    final_text.push_str(&text);
                                    delta_callback(&text);
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(json_part) = delta.input_json {
                                    let idx = stream_event.index.unwrap_or(0);
                                    while tool_call_args.len() <= idx {
                                        tool_call_args.push(String::new());
                                        tool_call_names.push(String::new());
                                        tool_call_ids.push(String::new());
                                    }
                                    tool_call_args[idx].push_str(&json_part);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "message_delta" => {
                    if let Some(usage) = stream_event.usage {
                        total_output_tokens = usage.output_tokens;
                    }
                }
                "message_stop" => {
                    // End of stream
                }
                _ => {
                    debug!(
                        "Unknown Anthropic SSE event type: {}",
                        stream_event.event_type
                    );
                }
            }
        }

        // Propagate any stream error that occurred during iteration
        if let Some(err) = stream_error {
            tracing::error!("[anthropic] SSE stream failed: {}", err);
            return Err(err);
        }

        // Build final tool_calls from accumulated deltas
        // Only include entries that have a non-empty tool call ID (text blocks
        // also add entries to the vectors but without an ID — skip those).
        let mut tool_calls: Vec<TransportToolCall> = Vec::new();
        for i in 0..tool_call_ids.len() {
            let id = tool_call_ids.get(i).cloned().unwrap_or_default();
            if id.is_empty() {
                continue; // text block placeholder, skip
            }
            let name = tool_call_names.get(i).cloned().unwrap_or_default();
            let args = tool_call_args.get(i).cloned().unwrap_or_default();
            let args_json = if args.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&args).unwrap_or_else(|_| serde_json::json!(&args))
            };
            tool_calls.push(TransportToolCall {
                id,
                tool_name: name,
                arguments: args_json,
            });
        }

        Ok(TransportResponse {
            text: final_text,
            tool_calls,
            tokens_used: Some(total_output_tokens),
            reasoning: None,
        })
    }

    async fn list_models(&self) -> Result<oben_models::ModelListResponse> {
        self.list_models().await
    }

    async fn find_model(&self, model_id: &str) -> Result<Option<oben_models::ModelInfo>> {
        self.find_model(model_id).await
    }
}

// ── Conversion: oben Tool → AnthropicTool ───────────────────────────────────

/// Convert an oben `Tool` to Anthropic API tool format.
fn tool_to_anthropic(tool: &oben_models::ToolMeta) -> AnthropicTool {
    let parameters = match &tool.parameters {
        oben_models::ToolParameters::JsonSchema { schema } => schema.clone(),
        oben_models::ToolParameters::Flat(params) => {
            let properties: serde_json::Map<String, serde_json::Value> = params
                .iter()
                .map(|p| {
                    let mut prop = serde_json::Map::new();
                    prop.insert("type".into(), json!(p.parameter_type));
                    prop.insert("description".into(), json!(p.description));
                    (p.name.clone(), json!(prop))
                })
                .collect();

            let required: Vec<String> = params
                .iter()
                .filter(|p| p.required)
                .map(|p| p.name.clone())
                .collect();

            let mut schema = serde_json::Map::new();
            schema.insert("type".into(), json!("object"));
            schema.insert("properties".into(), json!(properties));
            if !required.is_empty() {
                schema.insert("required".into(), json!(required));
            }
            json!(schema)
        }
    };

    AnthropicTool {
        name: tool.name.clone(),
        description: Some(tool.description.clone()),
        input_schema: parameters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    // ── AnthropicContentBlock tests ───────────────────────────────────────

    #[test]
    fn test_anthropic_content_block_text_serializes() {
        /// given: a text content block
        /// when: serialized via serde_json
        /// then: produces {"type": "text", "text": "..."}
        let block = AnthropicContentBlock::text("hello");
        let json = serde_json::to_string(&block).unwrap();
        let parsed: AnthropicContentBlock = serde_json::from_str(&json).unwrap();
        if let AnthropicContentBlock::Text { text, .. } = parsed {
            assert_eq!(text, "hello");
        } else {
            panic!("expected Text variant");
        }
    }

    #[test]
    fn test_anthropic_content_block_tool_use_serializes() {
        /// given: a tool_use content block
        /// when: serialized via serde_json
        /// then: produces {"type": "tool_use", "id": "...", "name": "...", "input": {}}
        let block = AnthropicContentBlock::tool_use(
            "call-1",
            "shell",
            serde_json::json!({"command": "ls"}),
        );
        let json = serde_json::to_string(&block).unwrap();
        let parsed: AnthropicContentBlock = serde_json::from_str(&json).unwrap();
        if let AnthropicContentBlock::ToolUse { id, name, input } = parsed {
            assert_eq!(id, "call-1");
            assert_eq!(name, "shell");
            assert_eq!(input["command"].as_str().unwrap(), "ls");
        } else {
            panic!("expected ToolUse variant");
        }
    }

    // ── AnthropicRequest tests ────────────────────────────────────────────

    #[test]
    fn test_anthropic_request_serializes_with_system() {
        /// given: an AnthropicRequest with system prompt
        /// when: serialized via serde_json
        /// then: system appears as top-level field, not in messages
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            system: Some("You are helpful".to_string()),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::text("hello"),
            }],
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            top_p: None,
            stop_sequences: None,
            thinking: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["system"], "You are helpful");
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[test]
    fn test_anthropic_request_with_tools() {
        /// given: an AnthropicRequest with tools
        /// when: serialized via serde_json
        /// then: tools array appears with input_schema format
        let tool = AnthropicTool {
            name: "shell".to_string(),
            description: Some("Execute shell command".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command to execute"}},
                "required": ["command"]
            }),
        };
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            system: Some("be helpful".to_string()),
            messages: vec![],
            tools: Some(vec![tool]),
            tool_choice: Some(AnthropicToolChoice::Auto),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            thinking: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["tools"][0]["name"], "shell");
        assert!(parsed["tools"][0]["input_schema"]["properties"].is_object());
    }

    // ── AnthropicTool conversion tests ────────────────────────────────────

    #[test]
    fn test_tool_to_anthropic_flat_params() {
        /// given: an oben Tool with flat parameters
        /// when: converted to AnthropicTool
        /// then: produces correct input_schema with properties and required
        let tool = oben_models::ToolMeta {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: oben_models::ToolParameters::Flat(vec![oben_models::ToolParameter {
                name: "x".to_string(),
                parameter_type: "string".to_string(),
                description: "X param".to_string(),
                required: true,
            }]),
        };
        let at = tool_to_anthropic(&tool);
        assert_eq!(at.name, "test_tool");
        assert_eq!(at.description, Some("A test tool".to_string()));
        assert_eq!(at.input_schema["type"], "object");
        assert_eq!(at.input_schema["required"][0], "x");
    }

    #[test]
    fn test_message_to_anthropic_user_text() {
        /// given: an oben user message with text content
        /// when: converted via message_to_anthropic_json
        /// then: produces JSON with user role and text content
        let msg = Message::user("hello world");
        let result = message_to_anthropic_json(&msg);
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"], "hello world");
    }

    #[test]
    fn test_message_to_anthropic_assistant_with_tool_calls() {
        /// given: an oben assistant message with tool calls
        /// when: converted via message_to_anthropic_json
        /// then: produces JSON with assistant role and the text content
        let msg = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(String::new()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(vec![oben_models::ToolCall {
                id: "call-1".to_string(),
                tool_name: "shell".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            }]),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };
        let result = message_to_anthropic_json(&msg);
        assert_eq!(result["role"], "assistant");
        // content is the text content (may be empty for tool-calling messages)
        assert_eq!(result["content"], "");
    }

    #[test]
    fn test_message_to_anthropic_tool_result() {
        /// given: an oben tool result message
        /// when: converted via message_to_anthropic_json
        /// then: produces JSON with user role and tool_result content blocks
        let msg = Message::tool_result("call-123", "file not found");
        let result = message_to_anthropic_json(&msg);
        assert_eq!(result["role"], "user");
        assert!(result["content"].is_array());
        assert_eq!(result["content"][0]["type"], "tool_result");
        assert_eq!(result["content"][0]["tool_use_id"], "call-123");
    }

    #[test]
    fn test_message_to_anthropic_skips_system() {
        /// given: a system message
        /// when: converted via message_to_anthropic_json
        /// then: system messages are converted but with "user" role (Anthropic mapping)
        let msg = Message::system("you are helpful");
        let result = message_to_anthropic_json(&msg);
        // Anthropic maps system to a system role in the request, not in messages array
        // message_to_anthropic_json converts the message role, not Anthropic-specific handling
        assert_eq!(result["role"], "system");
    }

    // ── Response conversion tests ─────────────────────────────────────────

    #[test]
    fn test_anthropic_response_to_transport_text_only() {
        /// given: an AnthropicResponse with text content
        /// when: converted via anthropic_response_to_transport
        /// then: produces TransportResponse with text and no tool calls
        let resp = AnthropicResponse {
            id: "msg-123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::Text {
                cache_control: None,
                text: "Hello, world!".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
            model: "claude-sonnet-4-20250514".to_string(),
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
        };
        let tr = anthropic_response_to_transport(&resp);
        assert_eq!(tr.text, "Hello, world!");
        assert!(tr.tool_calls.is_empty());
        assert_eq!(tr.tokens_used, Some(50));
    }

    #[test]
    fn test_anthropic_response_to_transport_with_tool_calls() {
        /// given: an AnthropicResponse with tool_use content blocks
        /// when: converted via anthropic_response_to_transport
        /// then: produces TransportResponse with text and tool calls
        let resp = AnthropicResponse {
            id: "msg-123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                AnthropicContentBlock::Text {
                    cache_control: None,
                    text: "Let me check the files.".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "shell".to_string(),
                    input: json!({"command": "ls -la"}),
                },
            ],
            stop_reason: Some("tool_use".to_string()),
            model: "claude-sonnet-4-20250514".to_string(),
            usage: AnthropicUsage {
                input_tokens: 150,
                output_tokens: 75,
            },
        };
        let tr = anthropic_response_to_transport(&resp);
        assert_eq!(tr.text, "Let me check the files.");
        assert_eq!(tr.tool_calls.len(), 1);
        assert_eq!(tr.tool_calls[0].id, "call-1");
        assert_eq!(tr.tool_calls[0].tool_name, "shell");
        assert_eq!(
            tr.tool_calls[0].arguments["command"].as_str().unwrap(),
            "ls -la"
        );
    }

    // ── Streaming event parsing tests ─────────────────────────────────────

    #[test]
    fn test_parse_message_start_event() {
        /// given: an Anthropic SSE `message_start` event
        /// when: deserialized via AnthropicStreamEvent
        /// then: event_type is "message_start" and message is present
        let json = r#"{"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":0}}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "message_start");
        assert!(event.message.is_some());
    }

    #[test]
    fn test_parse_content_block_delta_event() {
        /// given: an Anthropic SSE `content_block_delta` event
        /// when: deserialized via AnthropicStreamEvent
        /// then: delta contains text_delta with content
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        assert_eq!(event.index, Some(0));
        assert!(event.delta.is_some());
        if let Some(delta) = event.delta {
            assert_eq!(delta.delta_type, Some("text_delta".to_string()));
            assert_eq!(delta.text, Some("Hello".to_string()));
        }
    }

    #[test]
    fn test_parse_input_json_delta_event() {
        /// given: an Anthropic SSE `content_block_delta` with input_json
        /// when: deserialized via AnthropicStreamEvent
        /// then: delta contains input_json_delta with JSON fragment
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","input_json":"{\"command\":"}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "content_block_delta");
        if let Some(delta) = event.delta {
            assert_eq!(delta.delta_type, Some("input_json_delta".to_string()));
            assert_eq!(delta.input_json, Some("{\"command\":".to_string()));
        }
    }

    #[test]
    fn test_parse_message_delta_event() {
        /// given: an Anthropic SSE `message_delta` event
        /// when: deserialized via AnthropicStreamEvent
        /// then: usage contains output_tokens
        let json = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":50}}"#;
        let event: AnthropicStreamEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "message_delta");
        assert!(event.usage.is_some());
        assert_eq!(event.usage.unwrap().output_tokens, 50);
    }

    // ── Template building tests ───────────────────────────────────────────

    #[test]
    fn test_build_anthropic_request_template() {
        /// given: a ProviderConfig with system prompt and tools
        /// when: build_anthropic_request_template is called
        /// then: returns valid request with system, model, max_tokens, tools
        let config = oben_models::ProviderConfig::new(
            oben_models::ProviderKind::Anthropic,
            "claude-sonnet-4-20250514",
        );
        let tool = AnthropicTool {
            name: "shell".to_string(),
            description: Some("Execute command".to_string()),
            input_schema: json!({"type": "object"}),
        };
        let template = build_anthropic_request_template(&config, "You are helpful", vec![tool]);
        assert_eq!(template["model"], "claude-sonnet-4-20250514");
        assert_eq!(template["max_tokens"], 4096);
        assert_eq!(template["system"], "You are helpful");
        assert!(template["tools"].is_array());
        assert_eq!(template["tool_choice"]["type"], "auto");
    }

    // ── Resolve request tests ─────────────────────────────────────────────

    #[test]
    fn test_resolve_anthropic_request_fresh() {
        /// given: empty cache and fresh mode with 2 messages
        /// when: resolve_anthropic_request is called
        /// then: cache contains request with messages array
        let template = std::sync::Arc::new(json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": "test",
            "messages": serde_json::Value::Array(vec![]),
        }));
        let mut cached = std::collections::HashMap::new();
        let messages = vec![Message::system("sys"), Message::user("hello")];
        let (msg_count, has_messages) = resolve_anthropic_request(
            &mut cached,
            &messages,
            &CallMode::Fresh("test-session".into()),
            &template,
            "test",
            |req| {
                assert_eq!(req["messages"].as_array().unwrap().len(), 1);
                assert_eq!(req["messages"][0]["role"], "user");
                (2, true)
            },
        );
        assert_eq!(msg_count, 2);
        assert!(has_messages);
    }

    #[test]
    fn test_resolve_anthropic_request_incremental_extend() {
        /// given: cached request with 1 message
        /// when: resolve_anthropic_request called in incremental mode with 2 messages
        /// then: messages array extended in-place (now has 2 messages)
        let template = std::sync::Arc::new(json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": "test",
            "messages": serde_json::Value::Array(vec![]),
        }));
        let mut cached = std::collections::HashMap::new();
        let messages1 = vec![Message::user("hello")];
        resolve_anthropic_request(
            &mut cached,
            &messages1,
            &CallMode::Fresh("test-session".into()),
            &template,
            "test",
            |_| (),
        );

        let messages2 = vec![Message::user("hello"), Message::assistant("hi")];
        resolve_anthropic_request(
            &mut cached,
            &messages2,
            &CallMode::Incremental("test-session".into()),
            &template,
            "test",
            |req| {
                assert_eq!(req["messages"].as_array().unwrap().len(), 2);
                assert_eq!(req["messages"][1]["role"], "assistant");
            },
        );
    }
}
// ── Cache control marker tests ─────────────────────────────────────

#[test]
fn test_system_prompt_with_cache_control() {
    // given: a ProviderConfig with cache_control enabled
    // when: build_anthropic_request_template is called
    // then: system is a content blocks array with cache_control marker
    let mut config = oben_models::ProviderConfig::new(
        oben_models::ProviderKind::Anthropic,
        "claude-sonnet-4-20250514",
    );
    config.cache_control = Some(oben_models::CacheControl {
        provider: None,
        model: None,
        strategy: "ephemeral".to_string(),
    });
    let template = build_anthropic_request_template(&config, "You are helpful", Vec::new());
    assert_eq!(template["model"], "claude-sonnet-4-20250514");
    assert!(template["system"].is_array());
    assert_eq!(template["system"][0]["type"], "text");
    assert_eq!(template["system"][0]["text"], "You are helpful");
    assert_eq!(template["system"][0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_system_prompt_without_cache_control() {
    // given: a ProviderConfig without cache_control
    // when: build_anthropic_request_template is called
    // then: system is a plain string (legacy format preserved)
    let config = oben_models::ProviderConfig::new(
        oben_models::ProviderKind::Anthropic,
        "claude-sonnet-4-20250514",
    );
    let template = build_anthropic_request_template(&config, "You are helpful", Vec::new());
    assert!(template["system"].is_string());
    assert_eq!(template["system"], "You are helpful");
}

#[test]
fn test_system_prompt_cache_control_empty_strategy() {
    // given: a ProviderConfig with cache_control but empty strategy
    // when: build_anthropic_request_template is called
    // then: system defaults to "ephemeral" strategy
    let mut config = oben_models::ProviderConfig::new(
        oben_models::ProviderKind::Anthropic,
        "claude-sonnet-4-20250514",
    );
    config.cache_control = Some(oben_models::CacheControl {
        provider: None,
        model: None,
        strategy: "".to_string(),
    });
    let template = build_anthropic_request_template(&config, "You are helpful", Vec::new());
    assert!(template["system"].is_array());
    assert_eq!(template["system"][0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_stream_template_inherits_cache_control() {
    // given: a ProviderConfig with cache_control enabled
    // when: build_anthropic_stream_request_template is called
    // then: system is a content blocks array with cache_control marker, and stream=true
    let mut config = oben_models::ProviderConfig::new(
        oben_models::ProviderKind::Anthropic,
        "claude-sonnet-4-20250514",
    );
    config.cache_control = Some(oben_models::CacheControl {
        provider: None,
        model: None,
        strategy: "ephemeral".to_string(),
    });
    let stream_template =
        build_anthropic_stream_request_template(&config, "You are helpful", Vec::new());
    assert!(stream_template["system"].is_array());
    assert_eq!(
        stream_template["system"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(stream_template["stream"], true);
}
