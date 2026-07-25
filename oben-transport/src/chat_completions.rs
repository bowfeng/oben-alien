/// Chat Completions transport — OpenAI-compatible API (OpenRouter, OpenAI, NovitaAI, etc.)
/// Maps to `agent/transports/chat_completions.py`.
use anyhow::{anyhow, Result};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use oben_models::{
    Message, MessageRole, ReasoningEffort, ToolMeta, TransportResponse, TransportToolCall,
};
use serde_json::json;
use tracing::{debug, info};

/// Mask base64 data URLs in log output to avoid exposing image data in logs.
fn mask_data_urls(json_str: &str) -> String {
    static DATA_IMG: &[u8] = b"data:image/";
    if !json_str.contains("data:image/") {
        return json_str.to_string();
    }

    let mut result = String::with_capacity(json_str.len());
    let mut remaining = json_str.as_bytes();

    while let Some(pos) = remaining.windows(11).position(|w| w == DATA_IMG) {
        let before = unsafe { std::str::from_utf8_unchecked(&remaining[..pos]) };
        result.push_str(before);

        let content = &remaining[pos + 11..];
        let mut total = 0usize;
        let mut i = 0;
        while i < content.len() {
            if content[i] == 0x22 && (i == 0 || content[i - 1] != 0x5C) {
                break;
            } else {
                i += 1;
                total += 1;
            }
        }

        result.push_str(&format!("data:image/[data URL {} bytes]", total));
        remaining = &content[i + 1..];
    }

    result.push_str(unsafe { std::str::from_utf8_unchecked(remaining) });
    result
}

use super::base::{BaseTransport, ChatResponse};

/// For GPT-5 and Codex models, the API expects "developer" instead of "system".
/// Returns `true` if the model name should use "developer" role.
fn swap_system_to_developer(model: &str) -> bool {
    let lower = model.to_lowercase();
    lower.contains("gpt-5") || lower.contains("codex")
}

/// Convert an `oben_models::Tool` into OpenAI API tool format:
/// `{ "type": "function", "function": { "name": ..., "description": ..., "parameters": ... } }`
fn tool_to_openai(tool: &ToolMeta) -> serde_json::Value {
    let parameters = match &tool.parameters {
        oben_models::ToolParameters::JsonSchema { schema } => schema.clone(),
        oben_models::ToolParameters::Flat(params) => {
            // Build a JSON Schema object from flat parameter list
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

    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }
    })
}

/// Per-session cached request state.
///
/// Stores the full request as a `serde_json::Value` with `"messages"` as a
/// mutable array. On Fresh we replace it entirely; on Incremental we extend
/// the existing array in-place — no cloning, no rebuilding static parts.
///
/// The `template` field holds the static request shape (model, temperature,
/// max_tokens, system prompt, tools) wrapped in an `Arc` so that cloning
/// a `CachedRequest` for a new incremental call is a single pointer copy
/// instead of a deep clone of the entire static JSON.
struct CachedRequest {
    /// Full request object, e.g. `{ "model": ..., "messages": [...], ... }`
    request: serde_json::Value,
    /// Number of messages currently in the cached array.
    msg_count: usize,
}

/// Build JSON for a single message.
fn message_to_json(m: &Message) -> serde_json::Value {
    let role = match m.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };
    match &m.content {
        oben_models::MessageContent::Text(t) => {
            // Assistant with tool_calls: content=null, emit tool_calls array
            if m.role == MessageRole::Assistant {
                if let Some(tcs) = &m.tool_calls {
                    let calls: Vec<serde_json::Value> = tcs
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.tool_name,
                                    "arguments": tc.arguments.to_string()
                                }
                            })
                        })
                        .collect();
                    return json!({"role": role, "content": null, "tool_calls": calls});
                }
            }
            // Tool message: include tool_call_id
            if m.role == MessageRole::Tool {
                let call_id = m
                    .tool_call_ids
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                json!({"role": role, "content": t, "tool_call_id": call_id})
            } else {
                json!({"role": role, "content": t})
            }
        }
        oben_models::MessageContent::Image { url, detail } => {
            let mut img = json!({
                "type": "image_url",
                "image_url": { "url": url }
            });
            if let Some(d) = detail {
                img["image_url"]["detail"] = json!(d);
            }
            json!({
                "role": role,
                "content": [
                    { "type": "text", "text": "I see an image. Let me analyze it." },
                    img
                ]
            })
        }
        oben_models::MessageContent::Parts(parts) => {
            let parts_json: Vec<serde_json::Value> = parts
                .iter()
                .map(|p| match p {
                    oben_models::MessagePart::Text(t) => {
                        json!({"type": "text", "text": t})
                    }
                    oben_models::MessagePart::Image { url, detail } => {
                        let mut img = json!({"type": "image_url", "image_url": {"url": url}});
                        if let Some(d) = detail {
                            img["image_url"]["detail"] = json!(d);
                        }
                        img
                    }
                })
                .collect();
            json!({"role": role, "content": parts_json})
        }
    }
}

/// Build JSON for all messages (fresh call). Returns Vec in place.
fn build_all_messages_json(messages: &[Message]) -> Vec<serde_json::Value> {
    // Filter out system messages from the array — they are baked into the
    // template's messages[0] by `build_request_template`.  The caller
    // (e.g. `goal_start`) may accidentally include a system message in the
    // input Vec; including it alongside the template's system message causes
    // API errors like "System message must be at the beginning.".
    messages
        .iter()
        .filter(|m| m.role != MessageRole::System && m.include_in_prompt)
        .map(message_to_json)
        .collect()
}

/// Push new messages into an existing JSON array (incremental extend).
/// Returns the number of messages added.
fn extend_messages_json(arr: &mut Vec<serde_json::Value>, messages: &[Message], from: usize) {
    for msg in &messages[from..] {
        // Filter out system messages and messages not intended for prompt — same as build_all_messages_json
        if msg.role != MessageRole::System && msg.include_in_prompt {
            arr.push(message_to_json(msg));
        }
    }
}

/// Resolves the CallMode into a cached request and calls `f` with a reference.
///
/// **Fresh**: replaces the cached request entirely.
/// **Incremental**: extends the existing `"messages"` array in-place — zero
/// copies of the existing messages, zero rebuild of static parts.
///
/// The callback runs while the mutex is held, so the reference is always valid.
fn resolve_request<F, R>(
    cached: &mut std::collections::HashMap<String, CachedRequest>,
    messages: &[Message],
    mode: &oben_models::CallMode,
    template: &std::sync::Arc<serde_json::Value>,
    f: F,
) -> R
where
    F: FnOnce(&serde_json::Value) -> R,
{
    let session_id = match mode {
        oben_models::CallMode::Fresh(id) | oben_models::CallMode::Incremental(id) => id.clone(),
    };

    match mode {
        oben_models::CallMode::Fresh(_) => {
            // Build the full request from the template + all messages.
            // Arc::clone is a pointer copy — the template (system prompt, tools,
            // static config) is shared, so we clone only the messages part.
            let mut req = (**template).clone();
            let mut json_messages = build_all_messages_json(messages);

            // Track non-system message count for incremental extend comparison.
            let non_sys_count = json_messages.len();

            let arr = req["messages"].as_array_mut().unwrap();
            // Pre-allocate capacity to avoid reallocations.
            arr.reserve(json_messages.len() + 1); // pre-allocate to avoid reallocations including system prompt
                                                  // Move JSON values into the array (avoids per-element clone).
            arr.append(&mut json_messages);
            let sid = session_id.clone();
            cached.insert(
                sid,
                CachedRequest {
                    request: req,
                    msg_count: non_sys_count,
                },
            );
            f(&cached[&session_id].request)
        }
        oben_models::CallMode::Incremental(_) => {
            let entry = cached.entry(session_id).or_insert_with(|| {
                // Arc clone = single pointer copy, not deep clone
                let req = (**template).clone();
                CachedRequest {
                    request: req,
                    msg_count: 0,
                }
            });
            let cached_count = entry.msg_count;

            // Count non-system messages in current input (same as what
            // build_all_messages_json would produce).
            let non_sys_count = messages
                .iter()
                .filter(|m| m.role != MessageRole::System && m.include_in_prompt)
                .count();

            if non_sys_count <= cached_count {
                // Messages haven't grown — content was edited or removed.
                // Rebuild entirely: clear all and rebuild from scratch.
                let mut json_messages = build_all_messages_json(messages);
                // Extract system content before clearing
                let sys_content = entry.request["messages"]
                    .as_array()
                    .unwrap()
                    .first()
                    .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                    .map(|s| s.to_string());
                let arr = entry.request["messages"].as_array_mut().unwrap();
                arr.clear();
                if let Some(s) = sys_content {
                    arr.push(json!({"role": "system", "content": s}));
                }
                let len = json_messages.len();
                arr.reserve(len + 1);
                arr.append(&mut json_messages);
                entry.msg_count = non_sys_count;
            } else {
                // Messages grew — extend existing array in-place.
                let arr = entry.request["messages"].as_array_mut().unwrap();
                // cached_count is the previous non-system count.
                // Count non-system messages in the previous slice to find
                // the raw index where new messages start.
                let added_non_sys = non_sys_count - cached_count;
                // Starting index: skip (total - added_non_sys) messages from front
                // which is the same as total - added_non_sys from the back.
                let start_idx = messages.len() - added_non_sys;
                extend_messages_json(arr, messages, start_idx);
                entry.msg_count = non_sys_count;
            }

            f(&entry.request)
        }
    }
}

/// SSE event from streaming response — fields used only for serde deserialization.
#[derive(serde::Deserialize)]
struct StreamChunk {
    #[serde(default)]
    _id: String,
    #[serde(default)]
    _object: String,
    #[serde(default)]
    _created: u64,
    pub choices: Vec<StreamChoice>,
    #[serde(default)]
    pub usage: Option<StreamUsage>,
}

#[derive(serde::Deserialize)]
struct StreamChoice {
    pub delta: StreamDelta,
    #[serde(default)]
    _index: usize,
    #[serde(default)]
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct StreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// `reasoning_content` — used by thinking models (Qwen, DeepSeek, etc.)
    #[serde(default, rename = "reasoning_content")]
    pub reasoning_content: Option<String>,
    #[serde(default, rename = "tool_calls")]
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
    _role: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct StreamToolCallDelta {
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<StreamFunctionDelta>,
}

#[derive(Debug, serde::Deserialize)]
struct StreamFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(serde::Deserialize)]
struct StreamUsage {
    _prompt_tokens: Option<usize>,
    _completion_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
}

/// Build the static parts of the request template (non-streaming).
/// The system prompt is NOT embedded here — it must be in messages[0]
/// (prepended by `SystemPromptConfig::build_and_prepend`).
fn build_request_template(
    config: &oben_models::ProviderConfig,
    system_prompt: impl Into<String>,
    tools: Vec<serde_json::Value>,
) -> serde_json::Value {
    let system_role = if swap_system_to_developer(&config.model) {
        "developer"
    } else {
        "system"
    };
    let mut req = json!({
        "model": config.model,
        "messages": serde_json::Value::Array(vec![json!({
            "role": system_role,
            "content": system_prompt.into(),
        })]),
    });

    if let Some(t) = config.temperature {
        req["temperature"] = json!(t);
    }
    if let Some(m) = config.max_tokens {
        req["max_tokens"] = json!(m);
    }
    if let Some(p) = config.top_p {
        req["top_p"] = json!(p);
    }
    if let Some(k) = config.top_k {
        req["top_k"] = json!(k);
    }
    if let Some(fp) = config.frequency_penalty {
        req["frequency_penalty"] = json!(fp);
    }
    if let Some(pp) = config.presence_penalty {
        req["presence_penalty"] = json!(pp);
    }
    if let Some(ss) = &config.stop_sequences {
        req["stop"] = serde_json::Value::Array(
            ss.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        );
    }
    if let Some(rf) = &config.response_format {
        req["response_format"] = match rf {
            oben_models::ResponseFormat::Text => json!({"type": "text"}),
            oben_models::ResponseFormat::Json => json!({"type": "json_object"}),
            oben_models::ResponseFormat::JsonSchema { schema } => {
                json!({"type": "json_schema", "json_schema": schema})
            }
        };
    }
    if let Some(tc) = &config.tool_choice {
        req["tool_choice"] = match tc {
            oben_models::ToolChoice::None => json!({"type": "none"}),
            oben_models::ToolChoice::Auto => json!({"type": "auto"}),
            oben_models::ToolChoice::Any => json!({"type": "any"}),
            oben_models::ToolChoice::Tool { name } => {
                json!({"type": "function", "function": {"name": name}})
            }
        };
    }
    if let Some(re) = &config.reasoning_effort {
        req["reasoning_effort"] = json!(match re {
            oben_models::ReasoningEffort::Low => "low",
            oben_models::ReasoningEffort::Medium => "medium",
            oben_models::ReasoningEffort::High => "high",
            oben_models::ReasoningEffort::XHigh => "xhigh",
        });
    }
    if let Some(st) = &config.service_tier {
        req["service_tier"] = json!(st);
    }
    if let Some(up) = &config.provider_preferences {
        req["provider_preferences"] = json!(up);
    }
    if let Some(uid) = &config.user_id {
        req["user"] = json!(uid);
    }
    if let Some(md) = &config.metadata {
        req["metadata"] = md.clone();
    }
    if let Some(lb) = &config.logit_bias {
        req["logit_bias"] = lb.clone();
    }
    let b = &config.extra_body;
    if let Some(v) = &b.anthropic_max_output {
        req["anthropic_max_output"] = json!(v);
    }
    if let Some(t) = &b.thinking {
        req["thinking"] = json!(t);
    }
    if b.thinking.is_some() || b.thinking_config.is_some() {
        if let Some(tc) = &b.thinking_config {
            req["thinking_config"] = tc.clone();
            let mut eb = serde_json::Map::new();
            eb.insert("reasoning".into(), json!({"enabled": b.thinking.unwrap_or_default(), "effort": match &b.reasoning_effort {
                Some(re) => match re {
                    ReasoningEffort::Low => "low",
                    ReasoningEffort::Medium => "medium",
                    ReasoningEffort::High => "high",
                    ReasoningEffort::XHigh => "xhigh",
                },
                None => "medium",
            }}));
            eb.insert("thinking_config".into(), tc.clone());
            req["extra_body"] = json!(serde_json::Value::Object(eb));
        }
    }
    if let Some(ollama_ctx) = &b.ollama_num_ctx {
        req["num_ctx"] = json!(ollama_ctx);
    }
    if let Some(uid) = &b.user_id {
        if config.user_id.is_none() {
            req["user"] = json!(uid);
        }
    }
    if let Some(md) = &b.metadata {
        if config.metadata.is_none() {
            req["metadata"] = md.clone();
        }
    }

    if !tools.is_empty() {
        req["tools"] = serde_json::Value::Array(tools);
    }
    req
}

/// Build the streaming request template.
/// The system prompt is NOT embedded here — it must be in messages[0]
/// (prepended by `SystemPromptConfig::build_and_prepend`).
fn build_stream_request_template(
    config: &oben_models::ProviderConfig,
    system_prompt: impl Into<String>,
    tools: Vec<serde_json::Value>,
) -> serde_json::Value {
    let mut req = build_request_template(config, system_prompt, vec![]);
    req["stream"] = json!(true);
    req["stream_options"] = json!({"include_usage": true});

    if !tools.is_empty() {
        req["tools"] = serde_json::Value::Array(tools);
    }
    req
}

/// Transport that talks to any OpenAI-compatible API.
pub struct ChatCompletionsTransport {
    base: BaseTransport,
    /// Cached request state per session — contains the full request object
    /// with a mutable `"messages"` array.
    cached: std::sync::Mutex<std::collections::HashMap<String, CachedRequest>>,
    /// Static non-streaming request template (model, temperature, max_tokens, tools,
    /// system prompt). Wrapped in `Arc` so incremental calls clone only a pointer
    /// instead of deep-cloning the entire static JSON (often 100KB+ of system prompt
    /// + tool definitions).
    template: std::sync::Arc<serde_json::Value>,
    /// Static streaming request template (same as above + stream: true).
    stream_template: std::sync::Arc<serde_json::Value>,
}

impl ChatCompletionsTransport {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        let base = BaseTransport::new(base_url, api_key, model.into());
        let system_prompt = system_prompt.into();
        let tools: Vec<serde_json::Value> = Vec::new();
        let config = oben_models::ProviderConfig::new(
            oben_models::ProviderKind::Custom,
            "model-placeholder",
        );
        let template = build_request_template(&config, system_prompt.clone(), tools.clone());
        let stream_template = build_stream_request_template(&config, system_prompt, tools);
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
        tools: Vec<ToolMeta>,
    ) -> Self {
        let model: String = model.into();
        let base = BaseTransport::new(base_url, api_key, model.clone());
        let system_prompt = system_prompt.into();
        let tool_defs: Vec<serde_json::Value> = tools.iter().map(tool_to_openai).collect();
        let config = oben_models::ProviderConfig::new(oben_models::ProviderKind::Custom, model);
        let template = build_request_template(&config, system_prompt.clone(), tool_defs.clone());
        let stream_template = build_stream_request_template(&config, system_prompt, tool_defs);
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }

    /// Resolve the base URL for a provider config.
    ///
    /// Resolution order:
    /// 1. Provider-specific env var (e.g. `OPENAI_BASE_URL`)
    /// 2. Registry default base URL from `PROVIDER_META`
    /// 3. `config.base_url`
    /// 4. Empty string
    fn resolve_base_url(config: &oben_models::ProviderConfig) -> String {
        // Step 1: Provider-specific env var for custom base URL
        if let Some(env_var_name) = config.kind.base_url_env_var() {
            if let Ok(url) = std::env::var(env_var_name) {
                let url = url.trim().to_string();
                if !url.is_empty() {
                    return url;
                }
            }
        }

        // Step 2: Registry default base URL
        if let Some(default_url) = config.kind.default_base_url() {
            if !default_url.is_empty() {
                return default_url.to_string();
            }
        }

        // Step 3: Config-level override
        config.base_url.clone().unwrap_or_default()
    }

    /// Create from a ProviderConfig, with tools for structured tool calling.
    pub fn from_config_with_tools(
        config: &oben_models::ProviderConfig,
        system_prompt: impl Into<String>,
        tools: Vec<ToolMeta>,
    ) -> Self {
        let base_url = Self::resolve_base_url(config);
        let api_key = config.api_key.clone().unwrap_or_default();
        let tool_defs: Vec<serde_json::Value> = tools.iter().map(tool_to_openai).collect();
        let system_prompt = system_prompt.into();
        let template = build_request_template(config, system_prompt.clone(), tool_defs.clone());
        let stream_template = build_stream_request_template(config, system_prompt, tool_defs);
        let base = BaseTransport::with_timeout(base_url, api_key, &config.model, config.timeout);
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
        let base =
            BaseTransport::with_timeout(base_url, api_key, config.model.clone(), config.timeout);
        let system_prompt = system_prompt.into();
        let template = build_request_template(config, system_prompt.clone(), Vec::new());
        let stream_template = build_stream_request_template(config, system_prompt, Vec::new());
        Self {
            base,
            cached: std::sync::Mutex::new(std::collections::HashMap::new()),
            template: std::sync::Arc::new(template),
            stream_template: std::sync::Arc::new(stream_template),
        }
    }

    /// Fetch the list of available models from the provider.
    pub async fn list_models(&self) -> Result<oben_models::ModelListResponse> {
        self.base.list_models().await
    }

    /// Find a specific model by ID from the provider.
    pub async fn find_model(&self, model_id: &str) -> Result<Option<oben_models::ModelInfo>> {
        self.base.find_model(model_id).await
    }
}

#[async_trait::async_trait]
impl oben_models::providers::TransportProvider for ChatCompletionsTransport {
    fn name(&self) -> &str {
        "chat-completions"
    }

    async fn chat(
        &self,
        messages: &[Message],
        mode: &oben_models::CallMode,
    ) -> Result<TransportResponse> {
        let request = {
            let mut cached = self.cached.lock().unwrap();
            // system_prompt is already in messages[0] from build_and_prepend —
            // the parameter is kept for API compatibility but not re-embedded.
            resolve_request(&mut *cached, messages, mode, &self.template, |req| {
                req.clone()
            })
        };

        let url = format!("{}/chat/completions", self.base.base_url);

        debug!(
            "Requesting {}: model={}, messages={}",
            url,
            self.base.model,
            messages.len()
        );
        let logged = mask_data_urls(&serde_json::to_string(&request).unwrap_or_default());
        tracing::trace!("Prompt (truncated to 512 chars), messages={}", &logged[..logged.len().min(512)]);

        let mut req = self.base.client.post(&url).json(&request);
        if !self.base.api_key.is_empty() {
            req = req.bearer_auth(&self.base.api_key);
        }
        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!("API error {}: {}", status, body));
        }

        let resp: ChatResponse = response.json().await?;

        tracing::trace!(
            "LLM response: choices={:?}, usage={:?}",
            resp.choices
                .iter()
                .map(|c| {
                    format!(
                        "role={:?} content_len={:?} tool_calls={:?} finish={:?}",
                        c.message.role,
                        c.message.content.as_ref().map(|s| s.len()),
                        c.message.tool_calls.as_ref().map(|tcs| tcs.len()),
                        c.finish_reason,
                    )
                })
                .collect::<Vec<_>>(),
            resp.usage.as_ref().map(|u| format!(
                "prompt={:?} comp={:?} total={:?}",
                u.prompt_tokens, u.completion_tokens, u.total_tokens
            )),
        );

        let choice = resp
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("No response choices"))?;
        let text = choice.message.content.clone().unwrap_or_default();
        let preview: String = text.chars().take(100).collect();
        tracing::trace!(
            "Extracted text: len={}, first_100={:?}",
            text.len(),
            preview
        );
        let tool_calls: Vec<TransportToolCall> = choice
            .message
            .tool_calls
            .iter()
            .flatten()
            .map(|tc| TransportToolCall {
                id: tc.id.clone(),
                tool_name: tc.function.name.clone(),
                arguments: serde_json::from_str(&tc.function.arguments).unwrap_or_default(),
            })
            .collect();

        Ok(TransportResponse {
            text,
            tool_calls,
            tokens_used: resp.usage.and_then(|u| u.total_tokens),
            reasoning: None,
        })
    }

    async fn stream_chat(
        &self,
        messages: &[Message],
        mode: &oben_models::CallMode,
        mut delta_callback: oben_models::StreamDeltaCallback,
        mut reasoning_callback: Option<oben_models::StreamReasoningCallback>,
    ) -> Result<TransportResponse> {
        let request = {
            let mut cached = self.cached.lock().unwrap();
            resolve_request(&mut *cached, messages, mode, &self.stream_template, |req| {
                req.clone()
            })
        };

        let url = format!("{}/chat/completions", self.base.base_url);
        info!("Streaming request to {}", url);
        let logged = mask_data_urls(&serde_json::to_string(&request).unwrap_or_default());
        info!("Stream prompt (truncated to 512 chars), messages={}", &logged[..logged.len().min(512)]);

        let mut req = self.base.client.post(&url).json(&request);
        if !self.base.api_key.is_empty() {
            req = req.bearer_auth(&self.base.api_key);
        }
        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow!("API error {}: {}", status, body));
        }

        // Parse SSE stream
        let body = response.bytes_stream();
        let mut stream = body.eventsource();

        let mut final_text = String::new();
        let mut final_reasoning = String::new();
        let mut tool_call_names: Vec<String> = Vec::new();
        let mut tool_call_args: Vec<String> = Vec::new();
        let mut tool_call_ids: Vec<String> = Vec::new();
        let mut total_tokens: Option<usize> = None;
        let mut delta_count: usize = 0;
        let mut stream_error: Option<anyhow::Error> = None;

        while let Some(event_result) = stream.next().await {
            let event = match event_result {
                Ok(e) => e,
                Err(e) => {
                    let err = anyhow::anyhow!("SSE stream error: {}", e);
                    tracing::error!("[stream_chat] {}", err);
                    stream_error = Some(err);
                    break;
                }
            };

            // Check for stream termination
            if event.data.trim() == "[DONE]" {
                break;
            }

            // Deserialize the SSE event data
            let chunk: StreamChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(e) => {
                    debug!("Failed to parse SSE event: {}", e);
                    continue;
                }
            };

            // Capture usage info
            if let Some(usage) = chunk.usage {
                total_tokens = usage.total_tokens;
            }

            // Process choices
            for choice in chunk.choices {
                let delta = choice.delta;

                // Emit reasoning, content, and accumulate tool calls in a single match.
                match (&delta.content, &delta.reasoning_content) {
                    (Some(content), Some(reasoning))
                        if !content.trim().is_empty() && !reasoning.is_empty() =>
                    {
                        // Both present: emit content to text stream, reasoning to reasoning stream.
                        final_text.push_str(content);
                        delta_count += 1;
                        delta_callback(content.as_str());
                        final_reasoning.push_str(reasoning);
                        if let Some(ref mut cb) = reasoning_callback {
                            cb(reasoning.as_str());
                        }
                    }
                    (Some(content), None) if !content.trim().is_empty() => {
                        final_text.push_str(content);
                        delta_count += 1;
                        delta_callback(content.as_str());
                    }
                    (None, Some(reasoning)) if !reasoning.is_empty() => {
                        final_reasoning.push_str(reasoning);
                        if let Some(ref mut cb) = reasoning_callback {
                            cb(reasoning.as_str());
                        }
                    }
                    _ => {}
                }

                // Accumulate tool call deltas (indexed by field to handle out-of-order).
                if let Some(ref tool_deltas) = delta.tool_calls {
                    for tc in tool_deltas {
                        let idx = tc.index;
                        // Ensure vectors are large enough.
                        while tool_call_args.len() <= idx {
                            tool_call_args.push(String::new());
                            tool_call_names.push(String::new());
                            tool_call_ids.push(String::new());
                        }
                        if let Some(ref func) = tc.function {
                            if let Some(ref name) = func.name {
                                tool_call_names[idx] = name.clone();
                            }
                            if let Some(ref args) = func.arguments {
                                tool_call_args[idx].push_str(args);
                            }
                        }
                        if let Some(ref tc_id) = tc.id {
                            tool_call_ids[idx] = tc_id.clone();
                        }
                    }
                }
            }
        }

        // Propagate any stream error that occurred during iteration
        if let Some(err) = stream_error {
            tracing::error!("[stream_chat] SSE stream failed: {}", err);
            return Err(err);
        }

        // Build final tool_calls from accumulated deltas
        let mut tool_calls: Vec<TransportToolCall> = Vec::new();
        for (idx, (name, args)) in tool_call_names
            .iter()
            .zip(tool_call_args.iter())
            .enumerate()
        {
            let args_json = if args.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!(args))
            };
            let tc_id = if idx < tool_call_ids.len() && !tool_call_ids[idx].is_empty() {
                tool_call_ids[idx].clone()
            } else {
                format!("call_{}", idx)
            };
            tool_calls.push(TransportToolCall {
                id: tc_id,
                tool_name: name.clone(),
                arguments: args_json,
            });
        }

        // Fallback: if no structured tool_calls, try parsing text content
        if tool_calls.is_empty() && !final_text.trim().is_empty() {
            let fallback = super::text_tool_parser::parse_tool_calls_from_text(&final_text);
            if !fallback.is_empty() {
                tracing::debug!(
                    "Fallback: parsed {} tool call(s) from text content",
                    fallback.len()
                );
            }
            tool_calls = fallback;
        }

        tracing::trace!(
            delta_count,
            final_text_len = final_text.len(),
            final_text_preview = ?final_text.chars().take(80).collect::<String>(),
            "[ChatCompletions] stream_chat completed"
        );

        Ok(TransportResponse {
            text: final_text,
            tool_calls,
            tokens_used: total_tokens,
            reasoning: if final_reasoning.is_empty() { None } else { Some(final_reasoning) },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_data_url_long() {
        let input = r#"{"prompt":"hello","image":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}}"#;
        let output = mask_data_urls(input);
        assert!(output.contains("[data URL"));
        assert!(!output.contains("iVBORw0KGgo"));
        assert!(output.contains("bytes]"));
    }

    #[test]
    fn test_mask_data_url_short() {
        let input = r#"{"image":{"url":"data:image/png;base64,abc123def456"}}"#;
        let output = mask_data_urls(input);
        assert!(output.contains("[data URL"));
        assert!(!output.contains("abc123def456"));
    }

    #[test]
    fn test_mask_no_data_url() {
        let input = r#"{"prompt":"hello","role":"user"}"#;
        let output = mask_data_urls(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_message_role_json() {
        let msg = Message::system("sys");
        let json = message_to_json(&msg);
        assert_eq!(json["role"], "system");
        assert_eq!(json["content"], "sys");

        let msg = Message::tool_result("call-1", "result");
        let json = message_to_json(&msg);
        assert_eq!(json["role"], "tool");
        assert_eq!(json["content"], "result");
    }

    #[test]
    fn test_resolve_request_fresh() {
        let session_id = String::from("test-session");
        let messages = vec![Message::user("hello")];
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "be helpful"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        let (json_len, model) = resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Fresh(session_id.clone()),
            &template,
            |req| {
                let arr = req["messages"].as_array().unwrap();
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0]["role"], "system");
                assert_eq!(arr[1]["content"], "hello");
                assert_eq!(req["model"], "test-model");
                (1, req["model"].clone())
            },
        );
        assert_eq!(json_len, 1); // user message only (system is in template)
        assert_eq!(model, "test-model");
        assert_eq!(cached[&session_id].msg_count, 1);
    }

    #[test]
    fn test_resolve_request_incremental_grown() {
        let session_id = String::from("test-session");
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "be helpful"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        // Fresh: 1 user message
        let messages = vec![Message::user("hello")];
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Fresh(session_id.clone()),
            &template,
            |_| (),
        );

        // Incremental: add 2 more
        let mut messages = messages.clone();
        messages.push(Message::assistant("hi there"));
        messages.push(Message::user("follow up"));
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Incremental(session_id.clone()),
            &template,
            |req| {
                let arr = req["messages"].as_array().unwrap();
                assert_eq!(arr.len(), 4); // system + 3 messages
                assert_eq!(arr[0]["role"], "system");
                assert_eq!(arr[3]["content"], "follow up");
            },
        );
        assert_eq!(cached[&session_id].msg_count, 3);
    }

    #[test]
    fn test_resolve_request_incremental_reset() {
        let session_id = String::from("test-session");
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "be helpful"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        // Fresh: 2 messages (system in template + user)
        let messages = vec![
            Message::user("hello"),
            Message::assistant("hi"),
        ];
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Fresh(session_id.clone()),
            &template,
            |_| (),
        );

        // Incremental: removed one — should reset
        let mut messages = messages;
        messages.pop();
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Incremental(session_id.clone()),
            &template,
            |req| {
                assert_eq!(req["messages"].as_array().unwrap().len(), 2);
            },
        );
        assert_eq!(cached[&session_id].msg_count, 1);
    }

    #[test]
    fn test_resolve_request_incremental_equal() {
        let session_id = String::from("test-session");
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "be helpful"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        // Fresh: 1 message
        let messages = vec![Message::user("hello")];
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Fresh(session_id.clone()),
            &template,
            |_| (),
        );

        // Incremental: same count but content changed — should reset
        let messages = vec![Message::user("changed")];
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Incremental(session_id.clone()),
            &template,
            |req| {
                assert_eq!(req["messages"].as_array().unwrap().len(), 2);
                assert_eq!(req["messages"][1]["content"], "changed");
            },
        );
        assert_eq!(cached[&session_id].msg_count, 1);
    }

    #[test]
    fn test_per_session_isolation() {
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "default-system"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        let messages_a = vec![Message::user("hello-a")];
        resolve_request(
            &mut cached,
            &messages_a,
            &oben_models::CallMode::Fresh("session-a".into()),
            &template,
            |_| (),
        );

        let messages_b = vec![Message::user("hello-b")];
        resolve_request(
            &mut cached,
            &messages_b,
            &oben_models::CallMode::Fresh("session-b".into()),
            &template,
            |_| (),
        );

        assert_eq!(cached["session-a"].msg_count, 1);
        assert_eq!(cached["session-b"].msg_count, 1);
        // Both sessions share the same template system
        assert_eq!(
            cached["session-a"].request["messages"][0]["content"],
            "default-system"
        );
        assert_eq!(
            cached["session-b"].request["messages"][0]["content"],
            "default-system"
        );
    }

    #[test]
    fn test_in_place_extend_no_clone() {
        let session_id = String::from("test-session");
        let template = std::sync::Arc::new(json!({
            "model": "test-model",
            "messages": serde_json::Value::Array(vec![json!({"role": "system", "content": "be helpful"})]),
            "temperature": 0.7,
            "max_tokens": 8192,
        }));
        let mut cached = std::collections::HashMap::new();

        // Fresh: 2 non-system messages (system is in template)
        let messages = vec![Message::user("hello"), Message::assistant("hey")];
        resolve_request(
            &mut cached,
            &messages,
            &oben_models::CallMode::Fresh(session_id.clone()),
            &template,
            |_| (),
        );

        // Incremental: extend in-place by 1
        let mut messages2 = messages.clone();
        messages2.push(Message::user("follow up"));
        resolve_request(
            &mut cached,
            &messages2,
            &oben_models::CallMode::Incremental(session_id.clone()),
            &template,
            |req| {
                let arr = req["messages"].as_array().unwrap();
                assert_eq!(arr.len(), 4); // system + 3 user/assistant
                assert_eq!(arr[0]["role"], "system");
                assert_eq!(arr[2]["role"], "assistant");
                assert_eq!(arr[3]["role"], "user");
            },
        );
    }
}
