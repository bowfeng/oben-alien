//! Google Gemini native transport — Google AI Studio API (API key auth)
//!
//! Wraps Google's native Gemini REST API behind the TransportProvider trait,
//! converting OpenAI-style `messages[]` / `tools[]` into Gemini's native schema.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::debug;

use oben_models::{
    CallMode, Message, MessageContent, MessagePart, MessageRole, StreamDeltaCallback, ToolMeta,
    ToolParameters, TransportProvider, TransportResponse, TransportToolCall,
};

// ---------------------------------------------------------------------------
// Gemini types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum GeminiPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "functionCall")]
    FunctionCall {
        name: String,
        args: Option<serde_json::Value>,
    },
    #[serde(rename = "functionResponse")]
    FunctionResponse {
        name: String,
        response: Option<serde_json::Value>,
    },
    #[serde(rename = "inlineData")]
    InlineData {
        #[serde(rename = "mimeType")]
        mime_type: String,
        data: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiContentPart {
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiTools {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    fcc: GeminiFcc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFcc {
    #[serde(rename = "mode")]
    mode: String,
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiThinkingConfig {
    #[serde(rename = "includeThoughts")]
    include_thoughts: Option<bool>,
    #[serde(rename = "thinkingLevel")]
    thinking_level: Option<String>,
    #[serde(rename = "thinkingBudget")]
    thinking_budget: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiRequest {
    contents: Vec<GeminiContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTools>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum GeminiFinishReason {
    #[serde(rename = "STOP")]
    Stop,
    #[serde(rename = "MAX_TOKENS")]
    MaxTokens,
    #[serde(rename = "SAFETY")]
    Safety,
    #[serde(rename = "RECITATION")]
    Recitation,
    #[serde(rename = "OTHER")]
    Other,
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    Unspecified,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContentPart>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<GeminiFinishReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<usize>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<usize>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<usize>,
    #[serde(rename = "cachedContentTokenCount")]
    cached_content_token_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

type GeminiStreamEvent = GeminiResponse;

// ---------------------------------------------------------------------------
// Translate OpenAI messages → Gemini contents
// ---------------------------------------------------------------------------

fn translate_messages(
    messages: &[Message],
) -> (Vec<GeminiContentPart>, Option<GeminiSystemInstruction>) {
    let mut sys_lines: Vec<String> = Vec::new();
    let mut contents: Vec<GeminiContentPart> = Vec::new();

    for msg in messages {
        if !msg.include_in_prompt {
            continue;
        }
        match msg.role {
            MessageRole::System => {
                let text = match &msg.content {
                    MessageContent::Text(t) => t.clone(),
                    MessageContent::Parts(ps) => ps
                        .iter()
                        .filter_map(|p| match p {
                            MessagePart::Text(t) => Some(t.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    MessageContent::Image { .. } => String::new(),
                };
                if !text.is_empty() {
                    sys_lines.push(text);
                }
            }
            MessageRole::Tool => {
                let output = match &msg.content {
                    MessageContent::Text(t) => t.clone(),
                    _ => String::new(),
                };
                let name = msg
                    .tool_call_ids
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "tool".into());
                let response = if output.starts_with('{') || output.starts_with('[') {
                    serde_json::from_str(&output).unwrap_or(serde_json::json!({"output": output}))
                } else {
                    serde_json::json!({"output": output})
                };
                contents.push(GeminiContentPart {
                    role: Some("user".into()),
                    parts: vec![GeminiPart::FunctionResponse {
                        name,
                        response: Some(response),
                    }],
                });
            }
            _ => {
                let inner_role = if msg.role == MessageRole::Assistant {
                    "model"
                } else {
                    "user"
                };
                let mut parts: Vec<GeminiPart> = Vec::new();
                match &msg.content {
                    MessageContent::Text(t) => {
                        if !t.is_empty() {
                            parts.push(GeminiPart::Text { text: t.clone() });
                        }
                    }
                    MessageContent::Parts(ps) => {
                        for p in ps {
                            match p {
                                MessagePart::Text(t) => {
                                    if !t.is_empty() {
                                        parts.push(GeminiPart::Text { text: t.clone() });
                                    }
                                }
                                MessagePart::Image { url, .. } => {
                                    if let Some(b64) =
                                        url.splitn(2, ',').nth(1).map(|s| s.to_string())
                                    {
                                        let mime = url
                                            .splitn(2, ';')
                                            .next()
                                            .and_then(|h| h.strip_prefix("data:"))
                                            .unwrap_or("image/png")
                                            .trim()
                                            .to_string();
                                        parts.push(GeminiPart::InlineData {
                                            mime_type: mime,
                                            data: b64,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    MessageContent::Image { url, .. } => {
                        if let Some(b64) = url.splitn(2, ',').nth(1).map(|s| s.to_string()) {
                            parts.push(GeminiPart::InlineData {
                                mime_type: "image/png".into(),
                                data: b64,
                            });
                        }
                    }
                }
                // Tool calls on assistant messages
                if let Some(tc_list) = &msg.tool_calls {
                    for call in tc_list {
                        let args = match &call.arguments {
                            serde_json::Value::String(s) => serde_json::from_str(s)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            v => v.clone(),
                        };
                        parts.push(GeminiPart::FunctionCall {
                            name: call.tool_name.clone(),
                            args: Some(args),
                        });
                    }
                }
                if !parts.is_empty() {
                    contents.push(GeminiContentPart {
                        role: Some(inner_role.into()),
                        parts,
                    });
                }
            }
        }
    }

    let sys = if sys_lines.is_empty() {
        None
    } else {
        let text = sys_lines.join("\n");
        Some(GeminiSystemInstruction {
            parts: vec![GeminiPart::Text { text }],
        })
    };

    (contents, sys)
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

fn translate_tools(tools: &[ToolMeta]) -> Option<Vec<GeminiTools>> {
    let decls: Vec<GeminiFunctionDeclaration> = tools
        .iter()
        .filter_map(|t| {
            let params = match &t.parameters {
                ToolParameters::JsonSchema { schema } => schema.clone(),
                _ => serde_json::json!({"type": "object", "properties": {}}),
            };
            Some(GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: Some(t.description.clone()),
                parameters: Some(sanitize_schema(params)),
            })
        })
        .collect();
    if decls.is_empty() {
        None
    } else {
        Some(vec![GeminiTools {
            function_declarations: decls,
        }])
    }
}

fn sanitize_schema(schema: serde_json::Value) -> serde_json::Value {
    const KEYS: [&str; 22] = [
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
        "propertyOrdering",
        "default",
        "items",
        "minimum",
        "maximum",
    ];
    let allowed: std::collections::HashSet<&str> = KEYS.into_iter().collect();

    match schema {
        serde_json::Value::Object(m) => {
            let mut c = serde_json::Map::new();
            for (k, v) in m {
                if !allowed.contains(k.as_str()) {
                    continue;
                }
                if k == "properties" {
                    if let serde_json::Value::Object(ps) = v {
                        c.insert(
                            k.clone(),
                            serde_json::Value::Object(
                                ps.into_iter()
                                    .map(|(a, b)| (a, sanitize_schema(b)))
                                    .collect(),
                            ),
                        );
                    }
                } else if k == "items" {
                    c.insert(k.clone(), sanitize_schema(v));
                } else if k == "anyOf" {
                    if let serde_json::Value::Array(arr) = v {
                        c.insert(
                            k.clone(),
                            serde_json::Value::Array(
                                arr.into_iter()
                                    .filter_map(|x| {
                                        if x.is_object() {
                                            Some(sanitize_schema(x))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            ),
                        );
                    }
                } else {
                    c.insert(k, v);
                }
            }
            // Drop non-string enum entries for typed parents
            if let (Some(serde_json::Value::Array(en)), Some(serde_json::Value::String(tp))) =
                (c.get("enum"), c.get("type"))
            {
                if tp != "string" && en.iter().any(|x| !x.is_string()) {
                    c.remove("enum");
                }
            }
            serde_json::Value::Object(c)
        }
        _ => schema,
    }
}

fn translate_tool_choice(tc: Option<&str>) -> Option<GeminiToolConfig> {
    match tc {
        Some("auto") => Some(GeminiToolConfig {
            fcc: GeminiFcc {
                mode: "AUTO".into(),
                allowed_function_names: None,
            },
        }),
        Some("required") | Some("any") => Some(GeminiToolConfig {
            fcc: GeminiFcc {
                mode: "ANY".into(),
                allowed_function_names: None,
            },
        }),
        Some("none") => Some(GeminiToolConfig {
            fcc: GeminiFcc {
                mode: "NONE".into(),
                allowed_function_names: None,
            },
        }),
        Some(s) => {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(s) {
                if let Some(fn_obj) = obj.get("function") {
                    if let Some(name) = fn_obj.get("name").and_then(|v| v.as_str()) {
                        return Some(GeminiToolConfig {
                            fcc: GeminiFcc {
                                mode: "ANY".into(),
                                allowed_function_names: Some(vec![name.to_string()]),
                            },
                        });
                    }
                }
            }
            None
        }
        None => None,
    }
}

fn normalize_thinking(cfg: &serde_json::Value) -> Option<GeminiThinkingConfig> {
    if !cfg.is_object() {
        return None;
    }
    let o = cfg.as_object().unwrap();
    let mut nc = GeminiThinkingConfig {
        include_thoughts: None,
        thinking_level: None,
        thinking_budget: None,
    };
    if let Some(v) = o
        .get("thinkingBudget")
        .or_else(|| o.get("thinking_budget"))
        .and_then(|v| v.as_i64())
    {
        nc.thinking_budget = Some(v);
    }
    if let Some(v) = o
        .get("includeThoughts")
        .or_else(|| o.get("include_thoughts"))
        .and_then(|v| v.as_bool())
    {
        nc.include_thoughts = Some(v);
    }
    if let Some(s) = o
        .get("thinkingLevel")
        .or_else(|| o.get("thinking_level"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
    {
        nc.thinking_level = Some(s.trim().to_lowercase());
    }
    if nc.include_thoughts.is_some() || nc.thinking_level.is_some() || nc.thinking_budget.is_some()
    {
        Some(nc)
    } else {
        None
    }
}

fn build_gen(
    temperature: Option<f64>,
    max_tokens: Option<usize>,
    top_p: Option<f64>,
    stop: Option<Vec<String>>,
    thinking: Option<GeminiThinkingConfig>,
) -> Option<GeminiGenerationConfig> {
    let gc = GeminiGenerationConfig {
        temperature,
        max_output_tokens: max_tokens,
        top_p,
        stop_sequences: stop,
        thinking_config: thinking,
    };
    if gc.temperature.is_none()
        && gc.max_output_tokens.is_none()
        && gc.top_p.is_none()
        && gc.stop_sequences.is_none()
        && gc.thinking_config.is_none()
    {
        None
    } else {
        Some(gc)
    }
}

// ---------------------------------------------------------------------------
// Translate Gemini response → TransportResponse
// ---------------------------------------------------------------------------

fn translate_response(resp: &GeminiResponse, _model: &str) -> TransportResponse {
    if resp.candidates.is_empty() {
        return TransportResponse {
            text: String::new(),
            tool_calls: Vec::new(),
            tokens_used: None,
            reasoning: None,
        };
    }
    let cand = &resp.candidates[0];
    let parts = match &cand.content {
        Some(c) => &c.parts,
        None => {
            return TransportResponse {
                text: String::new(),
                tool_calls: Vec::new(),
                tokens_used: None,
                reasoning: None,
            }
        }
    };

    let mut texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<TransportToolCall> = Vec::new();

    for part in parts {
        match part {
            GeminiPart::Text { text } => {
                if !text.is_empty() {
                    texts.push(text.clone());
                }
            }
            GeminiPart::FunctionCall { name, args } => {
                let args_json = args
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                    .unwrap_or_else(|| "{}".into());
                let id = short_id(8);
                tool_calls.push(TransportToolCall {
                    id,
                    tool_name: name.clone(),
                    arguments: serde_json::Value::String(args_json),
                });
            }
            _ => {}
        }
    }

    let is_text_empty = texts.iter().all(|t| t.trim().is_empty());
    let r = if tool_calls.is_empty() {
        TransportResponse {
            text: texts.join("\n"),
            tool_calls: Vec::new(),
            tokens_used: None,
            reasoning: None,
        }
    } else if is_text_empty {
        TransportResponse {
            text: String::new(),
            tool_calls,
            tokens_used: None,
            reasoning: None,
        }
    } else {
        let mut r = TransportResponse {
            text: texts.join("\n"),
            tool_calls: Vec::new(),
            tokens_used: None,
            reasoning: None,
        };
        r.tool_calls = tool_calls;
        r
    };

    r
}

fn short_id(n: usize) -> String {
    let v = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    format!("{:016x}", v).chars().take(n).collect()
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

struct TSlot {
    _index: usize,
    id: String,
    last: String,
}

#[derive(Default)]
struct StreamUsage {
    total: usize,
}

fn stream_delta(
    event: &GeminiResponse,
    slots: &mut std::collections::HashMap<String, TSlot>,
    saw_tool: &mut bool,
) -> (Vec<String>, Vec<TransportToolCall>, StreamUsage) {
    let mut texts = Vec::new();
    let mut tcs = Vec::new();
    let u = StreamUsage {
        total: event
            .usage_metadata
            .as_ref()
            .and_then(|m| m.total_token_count)
            .unwrap_or(0),
    };

    for cand in &event.candidates {
        if let Some(ref content) = cand.content {
            for part in &content.parts {
                match part {
                    GeminiPart::Text { text } => {
                        if !text.is_empty() {
                            texts.push(text.clone());
                        }
                    }
                    GeminiPart::FunctionCall { name, args } => {
                        let a = args
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                            .unwrap_or_else(|| "{}".into());
                        let key = format!("{}_{}", 0, name);
                        let ni = slots.len();
                        let id = short_id(8);
                        let prev = slots.entry(key.clone()).or_insert(TSlot {
                            _index: ni,
                            id,
                            last: String::new(),
                        });
                        let emitted = if a.starts_with(&prev.last) {
                            a[prev.last.len()..].to_string()
                        } else {
                            a.clone()
                        };
                        prev.last = a;
                        *saw_tool = true;
                        tcs.push(TransportToolCall {
                            id: prev.id.clone(),
                            tool_name: name.clone(),
                            arguments: serde_json::Value::String(emitted),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    (texts, tcs, u)
}

// ---------------------------------------------------------------------------
// Transport implementation
// ---------------------------------------------------------------------------

struct CachedReq {
    _json_str: String,
}

/// Google Gemini native transport.
pub struct GeminiMessagesTransport {
    name: Arc<String>,
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    tools: Vec<ToolMeta>,
    cached: std::sync::Mutex<Option<CachedReq>>,
    extra_body: Option<serde_json::Value>,
}

impl GeminiMessagesTransport {
    pub fn new(api_key: String, base_url: String, model: String) -> Self {
        Self {
            name: Arc::new("gemini-native".into()),
            client: reqwest::Client::builder()
                .pool_max_idle_per_host(20)
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .tcp_nodelay(true)
                .http2_keep_alive_interval(std::time::Duration::from_secs(30))
                .http2_keep_alive_timeout(std::time::Duration::from_secs(10))
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|e| panic!("reqwest: {}", e)),
            base_url,
            api_key,
            model,
            tools: Vec::new(),
            cached: std::sync::Mutex::new(None),
            extra_body: None,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolMeta>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_extra_body(mut self, body: serde_json::Value) -> Self {
        self.extra_body = Some(body);
        self
    }

    fn build_request(&self, messages: &[Message]) -> Result<GeminiRequest> {
        let (contents, sys) = translate_messages(messages);
        let tools = translate_tools(&self.tools);
        let tconf = self
            .extra_body
            .as_ref()
            .and_then(|e| e.get("tool_choice").and_then(|v| v.as_str()))
            .map(|s| translate_tool_choice(Some(s)))
            .flatten();

        let mut temperature: Option<f64> = None;
        let mut max_tokens: Option<usize> = None;
        let mut top_p: Option<f64> = None;
        let mut stop: Option<Vec<String>> = None;
        let mut thinking: Option<GeminiThinkingConfig> = None;

        if let Some(ref extra) = self.extra_body {
            if let Some(v) = extra.get("temperature").and_then(|v| v.as_f64()) {
                temperature = Some(v);
            }
            if let Some(v) = extra
                .get("max_tokens")
                .and_then(|v| v.as_u64().map(|u| u as usize))
            {
                max_tokens = Some(v);
            }
            if let Some(v) = extra.get("top_p").and_then(|v| v.as_f64()) {
                top_p = Some(v);
            }
            if let Some(arr) = extra.get("stop_sequences").and_then(|v| v.as_array()) {
                stop = Some(
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect(),
                );
            }
            if let Some(tc) = extra.get("thinking_config").or(extra.get("thinkingConfig")) {
                thinking = normalize_thinking(tc);
            }
        }

        let gen = build_gen(temperature, max_tokens, top_p, stop, thinking);
        Ok(GeminiRequest {
            contents,
            tools,
            tool_config: tconf,
            system_instruction: sys,
            generation_config: gen,
        })
    }

    fn resolve(&self, messages: &[Message]) -> Result<serde_json::Value> {
        let mut guard = self.cached.lock().unwrap();
        let req = self.build_request(messages)?;
        let json = serde_json::to_value(&req)?;
        *guard = Some(CachedReq {
            _json_str: serde_json::to_string(&req)?,
        });
        Ok(json)
    }

    fn url(&self, model: &str, stream: bool) -> String {
        let base = self.base_url.trim_end_matches('/');
        if stream {
            format!("{base}/models/{model}:streamGenerateContent?alt=sse")
        } else {
            format!("{base}/models/{model}:generateContent")
        }
    }
}

#[async_trait::async_trait]
impl TransportProvider for GeminiMessagesTransport {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, messages: &[Message], _mode: &CallMode) -> Result<TransportResponse> {
        let json = self.resolve(messages)?;
        let url = self.url(&self.model, false);
        debug!("Gemini POST {url}");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&json)
            .send()
            .await?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!("Gemini error {status}: {body}"));
        }

        let gr: GeminiResponse =
            serde_json::from_str(&body).map_err(|e| anyhow!("Invalid Gemini response: {e}"))?;

        let mut r = translate_response(&gr, &self.model);
        if let Some(um) = &gr.usage_metadata {
            r.tokens_used = Some(um.total_token_count.unwrap_or(0));
        }
        Ok(r)
    }

    async fn stream_chat(
        &self,
        messages: &[Message],
        _mode: &CallMode,
        mut delta_callback: StreamDeltaCallback,
        _reasoning_callback: Option<oben_models::StreamReasoningCallback>,
    ) -> Result<TransportResponse> {
        let json = self.resolve(messages)?;
        let url = self.url(&self.model, true);
        debug!("Gemini stream {url}");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&json)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await?;
            return Err(anyhow!("Gemini stream error {status}: {body}"));
        }

        let mut full_text = String::new();
        let mut all_tc: Vec<TransportToolCall> = Vec::new();
        let mut slots: std::collections::HashMap<String, TSlot> = std::collections::HashMap::new();
        let mut usage = StreamUsage::default();
        let mut saw_tool = false;

        let mut stream = resp.bytes_stream();
        let mut buf = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| anyhow!("stream: {e}"))?;
            buf.extend_from_slice(&chunk);

            let s = String::from_utf8_lossy(&buf);
            if let Some(newline_pos) = s.find('\n') {
                let line = s[..newline_pos].to_string();
                buf.drain(..newline_pos + 1);
                if line.starts_with("data: ") {
                    let remaining = &line[6..];
                    if !remaining.trim().is_empty() && remaining.trim() != "[DONE]" {
                        match serde_json::from_str::<GeminiStreamEvent>(remaining.trim()) {
                            Ok(event) => {
                                let (texts, tcs, u) =
                                    stream_delta(&event, &mut slots, &mut saw_tool);
                                for t in &texts {
                                    full_text.push_str(t);
                                    delta_callback(t.as_str());
                                }
                                all_tc.extend(tcs);
                                if u.total > usage.total {
                                    usage = u;
                                }
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }

        // Aggregate tool calls by name
        let mut agg: std::collections::HashMap<String, TransportToolCall> =
            std::collections::HashMap::new();
        for tc in all_tc {
            let entry = agg
                .entry(tc.tool_name.clone())
                .or_insert(TransportToolCall {
                    id: tc.id,
                    tool_name: tc.tool_name.clone(),
                    arguments: serde_json::json!(""),
                });
            if let (serde_json::Value::String(ref mut base), serde_json::Value::String(args)) =
                (&mut entry.arguments, tc.arguments)
            {
                base.push_str(&args);
            }
        }

        let tools_vec: Vec<TransportToolCall> = agg.into_values().collect();
        let has_text = !full_text.is_empty();
        let mut r = if tools_vec.is_empty() {
            TransportResponse {
                text: full_text,
                tool_calls: Vec::new(),
                tokens_used: None,
                reasoning: None,
            }
        } else {
            TransportResponse {
                text: if has_text { full_text } else { String::new() },
                tool_calls: tools_vec,
                tokens_used: None,
                reasoning: None,
            }
        };

        r.tokens_used = Some(usage.total);
        Ok(r)
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| {
                let (len, has_img) = match &m.content {
                    MessageContent::Text(s) => (s.len(), false),
                    MessageContent::Parts(ps) => ps
                        .iter()
                        .map(|p| match p {
                            MessagePart::Text(s) => (s.len(), false),
                            MessagePart::Image { .. } => (500, true),
                        })
                        .fold((0usize, false), |(acc, hi), (l, h)| (acc + l, hi || h)),
                    MessageContent::Image { .. } => (500, true),
                };
                if has_img {
                    500
                } else {
                    len / 4 + 5
                }
            })
            .sum()
    }
}
