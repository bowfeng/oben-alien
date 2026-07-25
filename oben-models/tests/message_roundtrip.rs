use oben_models::{Message, MessageContent, MessagePart, MessageRole, ToolMeta, ToolCall, ToolResult};

// ─── Message serialization ──────────────────────────────────────────

#[test]
fn system_message_roundtrip_json() {
    let msg = Message::system("You are a helpful assistant.");
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.role, MessageRole::System);
    assert_eq!(
        restored.content,
        MessageContent::Text("You are a helpful assistant.".to_string())
    );
}

#[test]
fn user_message_roundtrip_json() {
    let msg = Message::user("Hello, world!");
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.role, MessageRole::User);
    assert_eq!(
        restored.content,
        MessageContent::Text("Hello, world!".to_string())
    );
}

#[test]
fn assistant_message_roundtrip_json() {
    let msg = Message::assistant("I can help with that.");
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.role, MessageRole::Assistant);
}

#[test]
fn tool_result_message_roundtrip_json() {
    let msg = Message::tool_result("call-123", "File written successfully");
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.role, MessageRole::Tool);
    assert_eq!(restored.tool_call_ids, vec!["call-123".to_string()]);
}

#[test]
fn message_with_image_content_roundtrip_json() {
    let msg = Message {
        role: MessageRole::User,
        content: MessageContent::Image {
            url: "https://example.com/cat.jpg".to_string(),
            detail: Some("high".to_string()),
        },
        id: None,
        tool_call_ids: vec![],
        tool_calls: None,
        reasoning: None,
        delegation_id: None,
        tool_error: false,
    };
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(
        restored.content,
        MessageContent::Image {
            url: "https://example.com/cat.jpg".to_string(),
            detail: Some("high".to_string()),
        }
    );
}

#[test]
fn message_with_parts_roundtrip_json() {
    let msg = Message {
        role: MessageRole::User,
        content: MessageContent::Parts(vec![
            MessagePart::Text("Describe this image".to_string()),
            MessagePart::Image {
                url: "https://example.com/photo.png".to_string(),
                detail: None,
            },
        ]),
        id: None,
        tool_call_ids: vec![],
        tool_calls: None,
        reasoning: None,
        delegation_id: None,
        tool_error: false,
    };
    let json = serde_json::to_string(&msg).unwrap();
    let restored: Message = serde_json::from_str(&json).unwrap();
    assert!(matches!(restored.content, MessageContent::Parts(_)));
    let parts = match restored.content {
        MessageContent::Parts(p) => p,
        _ => panic!("expected Parts"),
    };
    assert_eq!(parts.len(), 2);
}

// ─── Full conversation sequence ─────────────────────────────────────

#[test]
fn full_conversation_sequence_roundtrip() {
    let messages = vec![
        Message::system("You are helpful."),
        Message::user("What is 2+2?"),
        Message::assistant("Let me calculate."),
    ];
    let json = serde_json::to_string(&messages).unwrap();
    let restored: Vec<Message> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0].role, MessageRole::System);
    assert_eq!(restored[1].role, MessageRole::User);
    assert_eq!(restored[2].role, MessageRole::Assistant);
}

// ─── YAML round-trip ────────────────────────────────────────────────

#[test]
fn message_roundtrip_yaml() {
    let msg = Message::user("test yaml roundtrip");
    let yaml = serde_yaml::to_string(&msg).unwrap();
    let restored: Message = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(
        restored.content,
        MessageContent::Text("test yaml roundtrip".to_string())
    );
}

// ─── ToolCall and ToolResult round-trip ─────────────────────────────

#[test]
fn tool_call_roundtrip_json() {
    let call = ToolCall {
        id: "call-456".to_string(),
        tool_name: "shell".to_string(),
        arguments: serde_json::json!({"command": "ls -la", "cwd": "/tmp"}),
    };
    let json = serde_json::to_string(&call).unwrap();
    let restored: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.id, "call-456");
    assert_eq!(restored.tool_name, "shell");
    assert_eq!(restored.arguments["command"], "ls -la");
}

#[test]
fn tool_result_roundtrip_json() {
    let result = ToolResult {
        call_id: "call-456".to_string(),
        output: "total 0\n".to_string(),
        error: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    let restored: ToolResult = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.call_id, "call-456");
    assert_eq!(restored.output, "total 0\n");
    assert!(restored.error.is_none());
}

#[test]
fn tool_result_with_error_roundtrip_json() {
    let result = ToolResult {
        call_id: "call-789".to_string(),
        output: "".to_string(),
        error: Some("command not found".to_string()),
    };
    let json = serde_json::to_string(&result).unwrap();
    let restored: ToolResult = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.error, Some("command not found".to_string()));
}

// ─── ToolBuilder → build → serialize ────────────────────────────────

#[test]
fn tool_builder_flat_params_roundtrip() {
    let tool = ToolMeta::builder("shell", "Execute shell commands")
        .param("command", "Shell command", "string", true)
        .param("cwd", "Working directory", "string", false)
        .build();
    let json = serde_json::to_string(&tool).unwrap();
    let restored: ToolMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.name, "shell");
    assert_eq!(restored.description, "Execute shell commands");
}

#[test]
fn tool_builder_json_schema_roundtrip() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "url": {"type": "string"}
        },
        "required": ["url"]
    });
    let tool = ToolMeta::builder("http_get", "Fetch a URL")
        .json_schema(schema)
        .build();
    let json = serde_json::to_string(&tool).unwrap();
    let restored: ToolMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.name, "http_get");
}
