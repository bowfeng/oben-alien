use serde::{Deserialize, Serialize};

use super::ToolCall;

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: MessageContent,
    pub id: Option<i64>,
    /// Tool call IDs this message references (for tool results).
    pub tool_call_ids: Vec<String>,
    /// Tool calls made by the assistant in this message.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Reasoning/thinking chain-of-thought from models that support it.
    /// Separated from content so the TUI can render it as a distinct block.
    pub reasoning: Option<String>,
    /// Numeric parent task ID for subagent tool result messages, used by the TUI to group subagent output into visual blocks.
    pub delegation_id: Option<u32>,
    pub tool_error: bool,
    #[serde(default)]
    pub include_in_prompt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageContent {
    Text(String),
    Image {
        url: String,
        detail: Option<String>,
    },
    /// Multiple parts (text + image) in one message.
    Parts(Vec<MessagePart>),
}

impl MessageContent {
    /// Get the text content if this is a Text variant.
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(t) => t.clone(),
            _ => String::new(),
        }
    }

    /// Get the text content if this is a Text variant (zero-copy).
    pub fn to_text_ref(&self) -> Option<&str> {
        match self {
            MessageContent::Text(t) => Some(t.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessagePart {
    Text(String),
    Image { url: String, detail: Option<String> },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(text.into()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(text.into()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(String::new()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(tool_calls),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(text.into()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(output.into()),
            id: None,
            tool_call_ids: vec![tool_call_id.into()],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    pub fn tool_error_result(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(error.into()),
            id: None,
            tool_call_ids: vec![tool_call_id.into()],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: true,
            include_in_prompt: true,
        }
    }

    pub fn with_delegation_id(mut self, id: u32) -> Self {
        self.delegation_id = Some(id);
        self
    }

    pub fn set_include_in_prompt(mut self, include: bool) -> Self {
        self.include_in_prompt = include;
        self
    }

    pub fn delegation_id(&self) -> Option<u32> {
        self.delegation_id
    }

    pub fn is_tool_error(&self) -> bool {
        self.tool_error
    }

    pub fn is_included_in_prompt(&self) -> bool {
        self.include_in_prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_message_roundtrip_json() {
        let msg = Message::user("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn test_assistant_message_roundtrip_json() {
        let msg = Message::assistant("i can help with that");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn test_system_message_roundtrip_json() {
        let msg = Message::system("you are a helpful assistant");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn test_tool_result_message_roundtrip_json() {
        let msg = Message::tool_result("call-123", "output data");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn test_include_in_prompt_field() {
        let msg = Message::user("test");
        assert!(msg.include_in_prompt);
        assert!(msg.is_included_in_prompt());

        let msg = msg.set_include_in_prompt(false);
        assert!(!msg.include_in_prompt);
        assert!(!msg.is_included_in_prompt());
    }
}
