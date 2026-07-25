/// Message sanitization — preprocessing before API calls.
///
/// Mirrors Hermes' message sanitization pipeline:
/// 1. Drop "thinking-only" assistant messages
/// 2. Merge consecutive user/system messages
use tracing::info;
use tracing::debug;
use oben_models::{Message, MessageContent, MessagePart, MessageRole};

/// Run the full sanitization pipeline on a message list.
///
/// For history messages (all except the last user message), runs thinking-only
/// removal and consecutive user message merging.
/// The **last user message** is preserved as-is — no merging, no content loss.
/// This ensures the most recent user input (potentially containing images/base64
/// data URLs) is never stripped.
pub fn sanitize_messages(messages: &mut Vec<Message>) {
    drop_thinking_only_assistant(messages);

    // Find the index of the last user message.
    let last_user_idx = messages.iter().rposition(|m| m.role == MessageRole::User);

    if let Some(last_user_idx) = last_user_idx {
        // Split into history and remainder (last user message + everything after it)
        let mut history: Vec<Message> = messages.drain(..last_user_idx).collect();
        let remainder: Vec<Message> = messages.drain(..).collect();

        // Merge consecutive users in history only
        merge_consecutive_user_messages(&mut history);

        history.extend(remainder);
        *messages = history;
    } else {
        // No user messages at all — nothing to change
    }

    // Note: merge_consecutive_user_messages is also called after drop_thinking_only_assistant
    // on the history portion, since history messages may now have consecutive users.

    // Merge consecutive assistant messages to prevent "2 assistant messages at end" error
    merge_consecutive_assistant_messages(messages);
}

/// Merge assistant messages that are separated only by system messages.
/// This prevents "2 assistant messages at end" API errors.
fn merge_consecutive_assistant_messages(messages: &mut Vec<Message>) {
    debug!("merge_consecutive_assistant_messages: called with {} messages", messages.len());
    let before_count = messages.len();
    
    let mut merged: Vec<Message> = Vec::new();
    
    for msg in messages.iter().cloned() {
        if msg.role == MessageRole::Assistant {
            // Find the last assistant message (only if separated by system messages)
            let mut last_assistant_idx = None;
            for i in (0..merged.len()).rev() {
                match merged[i].role {
                    MessageRole::Assistant => {
                        last_assistant_idx = Some(i);
                        break;
                    }
                    MessageRole::System => {
                        // Keep looking backwards for an assistant message
                        continue;
                    }
                    _ => {
                        // Non-system, non-assistant message found - stop looking
                        break;
                    }
                }
            }
            
            if let Some(last_idx) = last_assistant_idx {
                // Merge with the last assistant message
                let last = &mut merged[last_idx];
                
                // Merge tool_calls
                if let Some(ref new_tcs) = msg.tool_calls {
                    if let Some(ref mut last_tcs) = last.tool_calls {
                        for tc in new_tcs.iter() {
                            if !last_tcs.iter().any(|t| t.id == tc.id) {
                                last_tcs.push(tc.clone());
                            }
                        }
                    } else {
                        last.tool_calls = Some(new_tcs.clone());
                    }
                }
                // Merge content
                let new_content = msg.content.to_text();
                if !new_content.is_empty() {
                    let last_content = last.content.to_text();
                    last.content = MessageContent::Text(format!("{}\n{}", last_content, new_content));
                }
                continue;
            }
        }
        merged.push(msg);
    }
    
    *messages = merged;
    
    let after_count = messages.len();
    if before_count != after_count {
        debug!("merge_consecutive_assistant_messages: merged {} messages", before_count - after_count);
    }
}

/// Drop assistant messages that are "thinking-only" — they have reasoning
/// (empty content) but no visible text and no tool calls.
///
/// These cause API errors on providers that convert reasoning into thinking blocks.
pub fn drop_thinking_only_assistant(messages: &mut Vec<Message>) {
    info!("drop_thinking_only_assistant: called with {} messages", messages.len());
    let before_count = messages.len();
    let mut dropped_count = 0;
    messages.retain(|msg| {
        let is_thinking = is_thinking_only_assistant(msg);
        if msg.role == MessageRole::Assistant {
            info!("drop_thinking_only_assistant: checking assistant msg, is_thinking={}", is_thinking);
            if is_thinking {
                let text = msg.content.to_text();
                info!("drop_thinking_only_assistant: dropping empty msg content=\"{}\" len={}", text, text.len());
                dropped_count += 1;
            }
        }
        !is_thinking
    });
    if dropped_count > 0 {
        info!("drop_thinking_only_assistant: removed {} thinking-only messages", dropped_count);
    }
    let after_count = messages.len();
    if before_count != after_count {
        info!("drop_thinking_only_assistant: before={} after={} removed={}", before_count, after_count, before_count - after_count);
    }
}

pub fn is_thinking_only_assistant(msg: &Message) -> bool {
    if msg.role != MessageRole::Assistant {
        return false;
    }

    let text = msg.content.to_text();
    let is_empty = text.trim().is_empty();
    let has_tool_calls = msg.tool_calls.as_ref().is_some_and(|calls| !calls.is_empty());

    info!("is_thinking_only_assistant: is_empty={} has_tool_calls={} tool_calls_count={:?} -> {}", is_empty, has_tool_calls, msg.tool_calls.as_ref().map(|c| c.len()), is_empty && !has_tool_calls);
    is_empty && !has_tool_calls
}

/// Check whether a user message contains non-text content (images/parts with images).
fn user_message_has_image(msg: &Message) -> bool {
    match &msg.content {
        MessageContent::Image { .. } => true,
        MessageContent::Parts(parts) => {
            parts.iter().any(|p| matches!(p, MessagePart::Image { .. }))
        }
        MessageContent::Text(_) => false,
    }
}

/// Flush the pending text-based user message into the merged list.
fn flush_pending_user(merged: &mut Vec<Message>, pending_text: String) {
    merged.push(Message {
        role: MessageRole::User,
        content: MessageContent::Text(pending_text),
        id: None,
        tool_call_ids: vec![],
        tool_calls: None,
        reasoning: None,
        delegation_id: None,
        tool_error: false,
        include_in_prompt: true,
    });
}

/// Merge consecutive user messages into single messages.
///
/// Merges consecutive user messages that contain only plain text into one
/// to avoid provider-specific issues with message role alternation.
///
/// User messages that contain images (MessageContent::Image or MessageContent::Parts
/// with image parts) are NOT merged — they are emitted as-is to preserve the
/// base64 data URLs. System messages are NOT merged — they are emitted separately
/// to preserve system prompt integrity.
pub fn merge_consecutive_user_messages(messages: &mut Vec<Message>) {
    if messages.len() <= 1 {
        return;
    }

    let mut merged = Vec::with_capacity(messages.len());
    let mut pending_user: Option<String> = None;

    for msg in messages.drain(..) {
        match msg.role {
            MessageRole::User => {
                if user_message_has_image(&msg) {
                    // Flush any accumulated text messages first
                    if let Some(text) = pending_user.take() {
                        flush_pending_user(&mut merged, text);
                    }
                    // Keep image message as-is
                    merged.push(msg);
                } else {
                    // Plain text user message — merge into pending
                    let text = match &msg.content {
                        MessageContent::Text(t) => t.clone(),
                        MessageContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                MessagePart::Text(t) => Some(t.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                        _ => String::new(),
                    };

                    if let Some(ref mut pending) = pending_user {
                        pending.push_str("\n\n");
                        pending.push_str(&text);
                    } else {
                        pending_user = Some(text);
                    }
                }
            }
            MessageRole::System => {
                // System messages are NOT merged — emit pending user first,
                // then emit the system message as its own entry.
                if let Some(text) = pending_user.take() {
                    flush_pending_user(&mut merged, text);
                }
                merged.push(msg);
            }
            _ => {
                // Non-user/system message — flush pending user message first
                if let Some(text) = pending_user.take() {
                    flush_pending_user(&mut merged, text);
                }
                merged.push(msg);
            }
        }
    }

    // Flush any remaining user message
    if let Some(text) = pending_user {
        flush_pending_user(&mut merged, text);
    }

    *messages = merged;
}

/// Strip surrogate characters from a string.
pub fn strip_surrogates(text: &str) -> String {
    text.chars()
        .filter(|c| {
            let code = *c as u32;
            !(0xD800..=0xDFFF).contains(&code)
        })
        .collect()
}

/// Strip non-ASCII characters from a string.
pub fn strip_non_ascii(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_assistant(text: &str) -> Message {
        Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(text.to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    fn make_assistant_with_tools() -> Message {
        Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text("Using tool...".to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(vec![]),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    fn make_user(text: &str) -> Message {
        Message::user(text)
    }

    fn make_system(text: &str) -> Message {
        Message::system(text)
    }

    #[test]
    fn test_is_thinking_only_assistant_empty() {
        let msg = make_assistant("");
        assert!(is_thinking_only_assistant(&msg));
    }

    #[test]
    fn test_is_thinking_only_assistant_whitespace() {
        let msg = make_assistant("   \n  ");
        assert!(is_thinking_only_assistant(&msg));
    }

    #[test]
    fn test_is_thinking_only_assistant_with_text() {
        let msg = make_assistant("Hello, how can I help?");
        assert!(!is_thinking_only_assistant(&msg));
    }

    #[test]
    fn test_is_thinking_only_assistant_with_tools() {
        let msg = make_assistant_with_tools();
        assert!(!is_thinking_only_assistant(&msg));
    }

    #[test]
    fn test_drop_thinking_only_messages() {
        let mut messages = vec![
            make_user("Hello"),
            make_assistant(""),         // thinking-only, drop
            make_assistant("Response"), // keep
            make_user("Follow up"),
        ];

        drop_thinking_only_assistant(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].content.to_text(), "Response");
        assert_eq!(messages[2].role, MessageRole::User);
    }

    #[test]
    fn test_merge_consecutive_user_messages() {
        let mut messages = vec![
            make_user("First"),
            make_user("Second"),
            make_user("Third"),
            make_system("System prompt"),
            make_user("After system"),
        ];

        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::User);
        let combined = messages[0].content.to_text();
        assert!(combined.contains("First"));
        assert!(combined.contains("Second"));
        assert!(combined.contains("Third"));
        assert_eq!(messages[1].role, MessageRole::System);
        assert_eq!(messages[2].role, MessageRole::User);
    }

    #[test]
    fn test_merge_preserves_image_messages() {
        let img_msg = Message {
            role: MessageRole::User,
            content: MessageContent::Image {
                url: "data:image/png;base64,abc123".into(),
                detail: None,
            },
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let text_msg = make_user("Hello");

        let mut messages = vec![text_msg, img_msg];
        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.to_text(), "Hello");
        assert!(
            matches!(messages[1].content, MessageContent::Image { ref url, .. } if url == "data:image/png;base64,abc123")
        );
    }

    #[test]
    fn test_merge_preserves_parts_with_image() {
        let parts_msg = Message {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                MessagePart::Text("分析下这个图片".into()),
                MessagePart::Image {
                    url: "data:image/jpg;base64,xyz".into(),
                    detail: None,
                },
            ]),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let text_msg = make_user("先看这张");
        let text_msg2 = make_user("再看那张");

        // Image in the middle means text1 and text2 are NOT consecutive
        let mut messages = vec![text_msg, parts_msg, text_msg2];
        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        // text_msg preserved as-is
        assert!(matches!(messages[0].content, MessageContent::Text(ref t) if t == "先看这张"));
        // parts_msg preserved with image intact
        assert!(
            matches!(messages[1].content, MessageContent::Parts(ref parts) if parts.len() == 2 && matches!(&parts[1], MessagePart::Image { .. }))
        );
        // text_msg2 preserved as-is
        assert!(matches!(messages[2].content, MessageContent::Text(ref t) if t == "再看那张"));
    }

    #[test]
    fn test_sanitize_preserves_latest_user_message() {
        // Mode A: latest user message is preserved intact
        let img_msg = Message {
            role: MessageRole::User,
            content: MessageContent::Image {
                url: "data:image/png;base64,abc123".into(),
                detail: None,
            },
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let mut messages = vec![make_user("previous"), make_assistant("hello"), img_msg];

        sanitize_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content.to_text(), "previous");
        assert_eq!(messages[1].content.to_text(), "hello");
        assert!(
            matches!(messages[2].content, MessageContent::Image { ref url, .. } if url == "data:image/png;base64,abc123")
        );
    }

    #[test]
    fn test_sanitize_preserves_latest_user_parts_with_image() {
        let parts_msg = Message {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                MessagePart::Text("分析下这个图片".into()),
                MessagePart::Image {
                    url: "data:image/png;base64,screenshot".into(),
                    detail: None,
                },
            ]),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let mut messages = vec![
            make_user("prev1"),
            make_user("prev2"),
            make_assistant("hi"),
            parts_msg,
        ];

        sanitize_messages(&mut messages);

        // "prev1" and "prev2" should be merged in history
        assert_eq!(messages.len(), 3);
        assert!(
            matches!(messages[0].content, MessageContent::Text(ref t) if t.contains("prev1") && t.contains("prev2"))
        );
        assert_eq!(messages[1].content.to_text(), "hi");
        // Latest user message preserved with image content intact
        assert!(
            matches!(messages[2].content, MessageContent::Parts(ref parts) if parts.len() == 2 && matches!(&parts[1], MessagePart::Image { .. }))
        );
    }

    #[test]
    fn test_sanitize_merging_history_user_messages() {
        let mut messages = vec![make_user("hi"), make_user("there"), make_user("world")];

        sanitize_messages(&mut messages);

        // Only the last message is exempt — history should merge
        // Actually with Mode A: history = [hi, there], last = [world]
        // merge history → user("hi\n\nthere")
        // result → [user("hi\n\nthere"), user("world")]
        assert_eq!(messages.len(), 2);
        assert!(
            matches!(messages[0].content, MessageContent::Text(ref t) if t.contains("hi") && t.contains("there"))
        );
        assert_eq!(messages[1].content.to_text(), "world");
    }

    #[test]
    fn test_merge_image_before_text_messages() {
        let img_msg = Message {
            role: MessageRole::User,
            content: MessageContent::Image {
                url: "data:image/png;base64,abc123".into(),
                detail: None,
            },
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let text_msg1 = make_user("先");
        let text_msg2 = make_user("后");

        let mut messages = vec![img_msg, text_msg1, text_msg2];
        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 2);
        // Image should be first
        assert!(matches!(messages[0].content, MessageContent::Image { .. }));
        // text_msg1 and text_msg2 should be merged
        assert!(
            matches!(messages[1].content, MessageContent::Text(ref t) if t.contains("先") && t.contains("后"))
        );
    }

    #[test]
    fn test_strip_surrogates() {
        let input = "Hello, world!";
        let output = strip_surrogates(input);
        assert_eq!(output, "Hello, world!");
    }

    #[test]
    fn test_strip_non_ascii() {
        let input = "Hello 世界 🌍";
        let output = strip_non_ascii(input);
        // Only ASCII chars kept: "Hello " + space before 世 (kept) + space before 🌍 (kept)
        assert_eq!(output, "Hello  ");
    }

    #[test]
    fn test_no_merge_when_alternating() {
        let mut messages = vec![
            make_user("User 1"),
            make_assistant("Assistant 1"),
            make_user("User 2"),
        ];

        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content.to_text(), "User 1");
        assert_eq!(messages[2].content.to_text(), "User 2");
    }

    #[test]
    fn test_merge_preserves_order() {
        let mut messages = vec![
            make_user("A"),
            make_user("B"),
            make_system("S"),
            make_user("C"),
            make_user("D"),
        ];

        merge_consecutive_user_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert!(messages[0].content.to_text().contains("A"));
        assert!(messages[0].content.to_text().contains("B"));
        assert_eq!(messages[1].role, MessageRole::System);
        assert!(messages[2].content.to_text().contains("C"));
        assert!(messages[2].content.to_text().contains("D"));
    }

    #[test]
    fn test_sanitize_messages_runs_full_pipeline() {
        let mut messages = vec![
            make_user("Hello"),
            make_assistant(""), // thinking-only
            make_assistant("Response"),
            make_user("Follow up"),
        ];

        sanitize_messages(&mut messages);

        // Thinking-only dropped, others preserved
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_merge_consecutive_assistant_messages() {
        let mut messages = vec![
            make_user("Hello"),
            make_assistant("Response 1"),
            make_assistant("Response 2"), // Should be merged with previous
            make_user("Follow up"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        let content = messages[1].content.to_text();
        assert!(content.contains("Response 1"));
        assert!(content.contains("Response 2"));
    }

    #[test]
    fn test_merge_consecutive_assistant_messages_with_tools() {
        let tool1 = oben_models::ToolCall {
            id: "tool-1".to_string(),
            tool_name: "test_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let tool2 = oben_models::ToolCall {
            id: "tool-2".to_string(),
            tool_name: "test_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let mut msg1 = make_assistant("Response 1");
        msg1.tool_calls = Some(vec![tool1]);

        let mut msg2 = make_assistant("Response 2");
        msg2.tool_calls = Some(vec![tool2]);

        let mut messages = vec![
            make_user("Hello"),
            msg1,
            msg2, // Should be merged with previous, tool_calls combined
            make_user("Follow up"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        let content = messages[1].content.to_text();
        assert!(content.contains("Response 1"));
        assert!(content.contains("Response 2"));
        // Tool calls should be combined
        let tool_calls = messages[1].tool_calls.as_ref();
        assert!(tool_calls.is_some());
        assert_eq!(tool_calls.unwrap().len(), 2);
    }

    #[test]
    fn test_merge_assistant_separated_by_system_message() {
        // Test case 1: Two assistants separated by a system message
        let mut messages = vec![
            make_user("Hello"),
            make_assistant("Response 1"),
            make_system("System instruction"),
            make_assistant("Response 2"),
            make_user("Follow up"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // System message should remain but assistants should be merged
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        let content = messages[1].content.to_text();
        assert!(content.contains("Response 1"));
        assert!(content.contains("Response 2"));
        assert_eq!(messages[2].role, MessageRole::System);
    }

    #[test]
    fn test_merge_multiple_assistants_separated_by_system_messages() {
        // Test case 2: Multiple assistants separated by system messages
        let mut messages = vec![
            make_user("Hello"),
            make_assistant("Response 1"),
            make_system("System instruction 1"),
            make_assistant("Response 2"),
            make_system("System instruction 2"),
            make_assistant("Response 3"),
            make_user("Follow up"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // All three assistants should be merged into one
        // System messages remain at their original positions
        // Final structure: [User, Assistant(merged), System, System, User]
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        let content = messages[1].content.to_text();
        assert!(content.contains("Response 1"));
        assert!(content.contains("Response 2"));
        assert!(content.contains("Response 3"));
        assert_eq!(messages[2].role, MessageRole::System);
        assert_eq!(messages[2].content.to_text(), "System instruction 1");
        assert_eq!(messages[3].role, MessageRole::System);
        assert_eq!(messages[3].content.to_text(), "System instruction 2");
        assert_eq!(messages[4].role, MessageRole::User);
    }

    #[test]
    fn test_merge_consecutive_assistants_original_behavior() {
        // Test case 3: Consecutive assistants (original behavior should still work)
        let mut messages = vec![
            make_user("Hello"),
            make_assistant("Response 1"),
            make_assistant("Response 2"),
            make_assistant("Response 3"),
            make_user("Follow up"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        let content = messages[1].content.to_text();
        assert!(content.contains("Response 1"));
        assert!(content.contains("Response 2"));
        assert!(content.contains("Response 3"));
    }

    #[test]
    fn test_merge_assistants_with_tool_calls() {
        // Test case 4: Assistants with tool_calls merging
        let tool1 = oben_models::ToolCall {
            id: "tool-1".to_string(),
            tool_name: "web_search".to_string(),
            arguments: serde_json::json!({"query": "weather"}),
        };

        let tool2 = oben_models::ToolCall {
            id: "tool-2".to_string(),
            tool_name: "file_read".to_string(),
            arguments: serde_json::json!({"path": "/tmp/data.txt"}),
        };

        let tool3 = oben_models::ToolCall {
            id: "tool-3".to_string(),
            tool_name: "web_search".to_string(),
            arguments: serde_json::json!({"query": "news"}),
        };

        let mut msg1 = make_assistant("I'll search the web");
        msg1.tool_calls = Some(vec![tool1]);

        let mut msg2 = make_assistant("Then I'll read the file");
        msg2.tool_calls = Some(vec![tool2]);

        let mut msg3 = make_assistant("Finally another web search");
        msg3.tool_calls = Some(vec![tool3]);

        let mut messages = vec![
            make_user("Task"),
            msg1,
            make_system("You can do multiple tool calls"),
            msg2,
            make_system("Go ahead"),
            msg3,
            make_user("Complete"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // All three assistants merged into one
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        
        // Content should be combined
        let content = messages[1].content.to_text();
        assert!(content.contains("I'll search the web"));
        assert!(content.contains("Then I'll read the file"));
        assert!(content.contains("Finally another web search"));
        
        // Tool calls should be combined (3 unique tools)
        let tool_calls = messages[1].tool_calls.as_ref();
        assert!(tool_calls.is_some());
        assert_eq!(tool_calls.unwrap().len(), 3);
        
        // Verify all tool IDs are present
        let tool_ids: Vec<&str> = tool_calls.unwrap().iter().map(|t| t.id.as_str()).collect();
        assert!(tool_ids.contains(&"tool-1"));
        assert!(tool_ids.contains(&"tool-2"));
        assert!(tool_ids.contains(&"tool-3"));
    }

    #[test]
    fn test_merge_empty_assistant_with_tool_calls() {
        // Test case 5: Empty assistant messages (content="") with tool_calls
        let tool1 = oben_models::ToolCall {
            id: "tool-1".to_string(),
            tool_name: "web_search".to_string(),
            arguments: serde_json::json!({"query": "weather"}),
        };

        let tool2 = oben_models::ToolCall {
            id: "tool-2".to_string(),
            tool_name: "file_read".to_string(),
            arguments: serde_json::json!({"path": "/tmp/data.txt"}),
        };

        // Empty assistant (only tool calls)
        let mut msg1 = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(String::new()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(vec![tool1]),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        // Assistant with both text and tool calls
        let mut msg2 = make_assistant("Processing...");
        msg2.tool_calls = Some(vec![tool2]);

        let mut messages = vec![
            make_user("Task"),
            msg1,
            make_system("System message between"),
            msg2,
            make_user("Complete"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // Empty and non-empty assistants should be merged
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        
        // Content should be combined (empty + non-empty)
        let content = messages[1].content.to_text();
        assert!(content.contains("Processing..."));
        
        // Tool calls should be combined
        let tool_calls = messages[1].tool_calls.as_ref();
        assert!(tool_calls.is_some());
        assert_eq!(tool_calls.unwrap().len(), 2);
    }

    #[test]
    fn test_merge_empty_content_only() {
        // Test case: Assistants with empty content but non-empty tool calls
        let tool1 = oben_models::ToolCall {
            id: "tool-1".to_string(),
            tool_name: "web_search".to_string(),
            arguments: serde_json::json!({"query": "weather"}),
        };

        let tool2 = oben_models::ToolCall {
            id: "tool-2".to_string(),
            tool_name: "file_read".to_string(),
            arguments: serde_json::json!({"path": "/tmp/data.txt"}),
        };

        let mut msg1 = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(String::new()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(vec![tool1]),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let mut msg2 = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Text(String::new()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: Some(vec![tool2]),
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let mut messages = vec![
            make_user("Task"),
            msg1,
            make_system("System message"),
            msg2,
            make_user("Complete"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        
        // Content should remain empty (both were empty)
        assert!(messages[1].content.to_text().is_empty());
        
        // Tool calls should be combined
        let tool_calls = messages[1].tool_calls.as_ref();
        assert!(tool_calls.is_some());
        assert_eq!(tool_calls.unwrap().len(), 2);
    }

    #[test]
    fn test_merge_assistants_not_merged_when_user_separates() {
        // Assistant messages separated by user messages should NOT be merged
        let mut messages = vec![
            make_user("First"),
            make_assistant("Response 1"),
            make_user("Second question"), // User message separates them
            make_assistant("Response 2"),
            make_user("Third"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // Should remain 5 messages - assistants should NOT be merged
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].content.to_text(), "Response 1");
        assert_eq!(messages[3].role, MessageRole::Assistant);
        assert_eq!(messages[3].content.to_text(), "Response 2");
    }

    #[test]
    fn test_merge_assistants_not_merged_when_tool_separates() {
        // Assistant messages separated by tool messages should NOT be merged
        let mut msg1 = make_assistant("Using tool");
        msg1.tool_calls = Some(vec![]);

        let tool_msg = Message {
            role: MessageRole::Tool,
            content: MessageContent::Text("Tool result".to_string()),
            id: None,
            tool_call_ids: vec!["tool-1".to_string()],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };

        let mut msg2 = make_assistant("Another response");

        let mut messages = vec![
            make_user("First"),
            msg1,
            tool_msg,
            msg2,
            make_user("Second"),
        ];

        merge_consecutive_assistant_messages(&mut messages);

        // Should remain 5 messages - assistants should NOT be merged
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert_eq!(messages[1].content.to_text(), "Using tool");
        assert_eq!(messages[3].role, MessageRole::Assistant);
        assert_eq!(messages[3].content.to_text(), "Another response");
    }
}
