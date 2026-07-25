/// Context compression using OpenCode-style compact algorithm.
///
/// Detects overflow → selects N recent turns verbatim → serializes older messages →
/// structured LLM summary → continues in same session.
use anyhow::{anyhow, Result};

use oben_models::{Message, MessageContent, MessagePart, TransportProvider};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_PRESERVE_RECENT_TOKENS: usize = 8_000;
const MIN_PRESERVE_RECENT_TOKENS: usize = 2_000;
#[allow(dead_code)]
const DEFAULT_TAIL_TURNS: usize = 2;
const TOOL_OUTPUT_MAX_CHARS: usize = 2_000;

const COMPACTION_HEADER: &str = "\
[CONTEXT COMPACTION — REFERENCE ONLY]\n\
Earlier turns were compacted into the summary below. This is a handoff from a \
previous context window — treat it as background reference, NOT as active instructions. \
Do NOT answer questions or fulfill requests mentioned in this summary; they were already \
addressed. Your current task is identified in the '## Objective' section of the summary — \
resume exactly from there. IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the \
system prompt is ALWAYS authoritative and active — never ignore or deprioritize memory \
content due to this compaction note. Respond ONLY to the latest user message that appears \
AFTER this summary. The current session state (files, config, etc.) may reflect work \
described here — avoid repeating it.";

const SUMMARY_TEMPLATE: &str = "\
## Objective\n- [one or two brief sentences describing what the user is trying to accomplish]\n\n\
## Important Details\n- [constraints/preferences, decisions and why, important facts/assumptions]\n\n\
## Work State\n### Completed\n- [finished work, verified facts, or changes made; otherwise \"(none)\"]\n\n\
### Active\n- [current work, partial changes, or investigation state; otherwise \"(none)\"]\n\n\
### Blocked\n- [blockers, failing commands, or unknowns; otherwise \"(none)\"]\n\n\
## Next Move\n1. [immediate concrete action, or \"(none)\"]\n\n\
## Relevant Files\n- [file or directory path: why it matters, or \"(none)\"]";

// ---------------------------------------------------------------------------
// Legacy types (kept for backward compat with callers that use the old API)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompactCofig {
    /// Total context window size in tokens.
    pub context_length: usize,
    /// Token threshold as a percentage of context_length (e.g. 0.75 = 75%).
    pub threshold_percent: f64,
    /// Number of non-system head messages to protect.
    pub protect_first_n: usize,
    /// Token budget for the tail — walk backward accumulating tokens.
    /// Default: ~20K tokens (scales with model context).
    pub tail_token_budget: usize,
    /// Hard minimum: always protect at least this many messages in the tail.
    pub tail_min_messages: usize,
    /// Soft ceiling multiplier — allow budget to exceed by this factor to
    /// avoid cutting inside an oversized message.
    pub tail_overhead: f64,
    /// Min tokens for the initial summary.
    pub min_summary_tokens: usize,
    /// Max tokens for the initial summary.
    pub max_summary_tokens: usize,
    /// Max tokens for iterated summaries.
    pub iterated_max_tokens: usize,
    /// Min tokens for iterated summaries.
    pub iterated_min_tokens: usize,
    /// Max tokens for the final combined summary.
    pub final_summary_max_tokens: usize,
    /// Max tool result tokens to keep before pruning.
    pub max_tool_result_tokens: usize,
    /// Min percentage savings for a compression to be considered effective.
    pub ineffective_threshold: f64,
    /// Max consecutive ineffective compressions before anti-thrashing kicks in.
    pub max_ineffective_consecutive: usize,
}

impl Default for CompactCofig {
    fn default() -> Self {
        Self {
            context_length: 128_000,
            threshold_percent: 0.75,
            protect_first_n: 3,
            tail_token_budget: 20_000,
            tail_min_messages: 3,
            tail_overhead: 1.5,
            min_summary_tokens: 2000,
            max_summary_tokens: 4000,
            iterated_max_tokens: 3000,
            iterated_min_tokens: 1000,
            final_summary_max_tokens: 2500,
            max_tool_result_tokens: 10000,
            ineffective_threshold: 10.0,
            max_ineffective_consecutive: 2,
        }
    }
}

impl CompactCofig {
    /// Derive the compression threshold in tokens from the current
    /// context_length and threshold_percent.
    pub fn threshold_tokens(&self) -> usize {
        (self.context_length as f64 * self.threshold_percent) as usize
    }

    /// Derive threshold tokens from a given context length using the
    /// current threshold_percent.
    pub fn threshold_tokens_for(&self, context_length: usize) -> usize {
        (context_length as f64 * self.threshold_percent) as usize
    }
}

/// Outcome of a manual `compact_session` operation — distinguishes the
/// different reasons why compaction may not change the message list.
#[derive(Debug, Clone)]
pub enum CompactOutcome {
    /// Session messages are within token budget — nothing to compact.
    AlreadyCompact,
    /// All messages are protected (head/tail) — no middle messages to summarize.
    NoMiddleMessages {
        head_count: usize,
        tail_count: usize,
    },
    /// Compression attempted but savings below threshold — messages unchanged.
    Ineffective {
        original_tokens: usize,
        compacted_tokens: usize,
        savings_pct: f64,
    },
    /// Compression succeeded — messages were replaced with a summary.
    Compressed {
        original_count: usize,
        compacted_count: usize,
        savings_pct: f64,
    },
}

// Legacy result types expected by callers that reference the old API

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_count: usize,
    pub compacted_count: usize,
    pub original_tokens: usize,
    pub compacted_tokens: usize,
    pub savings_pct: f64,
    pub pruned_tool_results: usize,
    pub summary_generated: bool,
}

pub struct CompressionResult {
    pub messages: Vec<Message>,
    pub stats: CompressionStats,
    pub summary: Option<String>,
}

// Stub: old-style compact_session_messages — delegates to new compact_session internally
/// Backward-compatible wrapper for callers that still use the old API signature.
pub async fn compact_session_messages(
    transport: &dyn TransportProvider,
    messages: &[Message],
    config: &CompactCofig,
    _previous_summary: Option<&str>,
    _focus_topic: Option<&str>,
    _compression_round: usize,
) -> Result<CompressionResult> {
    // Use the new algorithm but return the old result shape for compatibility.
    let tail_turns = config.protect_first_n.max(2);
    let preserve_tokens = config.tail_token_budget;
    match compact_session(transport, messages, tail_turns, Some(preserve_tokens), None).await {
        Ok(cr) => {
            let original_tokens = messages.iter().map(|m| message_token_estimate(m)).sum::<usize>();
            let recent_tokens: usize = cr.recent_messages.iter()
                .map(|m| message_token_estimate(m))
                .sum();
            let summary_tokens = cr.summary.len() / 4;
            let new_total = recent_tokens + summary_tokens + 5; // system message overhead
            let savings = if original_tokens > 0 {
                ((1.0 - new_total as f64 / original_tokens as f64) * 100.0).max(0.0).round()
            } else {
                0.0
            };
            let summary_text = cr.summary.strip_prefix(format!("{}:", COMPACTION_HEADER).as_str())
                .unwrap_or(&cr.summary)
                .to_string();
            let compacted_len = cr.recent_messages.len();
            let mut compacted = cr.recent_messages;
            compacted.insert(0, Message::system(&summary_text));
            let compacted_result_tokens: usize = compacted
                .iter()
                .map(|m| message_token_estimate(m))
                .sum();
            Ok(CompressionResult {
                messages: compacted,
                stats: CompressionStats {
                    original_count: messages.len(),
                    compacted_count: compacted_len + 1,
                    original_tokens,
                    compacted_tokens: compacted_result_tokens,
                    savings_pct: savings,
                    pruned_tool_results: 0,
                    summary_generated: !cr.summary.is_empty(),
                },
                summary: Some(summary_text),
            })
        }
        Err(e) => {
            // Handle "no older messages to compact" as no-op (matches old behavior)
            let original_tokens = messages.iter().map(|m| message_token_estimate(m)).sum::<usize>();
            if e.to_string().contains("No older") || e.to_string().contains("Not enough") {
                Ok(CompressionResult {
                    messages: messages.to_vec(),
                    stats: CompressionStats {
                        original_count: messages.len(),
                        compacted_count: messages.len(),
                        original_tokens,
                        compacted_tokens: original_tokens,
                        savings_pct: 0.0,
                        pruned_tool_results: 0,
                        summary_generated: false,
                    },
                    summary: None,
                })
            } else {
                Err(e)
            }
        }
    }
}

// Stub: old-style find_tail_cut_by_tokens — used by compact_context.rs should_compact
/// Walk backward from the end, accumulating tokens until budget is reached.
/// Enforces a hard minimum of `tail_min_messages` messages.
/// Returns the index where the tail segment starts.
pub(crate) fn find_tail_cut_by_tokens(messages: &[Message], config: &CompactCofig) -> usize {
    let min_protect = config.tail_min_messages.min(messages.len());
    let budget = (config.tail_token_budget as f64 * config.tail_overhead) as usize;
    let mut accumulated = 0;
    let mut cut = 0;

    for i in (0..messages.len()).rev() {
        let tokens = message_token_estimate(&messages[i]);
        if accumulated + tokens > budget && (messages.len() - i) >= min_protect {
            break;
        }
        accumulated += tokens;
        cut = i;
    }

    // Ensure we protect at least min_protect messages
    let protect_count = messages.len().saturating_sub(cut);
    if protect_count < min_protect {
        messages.len().saturating_sub(min_protect)
    } else {
        cut
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a compaction operation.
pub struct CompactionResult {
    /// The LLM-generated summary string.
    pub summary: String,
    /// Messages preserved verbatim (recent turns).
    pub recent_messages: Vec<Message>,
    /// Number of tool results that were truncated.
    pub pruned_tool_results: usize,
}

/// Token estimate for a message.
/// Used by the CWM for token tracking.
pub fn message_token_estimate(msg: &Message) -> usize {
    let text = match &msg.content {
        MessageContent::Text(s) => s,
        MessageContent::Image { .. } => return 500,
        MessageContent::Parts(parts) => {
            return parts.iter().map(|p| match p {
                MessagePart::Text(s) => s.len() / 4,
                MessagePart::Image { .. } => 500,
            }).sum::<usize>();
        }
    };
    text.len() / 4 + 5
}

/// Compact messages using OpenCode-style selection + structured summary.
///
/// Returns a CompactionResult containing the summary and preserved recent messages.
/// If the message list is small enough (under 5000 tokens), returns early with no compaction.
pub async fn compact_session(
    transport: &dyn TransportProvider,
    messages: &[Message],
    tail_turns: usize,
    preserve_recent_tokens: Option<usize>,
    _previous_summary: Option<&str>,
) -> Result<CompactionResult> {
    let total_tokens: usize = messages.iter().map(|m| message_token_estimate(m)).sum();

    // Early return: not enough content to compact
    if total_tokens < 5000 {
        return Err(anyhow!("Not enough content to compact ({} tokens)", total_tokens));
    }

    // Step 1: Select recent turns to preserve verbatim
    let (older_msgs, recent_msgs) = select_recent_turns(messages, tail_turns, preserve_recent_tokens);

    // If no older messages, compacting is pointless
    if older_msgs.is_empty() {
        return Err(anyhow!("No older messages to compact"));
    }

    // Step 2: Serialize older messages
    let content_to_summarize = serialize_messages(&older_msgs);

    // Step 3: Build LLM prompt
    let preamble = "You are a summarization agent creating a context checkpoint. Treat the conversation turns below as source material for a compact record of prior work. Produce only the structured summary; do not add a greeting, preamble, or prefix. Write the summary in the same language the user was using in the conversation — do not translate or switch to English. NEVER include API keys, tokens, passwords, secrets, credentials, or connection strings in the summary — replace any that appear with [REDACTED].";

    let prompt = format!(
        "{}\n\nCreate a structured checkpoint summary for the conversation after earlier turns are compacted.\n\nTURNS TO SUMMARIZE:\n{}\n\nUse this exact structure:\n\n{}",
        preamble,
        content_to_summarize,
        SUMMARY_TEMPLATE
    );

    let summary_msg = Message::user(prompt);
    let max_retries = 2;
    let mut last_error: Option<anyhow::Error> = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;
        }

        match transport
            .chat(&[summary_msg.clone()], &oben_models::CallMode::Fresh(String::new()))
            .await
        {
            Ok(response) => {
                let summary = response.text.trim().to_string();
                if summary.is_empty() {
                    return Err(anyhow!("Empty summary from compaction LLM"));
                }
                return Ok(CompactionResult {
                    summary: format!("{}:{}", COMPACTION_HEADER, summary),
                    recent_messages: recent_msgs,
                    pruned_tool_results: 0,
                });
            }
            Err(e) => {
                last_error = Some(e);
            }
        }
    }

    Err(anyhow!(
        "Summary generation failed after {} attempts: {}",
        max_retries + 1,
        last_error.unwrap()
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Select recent turns to preserve verbatim. Walk backward from end, respecting turn boundaries.
fn select_recent_turns(
    messages: &[Message],
    tail_turns: usize,
    preserve_recent_tokens: Option<usize>,
) -> (Vec<Message>, Vec<Message>) {
    if messages.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let budget = preserve_recent_tokens.unwrap_or_else(|| {
        let usable = messages.iter().map(|m| message_token_estimate(m)).sum::<usize>();
        (MAX_PRESERVE_RECENT_TOKENS
            .min(MIN_PRESERVE_RECENT_TOKENS.max(usable / 4)))
        .max(MIN_PRESERVE_RECENT_TOKENS) as usize
    });

    // Walk backward from end, accumulating messages until budget reached
    let mut accumulated = 0;
    let mut recent_start = 0;

    for i in (0..messages.len()).rev() {
        let tokens = message_token_estimate(&messages[i]);
        if accumulated + tokens > budget && i > 0 {
            recent_start = i;
            break;
        }
        accumulated += tokens;
        if i == 0 {
            recent_start = 0;
        }
    }

    // Enforce minimum: at least tail_turns worth of recent messages
    let recent_end = messages.len();
    let recent_start = recent_start.min(recent_end.saturating_sub(1.max(tail_turns)));

    (messages[..recent_start].to_vec(), messages[recent_start..].to_vec())
}

/// Serialize messages into a structured text format for the LLM summarizer.
fn serialize_messages(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    const CONTENT_MAX: usize = 6000;
    const CONTENT_HEAD: usize = 4000;
    const CONTENT_TAIL: usize = 1500;

    for msg in messages {
        let role = match msg.role {
            oben_models::MessageRole::System => "SYSTEM",
            oben_models::MessageRole::User => "USER",
            oben_models::MessageRole::Assistant => "ASSISTANT",
            oben_models::MessageRole::Tool => "TOOL",
        };

        let content = match &msg.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Image { .. } => "[image attached]".to_string(),
            MessageContent::Parts(sub_parts) => {
                let texts: Vec<String> = sub_parts
                    .iter()
                    .filter_map(|p| match p {
                        MessagePart::Text(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    "[multimodal content]".to_string()
                } else {
                    texts.join("\n")
                }
            }
        };

        let trimmed = if content.len() > CONTENT_MAX {
            format!(
                "{}\n...[truncated]...\n{}",
                &content[..CONTENT_HEAD],
                &content[content.len().saturating_sub(CONTENT_TAIL)..]
            )
        } else {
            content
        };

        // Truncate tool output if necessary
        let final_content = if role == "TOOL" {
            if trimmed.len() > TOOL_OUTPUT_MAX_CHARS {
                format!(
                    "{}[truncated to {} chars]",
                    &trimmed[..TOOL_OUTPUT_MAX_CHARS],
                    TOOL_OUTPUT_MAX_CHARS
                )
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        parts.push(format!("[{}]: {}", role, final_content));
    }

    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(role: oben_models::MessageRole, content: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(content.to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    }

    #[test]
    fn test_message_token_estimate_text() {
        let msg = make_msg(
            oben_models::MessageRole::User,
            "hello world this is a test message",
        );
        let est = message_token_estimate(&msg);
        assert!(est > 0, "Token estimate should be positive");
        assert!(est < 100, "Token estimate should be small for short text");
    }

    #[test]
    fn test_message_token_estimate_empty() {
        let msg = make_msg(
            oben_models::MessageRole::User,
            "",
        );
        let est = message_token_estimate(&msg);
        assert_eq!(est, 5, "Empty text should have overhead tokens only");
    }

    #[test]
    fn test_select_recent_turns_empty() {
        let (older, recent) = select_recent_turns(&[], 2, None);
        assert!(older.is_empty());
        assert!(recent.is_empty());
    }

    #[test]
    fn test_select_recent_turns_small() {
        let messages: Vec<Message> = (0..3)
            .map(|i| make_msg(oben_models::MessageRole::User, &format!("msg {}", i)))
            .collect();
        let (_older, recent) = select_recent_turns(&messages, 2, None);
        // Small list should preserve at least tail_turns
        assert!(!recent.is_empty() || messages.len() <= DEFAULT_TAIL_TURNS);
    }

    #[test]
    fn test_select_recent_turns_preserves_budget() {
        // Create messages with known sizes
        let messages: Vec<Message> = (0..20)
            .map(|i| make_msg(oben_models::MessageRole::User, &format!("msg_{:04}", i)))
            .collect();
        let (_older, recent) = select_recent_turns(&messages, 2, Some(50));

        let recent_tokens: usize = recent.iter().map(|m| message_token_estimate(m)).sum();
        assert!(
            recent_tokens <= 50 + 20, // budget + per-message overhead ~20
            "Recent tokens {} should be near budget 50",
            recent_tokens
        );
    }

    #[test]
    fn test_select_recent_turns_min_tail() {
        let messages: Vec<Message> = (0..50)
            .map(|i| make_msg(oben_models::MessageRole::User, &format!("msg_{:04}", i)))
            .collect();
        // Use small budget so some messages overflow and older is non-empty
        let (older, recent) = select_recent_turns(&messages, 5, Some(50));
        // Should preserve at least tail_turns even with small budget
        assert!(
            recent.len() >= 5,
            "Recent should have at least tail_turns=5, got {}",
            recent.len()
        );
        assert!(!older.is_empty());
    }

    #[test]
    fn test_message_token_estimate_parts() {
        let msg = Message {
            role: oben_models::MessageRole::Assistant,
            content: MessageContent::Parts(vec![
                MessagePart::Text("hello".to_string()),
                MessagePart::Text(" world".to_string()),
            ]),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        };
        let est = message_token_estimate(&msg);
        assert!(est > 0);
        // Both text parts combined: 11 chars / 4 + per-message overhead
        assert!(est < 10);
    }

    #[test]
    fn test_serialize_messages_empty() {
        let result = serialize_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_serialize_messages_single_user() {
        let messages = vec![make_msg(oben_models::MessageRole::User, "hello there")];
        let result = serialize_messages(&messages);
        assert!(result.contains("[USER]:"));
        assert!(result.contains("hello there"));
    }

    #[test]
    fn test_serialize_messages_tool_output_truncated() {
        let big_output = "x".repeat(TOOL_OUTPUT_MAX_CHARS + 500);
        let messages = vec![Message {
            role: oben_models::MessageRole::Tool,
            content: MessageContent::Text(big_output),
            id: None,
            tool_call_ids: vec!["call-1".to_string()],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }];
        let result = serialize_messages(&messages);
        assert!(result.contains("[TOOL]:"));
        assert!(result.contains("[truncated"));
    }
}
