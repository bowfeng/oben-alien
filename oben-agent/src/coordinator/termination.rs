/// Pluggable termination and remedy policies for turn execution.
///
/// Split into two concerns:
/// - **TurnTerminationPolicy** — evaluates API response to decide Continue/Return/ReturnLastToolResult
/// - **TurnRemedyPolicy** — handles budget exhaustion and empty response recovery
///
/// `execute_turn_with_config` in `turn_executor.rs` uses these as a two-phase
/// decision pipeline: first termination, then remedy.
use anyhow::Result;
use tracing::info;

// Config import intentionally unused — max_iterations comes from the caller.
use oben_models::{Message, MessageContent, MessageRole, TransportResponse};

// =========================================================================
// TurnTerminationPolicy (phase 1: response evaluation)
// =========================================================================

/// Context passed to TerminationPolicy during evaluation.
pub struct TurnTerminationContext<'a> {
    /// The LLM's response from this turn.
    pub response: &'a TransportResponse,
    /// Current session messages.
    pub messages: &'a [Message],
}

/// Decision returned by TerminationPolicy.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnTerminationDecision {
    /// More tool calls to dispatch.
    Continue,
    /// Return from the turn with the given text.
    Return(String),
    /// Return from the turn with the last tool result's text.
    ReturnLastToolResult,
}

/// Policy that evaluates an API response and decides what to do.
pub trait TurnTerminationPolicy: Send + Sync {
    /// Evaluate whether to continue, return with text, or return last tool result.
    fn evaluate(&self, ctx: &TurnTerminationContext<'_>) -> Result<TurnTerminationDecision>;
}

/// Default termination logic — the original process_response behavior:
/// - tool_calls non-empty → Continue
/// - text non-empty → Return(text)
/// - text empty + has tool messages → ReturnLastToolResult
/// - text empty + no tool messages → Return("")
pub struct DefaultTurnTerminationPolicy;

impl Default for DefaultTurnTerminationPolicy {
    fn default() -> Self { Self }
}

impl TurnTerminationPolicy for DefaultTurnTerminationPolicy {
    fn evaluate(&self, ctx: &TurnTerminationContext<'_>) -> Result<TurnTerminationDecision> {
        info!("Policy: evaluating with tool_calls={:?}, messages.len={}", ctx.response.tool_calls, ctx.messages.len());
        
        // If there are tool errors in messages, always return last tool result
        // to prevent LLM from generating text that ignores tool failures
        let has_tool_errors = ctx.messages.iter().any(|m| m.tool_error);
        info!("Policy: has_tool_errors={}", has_tool_errors);
        if has_tool_errors {
            info!("Policy: returning ReturnLastToolResult due to tool errors");
            return Ok(TurnTerminationDecision::ReturnLastToolResult);
        }
        
        if !ctx.response.tool_calls.is_empty() {
            info!("Policy: LLM wants to call more tools");
            return Ok(TurnTerminationDecision::Continue);
        }
        
        let text = ctx.response.text.trim().to_string();
        info!("Policy: response text=\"{}\" length={}", text, text.len());
        info!("Policy: response.tool_calls={:?}", ctx.response.tool_calls);
        if text.is_empty() {
            let has_tool_messages = ctx.messages.iter().any(|m| m.role == MessageRole::Tool);
            info!("Policy: empty response, has_tool_messages={}", has_tool_messages);
            if has_tool_messages {
                info!("Policy: returning ReturnLastToolResult due to empty response with tool messages");
                Ok(TurnTerminationDecision::ReturnLastToolResult)
            } else {
                info!("Policy: returning empty response (no tools involved)");
                Ok(TurnTerminationDecision::Return(String::new()))
            }
        } else {
            info!("Policy: returning normal response");
            Ok(TurnTerminationDecision::Return(text))
        }
    }
}

/// Group of termination policies. Returns the first non-Continue decision.
pub struct TurnTerminationPolicyGroup {
    policies: Vec<Box<dyn TurnTerminationPolicy>>,
}

impl TurnTerminationPolicyGroup {
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    pub fn with_policy(mut self, policy: Box<dyn TurnTerminationPolicy>) -> Self {
        self.policies.push(policy);
        self
    }

    pub fn add_policy(&mut self, policy: Box<dyn TurnTerminationPolicy>) {
        self.policies.push(policy);
    }
}

impl Default for TurnTerminationPolicyGroup {
    fn default() -> Self { Self::new() }
}

impl TurnTerminationPolicy for TurnTerminationPolicyGroup {
    fn evaluate(&self, ctx: &TurnTerminationContext<'_>) -> Result<TurnTerminationDecision> {
        for policy in &self.policies {
            match policy.evaluate(ctx)? {
                TurnTerminationDecision::Continue => continue,
                decision => return Ok(decision),
            }
        }
        // All policies said Continue → default to Continue
        Ok(TurnTerminationDecision::Continue)
    }
}

// =========================================================================
// TurnRemedyPolicy (phase 2: budget & empty recovery)
// =========================================================================

/// Decision from TurnRemedyPolicy after budget/response evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnRemedyAction {
    /// Everything is OK — dispatch tool calls and continue.
    Continue,
    /// Remediation applied (reminder hint injected) — continue loop.
    Remedy,
    /// Remediation exhausted — return last tool result text.
    RemedyExhausted,
}

/// Policy that handles budget exhaustion and empty response recovery.
pub trait TurnRemedyPolicy: Send {
    /// Evaluate whether to remedy, continue, or exhaust.
    ///
    /// - `max_calls` — maximum API calls allowed in this turn
    /// - `messages` — mutable message list (for injecting reminder hints)
    /// - `empty_count` — consecutive empty responses this turn
    fn evaluate(
        &mut self,
        max_calls: usize,
        messages: &mut Vec<Message>,
        empty_count: u32,
    ) -> Result<TurnRemedyAction>;
}

/// Budget-based remedy — stops after first budget exhaustion,
/// injects a reminder prompt on first exhaustion.
pub struct BudgetRemedyPolicy {
    remedyed: bool,
}

impl BudgetRemedyPolicy {
    pub fn new(_max_calls: usize) -> Self {
        Self { remedyed: false }
    }

    fn evaluate_inner(&mut self, _max_calls: usize, messages: &mut Vec<Message>) -> TurnRemedyAction {
        if !self.remedyed {
            self.remedyed = true;
            let reminder = "You have reached your iteration limit. Please provide a final answer now without using any more tools.";
            messages.push(Message::system(reminder.to_string()));
            return TurnRemedyAction::Remedy;
        }
        TurnRemedyAction::RemedyExhausted
    }
}

impl TurnRemedyPolicy for BudgetRemedyPolicy {
    fn evaluate(&mut self, max_calls: usize, messages: &mut Vec<Message>, _empty_count: u32) -> Result<TurnRemedyAction> {
        // Only activate when budget is actually exhausted
        if max_calls > 0 {
            Ok(TurnRemedyAction::Continue)
        } else {
            Ok(self.evaluate_inner(max_calls, messages))
        }
    }
}

pub struct ToolFailureRemedyPolicy {
    max_consecutive_failures: u32,
    consecutive_failures: u32,
}

impl ToolFailureRemedyPolicy {
    pub fn new(max_consecutive_failures: u32) -> Self {
        Self {
            max_consecutive_failures,
            consecutive_failures: 0,
        }
    }

    fn evaluate_inner(&mut self, _messages: &mut Vec<Message>) -> TurnRemedyAction {
        // The tool failure guidance is now in system prompt, so we don't need to
        // add retry hints to messages. Just track failures for the remedy decision.
        self.consecutive_failures += 1;

        if self.consecutive_failures < self.max_consecutive_failures {
            TurnRemedyAction::Remedy
        } else {
            TurnRemedyAction::RemedyExhausted
        }
    }

    fn is_tool_error(_text: &str, tool_error: bool) -> bool {
        tool_error
    }
}

impl Default for ToolFailureRemedyPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

impl TurnRemedyPolicy for ToolFailureRemedyPolicy {
    fn evaluate(&mut self, _max_calls: usize, messages: &mut Vec<Message>, _empty_count: u32) -> Result<TurnRemedyAction> {
        let has_tool_errors = messages
            .iter()
            .any(|m| m.role == MessageRole::Tool && Self::is_tool_error(&m.content.to_text(), m.tool_error));
        
        info!("ToolFailureRemedyPolicy: has_tool_errors={}, messages.len={}", has_tool_errors, messages.len());
        for (i, m) in messages.iter().enumerate() {
            info!("  msg[{}]: role={:?} tool_error={}", i, m.role, m.tool_error);
        }

        if !has_tool_errors {
            return Ok(TurnRemedyAction::Continue);
        }

        Ok(self.evaluate_inner(messages))
    }
}

/// Empty response remedy — handles repeated empty LLM responses.
/// Only activates when `empty_count > 0`.
pub struct EmptyResponseRemedyPolicy {
    max_consecutive: u32,
}

impl EmptyResponseRemedyPolicy {
    pub fn new(max_consecutive: u32) -> Self {
        Self { max_consecutive }
    }

    fn evaluate_inner(&mut self, empty_count: u32, messages: &mut Vec<Message>) -> TurnRemedyAction {
        if empty_count <= self.max_consecutive {
            let hint = "Your previous response was completely empty and will be skipped. Please summarize what you learned from the tool results above, or use a tool call if you need more information.";
            messages.push(Message::system(hint.to_string()));
            TurnRemedyAction::Remedy
        } else {
            TurnRemedyAction::RemedyExhausted
        }
    }
}

impl TurnRemedyPolicy for EmptyResponseRemedyPolicy {
    fn evaluate(&mut self, _max_calls: usize, messages: &mut Vec<Message>, empty_count: u32) -> Result<TurnRemedyAction> {
        if empty_count == 0 {
            return Ok(TurnRemedyAction::Continue);
        }
        Ok(self.evaluate_inner(empty_count, messages))
    }
}

/// Group of remedy policies. Evaluates in order, returns first action.
pub struct TurnRemedyPolicyGroup {
    policies: Vec<Box<dyn TurnRemedyPolicy>>,
}

impl TurnRemedyPolicyGroup {
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    pub fn with_policy(mut self, policy: Box<dyn TurnRemedyPolicy>) -> Self {
        self.policies.push(policy);
        self
    }

    pub fn add_policy(&mut self, policy: Box<dyn TurnRemedyPolicy>) {
        self.policies.push(policy);
    }
}

impl Default for TurnRemedyPolicyGroup {
    fn default() -> Self { Self::new() }
}

impl TurnRemedyPolicy for TurnRemedyPolicyGroup {
    fn evaluate(&mut self, max_calls: usize, messages: &mut Vec<Message>, empty_count: u32) -> Result<TurnRemedyAction> {
        for policy in &mut self.policies {
            let action = policy.evaluate(max_calls, messages, empty_count)?;
            info!("TurnRemedyPolicyGroup: policy returned {:?}", action);
            match action {
                TurnRemedyAction::Continue => continue,
                action => return Ok(action),
            }
        }
        Ok(TurnRemedyAction::Continue)
    }
}

/// Default remedy policy group — BudgetRemedyPolicy + EmptyResponseRemedyPolicy.
/// Used when no custom policy is provided.
pub struct DefaultTurnRemedyPolicy {
    inner: TurnRemedyPolicyGroup,
}

impl Default for DefaultTurnRemedyPolicy {
    fn default() -> Self {
        Self {
            inner: TurnRemedyPolicyGroup::new()
                .with_policy(Box::new(BudgetRemedyPolicy::new(100)))
                .with_policy(Box::new(ToolFailureRemedyPolicy::default()))
                .with_policy(Box::new(EmptyResponseRemedyPolicy::new(3))),
        }
    }
}

impl TurnRemedyPolicy for DefaultTurnRemedyPolicy {
    fn evaluate(&mut self, max_calls: usize, messages: &mut Vec<Message>, empty_count: u32) -> Result<TurnRemedyAction> {
        self.inner.evaluate(max_calls, messages, empty_count)
    }
}

// =========================================================================
// Deprecated types (kept for reference — old unified design)
// =========================================================================

// #[deprecated(note = "Use TurnTerminationPolicy + TurnRemedyPolicy instead")]
// pub struct TerminationPolicyGroup { ... }
// pub trait TerminationPolicy: ... { fn evaluate(&self) -> Option<ExitReason>; }
// pub enum ExitReason { UserQuit, BudgetExhausted, Interrupted, Timeout, LlmDecidedDone, Other(String) }
// pub enum TurnDecision { Continue, Return(String), ReturnLastToolResult }
// pub struct TurnContext<'a> { ... }

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oben_models::MessageContent;

    fn mk_response(text: &str, tool_calls: Vec<oben_models::TransportToolCall>) -> TransportResponse {
        TransportResponse {
            text: text.to_string(),
            tool_calls,
            tokens_used: Some(10),
            reasoning: None,
        }
    }

    fn mk_ctx<'a>(response: &'a TransportResponse, messages: &'a [Message]) -> TurnTerminationContext<'a> {
        TurnTerminationContext { response, messages }
    }

    // ------------------------------------------------------------------
    // DefaultTurnTerminationPolicy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_default_policy_continues_on_tool_calls() {
        let policy = DefaultTurnTerminationPolicy::default();
        let resp = mk_response("", vec![oben_models::TransportToolCall {
            id: "tc1".into(), tool_name: "test".into(), arguments: serde_json::json!({}),
        }]);
        let ctx = mk_ctx(&resp, &[]);
        assert_eq!(policy.evaluate(&ctx).unwrap(), TurnTerminationDecision::Continue);
    }

    #[test]
    fn test_default_policy_returns_text() {
        let policy = DefaultTurnTerminationPolicy::default();
        let resp = mk_response("Hello, world!", vec![]);
        let ctx = mk_ctx(&resp, &[]);
        let result = policy.evaluate(&ctx).unwrap();
        assert_eq!(result, TurnTerminationDecision::Return("Hello, world!".to_string()));
    }

    #[test]
    fn test_default_policy_returns_remedy_exhausted_on_empty_with_tools() {
        let policy = DefaultTurnTerminationPolicy::default();
        let resp = mk_response("", vec![]);
        let msgs = [
            Message { role: MessageRole::Tool, content: MessageContent::Text("tool output".into()), id: None, tool_call_ids: vec![], tool_calls: None, reasoning: None, delegation_id: None, tool_error: false, include_in_prompt: true },
        ];
        let ctx = mk_ctx(&resp, &msgs);
        assert_eq!(policy.evaluate(&ctx).unwrap(), TurnTerminationDecision::ReturnLastToolResult);
    }

    #[test]
    fn test_default_policy_returns_empty_on_empty_no_tools() {
        let policy = DefaultTurnTerminationPolicy::default();
        let resp = mk_response("", vec![]);
        let ctx = mk_ctx(&resp, &[]);
        assert_eq!(policy.evaluate(&ctx).unwrap(), TurnTerminationDecision::Return(String::new()));
    }

    // ------------------------------------------------------------------
    // TurnTerminationPolicyGroup tests
    // ------------------------------------------------------------------

    #[test]
    fn test_group_returns_first_non_continue() {
        let group = TurnTerminationPolicyGroup::new()
            .with_policy(Box::new(DefaultTurnTerminationPolicy::default()));

        let resp = mk_response("done", vec![]);
        let ctx = mk_ctx(&resp, &[]);
        let result = group.evaluate(&ctx).unwrap();
        assert_eq!(result, TurnTerminationDecision::Return("done".to_string()));
    }

    // ------------------------------------------------------------------
    // BudgetRemedyPolicy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_budget_allows_within_limit() {
        let mut policy = BudgetRemedyPolicy::new(50);
        let mut messages = vec![];
        let action = policy.evaluate(25, &mut messages, 0).unwrap();
        assert_eq!(action, TurnRemedyAction::Continue);
    }

    #[test]
    fn test_budget_remies_on_first_exhaustion() {
        let mut policy = BudgetRemedyPolicy::new(50);
        let mut messages = vec![];
        let action = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action, TurnRemedyAction::Remedy);
        assert_eq!(messages.last().unwrap().content.to_text_ref().unwrap(),
            "You have reached your iteration limit. Please provide a final answer now without using any more tools.");
    }

    #[test]
    fn test_budget_remies_exhausted_after_first() {
        let mut policy = BudgetRemedyPolicy::new(50);
        let mut messages = vec![];
        policy.evaluate(0, &mut messages, 0).unwrap(); // first → Remedy
        let action = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action, TurnRemedyAction::RemedyExhausted);
    }

    // ------------------------------------------------------------------
    // EmptyResponseRemedyPolicy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_policy_continues_on_zero() {
        let mut policy = EmptyResponseRemedyPolicy::new(3);
        let mut messages = vec![];
        let action = policy.evaluate(50, &mut messages, 0).unwrap();
        assert_eq!(action, TurnRemedyAction::Continue);
    }

    #[test]
    fn test_empty_policy_remies_within_limit() {
        let mut policy = EmptyResponseRemedyPolicy::new(3);
        let mut messages = vec![];
        let action = policy.evaluate(50, &mut messages, 1).unwrap();
        assert_eq!(action, TurnRemedyAction::Remedy);
        assert!(messages.last().unwrap().content.to_text_ref().unwrap()
            .contains("Your previous response was completely empty"));
    }

    #[test]
    fn test_empty_policy_remies_when_exceeded() {
        let mut policy = EmptyResponseRemedyPolicy::new(3);
        let mut messages = vec![];
        // Exhaust limit
        for i in 1..=4 {
            let action = policy.evaluate(50, &mut messages, i).unwrap();
            if i <= 3 {
                assert_eq!(action, TurnRemedyAction::Remedy);
            } else {
                assert_eq!(action, TurnRemedyAction::RemedyExhausted);
            }
        }
    }

    // ------------------------------------------------------------------
    // DefaultTurnRemedyPolicy tests
    // ------------------------------------------------------------------

    #[test]
    fn test_default_remedy_budget_ok() {
        let mut policy = DefaultTurnRemedyPolicy::default();
        let mut messages = vec![];
        let action = policy.evaluate(50, &mut messages, 0).unwrap();
        assert_eq!(action, TurnRemedyAction::Continue);
    }

    #[test]
    fn test_tool_failure_remedy_policy_tracking() {
        let mut policy = ToolFailureRemedyPolicy::new(3);
        
        let tool_error_msg = Message {
            role: MessageRole::Tool,
            content: MessageContent::Text("tool failed".to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: true,
            include_in_prompt: true,
        };
        
        let mut messages = vec![tool_error_msg.clone()];
        let action1 = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action1, TurnRemedyAction::Remedy);
        assert_eq!(policy.consecutive_failures, 1);

        policy.consecutive_failures = 1;
        let mut messages = vec![tool_error_msg.clone()];
        let action2 = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action2, TurnRemedyAction::Remedy);
        assert_eq!(policy.consecutive_failures, 2);

        policy.consecutive_failures = 2;
        let mut messages = vec![tool_error_msg];
        let action3 = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action3, TurnRemedyAction::RemedyExhausted);
        assert_eq!(policy.consecutive_failures, 3);
    }

    #[test]
    fn test_tool_failure_remedy_policy_max_failures() {
        let mut policy = ToolFailureRemedyPolicy::new(3);
        
        let tool_msg = Message {
            role: MessageRole::Tool,
            content: MessageContent::Text("web_search error".to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: true,
            include_in_prompt: true,
        };

        let mut messages = vec![tool_msg.clone()];
        let action1 = policy.evaluate(0, &mut messages, 0).unwrap();
        assert_eq!(action1, TurnRemedyAction::Remedy);

        policy.consecutive_failures = 1;
        let mut messages2 = vec![tool_msg.clone()];
        let action2 = policy.evaluate(0, &mut messages2, 0).unwrap();
        assert_eq!(action2, TurnRemedyAction::Remedy);

        policy.consecutive_failures = 2;
        let mut messages3 = vec![tool_msg.clone()];
        let action3 = policy.evaluate(0, &mut messages3, 0).unwrap();
        assert_eq!(action3, TurnRemedyAction::RemedyExhausted);
    }
}
