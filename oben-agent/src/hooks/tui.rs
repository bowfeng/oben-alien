use std::sync::Arc;

use parking_lot::Mutex as PlMutex;

use super::kind::*;

/// Shared state reference for TUI hook adapters.
/// All adapters write directly to the same Arc<Mutex<TurnState>>.
struct TuiState {
    state: Arc<PlMutex<TurnState>>,
    next_tool_id: std::sync::atomic::AtomicU32,
}

impl TuiState {
    fn new(state: Arc<PlMutex<TurnState>>) -> Self {
        Self {
            state,
            next_tool_id: std::sync::atomic::AtomicU32::new(1),
        }
    }

    fn next_tool_id(&self) -> String {
        self.next_tool_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed).to_string()
    }
}

// ---------------------------------------------------------------------------
// Streaming Adapter — writes deltas directly to TurnState
// ---------------------------------------------------------------------------

pub struct TuiStreamingAdapter {
    state: Arc<TuiState>,
}

impl TuiStreamingAdapter {
    pub fn new(state: Arc<PlMutex<super::kind::TurnState>>) -> Self {
        Self {
            state: Arc::new(TuiState::new(state)),
        }
    }
}

impl Hook for TuiStreamingAdapter {
    fn id(&self) -> &str { "tui_streaming" }
    fn priority(&self) -> u32 { 10 }
}

impl StreamingHooks for TuiStreamingAdapter {
    fn on_stream_delta(&self, text: &str) {
        let mut ts = self.state.state.lock();
        let total_before = ts.streaming_text.len();
        let delta_len = text.len();
        let total_after = total_before + delta_len;
        
        // Log every delta to trace streaming behavior
        tracing::debug!(
            delta_len = delta_len,
            total_before = total_before,
            total_after = total_after,
            phase = ?ts.phase,
            text_preview = ?text.chars().take(20).collect::<String>(),
            "[TuiStreamingAdapter] on_stream_delta"
        );
        
        ts.on_stream_delta(text);
        
        // Log after update for verification
        let final_len = ts.streaming_text.len();
        if final_len % 50 == 0 && final_len > 0 {
            tracing::info!(
                final_len = final_len,
                phase = ?ts.phase,
                "[TuiStreamingAdapter] After update: streaming_text.len={}",
                final_len
            );
        }
    }

    fn on_reasoning(&self, text: &str) {
        let mut ts = self.state.state.lock();
        ts.on_reasoning(text);
    }

    fn on_thinking(&self, _text: &str) {}

    fn on_interim_assistant(&self, _text: &str) {}
}

// ---------------------------------------------------------------------------
// Tool Lifecycle Adapter — writes tool events directly to TurnState
// ---------------------------------------------------------------------------

pub struct TuiToolLifecycleAdapter {
    state: Arc<TuiState>,
}

impl TuiToolLifecycleAdapter {
    pub fn new(state: Arc<PlMutex<super::kind::TurnState>>) -> Self {
        Self {
            state: Arc::new(TuiState::new(state)),
        }
    }
}

impl Hook for TuiToolLifecycleAdapter {
    fn id(&self) -> &str { "tui_tools" }
    fn priority(&self) -> u32 { 10 }
}

impl ToolLifecycleHooks for TuiToolLifecycleAdapter {
    fn on_tool_gen(&self, tool_name: &str, call_id: &str) {
        let _ = (tool_name, call_id);
    }

    fn on_tool_start(&self, tool_name: &str, args: &str) {
        let tid = self.state.next_tool_id();
        let mut ts = self.state.state.lock();
        ts.on_tool_start(&tid, tool_name, args);
    }

    fn on_tool_complete(&self, tool_name: &str, _args: &str, result: &str) {
        let tid = self.state.next_tool_id();
        let mut ts = self.state.state.lock();
        ts.on_tool_complete(&tid, tool_name, result);
    }

    fn on_tool_error(
        &self,
        _tool_name: &str,
        _args: &str,
        _error: &str,
    ) {
        // Tool errors are captured during tool execution in TurnState.
        // We leave the tool in active_tools to reflect that it didn't complete.
    }

    fn on_tool_progress(&self, _tool_name: &str, _preview: &str) {}
}

// ---------------------------------------------------------------------------
// Agent Loop Adapter — writes turn start/end to TurnState
// ---------------------------------------------------------------------------

pub struct TuiAgentLoopAdapter {
    state: Arc<TuiState>,
}

impl TuiAgentLoopAdapter {
    pub fn new(state: Arc<PlMutex<super::kind::TurnState>>) -> Self {
        Self {
            state: Arc::new(TuiState::new(state)),
        }
    }
}

impl Hook for TuiAgentLoopAdapter {
    fn id(&self) -> &str { "tui_agent_loop" }
    fn priority(&self) -> u32 { 10 }
}

impl AgentLoopHooks for TuiAgentLoopAdapter {
    fn on_loop_start(&self) {
        tracing::debug!("[tui_agent_loop] on_loop_start: phase -> Idle");
        let mut ts = self.state.state.lock();
        ts.phase = TurnPhase::Idle;
    }

    fn on_loop_end(&self, outcome: &str) {
        tracing::debug!("[tui_agent_loop] on_loop_end: outcome={}", outcome);
        let mut ts = self.state.state.lock();
        // Only complete if not already in error state — don't overwrite TurnPhase::Error
        if matches!(ts.phase, TurnPhase::Completed) {
            ts.on_completed(outcome);
        }
    }
}

// ---------------------------------------------------------------------------
// Turn Lifecycle Adapter — writes turn start/complete/error to TurnState
// ---------------------------------------------------------------------------

pub struct TuiTurnLifecycleAdapter {
    state: Arc<TuiState>,
}

impl TuiTurnLifecycleAdapter {
    pub fn new(state: Arc<PlMutex<super::kind::TurnState>>) -> Self {
        Self {
            state: Arc::new(TuiState::new(state)),
        }
    }
}

impl Hook for TuiTurnLifecycleAdapter {
    fn id(&self) -> &str { "tui_turn" }
    fn priority(&self) -> u32 { 10 }
}

impl TurnLifecycleHooks for TuiTurnLifecycleAdapter {
    fn on_pre_turn(&self) {
        tracing::info!("[tui_turn] on_pre_turn: phase -> Streaming");
        let mut ts = self.state.state.lock();
        ts.on_turn_start();
    }

    fn on_post_turn(&self, response: &str, success: bool, _turn_count: u32) {
        tracing::info!("[tui_turn] on_post_turn: response.len={}, success={}", response.len(), success);
        let mut ts = self.state.state.lock();
        if success {
            ts.on_completed(response);
        } else {
            ts.on_error(response);
        }
    }
}

// ---------------------------------------------------------------------------
// Subagent Lifecycle Adapter — intercepts delegation events via ToolLifecycleHooks
// ---------------------------------------------------------------------------

/// Trait implemented by SharedAgentState to accept subagent lifecycle events.
pub trait SubagentLifecycleCallback: Send + Sync {
    fn on_start(
        &self,
        delegation_id: u32,
        parent_session_id: &str,
        goal: &str,
    );
    fn on_complete(
        &self,
        delegation_id: u32,
        result: &str,
        status: &str,
        tool_calls: Vec<SubagentToolInfo>,
    );
}

/// Adapter that listens to tool hook events and updates SharedAgentState.subagents.
///
/// Intercepts `delegate_task` tool calls by parsing `delegation_id` from args.
/// Registers subagents on tool start, completes them on tool complete.
pub struct TuiSubagentAdapter {
    callback: Arc<dyn SubagentLifecycleCallback>,
}

impl TuiSubagentAdapter {
    pub fn new(callback: Arc<dyn SubagentLifecycleCallback>) -> Self {
        Self {
            callback,
        }
    }

    fn parse_delegation_id(args: &str) -> Option<u32> {
        serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| v.get("delegation_id").and_then(|d| d.as_u64()))
            .map(|n| n as u32)
    }

    fn parse_field(args: &str, field: &str) -> String {
        serde_json::from_str::<serde_json::Value>(args)
            .ok()
            .and_then(|v| {
                let s = v.get(field).and_then(|f| f.as_str());
                s.map(|s| s.to_string())
            })
            .unwrap_or_default()
    }
}

impl Hook for TuiSubagentAdapter {
    fn id(&self) -> &str { "tui_subagent" }
    fn priority(&self) -> u32 { 5 }
}

impl ToolLifecycleHooks for TuiSubagentAdapter {
    fn on_tool_start(&self, tool_name: &str, args: &str) {
        if tool_name != "delegate_task" {
            return;
        }
        if let Some(delegation_id) = Self::parse_delegation_id(args) {
            let parent = Self::parse_field(args, "parent_session_id");
            let goal = Self::parse_field(args, "goal");
            let callback_ref = Arc::clone(&self.callback);
            callback_ref.on_start(delegation_id, &parent, &goal);
        }
    }

    fn on_tool_complete(&self, tool_name: &str, args: &str, result: &str) {
        if tool_name != "delegate_task" {
            return;
        }
        let delegate_id = Self::parse_delegation_id(args).unwrap_or(0);

        if let Ok(v) = serde_json::from_str::<serde_json::Value>(result) {
            let status = v
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string();
            let sessions = v
                .get("sessions")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|s| {
                            let name = s
                                .get("tool_name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("");
                            let args_val = s.get("args").cloned().unwrap_or(serde_json::Value::Null);
                            let output = s
                                .get("output")
                                .and_then(|o| o.as_str())
                                .unwrap_or("");
                            let preview = output.chars().take(80).collect::<String>();
                            if name.is_empty() {
                                return None;
                            }
                            Some(SubagentToolInfo {
                                name: name.to_string(),
                                args: args_val,
                                output_preview: preview,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let callback_ref = Arc::clone(&self.callback);
            callback_ref.on_complete(delegate_id, result, &status, sessions);
        }
    }

    fn on_tool_error(&self, _tool_name: &str, _args: &str, _error: &str) {}
    fn on_tool_gen(&self, _tool_name: &str, _call_id: &str) {}
    fn on_tool_progress(&self, _tool_name: &str, _preview: &str) {}
}
