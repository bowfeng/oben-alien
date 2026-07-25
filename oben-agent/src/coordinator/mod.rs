/// The main conversation loop that all modes implement.

use std::sync::Arc;

use anyhow::Result;
use tracing::info;

use crate::context::ContextWindowManager;
use crate::hooks::HookEngine;
use crate::interrupt::SharedInterrupt;
use crate::turn_executor::TurnExecutor;
use oben_models::{SessionManager, TransportProvider};

/// Shared config for both CLI and TUI conversation loops.
pub mod config;
pub use config::*;

/// Pluggable termination policies used by concrete coordinators.
pub mod termination;

// -- Shared turn execution utilities --

/// Execute a single turn.
pub async fn execute_turn(
    context_window_manager: &mut dyn ContextWindowManager,
    transport: &dyn TransportProvider,
    tools: &Arc<oben_tools::ToolRegistry>,
    session_manager: &mut dyn SessionManager,
    session_id: &str,
    user_message: oben_models::Message,
    call_mode: &oben_models::CallMode,
    conversation_config: &ConversationConfig,
) -> Result<String> {
    let turn_config = crate::turn_executor::TurnConfig {
        retry_config: conversation_config.retry_config.clone(),
        hooks: None,
        fallback_chain: if conversation_config.fallback_configs.is_empty() {
            None
        } else {
            Some(crate::fallback::FallbackChain::new(
                conversation_config
                    .fallback_configs
                    .iter()
                    .map(|fb| crate::fallback::FallbackConfig {
                        provider: fb.provider.clone(),
                        model: fb.model.clone(),
                        api_key: fb.api_key.clone(),
                        base_url: fb.base_url.clone(),
                    })
                    .collect(),
            ))
        },
        dispatch_config: conversation_config.dispatch_config.clone(),
        max_iterations: conversation_config.max_iterations,
        memory_context: conversation_config.memory_context.clone(),
    };

    let result = TurnExecutor::execute_turn_with_config(
        context_window_manager,
        transport,
        tools,
        session_manager,
        session_id,
        user_message,
        call_mode,
        None,
        None,
        turn_config,
    )
    .await?;

    Ok(result.text)
}

/// Execute a single turn with full agent integration (hooks, interrupt).
pub async fn execute_turn_full(
    context_window_manager: &mut dyn ContextWindowManager,
    transport: &dyn TransportProvider,
    tools: &Arc<oben_tools::ToolRegistry>,
    session_manager: &mut dyn SessionManager,
    session_id: &str,
    user_message: oben_models::Message,
    call_mode: &oben_models::CallMode,
    conversation_config: &ConversationConfig,
    hooks: Option<Arc<HookEngine>>,
    interrupt: Option<SharedInterrupt>,
) -> Result<String> {
    info!("execute_turn_full: starting with session_id={}", session_id);
    let turn_config = crate::turn_executor::TurnConfig {
        retry_config: conversation_config.retry_config.clone(),
        hooks,
        fallback_chain: if conversation_config.fallback_configs.is_empty() {
            None
        } else {
            Some(crate::fallback::FallbackChain::new(
                conversation_config
                    .fallback_configs
                    .iter()
                    .map(|fb| crate::fallback::FallbackConfig {
                        provider: fb.provider.clone(),
                        model: fb.model.clone(),
                        api_key: fb.api_key.clone(),
                        base_url: fb.base_url.clone(),
                    })
                    .collect(),
            ))
        },
        dispatch_config: conversation_config.dispatch_config.clone(),
        max_iterations: conversation_config.max_iterations,
        memory_context: conversation_config.memory_context.clone(),
    };
    info!("execute_turn_full: Build turn_config");

    info!("execute_turn_full: calling TurnExecutor::execute_turn_with_config");
    let result = TurnExecutor::execute_turn_with_config(
        context_window_manager,
        transport,
        tools,
        session_manager,
        session_id,
        user_message,
        call_mode,
        None,
        None,
        turn_config,
    )
    .await?;
    info!("execute_turn_full: returned result with text=\"{}\"", result.text);

    // Note: interrupt parameter was removed — interrupt handling moved to the
    // concrete coordinators (CliCoordinator/TuiCoordinator) via the new flow.
    let _ = interrupt;

    Ok(result.text)
}

/// Result of a conversation loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversationResult {
    /// User chose to exit normally.
    Exit,
    /// Loop ended due to budget/iteration limit.
    BudgetExhausted,
    /// Loop ended due to external stop signal (Ctrl+C, disconnect).
    Interrupted,
    /// Loop ended due to a goal being satisfied.
    GoalDone,
    /// Loop ended due to an unrecoverable error.
    Error(String),
}

/// The conversation coordinator.
///
/// The agent owns the turn loop. Coordinators only provide I/O callbacks:
/// - `on_loop_start` / `on_loop_end` for lifecycle events
/// - `on_turn_complete` to handle turn results and decide loop continuation
#[async_trait::async_trait]
pub trait ConversationCoordinator: Send + Sync {
    /// Called once at loop start.
    /// Implementations should emit pre-turn hooks, set up state, etc.
    fn on_loop_start(&mut self) {}

    /// Read next user turn input.
    /// Returns `Some(input)` with user text on success, `None` when no more input is available.
    async fn next_turn(&mut self) -> Option<String> { None }

    /// Called after each turn completes.
    /// Returns `true` to continue the loop, `false` to exit.
    ///
    /// For streaming mode, the coordinator should handle output formatting.
    /// For non-streaming mode, it should print the full response.
    fn on_turn_complete(&mut self, response: &str, msg_count: usize, turn_count: u32, success: bool) -> bool;

    /// Called when the loop exits (no more input, user quit, etc.).
    fn on_loop_end(&mut self, _outcome: &ConversationResult) {}
}
