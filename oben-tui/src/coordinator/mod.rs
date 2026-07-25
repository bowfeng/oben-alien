//! TUI conversation coordinator.
//!
//! The TUI coordinator is a thin wrapper around channels. It implements
//! `ConversationCoordinator` so `Agent::run()` can drive the turn loop via the
//! trait methods — no custom loop (`drive()` / `run_tui()`) needed. The event
//! loop receives status updates on `done_tx`.

use super::app::TurnCompletion;
use oben_agent::coordinator::{ConversationCoordinator, ConversationResult};

// ── TUI Commands ──────────────────────────────────────────────────────────

/// Commands from the event loop to TUI panels. Sent from the event loop,
/// received by panels to update UI state.
pub enum TuiCommand {
    /// Signal the event loop to prepare ChatPanel for streaming.
    StartTurn {
        input: String,
        session_name: Option<String>,
    },
    /// Append the user input to the session history.
    AppendInputHistory {
        input: String,
    },
}

// ── TUI Coordinator ──────────────────────────────────────────────────────

/// TUI coordinator — the minimal channel bridge for `Agent::run()`.
pub struct TuiCoordinator {
    chat_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    command_tx: tokio::sync::mpsc::UnboundedSender<TuiCommand>,
    done_tx: tokio::sync::mpsc::UnboundedSender<TurnCompletion>,
}

impl TuiCoordinator {
    pub fn new(
        chat_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
        command_tx: tokio::sync::mpsc::UnboundedSender<TuiCommand>,
        done_tx: tokio::sync::mpsc::UnboundedSender<TurnCompletion>,
    ) -> Self {
        Self {
            chat_rx,
            command_tx,
            done_tx,
        }
    }
}

#[::async_trait::async_trait]
impl ConversationCoordinator for TuiCoordinator {
    fn on_loop_start(&mut self) {
        // TUI sets up state in splash init; nothing special at trait level.
    }

    async fn next_turn(&mut self) -> Option<String> {
        tracing::info!("[TuiCoordinator] next_turn: waiting for chat_rx");
        let input = self.chat_rx.recv().await;
        tracing::info!("[TuiCoordinator] next_turn: received {:?}", input.as_deref().unwrap_or(""));
        tracing::info!("[TuiCoordinator] next_turn: is_closed={}, chat_rx_closed={}", self.chat_rx.is_closed(), self.chat_rx.is_closed());
        if let Some(ref text) = input {
            tracing::info!("[TuiCoordinator] next_turn: sending StartTurn with text.len={}", text.len());
            let _ = self.command_tx.send(TuiCommand::StartTurn {
                input: text.clone(),
                session_name: None,
            });
        }
        input
    }

    fn on_turn_complete(
        &mut self,
        response: &str,
        msg_count: usize,
        _turn_count: u32,
        success: bool,
    ) -> bool {
        tracing::debug!(
            "[TuiCoordinator] on_turn_complete: success={}, msg_count={}, response.len()={}, continuing={}",
            success,
            msg_count,
            response.len(),
            !self.done_tx.is_closed(),
        );

        let status = if success {
            format!(
                "Turn {} ({} chars)",
                if msg_count > 0 { "completed" } else { "empty" },
                response.len()
            )
        } else {
            format!("Turn failed: {response}")
        };

        let completion = TurnCompletion {
            success,
            status,
            session_name: None,
            message_count: msg_count,
        };
        tracing::debug!("[TuiCoordinator] sending TurnCompletion: success={}, response_len={}", success, response.len());
        let _ = self.done_tx.send(completion);

        true
    }

    fn on_loop_end(&mut self, _outcome: &ConversationResult) {
        // TUI exits naturally when `next_turn` returns None.
    }
}

#[::async_trait::async_trait]
impl ConversationCoordinator for &mut TuiCoordinator {
    fn on_loop_start(&mut self) {
        (**self).on_loop_start();
    }

    async fn next_turn(&mut self) -> Option<String> {
        (**self).next_turn().await
    }

    fn on_turn_complete(&mut self, response: &str, msg_count: usize, turn_count: u32, success: bool) -> bool {
        (**self).on_turn_complete(response, msg_count, turn_count, success)
    }

    fn on_loop_end(&mut self, outcome: &ConversationResult) {
        (**self).on_loop_end(outcome);
    }
}


