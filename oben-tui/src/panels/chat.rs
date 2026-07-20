//! Chat panel — message history, streaming, input bar.

use super::{KeyAction, Panel};
use crate::widgets::conversation::{ConversationState, ConversationWidget};
use crate::widgets::input_bar::{InputBarResult, InputBarWidget, InputState};
use crate::widgets::message_renderer::{MessageRenderEntry, MessageRenderer, StyledLine, render_body_lines};
use crate::widgets::subagent_accordion::SubagentAccordion;
use crate::shared::SubagentInfo;
use crossterm::event::KeyEvent;
use std::sync::atomic::Ordering;
use parking_lot::Mutex as PlMutex;
use ratatui::layout::Rect;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use std::sync::Arc;
use std::time::Instant;
use oben_agent::{TurnPhase, TurnState};
use oben_models::Message;

/// Chat panel — message history, input bar, and streaming control.
pub struct ChatPanel {
    pub session_name: Option<String>,
    pub streaming: bool,
    pub input: InputState,
    pub message_state: ConversationState,
    pub message_count: usize,
    renderer: MessageRenderer,
    message_display: ConversationWidget,
    /// Turn state reference for polling during draw.
    turn_state_ref: Arc<PlMutex<TurnState>>,
    /// Shared agent state reference for subagent data during draw.
    shared_state_ref: Arc<PlMutex<crate::shared::SharedAgentState>>,
    /// Channel to send drained messages back to the event loop.  Set via
    /// `set_input_sender()` during app init.
    input_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::TuiEvent>>,
    /// Previous turn phase — used to detect transitions into a settled state
    /// for queue auto-drain.  Only the *transition* into Completed/Error
    /// triggers a drain; steady-state draws do not fire again.
    prev_phase: TurnPhase,
    /// Whether auto-drain has already fired for this completion.
    /// Prevents draining the next queued message when a new turn starts
    /// but the phase hasn't yet moved from Completed (e.g., while delta
    /// callbacks are still firing for the turn that just ended).
    drained_this_turn: bool,
    /// Subagent accordion state (track which subagents are expanded).
    subagent_accordion: SubagentAccordion,
}

impl ChatPanel {
    pub fn new(session_name: Option<String>) -> Self {
        let message_state = ConversationState::new();
        Self {
            session_name,
            streaming: false,
            input: InputState::new(),
            message_state,
            message_count: 0,
            renderer: MessageRenderer::new(),
            message_display: ConversationWidget,
            turn_state_ref: Arc::new(PlMutex::new(TurnState::new())),
            shared_state_ref: Arc::new(PlMutex::new(
                crate::shared::SharedAgentState::new_empty(),
            )),
            input_tx: None,
            prev_phase: TurnPhase::Idle,
            drained_this_turn: false,
            subagent_accordion: SubagentAccordion::new(),
        }
    }

    /// Create a chat panel with a specific theme from config.
    pub fn new_with_theme(session_name: Option<String>, theme: &str, turn_state: Arc<PlMutex<TurnState>>) -> Self {
        let mut panel = Self::new(session_name);
        panel.renderer.set_theme_from_str(theme);
        // Both references must point to the same Arc that the TuiStreamingAdapter writes to
        panel.turn_state_ref = Arc::clone(&turn_state);
        panel.set_turn_state_ref(turn_state);
        panel
    }

    /// Set the shared state reference for subagent rendering.
    pub fn set_shared_state_ref(&mut self, state: Arc<PlMutex<crate::shared::SharedAgentState>>) {
        self.shared_state_ref = state;
    }

    /// Set the input event sender for queue drain — drained messages are
    /// sent through here so they go through the normal event loop.
    pub fn set_input_sender(&mut self, tx: tokio::sync::mpsc::UnboundedSender<crate::TuiEvent>) {
        self.input_tx = Some(tx);
    }

    /// Cycle to the next theme, returning the new theme name for persistence.
    pub fn cycle_theme(&mut self) -> String {
        let current = self.renderer.current_theme();
        let next = current.next();
        self.renderer.set_theme(next);
        next.slug().to_string()
    }

    /// Update message display state from session messages.
    pub fn update_from_messages(&mut self, messages: &[Message], session_name: Option<String>) {
        if messages.is_empty() {
            // No messages from coordinator — entries were built during
            // the turn via TuiStreamingAdapter + append_user_message.
            // Don't wipe them.
            return;
        }
        self.message_display.rebuild_from_messages(
            &mut self.message_state,
            messages,
            &self.renderer,
            false,
        );
        self.session_name = session_name;
        self.message_count = messages.len();
        self.streaming = false;
    }

    /// Set session data during panel activation.
    pub fn set_session_data(&mut self, session_name: Option<String>, messages: Vec<Message>) {
        if !messages.is_empty() || session_name.is_some() {
            self.update_from_messages(&messages, session_name);
        }
    }

    /// Append a user message to the display (shown immediately during a turn).
    pub fn append_user_message(&mut self, text: &str) {
        self.message_display
            .append_user_message(&mut self.message_state, text);
    }

    /// Append an info/system message to the display (used for slash commands like /help).
    pub fn append_info_message(&mut self, text: &str) {
        self.message_display
            .append_info_message(&mut self.message_state, text);
    }

    /// Update from turn state and sync stream_info in display.
    pub fn update_from_turn_state(&mut self) {
        // Use try_lock to avoid deadlock: during LLM streaming the callback
        // holds turn_state_ref.lock() and subsequently needs arc_app.async_lock();
        // the draw closure holds arc_app.async_lock() and needs turn_state_ref —
        // try_lock breaks the cycle by skipping the draw instead of blocking.
        let ts = self.turn_state_ref.try_lock();
        let current = ts.as_ref().map(|ts| ts.phase.clone()).unwrap_or(TurnPhase::Idle);
        let agent_idle = ts.as_ref().map(|ts| !ts.is_active()).unwrap_or(true);
        // Drop ts before we start using &mut self to avoid borrow conflicts
        drop(ts);

        let prev = self.prev_phase.clone();
        let settled = matches!(
            current,
            TurnPhase::Completed | TurnPhase::Error(_)
        );
        let transitioning = !matches!(
            prev,
            TurnPhase::Completed | TurnPhase::Error(_)
        );
        
        let new_turn_started = matches!(current, TurnPhase::Streaming) 
            && matches!(prev, TurnPhase::Completed | TurnPhase::Error(_));
        if new_turn_started {
            self.message_count = 0;
        }

        self.prev_phase = current.clone();

        let drain_trigger = settled && transitioning && agent_idle && !self.drained_this_turn;
        if settled || transitioning {
            tracing::info!(
                "[chat_panel] update_from_turn_state: settled={}, transitioning={}, agent_idle={}, drained_this_turn={}, prev_phase={:?}, current_phase={:?}, drain={}",
                settled, transitioning, agent_idle, self.drained_this_turn, prev, current, drain_trigger
            );
        }
        tracing::debug!(
            "[chat_panel] update_from_turn_state: settled={}, transitioning={}, agent_idle={}, drained_this_turn={}, prev_phase={:?}, current_phase={:?}, drain={}",
            settled, transitioning, agent_idle, self.drained_this_turn, prev, current, drain_trigger
        );

        if drain_trigger {
            tracing::debug!(
                "[chat_panel] auto-drain trigger: queue_len={}",
                self.input.queue_len()
            );
            let drain_time = Instant::now();
            // Send a single QueueDrain event; the event loop will
            // dequeue ALL messages from the queue in one pass and
            // process each one sequentially through handle_chat_input.
            // This avoids the race condition that used to occur when
            // sending individual ChatInput events here.
            if let Some(ref tx) = &self.input_tx {
                if tx.send(crate::TuiEvent::QueueDrain).is_err() {
                    tracing::warn!("[chat_panel] failed to send QueueDrain");
                }
                self.message_state
                    .scroll_to_bottom
                    .store(true, Ordering::SeqCst);
                tracing::debug!(
                    "[chat_panel] drain: sent QueueDrain in {:?}, queue_len={}",
                    drain_time.elapsed(),
                    self.input.queue_len()
                );
            } else {
                tracing::debug!("[chat_panel] no input_tx for drained message");
            }
            self.drained_this_turn = true;
        } else if !settled || !transitioning {
            // Reset on any non-transition, but more importantly reset when we
            // start a fresh turn (idle / completed → idle → streaming).
            self.drained_this_turn = false;
        }

        // Clear subagent data during draw cycle
        self.shared_state_ref.try_lock().map(|guard| {
            guard.clear_subagents();
        });

        // Update streaming flag on phase transition
        let is_entering_streaming = matches!(current, TurnPhase::Streaming)
            && (matches!(prev, TurnPhase::Idle)
                || matches!(prev, TurnPhase::Completed)
                || matches!(prev, TurnPhase::Error(_)));
        if is_entering_streaming {
            self.streaming = true;
            self.input.streaming = true;
        } else if matches!(prev, TurnPhase::Streaming)
            && !matches!(current, TurnPhase::Streaming)
        {
            self.streaming = false;
            self.input.streaming = false;
        }
        self.prev_phase = current;
    }

    /// Copy selection to system clipboard, then clear selection.
    pub fn copy_selection_to_clipboard(&mut self) {
        tracing::debug!(
            "[selection] copy: start={:?} end={:?} body_width={} scroll_pos={}",
            self.message_state.selection_start,
            self.message_state.selection_end,
            self.message_state.body_width,
            self.message_state.scroll_pos.load(Ordering::SeqCst)
        );
        if let Some(text) = self.message_display.get_selected_text(&self.message_state) {
            tracing::debug!("[selection] copy: got {} chars", text.len());
            if !text.is_empty() && crate::clipboard::write_clipboard(&text) {
                self.message_state.clear_selection();
            }
        } else {
            tracing::debug!("[selection] copy: no text returned");
        }
    }

    /// Set turn_state_ref so message display can read streaming text in real-time.
    pub fn set_turn_state_ref(
        &mut self,
        turn_state: Arc<PlMutex<TurnState>>,
    ) {
        self.message_state.turn_state_ref = Some(turn_state);
    }

    /// Clear all messages from the display and reset the message count.
    pub fn clear_display(&mut self) {
        let entries_before = self.message_state.message_entries.lock().unwrap().len();
        self.message_state.message_entries.lock().unwrap().clear();
        let entries_after = self.message_state.message_entries.lock().unwrap().len();
        tracing::debug!("[chat_panel] clear_display: entries_before={} -> after={} (should be 0)", entries_before, entries_after);
    }

    /// Render the input bar widget.
    fn render_input_bar(&self, frame: &mut Frame, area: Rect, state: &InputState) {
        let palette = self.renderer.current_palette();
        InputBarWidget.render(frame, area, state, &palette);
    }

    /// Render the subagers sidebar panel.
    fn render_subagers_panel(&self, frame: &mut Frame, area: Rect, subagers: &[SubagentInfo]) {
        let header = Line::from(vec![
            Span::styled("\u{25c6}", Style::default().fg(Color::Cyan).bold()),
            Span::raw(" Subagers"),
            Span::styled(
                format!(" ({})", subagers.len()),
                Style::default().fg(Color::DarkGray),
            ),
        ]);

        let block = Block::default()
            .title(header)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        frame.render_widget(block, area);

        let inner = area.inner(Margin { horizontal: 1, vertical: 1 });
        if inner.height == 0 || inner.width == 0 {
            return;
        }

        let content_width = inner.width as usize;

        let mut lines: Vec<Line> = Vec::new();

        for sub in subagers.iter() {
            let is_expanded = self.subagent_accordion.is_expanded(sub.delegation_id);

            let status_color = match sub.status.as_str() {
                "running" => Color::Yellow,
                "completed" => Color::Green,
                "error" => Color::Red,
                _ => Color::Gray,
            };

            let status_marker = match sub.status.as_str() {
                "running" => "[running]",
                "completed" => "[done]",
                "error" => "[failed]",
                _ => "[?]",
            };

            let arrow = if is_expanded { "\u{25be}" } else { "\u{25b8}" };
            lines.push(Line::from(vec![
                Span::styled(arrow, Style::default().fg(Color::DarkGray)),
                Span::styled(status_marker, Style::default().fg(status_color)),
                Span::styled(
                    format!(" #{}", sub.delegation_id),
                    Style::default().fg(status_color),
                ),
            ]));

            // Split goal into lines that fit within the panel width.
            let goal_prefix = "  goal: ";
            let goal_text_width = content_width.saturating_sub(goal_prefix.len());
            let goals: Vec<String> = if goal_text_width > 0 {
                sub.goal
                    .chars()
                    .collect::<Vec<_>>()
                    .chunks(goal_text_width)
                    .map(|c| c.iter().collect())
                    .collect()
            } else {
                vec![sub.goal.clone()]
            };
            for (i, g) in goals.into_iter().enumerate() {
                lines.push(Line::from(vec![
                    Span::styled(
                        if i == 0 { "  goal: " } else { "          " },
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        g,
                        Style::default().fg(Color::White).add_modifier(Modifier::DIM),
                    ),
                ]));
            }

            if sub.status == "running" {
                lines.push(Line::from(vec![
                    Span::styled(
                        "⠋ executing",
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    ),
                ]));

                // Show tool call previews
                for preview in sub.tool_call_previews.iter().take(2) {
                    if !preview.is_empty() {
                        let truncated: String = preview.chars().take(60).collect();
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(truncated, Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM)),
                        ]));
                    }
                }

                // Show stats
                if sub.stats.tool_count > 0 {
                    let stat_text = if sub.stats.token_count > 0 {
                        format!("{} tools, {} tokens", sub.stats.tool_count, sub.stats.token_count)
                    } else {
                        format!("{} tool calls", sub.stats.tool_count)
                    };
                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(stat_text, Style::default().fg(Color::DarkGray)),
                    ]));
                }

                // Show duration
                if !sub.stats.duration.is_empty() {
                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(
                            format!("elapsed: {}", sub.stats.duration),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                }
            }

            if is_expanded {
                if let Some(ref t) = sub.start_time {
                    lines.push(Line::from(vec![
                        Span::raw("  start: "),
                        Span::styled(t, Style::default().fg(Color::Blue)),
                    ]));
                }
                if let Some(ref t) = sub.end_time {
                    lines.push(Line::from(vec![
                        Span::raw("  end: "),
                        Span::styled(t, Style::default().fg(Color::Blue)),
                    ]));
                }
                if !sub.summary.is_empty() {
                    let resp = if sub.summary.chars().count() > content_width.saturating_sub(10) {
                        sub.summary
                            .chars()
                            .take(content_width.saturating_sub(10))
                            .collect::<String>()
                    } else {
                        sub.summary.clone()
                    };
                    lines.push(Line::from(vec![
                        Span::raw("  summary: "),
                        Span::styled(resp, Style::default().fg(Color::Gray)),
                    ]));
                }
                if !sub.children.is_empty() {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            format!("({} child subagers)", sub.children.len()),
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ]));
                }

                // Render nested children if expanded
                for child in &sub.children {
                    let child_expanded = self.subagent_accordion.is_expanded(child.delegation_id);
                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(
                            if child_expanded { "\u{25be}" } else { "\u{25b8}" },
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::DIM),
                        ),
                        Span::styled(
                            format!(" [child] #{}", child.delegation_id),
                            Style::default()
                                .fg(Color::Blue)
                                .add_modifier(Modifier::DIM),
                        ),
                    ]));
                }
            }
        }

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);
    }

}

#[async_trait::async_trait]
impl Panel for ChatPanel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&self, frame: &mut Frame, area: Rect) {
        let subagents = self.shared_state_ref.try_lock()
            .map(|s| s.get_subagents())
            .unwrap_or_default();

        let input_height = InputBarWidget.calculate_input_height(&self.input, area.width);

        if subagents.is_empty() {
            // Only messages + input bar
            let msg_area = area.height.saturating_sub(input_height);
            if msg_area > 0 {
                self.message_display.render(
                    frame,
                    Rect::new(area.x, area.y, area.width, msg_area),
                    &self.message_state,
                    &self.renderer.current_palette(),
                    self.streaming,
                );
                if self.message_state.selection_start.is_some() {
                    self.message_display.render_selection(
                        frame,
                        Rect::new(area.x, area.y, area.width, msg_area),
                        &self.message_state,
                        &self.renderer.current_palette(),
                    );
                }
            }
            self.render_input_bar(
                frame,
                Rect::new(
                    area.x,
                    area.y + area.height.saturating_sub(input_height),
                    area.width,
                    input_height,
                ),
                &self.input,
            );
        } else {
            // ── ALWAYS split: chat (left) | subager panel (right) ──
            let right_w = std::cmp::min(45u16, area.width.saturating_sub(20));
            let chat_area = Rect::new(area.x, area.y, area.width.saturating_sub(right_w), area.height);
            let sidebar_area = Rect::new(
                area.x + area.width.saturating_sub(right_w),
                area.y,
                right_w,
                area.height,
            );

            // Left: messages + input
            let msg_area = chat_area.height.saturating_sub(input_height);
            if msg_area > 0 {
                self.message_display.render(
                    frame,
                    Rect::new(chat_area.x, chat_area.y, chat_area.width, msg_area),
                    &self.message_state,
                    &self.renderer.current_palette(),
                    self.streaming,
                );
                if self.message_state.selection_start.is_some() {
                    self.message_display.render_selection(
                        frame,
                        Rect::new(chat_area.x, chat_area.y, chat_area.width, msg_area),
                        &self.message_state,
                        &self.renderer.current_palette(),
                    );
                }
            }
            self.render_input_bar(
                frame,
                Rect::new(
                    chat_area.x,
                    chat_area.y + chat_area.height.saturating_sub(input_height),
                    chat_area.width,
                    input_height,
                ),
                &self.input,
            );

            // Right: subagers always rendered as sidebar
            self.render_subagers_panel(frame, sidebar_area, &subagents);
        }
    }

     async fn handle_key(&mut self, key: KeyEvent) -> KeyAction {
         let result = InputBarWidget.handle_key(&mut self.input, key);
         tracing::debug!(
             "[chat_panel] handle_key: code={:?} result={:?}",
             key.code,
             result
         );
         match result {
             InputBarResult::Consumed => KeyAction::None,
             InputBarResult::PassedThrough => KeyAction::None,
             InputBarResult::ChatInput(text) => KeyAction::ChatInput(text),
             InputBarResult::SlashCommand { cmd_name, extra } => match cmd_name.as_str() {
                 "clear" => KeyAction::Clear,
                 "new" => KeyAction::New,
                 "compact" => KeyAction::Compact,
                 "quit" => KeyAction::Quit,
                 "reasoning" => KeyAction::Reasoning,
                 "theme" => KeyAction::Theme,
                 _ => KeyAction::Command { cmd_name, extra },
             },
              InputBarResult::Interrupt => KeyAction::Interrupt,
              InputBarResult::Steer(text) => KeyAction::Steer(text),
          }
      }

    fn handle_mouse(&mut self, area: Rect, event: &crossterm::event::MouseEvent) -> Option<String> {
        use crossterm::event::MouseButton;
        use crossterm::event::MouseEventKind;

        let scroll_step: i32 = 3;
        let actual_body_width = area.width.saturating_sub(6);
        let content_x_offset = area.x.saturating_add(4);
        tracing::info!(
            "[selection] mouse event: row={} col={} kind={:?} msg_area_y={} content_x={} content_x_offset={} body_width={} old_body_width={}",
            event.row, event.column, event.kind, area.y,
            self.message_state.content_x,
            content_x_offset,
            actual_body_width,
            self.message_state.body_width
        );
        if actual_body_width > 0 {
            self.message_state.body_width = actual_body_width as usize;
        }
        self.message_state.wrap_width = actual_body_width as usize;
        self.message_state.content_x = content_x_offset;
        self.message_state.msg_area_y = area.y;
        self.message_state.content_y = area.y.saturating_add(2);

        match event.kind {
            MouseEventKind::ScrollDown => {
                self.message_state
                    .scroll_to_bottom
                    .store(false, Ordering::SeqCst);
                let old = self.message_state.user_scroll_offset.load(Ordering::SeqCst);
                self.message_state
                    .user_scroll_offset
                    .fetch_sub(scroll_step, Ordering::SeqCst);
                tracing::info!(
                    "[mouse] ScrollDown: old_offset={} new_offset={} scroll_to_bottom=false",
                    old,
                    self.message_state.user_scroll_offset.load(Ordering::SeqCst)
                );
                return None;
            }
            MouseEventKind::ScrollUp => {
                self.message_state
                    .scroll_to_bottom
                    .store(false, Ordering::SeqCst);
                let old = self.message_state.user_scroll_offset.load(Ordering::SeqCst);
                self.message_state
                    .user_scroll_offset
                    .fetch_add(scroll_step, Ordering::SeqCst);
                tracing::info!(
                    "[mouse] ScrollUp: old_offset={} new_offset={} scroll_to_bottom=false",
                    old,
                    self.message_state.user_scroll_offset.load(Ordering::SeqCst)
                );
                return None;
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let row = event.row;
                let col = event.column;
                self.message_state.selection_start = Some((row, col));
                self.message_state.selection_end = None;
                tracing::debug!(
                    "[selection] MOUSE_DOWN: row={} col={} → selection=({},{})",
                    row,
                    col,
                    row,
                    col
                );
                return None;
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                let row = event.row;
                let col = event.column;
                self.message_state.selection_end = Some((row, col));
                return None;
            }
            MouseEventKind::Up(MouseButton::Left) => {
                let row = event.row;
                let col = event.column;
                self.message_state.selection_end = Some((row, col));
                tracing::debug!(
                    "[mouse] LeftRelease: sel=(row={},col={}) content_y={} scroll_pos={}",
                    row,
                    col,
                    self.message_state.content_y,
                    self.message_state.scroll_pos.load(Ordering::SeqCst)
                );
                if self.message_state.selection_start.is_some()
                    && self.message_state.selection_end.is_some()
                {
                    if let Some(text) = self.message_display.get_selected_text(&self.message_state)
                    {
                        tracing::debug!("[mouse] LeftRelease: got {} chars", text.len());
                        if !text.is_empty() && crate::clipboard::write_clipboard(&text) {
                            tracing::info!("[mouse] auto-copied {} chars", text.len());
                            self.message_state.clear_selection();
                            return Some(text);
                        } else {
                            tracing::debug!("[mouse] LeftRelease: write_clipboard failed or empty");
                        }
                    } else {
                        tracing::debug!("[mouse] LeftRelease: get_selected_text returned None");
                    }
                }
                return None;
            }
            _ => None,
        }
    }
}

impl ChatPanel {
    /// Toggle expansion of the first (most recent) subagent.
    pub fn toggle_first_subagent(&mut self, subagents: &[SubagentInfo]) -> bool {
        if let Some(sub) = subagents.iter().rev().next() {
            self.subagent_accordion.toggle(sub.delegation_id);
            true
        } else {
            false
        }
    }

    /// Get a mutable reference to the subagent accordion.
    pub fn subagent_accordion_mut(&mut self) -> &mut SubagentAccordion {
        &mut self.subagent_accordion
    }

}

fn tool_name_to_title_case(name: &str) -> String {
    name.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
