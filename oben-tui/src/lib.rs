//! ObenAgent TUI — a full terminal UI for chat, sessions, config, and setup.
//!
//! Replaces the CLI-based `oben chat`, `oben setup`, `oben config`, and
//! `oben sessions` with a ratatui-driven interface.

pub mod app;
pub mod clipboard;
pub mod commands;
pub mod coordinator;
pub mod history;
pub mod image;

pub mod panels;
pub mod shared;
pub mod widgets;

pub use app::App;
pub use coordinator::{TuiCommand, TuiCoordinator};

use anyhow::Result;
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
    MouseEvent, MouseEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, size, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use panels::PanelId;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph, Tabs};
use ratatui::{Frame, Terminal};
use ratatui_toaster::ToastType;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::app::TurnCompletion;
use oben_agent::Agent;
use oben_models::SessionManager;
use tokio::sync::mpsc::unbounded_channel;
use tokio::time::MissedTickBehavior;
use tracing::info;

pub struct Layouts {
    pub header: Rect,
    pub body: Rect,
    pub statusbar: Rect,
}

impl Layouts {
    pub fn new(area: Rect) -> Self {
        let chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(area);
        Self {
            header: chunks[0],
            body: chunks[1],
            statusbar: chunks[2],
        }
    }
}

pub enum TuiEvent {
    Key(KeyEvent),
    ChatInput(String),
    CompactSession,
    Mouse(MouseEvent),
    Resize(u16, u16),
    Interrupt,
    /// Inject a message into the next tool call without interrupting.
    /// (Mirrors `/steer` from the Hermes CLI.)
    Steer(String),
    /// Signal the event loop to dequeue messages from ChatPanel's input queue
    /// and submit them as ChatInput events. Replaces the old direct TuiEvent::ChatInput
    /// sends from ChatPanel for auto-drain.
    QueueDrain,
    /// Signal shutdown — sent when the keyboard reader task exits.
    Quit,
}

pub async fn run_tui(session_name: Option<&str>, agent_name: Option<&str>) -> Result<()> {
    let app = App::new()?;
    let agent_name_owned = agent_name.map(|s| s.to_string());

    // Raw mode + alternate screen — needed for splash loop to draw
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(EnableMouseCapture)?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Wrap app in Arc<TokioMutex<>> — needed because init task and splash loop
    // both need mutable access to app.panels during splash phase.
    let arc_app = Arc::new(tokio::sync::Mutex::new(app));



    // --- SPLASH LOOP ---
    // Use oneshot channel so init and draw don't contend for the app mutex.
    let (init_done_tx, init_done_rx) =
        tokio::sync::oneshot::channel::<Result<(), anyhow::Error>>();

    // Minimum splash display duration — gives time for agent init.
    let splash_start = std::time::Instant::now();
    let min_display_ms: u64 = 1500;

    {
        let init_arc_app = Arc::clone(&arc_app);
        let tx = init_done_tx;
        tokio::spawn(async move {
            let mut a = init_arc_app.lock().await;
            let result = a.init_agent(agent_name_owned.as_deref()).await;
            if result.is_ok() {
                // Register TUI adapters on the agent's internal HookEngine.
                // AgentBuilder creates HookEngine internally (from config), then we register TUI adapters via Agent::hooks() and share it with subagents through SubagentSpawner::new().with_hooks().
                let ts = a.shared_state.lock().turn_state.clone();
                // Clone the agent arc before the lock is dropped, to avoid
                // holding a borrow from the lock guard across the .await below.
                let agent_opt = {
                    let guard = a.shared_state.lock();
                    guard.agent.clone()
                };
                if let Some(agent) = agent_opt {
                    let hooks = Arc::clone(agent.lock().await.hooks());
                    hooks.register_tool(Box::new(
                        oben_agent::TuiSubagentAdapter::new(std::sync::Arc::new(
                            crate::shared::SubagentCallback::new(Arc::clone(&a.shared_state))
                        ))
                    ));
                    hooks.register_streaming(Box::new(
                        oben_agent::TuiStreamingAdapter::new(Arc::clone(&ts))
                    ));
                    hooks.register_tool(Box::new(
                        oben_agent::TuiToolLifecycleAdapter::new(Arc::clone(&ts))
                    ));
                    hooks.register_agent_loop(Box::new(
                        oben_agent::TuiAgentLoopAdapter::new(ts.clone())
                    ));
                    hooks.register_turn(Box::new(
                        oben_agent::TuiTurnLifecycleAdapter::new(ts)
                    ));
                }
            }
            let _ = tx.send(result);
        });
    }

    loop {
        // Draw splash screen
        {
            let a = arc_app.lock().await;
            if let Some(splash) = a.panels.get(&PanelId::Splash) {
                terminal
                    .draw(|frame| {
                        splash.draw(frame, frame.area());
                    })
                    .ok();
            }
        }
        terminal.backend_mut().flush().ok();

        // Break after minimum display or any key press
        let elapsed_ms: u128 = std::time::Instant::now()
            .duration_since(splash_start)
            .as_millis();
        if elapsed_ms >= min_display_ms as u128 {
            break;
        }

        if crossterm::event::poll(Duration::from_millis(32)).unwrap_or(false) {
            // Any key press skips splash early
            if crossterm::event::read().is_ok() {
                break;
            }
        }
    }

    let init_done = tokio::time::timeout(
        Duration::from_secs(3), init_done_rx,
    ).await.is_ok();

    arc_app
        .lock()
        .await
        .init_active_panel(session_name)
        .await
        .ok();

    // --- MAIN EVENT LOOP SETUP ---
    let agent = if init_done {
        arc_app.lock().await.shared_state.lock().agent.clone()
    } else {
        None
    };

    if agent.is_none() {
        // Init failed — show error splash until user quits
        arc_app
            .lock()
            .await
            .panels
            .get_mut(&PanelId::Splash)
            .and_then(|p| p.downcast_mut::<panels::splash::SplashPanel>())
            .map(|s| s.set_error("Agent failed to initialize".to_string()));
        // Wait forever for user to quit
        loop {
            let a = arc_app.lock().await;
            if let Some(splash) = a.panels.get(&PanelId::Splash) {
                terminal
                    .draw(|frame| splash.draw(frame, frame.area()))
                    .ok();
            }
            drop(a);
            terminal.backend_mut().flush().ok();
            if crossterm::event::poll(Duration::from_millis(32)).unwrap_or(false) {
                let ev = crossterm::event::read()?;
                if let crossterm::event::Event::Key(k) = ev {
                    if k.code == crossterm::event::KeyCode::Char('c')
                        && k.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        break;
                    }
                }
            }
        }
        drop(terminal);
        io::stdout().execute(LeaveAlternateScreen)?;
        io::stdout().execute(DisableMouseCapture)?;
        disable_raw_mode()?;
        return Ok(());
    }

    let agent = agent.unwrap();

    // --- EVENT LOOP CHANNELS ---
    let (chat_tx, chat_rx) = unbounded_channel::<String>();
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel::<TurnCompletion>();
    let (command_tx, mut command_rx) = unbounded_channel::<TuiCommand>();

    // Initial draw — needs_redraw starts as true from App::new()
    {
        let mut app = arc_app.lock().await;
        if app.needs_redraw {
            terminal.draw(|frame| draw_ui(frame, &mut app))?;
        }
    }

    // Create TuiCoordinator — the bridge between Agent::run() and the TUI event loop.
    let tui_coordinator = TuiCoordinator::new(
        chat_rx,
        command_tx,
        done_tx,
    );
    let coordinator_handle = tokio::spawn(async move {
        // Agent::run() consumes the coordinator and drives the conversation loop.
        let _ = Agent::run(agent.clone(), tui_coordinator).await;
    });

    // --- SPAWN KEYBOARD READER TASK ---
    let (event_tx, mut event_rx) = unbounded_channel();
    let _event_tx_clone = event_tx.clone(); // For event loop (reader moves original)
    let input_tx = event_tx.clone(); // For command events from app.handle_key()

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    let event_tx_for_reader = event_tx.clone();

    let reader_handle = tokio::task::spawn_blocking(move || {
        while running_clone.load(Ordering::SeqCst) {
            match crossterm::event::poll(Duration::from_millis(16)) {
                Ok(true) => match crossterm::event::read() {
                    Ok(crossterm::event::Event::Key(key)) => {
                        if key.kind == KeyEventKind::Press || key.kind == KeyEventKind::Repeat {
                            if key.code == KeyCode::Char('c')
                                && key.modifiers.contains(KeyModifiers::CONTROL)
                            {
                                tracing::info!("[reader_task] Ctrl+C detected, sending Quit");
                                let _ = event_tx.send(TuiEvent::Quit);
                            } else {
                                tracing::debug!("[reader_task] key event: {:?}", key.code);
                                let _ = event_tx.send(TuiEvent::Key(key));
                            }
                        }
                    }
                    Ok(crossterm::event::Event::Mouse(mouse)) => {
                        let _ = event_tx.send(TuiEvent::Mouse(mouse));
                    }
                    Ok(crossterm::event::Event::Resize(w, h)) => {
                        let _ = event_tx.send(TuiEvent::Resize(w, h));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("[reader_task] crossterm read error: {e}");
                        std::thread::sleep(Duration::from_millis(100));
                    }
                },
                Ok(false) => {}
                Err(e) => {
                    tracing::error!("[reader_task] crossterm poll error: {e}");
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
        // Reader task exiting — signal event loop to quit.
        tracing::info!("[reader_task] exiting, sending Quit event");
        let _ = event_tx_for_reader.send(TuiEvent::Quit);
    });

    // Wire input_tx for command events (/compact, interrupt, steer) and auto-drain.
    arc_app.lock().await.input_tx = Some(input_tx.clone());
    if let Some(chat) = arc_app.lock().await.get_chat_mut() {
        chat.set_input_sender(input_tx.clone());
    }

    // Periodic redraw timer — ensures UI updates during streaming when no
    // keyboard/mouse events fire. Without this, `TurnState.streaming_text`
    // gets populated by transport callbacks but never reaches the screen.
    // Use 50ms interval for ~20fps — fast enough for smooth streaming,
    // avoids fighting the terminal with 16ms redraws.
    let mut redraw_interval = tokio::time::interval(Duration::from_millis(50));
    redraw_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    // --- MAIN EVENT LOOP (pure UI: keyboard/mouse -> display) ---
    loop {
        // Use `select!` with a default_delay arm so the loop yields to `terminal.draw()`
        // even when no channel fires. All arms set `app.needs_redraw = true`; the
        // default arm checks the flag and renders if true, then re-enters select!.
        tokio::select! {
            _ = redraw_interval.tick() => {
                let mut app = arc_app.lock().await;
                app.needs_redraw = true;
            }
            Some(cmd) = command_rx.recv() => {
                let mut app = arc_app.lock().await;
                match cmd {
                    TuiCommand::StartTurn { input, session_name: _ } => {
                        tracing::info!("[event_loop] TuiCommand::StartTurn, input.len={}, active_panel={:?}", input.len(), app.active_panel);
                        if let Some(chat) = app.get_chat_mut() {
                            let entries_before = chat.message_state.message_entries.lock().unwrap().len();
                            tracing::debug!("[event_loop] Clearing entries before: {}", entries_before);
                            chat.streaming = true;
                            chat.clear_display();
                            let entries_after_clear = chat.message_state.message_entries.lock().unwrap().len();
                            tracing::debug!("[event_loop] After clear_display: {}", entries_after_clear);
                            chat.append_user_message(&input);
                            let count = chat.message_state.message_entries.lock().unwrap().len();
                            tracing::info!("[event_loop] After StartTurn, message_entries count={}", count);
                        } else {
                            tracing::error!("[event_loop] StartTurn: no chat panel found!");
                        }
                        app.status = "Busy".into();
                        app.needs_redraw = true;
                    }
                    TuiCommand::AppendInputHistory { input } => {
                        tracing::info!("[event_loop] TuiCommand::AppendInputHistory");
                        app.input_history.append(&input);
                        app.needs_redraw = true;
                    }
                }
            }
            completion = done_rx.recv() => {
                if let Some(completion) = completion {
                    tracing::debug!("[event_loop] done_rx: channel received");
                    tracing::info!(
                        "[event_loop] done_rx: success={} session_name={}",
                        completion.success,
                        completion.session_name.as_deref().unwrap_or("<none>"),
                    );
                    let mut app = arc_app.lock().await;
                    // Refresh session_id from agent's CWM — sessions are created
                    // lazily on first turn, so the init-time capture may be None.
                    app.shared_state.lock().sync_session_id().await;
                    
                    // Reload messages from session to overwrite streaming entries
                    let messages = {
                        let ss = app.shared_state.lock();
                        if let Some(ref agent) = ss.agent {
                            let sid = ss.session_id.clone();
                            if let Some(ref sid_val) = sid {
                                let agent_guard = agent.lock().await;
                                let sm = agent_guard.session_manager();
                                let sm_guard = sm.lock().await;
                                sm_guard.session(sid_val)
                                    .map(|s| s.messages.clone())
                                    .unwrap_or_default()
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        }
                    };
                    
                    if let Some(chat) = app.get_chat_mut() {
                        chat.streaming = false;
                        chat.message_count = completion.message_count;
                        chat.update_from_messages(&messages, completion.session_name.clone());
                        // Also set session_name so ChatPanel state matches TurnCompletion.
                        if let Some(ref name) = completion.session_name {
                            chat.session_name = Some(name.clone());
                        }
                    }
                    if completion.success {
                        app.status = "Ready".into();
                    } else {
                        app.status = format!("Error: {}", completion.status);
                        tracing::error!("[event_loop] Turn error: {}", completion.status);
                    }
                    app.needs_redraw = true;
                }
            }
            event = event_rx.recv() => {
                match event {
                    Some(TuiEvent::Key(key)) => {
                        // Compute key action (needs mutable app)
                        let action = {
                            let app = &mut arc_app.lock().await;
                            app.handle_key(key).await
                        };
                        match action {
                            panels::KeyAction::ChatInput(text) => {
                                tracing::debug!("[event_loop] KeyAction::ChatInput sending to chat_tx, len={}", text.len());
                                let _ = chat_tx.send(text);
                                {
                                    let mut app = arc_app.lock().await;
                                    app.needs_redraw = true;
                                }
                            }
                            panels::KeyAction::Quit => {
                                let _ = _event_tx_clone.send(TuiEvent::Quit);
                            }
                            panels::KeyAction::Steer(text) => {
                                let _ = input_tx.send(TuiEvent::Steer(text));
                            }
                            panels::KeyAction::Interrupt => {
                                let _ = _event_tx_clone.send(TuiEvent::Interrupt);
                            }
                            panels::KeyAction::None => {
                                // No action
                            }
                            _ => {}
                        }
                    }
                    Some(TuiEvent::Mouse(mouse_event)) => {
                        let (term_w, term_h) = size().unwrap_or((80, 24));
                        let body_area = Rect::new(0, 1, term_w, term_h.saturating_sub(2));
                        {
                            let mut app = arc_app.lock().await;
                            if let Some(expiry) = app.toast_expires_at {
                                if std::time::Instant::now() >= expiry {
                                    app.toast_engine.hide_toast();
                                    app.toast_expires_at = None;
                                    app.needs_redraw = true;
                                }
                            }
                        }
                        let click_on_tabs = matches!(
                            mouse_event.kind,
                            MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        ) && mouse_event.row == 0;
                        if click_on_tabs {
                            let tab_names = ["Chat", "Sessions"];
                            let tab_widths: Vec<usize> = tab_names.iter().map(|n| n.len() + 2).collect();
                            let mut consumed = 0usize;
                            for (i, &pw) in tab_widths.iter().enumerate() {
                                if mouse_event.column >= consumed as u16
                                    && mouse_event.column < (consumed + pw) as u16
                                {
                                    let target = match i {
                                        0 => PanelId::Chat,
                                        1 => PanelId::Sessions,
                                        _ => break,
                                    };
                                    let mut app = arc_app.lock().await;
                                    if app.active_panel != target {
                                        app.activate_panel(target).await;
                                    }
                                    app.active_panel = target;
                                    app.needs_redraw = true;
                                    break;
                                }
                                consumed += pw + 1;
                            }
                            continue;
                        }
                        match mouse_event.kind {
                            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                                let mut app = arc_app.lock().await;
                                let panel_id = app.active_panel.clone();
                                if let Some(panel) = app.panels.get_mut(&panel_id) {
                                    let _ = panel.handle_mouse(body_area, &mouse_event);
                                    app.needs_redraw = true;
                                }
                            }
                            _ => {
                                let mut app = arc_app.lock().await;
                                let panel_id = app.active_panel.clone();
                                if let Some(panel) = app.panels.get_mut(&panel_id) {
                                    if let Some(text) = panel.handle_mouse(body_area, &mouse_event) {
                                        tracing::debug!(
                                            "[lib] handle_mouse returned text, about to show toast"
                                        );
                                        let lines = text.lines().count();
                                        let msg = if lines == 1 {
                                            "Copied selection.".to_string()
                                        } else {
                                            format!("Copied {} lines.", lines)
                                        };
                                        app.show_toast(msg, ratatui_toaster::ToastType::Success);
                                        app.needs_redraw = true;
                                    } else {
                                        tracing::debug!(
                                            "[lib] handle_mouse returned None for event={:?}",
                                            mouse_event.kind
                                        );
                                    }
                                    if let Some(chat) = app.get_chat() {
                                        let selecting = chat.message_state.selection_start.is_some()
                                            && chat.message_state.selection_end.is_some();
                                        if selecting {
                                            app.needs_redraw = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(TuiEvent::ChatInput(input)) => {
                        tracing::debug!(
                            "[event_loop] ChatInput event, input.len()={}",
                            input.len()
                        );
                        let _ = chat_tx.send(input);
                        {
                            let mut app = arc_app.lock().await;
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::QueueDrain) => {
                        let messages: Vec<String> = {
                            let mut app = arc_app.lock().await;
                            std::mem::take(
                                &mut app.panels
                                    .get_mut(&PanelId::Chat)
                                    .unwrap()
                                    .downcast_mut::<crate::panels::chat::ChatPanel>()
                                    .unwrap()
                                    .input
                                    .input_queue,
                            )
                        };
                        for msg in messages {
                            tracing::info!("[event_loop] QueueDrain: draining msg: {msg}");
                            let _ = chat_tx.send(msg);
                        }
                        {
                            let mut app = arc_app.lock().await;
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::Resize(_w, _h)) => {
                        {
                            let mut app = arc_app.lock().await;
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::Interrupt) => {
                        let is = {
                            let a_guard = arc_app.lock().await;
                            let agent = a_guard.shared_state.lock().agent.clone();
                            match agent {
                                Some(agent) => {
                                    Some(Arc::clone(&agent.blocking_lock().get_interrupt_state()))
                                }
                                None => None,
                            }
                        };
                        if let Some(is) = is {
                            is.request_interrupt(Some("interrupted".to_string()));
                            is.reset_for_turn();
                        }
                        {
                            let mut app = arc_app.lock().await;
                            app.status = "Interrupted".into();
                            if let Some(chat) = app.get_chat_mut() {
                                chat.input.text.clear();
                                chat.input.cursor = 0;
                            }
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::Steer(text)) => {
                        let agent_opt = {
                            let a = arc_app.lock().await;
                            let ss = a.shared_state.lock();
                            ss.agent.clone()
                        };
                        if let Some(agent) = agent_opt {
                            let accepted = agent.lock().await.steer(&text);
                            if accepted {
                                tracing::info!("[steer] queued: %text");
                                let mut app = arc_app.lock().await;
                                app.show_toast(
                                    "Steer queued - arrives after next tool call",
                                    ToastType::Info,
                                );
                            } else {
                                let mut app = arc_app.lock().await;
                                app.show_toast(
                                    "Steer rejected (empty payload).",
                                    ToastType::Warning,
                                );
                            }
                        } else {
                            if let Some(chat) = arc_app.lock().await.get_chat_mut() {
                                chat.input.enqueue_msg(text.clone());
                                tracing::info!(
                                    "[steer] no agent - queued message instead: %text"
                                );
                                let mut app = arc_app.lock().await;
                                app.show_toast(
                                    "Agent not running - queued for next turn",
                                    ToastType::Info,
                                );
                            }
                        }
                        if let Some(chat) = arc_app.lock().await.get_chat_mut() {
                            if chat.input.pending_steer_count == 0 {
                                chat.append_user_message(&text);
                                let count = chat.message_state.message_entries.lock().unwrap().len();
                                tracing::info!("[event_loop] After steer append_user_message, message_entries count={}", count);
                                chat.input.pending_steer_count += 1;
                            }
                        }
                        {
                            let mut app = arc_app.lock().await;
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::CompactSession) => {
                        let agent_arc = {
                            let a = arc_app.lock().await;
                            let ss = a.shared_state.lock();
                            ss.agent.clone()
                        };
                        if let Some(agent_arc) = agent_arc {
                            let outcome = agent_arc.lock().await.compact_session().await;
                            let sid = {
                                let a = arc_app.lock().await;
                                let ss = a.shared_state.lock();
                                ss.session_id.clone()
                            };
                            let messages = {
                                let a = arc_app.lock().await;
                                let ss = a.shared_state.lock();
                                let agent_opt = ss.agent.clone();
                                match agent_opt {
                                    Some(agent_arc) => {
                                        let guard = agent_arc.lock().await;
                                        let sm_arc = guard.session_manager();
                                        let sm = sm_arc.lock().await;
                                        guard
                                            .context_window_manager()
                                            .session_id()
                                            .and_then(|s| sm.session(&s))
                                            .map(|s| s.messages.clone())
                                            .unwrap_or_default()
                                    }
                                    None => Vec::new(),
                                }
                            };
                            if let Some(chat) = arc_app.lock().await.get_chat_mut() {
                                chat.update_from_messages(&messages, sid);
                            }
                            match outcome {
                                oben_agent::compact::CompactOutcome::Compressed { .. } => {
                                    let mut app = arc_app.lock().await;
                                    app.show_toast(
                                        "Session context compressed successfully.",
                                        ToastType::Success,
                                    );
                                }
                                oben_agent::compact::CompactOutcome::Ineffective { .. } => {
                                    let mut app = arc_app.lock().await;
                                    app.show_toast(
                                        "Compaction skipped: no savings (all messages protected in head/tail zones).",
                                        ToastType::Info,
                                    );
                                }
                                oben_agent::compact::CompactOutcome::AlreadyCompact => {
                                    let mut app = arc_app.lock().await;
                                    app.show_toast(
                                        "Session context already within budget.",
                                        ToastType::Info,
                                    );
                                }
                                oben_agent::compact::CompactOutcome::NoMiddleMessages { .. } => {
                                    let mut app = arc_app.lock().await;
                                    app.show_toast(
                                        "Compaction skipped: all messages in protected head/tail zones.",
                                        ToastType::Info,
                                    );
                                }
                            }
                        } else {
                            let mut app = arc_app.lock().await;
                            app.show_toast(
                                "Cannot compact: agent not initialized",
                                ToastType::Warning,
                            );
                        }
                        {
                            let mut app = arc_app.lock().await;
                            app.needs_redraw = true;
                        }
                    }
                    Some(TuiEvent::Quit) => {
                        tracing::info!("[event_loop] Quit event received, signaling shutdown");
                        running.store(false, Ordering::SeqCst);
                        tracing::info!("[event_loop] Dropping chat_tx to shut down coordinator");
                        drop(chat_tx);
                        break;
                    }
                    None => break,
                }
            }
            _default_delay = async { tokio::time::sleep(std::time::Duration::from_millis(16)).await } => {
                // Yield periodically so the `terminal.draw()` below executes even during idle.
            }
        }

        if { let app = arc_app.lock().await; app.needs_redraw } {
            let mut app = arc_app.lock().await;
            let _ = terminal.draw(|frame| draw_ui(frame, &mut app));
        }
    }

    let _ = coordinator_handle.await;
    let _ = reader_handle.await;
    drop(terminal);
    io::stdout().execute(LeaveAlternateScreen)?;
    io::stdout().execute(DisableMouseCapture)?;
    disable_raw_mode()?;
    info!("TUI exited normally.");
    Ok(())
}

fn draw_ui(frame: &mut Frame, app: &mut App) {
    app.needs_redraw = false;
    let area = frame.area();
    let layout = Layouts::new(area);

    // update ChatPanel from turn_state (polling pattern via Arc<Mutex<TurnState>>)
    let (entry_count, entry_roles, is_streaming) = {
        if let Some(chat) = app.get_chat_mut() {
            let ec = chat.message_state.message_entries.lock().unwrap().len();
            let er: Vec<String> = chat.message_state.message_entries.lock().unwrap()
                .iter().map(|e| format!("{:?}", e.role)).collect();
            chat.update_from_turn_state();
            let s = chat.streaming;
            (ec, er, s)
        } else {
            (0, Vec::new(), false)
        }
    };
    if entry_count > 0 || is_streaming {
        tracing::info!(
            "[draw_ui] drawing chat: entries={} roles={:?} streaming={}",
            entry_count, entry_roles, is_streaming
        );
    } else {
        tracing::trace!(
            "[draw_ui] drawing chat: entries={} roles={:?} streaming={}",
            entry_count, entry_roles, is_streaming
        );
    }

    let panel_names: [&str; 2] = ["Chat", "Sessions"];
    let panel_index = match app.active_panel {
        PanelId::Chat => 0,
        PanelId::Sessions => 1,
        _ => 0,
    };
    let tabs = Tabs::new(panel_names)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(Style::default().fg(Color::Cyan).bold())
        .divider(" ")
        .select(panel_index)
        .block(Block::default().style(Style::default().bg(Color::Blue)));
    frame.render_widget(tabs, layout.header);

    if let Some(panel) = app.panels.get(&app.active_panel) {
        panel.draw(frame, layout.body);
    }

    // Toast area: top-right of the body region, 2 lines tall.
    let toast_height = 2u16;
    let toast_width: u16 = 50;
    let toast_rect = Rect::new(
        layout.body.x + layout.body.width.saturating_sub(toast_width),
        layout.body.y,
        toast_width,
        toast_height,
    );
    app.toast_engine.set_area(toast_rect);

    // Render toast above the status bar if one is active.
    // Auto-expire after 3 seconds (ratatui-toaster has no built-in timeout).
    let has_active_toast = if let Some(expiry) = app.toast_expires_at {
        if std::time::Instant::now() >= expiry {
            app.toast_engine.hide_toast();
            app.toast_expires_at = None;
            false
        } else {
            app.toast_engine.has_toast()
        }
    } else {
        app.toast_engine.has_toast()
    };
    if has_active_toast {
        frame.render_widget(&app.toast_engine, toast_rect);
    }

    // Derive session info from stored ChatPanel fields — no Agent locking.
    let (session_name, msg_count) = match app.active_panel {
        PanelId::Sessions => match app.get_sessions() {
            Some(sessions) => (
                sessions.get_session_name().unwrap_or_default(),
                sessions.get_message_count().unwrap_or(0),
            ),
            None => (String::new(), 0),
        },
        _ => match app.get_chat() {
            Some(chat) => {
                if let Some(ref sid) = chat.session_name {
                    (sid.clone(), chat.message_count)
                } else {
                    // Fallback: pull from shared state when ChatPanel
                    // session_name/message_count are still at their defaults.
                    (
                        app.shared_state.lock()
                            .session_id
                            .clone()
                            .unwrap_or_default(),
                        chat.message_count,
                    )
                }
            }
            None => (String::new(), 0),
        },
    };

    let session_text = match &session_name {
        s if !s.is_empty() => format!(" Session: {} ({} msgs)", s, msg_count),
        _ => " No session".to_string(),
    };
    let mode_text = match app.status.as_str() {
        s if s.starts_with("Error") => "Error",
        "Busy" => "Streaming",
        _ => "Ready",
    };
    let model_text = {
        let kind_str = app.config.model.kind.as_str();
        let model_name = app
            .config
            .model
            .model
            .splitn(2, '/')
            .nth(1)
            .unwrap_or(&app.config.model.model);
        if model_name.trim().is_empty() {
            format!("{} model", kind_str)
        } else {
            format!("{} ({})", model_name, kind_str)
        }
    };
    let model_label = if model_text.is_empty() { String::new() } else { format!("  {}", model_text) };
    let status_lines: Vec<Line> =
        vec![Line::from(format!(" [{}]{}{}", mode_text, model_label, session_text))];
    let status_para = Paragraph::new(status_lines);
    let status_area = Rect::new(
        layout.statusbar.x,
        layout.statusbar.y,
        layout.statusbar.width,
        layout.statusbar.height,
    );
    frame.render_widget(status_para, status_area);
}
