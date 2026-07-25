//! Agent — shared between CLI and TUI.
//!
//! **Resource Ownership:**
//! Agent owns all runtime resources — Transport, Tools, Skills, SystemPrompt,
//! ContextWindowManager, and SessionManager. It delegates turn execution to
//! `coordinator::execute_turn_full`.
//!
//! **Responsibilities:**
//! - Own Transport, Tools, Skills, SystemPrompt, ContextWindowManager (resource management)
//! - Own SessionManager (session lifecycle, persistence)
//! - Delegate turn cycle to shared `execute_turn_full`
//! - Manage session switching
//! - Interactive chat loop via `run(coordinator)` — passes resources to a `ConversationCoordinator`
//! - Cross-thread interrupt and steer (Tier 1)
//! - Fallback model chain (Tier 2)
//! - Rich callback system (Tier 2)
//! - Activity tracking (Tier 2)
//! - Concurrent tool dispatch (Tier 2)
//! - Nudge / background review (Tier 2)

use anyhow::Result;
use crate::agent_builder::AgentBuilder;
use crate::fallback::FallbackChain;
use crate::hooks::HookEngine;
use std::sync::Arc;
use tokio::sync::Mutex;

pub fn generate_session_name() -> String {
    let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let r = rand::random::<u64>() % 1_000_000;
    format!("{}-{:06}", ts, r)
}
use oben_models::{CallMode, Message};
use oben_sessions::{memory_provider::MemoryManager, SessionManager, SessionStore};


use crate::coordinator::execute_turn_full;
use crate::coordinator::{ConversationCoordinator, ConversationResult};
use crate::interrupt::InterruptState;
use crate::system_prompt::build_volatile_block;

/// Dummy coordinator for testing.
pub struct DummyCoordinator;

impl Default for DummyCoordinator {
    fn default() -> Self {
        Self
    }
}

#[::async_trait::async_trait]
impl ConversationCoordinator for DummyCoordinator {
    fn on_loop_start(&mut self) {}
    fn on_turn_complete(&mut self, _response: &str, _msg_count: usize, _turn_count: u32, _success: bool) -> bool { false }
    fn on_loop_end(&mut self, _outcome: &ConversationResult) {}
    async fn next_turn(&mut self) -> Option<String> { None }
}

/// An interactive agent — owns all resources, delegates turns to shared execute_turn_full.
pub struct Agent {
    pub(crate) transport: Arc<dyn oben_models::providers::TransportProvider + Send + Sync>,
    pub(crate) tools: Arc<oben_tools::ToolRegistry>,
    pub(crate) context_window_manager: Box<dyn crate::context::ContextWindowManager>,
    pub(crate) call_mode: Option<oben_models::CallMode>,
    pub(crate) session_manager: Arc<Mutex<SessionStore>>,
    pub(crate) interrupt_state: Arc<InterruptState>,
    pub(crate) config: oben_config::AppConfig,
    pub(crate) fallback_chain: Option<crate::fallback::FallbackChain>,
    pub(crate) system_prompt: String,
    pub(crate) hooks: Arc<super::hooks::HookEngine>,
    pub(crate) memory_manager: Arc<std::sync::Mutex<MemoryManager>>,
}

impl Agent {
    pub async fn new(
        config: oben_config::AppConfig,
        system_prompt: String,
        tools: Arc<oben_tools::ToolRegistry>,
    ) -> Result<Self> {
        AgentBuilder::new()
            .with_config(config)
            .with_system_prompt(system_prompt)
            .with_tools(tools)
            .build()
            .await
    }

    /// Create a minimal Agent for testing. Does NOT call real transport or LLMs.
    pub async fn new_for_test() -> Self {
        use oben_models::providers::TransportProvider;

        let session_manager = Arc::new(Mutex::new(SessionStore::new(
            oben_models::SessionStoreKind::Memory,
        )
        .unwrap()));
        let tools = Arc::new(oben_tools::ToolRegistry::new());

        let transport = Arc::new(
            oben_transport::Transport::from_config_with_tools_via_registry(
                &oben_config::AppConfig::default().model,
                "",
                &[],
            )
        ) as Arc<dyn TransportProvider + Send + Sync>;

        Self {
            transport,
            tools,
            context_window_manager: Box::new(crate::compact_context::BuiltinContextWindowManager::new()),
            call_mode: None,
            session_manager,
            interrupt_state: Arc::new(InterruptState::new()),
            config: oben_config::AppConfig::default(),
            fallback_chain: None,
            system_prompt: String::new(),
            hooks: Arc::new(super::hooks::HookEngine::new()),
            memory_manager: Arc::new(std::sync::Mutex::new(MemoryManager::new())),
        }
    }

    /// Run a conversation loop driven by the given coordinator.
    ///
    /// The agent provides all runtime resources. The mutex is released during
    /// `coordinator.next_turn().await` to prevent deadlocks with other tasks
    /// accessing agent-derived state from the TUI event loop.
    pub async fn run(
        agent: Arc<Mutex<Self>>,
        mut coordinator: impl ConversationCoordinator,
    ) -> Result<ConversationResult> {
        let my = agent.lock().await;
        let max_iterations = my.config.max_iterations;
        let hooks = Arc::clone(&my.hooks);
        drop(my);

        hooks.emit_loop_start();
        coordinator.on_loop_start();

        let mut turn_count = 0usize;

        loop {
            // Release the agent lock during next_turn().await so the TUI event
            // loop can access agent-derived state (turn_state, session manager)
            // without deadlocking.
            let user_input = coordinator.next_turn().await;
            let user_input = match user_input {
                Some(input) => input,
                None => {
                    hooks.emit_loop_end("input_end");
                    return Ok(ConversationResult::Exit);
                }
            };

            {
                let mut me = agent.lock().await;
                me.hooks.emit_pre_turn();
                let response = me.turn(&user_input, true, None).await;
                let msg_count = me.session_manager.lock().await
                    .session(&me.context_window_manager.session_id().unwrap_or_default())
                    .map(|s| s.messages.len()).unwrap_or(0);
                let turn_count_v = me.session_manager.lock().await
                    .session(&me.context_window_manager.session_id().unwrap_or_default())
                    .map(|s| s.metadata.turn_count).unwrap_or(turn_count as u32);
                let turn_success = response.is_ok();
                // When error occurs, extract the error message to pass to coordinator
                let turn_text: String = if turn_success {
                    response.as_deref().unwrap_or("").to_string()
                } else {
                    response.as_ref().unwrap_err().to_string()
                };
                let is_error_content = turn_text.starts_with("Error:");
                let turn_success_final = turn_success && !is_error_content;
                tracing::debug!("[agent] calling coordinator.on_turn_complete: text_len={}, success={}", turn_text.len(), turn_success_final);
                let cont = coordinator.on_turn_complete(turn_text.as_str(), msg_count, turn_count_v, turn_success_final);
                tracing::debug!("[agent] coordinator.on_turn_complete returned: cont={}", cont);
                if !cont {
                    me.hooks.emit_loop_end("user_exit");
                    return Ok(ConversationResult::Exit);
                }
                if turn_success_final {
                    me.hooks.emit_turn_complete(turn_text.as_str(), turn_count_v, msg_count);
                } else {
                    let error = if response.is_err() {
                        response.unwrap_err()
                    } else {
                        anyhow::anyhow!("{}", turn_text)
                    };
                    me.hooks.emit_turn_error(&error, turn_count_v);
                }
                turn_count += 1;
                if let Some(max) = max_iterations {
                    if turn_count >= max {
                        me.hooks.emit_loop_end("max_turns_reached");
                        return Ok(ConversationResult::BudgetExhausted);
                    }
                }
            }
        }
    }

    pub(crate) async fn eager_load_active_session(&mut self) {
        let sid = self.context_window_manager.session_id();
        if let Some(sid) = sid {
            let _ = self.session_manager.lock().await.switch_session(&sid);
        }
    }

    /// Access the shared HookEngine so CliCoordinator can reuse it
    /// instead of creating its own duplicate instance.
    pub fn hooks(&self) -> &Arc<HookEngine> {
        &self.hooks
    }

    /// Access the AppConfig for constructing ConversationConfig and other coordinators.
    pub fn config(&self) -> &oben_config::AppConfig {
        &self.config
    }

    /// Access the session manager for listing/saving outside the turn cycle.
    pub fn session_manager(&self) -> Arc<Mutex<SessionStore>> {
        Arc::clone(&self.session_manager)
    }

    /// Mutably access the session manager (for admin ops: load, delete, new).
    pub fn session_manager_mut(&mut self) -> Arc<Mutex<SessionStore>> {
        Arc::clone(&self.session_manager)
    }

    /// Access the CWM for external turn loop coordination.
    pub fn context_window_manager(&self) -> &dyn crate::context::ContextWindowManager {
        self.context_window_manager.as_ref()
    }

    /// Mutably access the CWM for external turn loop coordination.
    pub fn context_window_manager_mut(&mut self) -> &mut dyn crate::context::ContextWindowManager {
        self.context_window_manager.as_mut()
    }

    /// Access the transport for external turn loop coordination.
    pub fn transport(&self) -> Arc<dyn oben_models::providers::TransportProvider + Send + Sync> {
        Arc::clone(&self.transport)
    }

    /// Access the tools registry for external turn loop coordination.
    pub fn tools_arc(&self) -> Arc<oben_tools::ToolRegistry> {
        Arc::clone(&self.tools)
    }

    // ── Tier 1: Interrupt & Steer ────────────────────────────────────────

    /// Interrupt the agent's current tool-calling loop.
    pub fn interrupt(&self, message: Option<&str>) {
        let msg = message.map(|s| s.to_string());
        self.interrupt_state.request_interrupt(msg);
    }

    /// Clear any pending interrupt request.
    pub fn clear_interrupt(&self) {
        self.interrupt_state.clear_interrupt();
    }

    /// Inject a user note into the next tool result without interrupting.
    pub fn steer(&self, text: &str) -> bool {
        self.interrupt_state.steer(text)
    }

    /// Get a shared reference to the interrupt state.
    ///
    /// This allows external code (e.g. TUI event loop) to call
    /// `request_interrupt()` without acquiring the tokio::sync::Mutex
    /// that the spawn task holds across `turn()`.  Prevents deadlock.
    pub fn get_interrupt_state(&self) -> Arc<crate::interrupt::InterruptState> {
        Arc::clone(&self.interrupt_state)
    }

    // ── Tier 2: Fallback Models ──────────────────────────────────────────

    /// Set the fallback model chain.
    pub fn set_fallback_chain(&mut self, chain: Vec<crate::fallback::FallbackConfig>) {
        self.fallback_chain = Some(FallbackChain::new(chain));
    }

    /// Check if a fallback was activated.
    pub fn fallback_activated(&self) -> bool {
        self.fallback_chain.as_ref().map(|fc| fc.is_activated()).unwrap_or(false)
    }

    /// Get the active fallback config (if any).
    pub fn active_fallback(&self) -> Option<&crate::fallback::FallbackConfig> {
        self.fallback_chain.as_ref().and_then(|fc| fc.active_fallback())
    }

    /// Get fallback chain status for diagnostics.
    pub fn fallback_status(&self) -> String {
        if self.fallback_activated() {
            if let Some(fb) = self.active_fallback() {
                format!("Fallback active: {}/{}", fb.provider, fb.model)
            } else {
                "Fallback active (unknown)".to_string()
            }
        } else {
            "Primary model active".to_string()
        }
    }

    // ── Tier 2: Callbacks ────────────────────────────────────────────────

    /// Set the agent hooks.
    pub fn set_hooks(&mut self, hooks: Arc<super::hooks::HookEngine>) {
        self.hooks = hooks;
    }


    // ── Tier 2: System Prompt ──────────────────────────────────────────────

    /// Update the cached system prompt after building a new one.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    /// Get the cached system prompt, if available.
    pub fn get_system_prompt(&self) -> Option<&str> {
        if self.system_prompt.is_empty() {
            None
        } else {
            Some(&self.system_prompt)
        }
    }

    /// Check if we have a cached prompt.
    pub fn has_system_prompt(&self) -> bool {
        !self.system_prompt.is_empty()
    }

    // ── Tier 2: Activity Tracking ────────────────────────────────────────

    /// Get activity summary for diagnostics.
    pub fn get_activity_summary(&self) -> crate::interrupt::ActivitySummary {
        self.interrupt_state.get_activity_summary(None, 0, 0)
    }

    // ── Tier 2: Concurrent Dispatch ──────────────────────────────────────

    /// Get the concurrent dispatch config.
    pub fn dispatch_config(&self) -> crate::concurrent_dispatch::ConcurrentDispatchConfig {
        crate::concurrent_dispatch::ConcurrentDispatchConfig {
            max_concurrency: self.config.concurrency.max_concurrency,
            serial_only_tools: self.config.concurrency.serial_only_tools.clone(),
            destructive_tools: self.config.concurrency.destructive_tools.clone(),
        }
    }

    /// Execute one conversation turn.
    pub async fn turn(
        &mut self,
        input: &str,
        _stream: bool,
        interrupt: Option<Arc<crate::interrupt::InterruptState>>,
    ) -> Result<String> {
        let sid = self.resolve_session().await;

        let call_mode = match &self.call_mode {
            Some(m) => m.clone(),
            None => {
                let mode = oben_models::CallMode::Fresh(sid.clone());
                self.call_mode = Some(mode.clone());
                mode
            }
        };

        let input_msg = oben_models::Message::user(input);
        let sm = Arc::clone(&self.session_manager);
        let input_text = input.to_string();

        // Pre-turn memory lifecycle
        let turn_count = {
            let guard = sm.lock().await;
            guard
                .session(&sid)
                .map(|s| s.metadata.turn_count)
                .unwrap_or(0)
        };
        if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            mm.lock().unwrap().on_turn_start(turn_count as usize, &input_text);
        }

        // Build memory context for this turn.
        // `prefetch_all` fetches relevant context for the input; `build_volatile_block`
        // wraps it (timestamp, session ID) so it's injected as system messages before
        // each API call via TurnExecutor.
        let memory_context = if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            let mem = mm.lock().unwrap().prefetch_all(&input_text, &sid);
            build_volatile_block(Some(&mem), Some(&sid), Some(&self.config.model.model))
        } else {
            build_volatile_block(None, Some(&sid), Some(&self.config.model.model))
        };

        let conversation = crate::coordinator::ConversationConfigBuilder::from_app_config(&self.config)
            .with_memory_context(memory_context)
            .build();

        let response = execute_turn_full(
            &mut self.context_window_manager,
            &self.transport,
            &self.tools,
            &mut *sm.lock().await,
            &sid,
            input_msg,
            &call_mode,
            &conversation,
            Some(Arc::clone(&self.hooks)),
            interrupt.map(|x| Arc::clone(&x)),
        )
        .await?;

        // Post-turn memory lifecycle
        if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            let sync_input = input_text.clone();
            let sync_response = response.clone();
            let sync_sid = sid.clone();
            let mm_cloned = Arc::clone(&mm);
            tokio::task::spawn_blocking(move || {
                mm_cloned.lock().unwrap().sync_all(&sync_input, &sync_response, &sync_sid);
            });
            mm.lock().unwrap().queue_prefetch_all(&input_text, &sid);
        }

        Ok(response)
    }

    /// Variant of [`Self::turn`] that accepts a pre-built [`Message`].
    ///
    ///
    /// Used by the TUI layer to send image messages when the user drags an
    /// image into the input bar — the URL is detected and wrapped in
    /// `MessageContent::Image` or `MessageContent::Parts` before reaching the
    /// transport layer.
    pub async fn turn_with_message(
        &mut self,
        input_msg: Message,
        interrupt: Option<Arc<crate::interrupt::InterruptState>>,
    ) -> Result<String> {
        let sid = self.resolve_session().await;

        let call_mode = match &self.call_mode {
            Some(m) => m.clone(),
            None => {
                let mode = oben_models::CallMode::Fresh(sid.clone());
                self.call_mode = Some(mode.clone());
                mode
            }
        };

        let sm = Arc::clone(&self.session_manager);
        let input_text = input_msg.content.to_text();

        // Pre-turn memory lifecycle
        let turn_count = {
            let guard = sm.lock().await;
            guard
                .session(&sid)
                .map(|s| s.metadata.turn_count)
                .unwrap_or(0)
        };
        if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            mm.lock().unwrap().on_turn_start(turn_count as usize, &input_text);
        }

        let memory_context = if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            let mem = mm.lock().unwrap().prefetch_all(&input_text, &sid);
            build_volatile_block(Some(&mem), Some(&sid), Some(&self.config.model.model))
        } else {
            build_volatile_block(None, Some(&sid), Some(&self.config.model.model))
        };

        let conversation = crate::coordinator::ConversationConfigBuilder::from_app_config(&self.config)
            .with_memory_context(memory_context)
            .build();

        let response = execute_turn_full(
            &mut self.context_window_manager,
            &self.transport,
            &self.tools,
            &mut *sm.lock().await,
            &sid,
            input_msg,
            &call_mode,
            &conversation,
            Some(Arc::clone(&self.hooks)),
            interrupt.map(|x| Arc::clone(&x)),
        )
        .await?;

        // Post-turn memory lifecycle
        if self.config.memory.enabled {
            let mm = Arc::clone(&self.memory_manager);
            let sync_input = input_text.clone();
            let sync_response = response.clone();
            let sync_sid = sid.clone();
            let mm_cloned = Arc::clone(&mm);
            tokio::task::spawn_blocking(move || {
                mm_cloned.lock().unwrap().sync_all(&sync_input, &sync_response, &sync_sid);
            });
            mm.lock().unwrap().queue_prefetch_all(&input_text, &sid);
        }

        Ok(response)
    }

    /// Resolve session ID (lazy create if no active session).
    async fn resolve_session(&mut self) -> String {
        let sid = match self.context_window_manager.session_id() {
            Some(sid) => match self.session_manager.lock().await.switch_session(&sid) {
                Ok(s) => s.id.clone(),
                Err(_) => {
                    match self.session_manager.lock().await.new_session(&generate_session_name()) {
                        Ok(s) => s.id.clone(),
                        Err(_) => generate_session_name(),
                    }
                }
            },
            None => match self.session_manager.lock().await.new_session(&generate_session_name()) {
                Ok(s) => s.id.clone(),
                Err(_) => generate_session_name(),
            },
        };
        sid
    }

    /// Switch to an existing session by ID or name.
    pub async fn continue_session(&mut self, key: &str) -> Result<String> {
        let sm = Arc::clone(&self.session_manager);
        sm.lock().await.init()?;
        let sid = {
            sm.lock().await.find_key(key).ok_or_else(|| {
                anyhow::anyhow!(
                    "Session not found: {}. Run `oben sessions list` to see available sessions.",
                    key
                )
            })?
        };
        sm.lock().await.switch_session(&sid)?;
        let name = match self.context_window_manager.session_id() {
            Some(sid) => sm.lock().await.session(&sid).map(|s| s.name.clone()),
            None => None,
        }.unwrap_or(key.to_string());
        Ok(name)
    }

    /// Reset the current session: delete it (messages + DB record) and enter
    /// a no-session state. The next [turn] will lazily create a new session
    /// via [resolve_session].
    pub async fn reset(&mut self) -> Result<()> {
        let sm = Arc::clone(&self.session_manager);
        let mm = Arc::clone(&self.memory_manager);

        let sid = self.context_window_manager.session_id();
        if let Some(ref sid) = sid {
            // Notify memory manager before deletion.
            let msgs = {
                let guard = sm.lock().await;
                guard.session(sid).map(|s| s.messages.clone()).unwrap_or_default()
            };
            let msgs_json: Vec<serde_json::Value> = msgs
                .iter()
                .filter_map(|m| serde_json::to_value(m).ok())
                .collect();
            mm.lock().unwrap().on_session_end(&msgs_json);
            self.session_manager.lock().await.delete_session(sid)?;
        }

        // Notify of switch to no-session state.
        let old_sid = sid.unwrap_or_default();
        mm.lock().unwrap().on_session_switch("", &old_sid, true);

        // Reset call mode so the next turn starts as Fresh.
        self.call_mode = None;
        Ok(())
    }

    /// Compact (summarize) the current session context.
    ///
    /// Retrieves the active session messages, mutates them in-place via the
    /// ContextWindowManager's compaction logic, then saves the session back.
    ///
    /// Returns a `CompactOutcome` describing what happened:
    /// - `AlreadyCompact` — messages within budget, no compaction needed
    /// - `NoMiddleMessages` — all messages protected (head/tail), no LLM call
    /// - `Ineffective` — compression attempted but savings below threshold
    /// - `Compressed` — messages successfully compacted with a summary
    pub async fn compact_session(&mut self) -> crate::compact::CompactOutcome {
        let sm = Arc::clone(&self.session_manager);
        // Ensure session manager is initialized
        if let Err(e) = sm.lock().await.init() {
            tracing::error!("Session manager init failed: {e}");
            return crate::compact::CompactOutcome::AlreadyCompact;
        }

        // Get the active session
        let active_id = self.context_window_manager.session_id();
        let sid = match active_id {
            Some(id) => id,
            None => return crate::compact::CompactOutcome::AlreadyCompact,
        };
        let mut guard = sm.lock().await;
        let mut messages = match guard.session_mut(&sid) {
            Some(s) => s.messages.clone(),
            None => return crate::compact::CompactOutcome::AlreadyCompact,
        };

        // Check if compaction is needed
        if !self.context_window_manager.should_compact(&messages) {
            return crate::compact::CompactOutcome::AlreadyCompact;
        }

        // Perform compaction
        let status = match self
            .context_window_manager
            .compact(&mut messages, Some(self.transport.as_ref()), None)
            .await
        {
            Ok(status) => status,
            Err(e) => {
                tracing::error!("ContextWindowManager::compact failed: {e}");
                return crate::compact::CompactOutcome::AlreadyCompact;
            }
        };

        // Notify memory manager before saving compaction.
        let msgs_json: Vec<serde_json::Value> = messages
            .iter()
            .filter_map(|m| serde_json::to_value(m).ok())
            .collect();
        let mm = Arc::clone(&self.memory_manager);
        let pre_compress = mm.lock().unwrap().on_pre_compress(&msgs_json);
        if !pre_compress.is_empty() {
            if let Some(s) = sm.lock().await.session_mut(&sid) {
                if let Some(ref mut mc) = s.memory_context {
                    mc.push_str("\n\n");
                    mc.push_str(&pre_compress);
                } else {
                    s.memory_context = Some(pre_compress);
                }
            }
        }

        match status {
            crate::context::CompactStatus::Compacted => {
                // Save the compacted messages back to the session.
                // The incremental save() path (tail.append) can't handle compaction
                // because persisted_message_count > messages.len() after compression.
                // save_compacted() handles: clear old DB messages, insert compacted set,
                // update in-memory session and persisted_message_count.
                let compact_ok = sm.lock().await.save_compacted(&sid, &messages).is_ok();
                if !compact_ok {
                    tracing::error!("Failed to save compacted session {sid}");
                }
                // Notify memory manager that the session was compacted.
                mm.lock().unwrap().on_session_switch(&sid, &sid, false);
                crate::compact::CompactOutcome::Compressed {
                    original_count: 0,
                    compacted_count: 0,
                    savings_pct: 0.0,
                }
            }
            crate::context::CompactStatus::Unchanged => {
                // Compression was ineffective — messages unchanged, nothing to persist.
                tracing::warn!(
                    "Manual compaction ineffective (session {}), skipping DB update",
                    sid
                );
                crate::compact::CompactOutcome::Ineffective {
                    original_tokens: 0,
                    compacted_tokens: 0,
                    savings_pct: 0.0,
                }
            }
        }
    }

    /// Create a new session and switch to it. The old session is preserved
    /// and can be restored later via [continue_session].
    pub async fn new_session(&mut self) -> Result<String> {
        let sm = Arc::clone(&self.session_manager);
        let mm = Arc::clone(&self.memory_manager);

        let old_sid = self.context_window_manager.session_id();
        let new_name = generate_session_name();
        let new_id = sm
            .lock()
            .await
            .new_session(&new_name)
            .map(|s| s.id.clone())
            .unwrap_or_else(|_| generate_session_name());

        // Notify memory manager of new session.
        let old_sid_v = old_sid.unwrap_or_default();
        mm.lock().unwrap().on_session_switch(&new_id, &old_sid_v, false);

        // Reset call mode so next turn starts Fresh.
        self.call_mode = None;
        Ok(new_id)
    }

    /// Get the currently loaded session display name (title if set, else internal name).
    pub async fn loaded_session_name(&self) -> Option<String> {
        let guard = self.session_manager.lock().await;
        self.context_window_manager
            .session_id()
            .and_then(|sid| guard.session(&sid))
            .map(|s| s.metadata.title.as_deref().unwrap_or(&s.name).to_string())
    }

    /// Get the active session name.
    pub async fn active_session_name(&self) -> Option<String> {
        self.loaded_session_name().await
    }

    /// Get the active session messages.
    pub async fn loaded_session_messages(&self) -> Result<Vec<oben_models::Message>> {
        let guard = self.session_manager.lock().await;
        let msgs = self
            .context_window_manager
            .session_id()
            .and_then(|sid| guard.session(&sid))
            .map(|s| s.messages.clone())
            .unwrap_or_default();
        Ok(msgs)
    }

    /// Initialize the session manager (for admin ops in async context).
    pub async fn init_session_manager(&mut self) -> Result<()> {
        let sm = Arc::clone(&self.session_manager);
        let res = sm.lock().await.init();
        res
    }

    /// Find a session key by name (for admin ops in async context).
    pub async fn find_session_key(&self, key: &str) -> Option<String> {
        let sm = Arc::clone(&self.session_manager);
        let result = sm.lock().await.find_key(key);
        result
    }

    /// Switch to a session by ID (for app-level session loading).
    pub async fn switch_session_to(&mut self, id: &str) -> Result<()> {
        let new_id = id.to_string();
        let sm = Arc::clone(&self.session_manager);
        let mm = Arc::clone(&self.memory_manager);

        let old_sid = self.context_window_manager.session_id();

        {
            let mut guard = sm.lock().await;
            guard.switch_session(id)?;
        }

        mm.lock().unwrap().on_session_switch(
            &new_id,
            old_sid.as_deref().unwrap_or(""),
            false,
        );
        Ok(())
    }

    /// List all sessions (wrapper for SessionsPanel).
    pub async fn list_sessions_full(&self) -> Vec<oben_models::Session> {
        let sessions = self.session_manager.lock().await.list_sessions_full();
        sessions
    }

    /// Get session messages (wrapper for SessionsPanel preview).
    pub async fn get_session_messages(
        &self,
        session_id: &str,
    ) -> Result<Vec<oben_models::Message>> {
        let msgs = self
            .session_manager
            .lock()
            .await
            .get_session_messages(session_id)?;
        Ok(msgs)
    }

    /// Load a session's messages into the in-memory cache (for refresh_list).
    pub async fn load_session_messages(&mut self, session_id: &str) -> Result<()> {
        self.session_manager
            .lock()
            .await
            .ensure_session_loaded(session_id)?;
        Ok(())
    }

    /// Delete a session (wrapper for SessionsPanel).
    pub async fn delete_session(&mut self, session_id: &str) -> Result<()> {
        self.session_manager.lock().await.delete_session(session_id)?;
        Ok(())
    }

    /// Switch to a session (wrapper for SessionsPanel compact/switch).
    pub async fn switch_session(&mut self, session_id: &str) -> Result<()> {
        let _session = self
            .session_manager
            .lock()
            .await
            .switch_session(session_id)?;
        Ok(())
    }

    /// Close current session (wrapper for SessionsPanel).
    pub async fn close_session(&mut self) -> Result<()> {
        self.session_manager.lock().await.close()?;
        Ok(())
    }

    /// Run a background memory/skill review turn (nudge).
    ///
    /// This is the internal implementation of C.15 — the "turn nudge" that
    /// automatically reviews conversation history and optionally updates
    /// MEMORY.md / USER.md via the memory tool.
    ///
    /// Mirrors Hermes' `_spawn_background_review` + `_run_review_in_thread`:
    /// executes a bounded review turn on the same agent with a custom
    /// prompt. The result is delivered via the provided callback.
    pub async fn trigger_nudge(
        &mut self,
        _user_message: &str,
        memory_interval: usize,
        skill_interval: usize,
        on_complete: impl FnOnce(bool, String) + Send + 'static,
    ) {
        // Quick-exit if nudge is disabled.
        if memory_interval == 0 && skill_interval == 0 {
            return;
        }

        let memory_enabled = memory_interval > 0;
        let skill_enabled = skill_interval > 0;

        let sm = Arc::clone(&self.session_manager);
        let has_memory_tool = {
            let guard = sm.lock().await;
            self.context_window_manager
                .session_id()
                .and_then(|sid| guard.session(&sid))
                .map(|s| s.messages.iter().any(|m| !m.tool_calls.is_none()))
                .unwrap_or(true)
        };

        if !has_memory_tool {
            on_complete(
                false,
                "Nudge skipped: no memory tools available.".to_string(),
            );
            return;
        }

        // Build the nudge prompt (mirrors Hermes _MEMORY_REVIEW_PROMPT).
        let prompt = self.build_nudge_prompt(memory_enabled, skill_enabled);

        let nudge_config = crate::coordinator::ConversationConfigBuilder::from_app_config(&self.config)
            .with_max_iterations(16)
            .build();

        let sm = Arc::clone(&self.session_manager);
        let sid = {
            let active_id = self.context_window_manager.session_id();
            let sid = match active_id {
                Some(sid) => sid,
                None => match sm.lock().await.new_session(&generate_session_name()) {
                    Ok(s) => s.id.clone(),
                    Err(_) => generate_session_name(),
                },
            };
            sm.lock()
                .await
                .switch_session(&sid)
                .map(|s| s.id.clone())
                .unwrap_or_else(|_| generate_session_name())
        };
        let call_mode = self
            .call_mode
            .clone()
            .unwrap_or(CallMode::Fresh(sid.clone()));

        let review_msg = Message::user(&prompt);
        let sm = Arc::clone(&self.session_manager);
        let response_text = execute_turn_full(
            &mut self.context_window_manager,
            &self.transport,
            &self.tools,
            &mut *sm.lock().await,
            &sid,
            review_msg,
            &call_mode,
            &nudge_config,
            None,
            None,
        )
        .await;

        let _result = match response_text {
            Ok(text) => {
                let text_lower = text.to_lowercase();
                let is_noop = text_lower.contains("nothing to")
                    || text_lower.contains("nothing worth")
                    || text_lower.contains("no changes needed");

                let updated = !is_noop;
                let summary = if is_noop {
                    "Review: nothing worth saving this session.".to_string()
                } else {
                    "Review: checked memory — may have updated.".to_string()
                };
                on_complete(updated, summary);
            }
            Err(e) => {
                tracing::info!("Nudge review failed (non-fatal): {}", e);
                on_complete(false, format!("Review failed: {}", e));
            }
        };
    }

    fn build_nudge_prompt(&self, memory_enabled: bool, skill_enabled: bool) -> String {
        if memory_enabled && skill_enabled {
            format!(
                "Review the conversation above and consider the following:\n\n\
                 MEMORY REVIEW:\n\
                 1. Has the user revealed things about themselves — their persona, desires, \
                 preferences, or personal details worth remembering?\n\
                 2. Has the user expressed expectations about how you should behave, their work \
                 preferences, or their communication style?\n\
                 3. What are the most important lessons from recent interactions?\n\n\
                 SKILL REVIEW:\n\
                 1. Did the user reveal a recurring task or workflow that should become a skill?\n\
                 2. Are there any tools or integrations worth setting up?\n\n\
                 Only suggest changes if there's something genuinely useful to save. \
                 If nothing worth saving, say so briefly and stop.\n\n\
                 Think through your reasoning before deciding."
            )
        } else if memory_enabled {
            format!(
                "Review the conversation above and consider the following:\n\n\
                 MEMORY REVIEW:\n\
                 1. Has the user revealed things about themselves — their persona, desires, \
                 preferences, or personal details worth remembering?\n\
                 2. Has the user expressed expectations about how you should behave, their work \
                 preferences, or their communication style?\n\
                 3. What are the most important lessons from recent interactions?\n\n\
                 Only suggest changes if there's something genuinely useful to save. \
                 If nothing worth saving, say so briefly and stop.\n\n\
                 Think through your reasoning before deciding."
            )
        } else {
            format!(
                "Review the conversation above and consider the following:\n\n\
                 SKILL REVIEW:\n\
                 1. Did the user reveal a recurring task or workflow that should become a skill?\n\
                 2. Are there any tools or integrations worth setting up?\n\n\
                 Only suggest changes if there's something genuinely useful to save. \
                 If nothing worth saving, say so briefly and stop.\n\n\
                 Think through your reasoning before deciding."
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_session_creates_fresh_on_empty() {
        let mut agent = Agent::new_for_test().await;
        let sid = agent.resolve_session().await;
        assert!(!sid.is_empty());
    }

    #[test]
    fn test_hook_engine_receives_tool_error() {
        use super::super::hooks::kind::ToolLifecycleHooks;
        use super::super::hooks::kind::Hook;
        use std::sync::{Arc, Mutex};

        let called = Arc::new(Mutex::new(false));
        let called_clone = called.clone();

        struct ErrorCapturingHook {
            called: Arc<Mutex<bool>>,
        }

        impl ToolLifecycleHooks for ErrorCapturingHook {
            fn on_tool_error(&self, _tool_name: &str, _args: &str, error: &str) {
                assert_eq!(error, "connection failed");
                *self.called.lock().unwrap() = true;
            }
        }

        impl Hook for ErrorCapturingHook {
            fn id(&self) -> &str { "test_error" }
        }

        let engine = super::super::hooks::HookEngine::new();
        engine.register_tool(Box::new(ErrorCapturingHook {
            called: called_clone,
        }));

        engine.emit_tool_error("shell", "{}", "connection failed");
        assert!(*called.lock().unwrap(), "on_tool_error should have been called");
    }
}

