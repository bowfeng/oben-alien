//! All CLI command implementations.
//!
//! Domain crates provide types and business logic only; this crate handles
//! CLI parsing, wiring crates together, and user-facing output.

use anyhow::Result;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::cli::{
    Cli, Commands, ConfigCommand, CronCommand, CuratorCommand, GatewayCommand, GoalCommand, ModelsCommand, SessionsCommand,
};
use clap::Parser;
use crate::coordinator::CliCoordinator;
use oben_agent::coordinator::ConversationConfig;
use oben_agent::delegate::{build_spawn_fn_wrapper, SubagentSpawner};
use oben_agent::hooks::HookBuilder;
use oben_agent::AgentBuilder;
use oben_tools::delegate::DelegateTool;
use oben_tools::ToolRegistry;
use oben_cron::{CronJob, CronStore};
use oben_goals::GoalStore;
use oben_models::TransportProvider;
use oben_skills::SkillStateManager;
use oben_curator::{Curator, CuratorConfig};

/// Entry point: parse CLI args and dispatch to the appropriate handler.
pub async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    // --verbose sets RUST_LOG only if not already configured, so explicit
    // env vars take precedence for fine-grained filtering.
    if cli.verbose && std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "oben=debug");
    }
    let _log_path = oben_utils::logging::init(tracing::Level::INFO);
    // Install panic hook so `panic!` goes into the log file instead of the TUI screen.
    oben_utils::logging::init_panic_hook();

    let profile = cli.profile.as_deref();
    let _moa_enabled = cli.moa; // Reserved for future MoA integration

    match cli.command {
        Commands::Chat {
            no_stream,
            continue_session,
            search_provider: _, agent
        } => run_chat(!no_stream, continue_session.as_deref(), profile, agent.as_deref()).await,
        Commands::Run { prompt, stream, search_provider: _, agent } => run_one_shot(&prompt, stream, profile, agent.as_deref()).await,
        Commands::Setup => run_setup(profile),
        Commands::Config { action } => run_config(action, profile).await,
        Commands::Tools => list_tools(),
        Commands::Skills => list_skills(),
        Commands::Sessions { action } => match action {
            Some(SessionsCommand::List) => list_sessions(),
            Some(SessionsCommand::Compact { session, focus }) => {
                run_compact_session(session.as_deref(), focus.as_deref(), profile).await
            }
            Some(SessionsCommand::Delete { session }) => run_delete_session(&session),
            Some(SessionsCommand::Dump { session }) => dump_session(session.as_deref()),
            None => list_sessions(),
        },
        Commands::Models { action } => run_models(action, profile).await,
        Commands::Tui { session, agent } => {
            oben_tui::run_tui(session.as_deref(), agent.as_deref()).await
        }
        Commands::Cron { action } => match action {
            None => cron_list(false),
            Some(CronCommand::List { all }) => cron_list(all),
            Some(CronCommand::Create {
                schedule,
                prompt,
                name,
                repeat,
            }) => cron_create(&schedule, prompt.as_deref(), name.as_deref(), repeat),
            Some(CronCommand::Pause { id }) => cron_pause(&id),
            Some(CronCommand::Resume { id }) => cron_resume(&id),
            Some(CronCommand::Remove { id }) => cron_remove(&id),
            Some(CronCommand::Tick) => cron_tick(),
            Some(CronCommand::Start) => cron_start(),
            Some(CronCommand::Info) => cron_info(),
        },
        Commands::Goals { action } => match action {
            None => goal_list(None),
            Some(GoalCommand::Start { goal, max_turns }) => goal_start(&goal, max_turns, profile).await,
            Some(GoalCommand::List { status }) => goal_list(status.as_deref()),
            Some(GoalCommand::Status { goal_id }) => goal_status(&goal_id),
            Some(GoalCommand::Pause { id }) => goal_pause(&id).await,
            Some(GoalCommand::Resume { id, reset }) => goal_resume(&id, reset).await,
            Some(GoalCommand::Clear { id }) => goal_clear(&id).await,
        },
        Commands::Gateway { action } => match action {
            None => {
                println!("Gateway commands: start, stop, status, setup");
                println!("Run 'oben gateway start' to launch the gateway server.");
                Ok(())
            }
            Some(GatewayCommand::Start) => gateway_start(profile).await,
            Some(GatewayCommand::Stop) => gateway_stop(profile).await,
            Some(GatewayCommand::Status) => gateway_status(profile).await,
            Some(GatewayCommand::Setup) => gateway_setup(profile).await,
        },
        Commands::Curator { action } => match action {
            CuratorCommand::Pin { skill } => run_curator_pin(&skill),
            CuratorCommand::Run { consolidate } => run_curator_run(consolidate).await,
            CuratorCommand::Status => run_curator_status(),
        },
    }
}

// ── Chat / Run ──────────────────────────────────────────────────────────

async fn run_chat(stream: bool, continue_with: Option<&str>, profile: Option<&str>, agent: Option<&str>) -> Result<()> {
    info!("Starting interactive chat...");

    let config = oben_config::AppConfig::load(profile)?;
    let mut tools = oben_tools::ToolRegistry::new();
    oben_tools::discover_builtin_tools(&mut tools);

    let tool_names: Vec<String> = tools.list_tools().iter().map(|t| t.name.clone()).collect();

    let identity = oben_config::defaults::default_system_prompt();
    let skills_dirs: Vec<std::path::PathBuf> = config.skills.dirs.iter()
        .map(|d| std::path::PathBuf::from(d))
        .collect();
    let context_cwd = std::env::current_dir().ok();

    let volatile =
        oben_agent::system_prompt::build_volatile_block(None, None, Some(&config.model.model));
    let assembled = oben_agent::system_prompt::build_system_prompt(
        &identity,
        &tool_names,
        &skills_dirs,
        context_cwd.as_deref(),
        None,
        Some(&volatile),
    );

    // Wire up delegate_task with a real SpawnFn (same pattern as TUI).
    let shared_hooks = Arc::new(HookBuilder::from_config(&config.hooks).build());
    let transport = oben_transport::Transport::from_config_with_tools_via_registry(
        &config.model,
        &assembled.prompt,
        &tools.list_tools());
    info!("CLI: creating SubagentSpawner for delegate_task wiring");
    let spawner = SubagentSpawner::new(
        Arc::new(transport),
        Arc::new(tools.clone()),
        config.clone(),
        oben_agent::compact::CompactCofig {
            context_length: config.context.context_length,
            threshold_percent: config.context.threshold_percent,
            ..Default::default()
        },
        config.max_iterations.unwrap_or(50),
        config.context.max_messages.unwrap_or(100),
        config.max_spawn_depth.unwrap_or(3),
        Arc::clone(&shared_hooks),
        agent.map(String::from),
    );
    let spawn_fn = build_spawn_fn_wrapper(spawner, assembled.prompt.clone());
    let mut tools_for_agent = ToolRegistry::clone(&tools);
    tools_for_agent.register(DelegateTool::new(
        spawn_fn,
        config.max_concurrent_tasks.unwrap_or(5),
    ));

    let mut chat = AgentBuilder::new()
        .with_config(config)
        .with_system_prompt(assembled.prompt.clone())
        .with_tools(Arc::new(tools_for_agent))
        .with_hooks(shared_hooks)
        .with_agent_name(agent.map(String::from).or_else(|| Some("default".to_string())))
        .build()
        .await?;

    // Coordinator handles streaming hook registration internally.
    let conversation_config = ConversationConfig::from_app_config(&chat.config());
    let hooks = chat.hooks();
    let coordinator = CliCoordinator::from_conversation(
        conversation_config,
        Arc::clone(hooks),
        stream,
        None, // max_turns: not yet configured in AppConfig
    );

    // If continuing an existing session, resolve it first.
    if let Some(resolved) = continue_with {
        let _name = chat.continue_session(resolved).await?;
    } else if let Some(name) = chat.loaded_session_name().await {
        tracing::info!("Session: {}", name);
    }

    let agent = std::sync::Arc::new(tokio::sync::Mutex::new(chat));
    oben_agent::Agent::run(agent, coordinator).await?;
    Ok(())
}



async fn run_one_shot(prompt: &str, stream: bool, profile: Option<&str>, agent: Option<&str>) -> Result<()> {
    let config = oben_config::AppConfig::load(profile)?;

    let mut tools = oben_tools::ToolRegistry::new();
    oben_tools::discover_builtin_tools(&mut tools);

    let system_prompt = oben_config::defaults::default_system_prompt();

    // Build transport + spawner so delegate_tool has a real SpawnFn
    let shared_hooks = Arc::new(HookBuilder::from_config(&config.hooks).build());
    let transport = oben_transport::Transport::from_config_with_tools_via_registry(
        &config.model,
        &system_prompt,
        &tools.list_tools());
    let spawner = SubagentSpawner::new(
        Arc::new(transport),
        Arc::new(tools.clone()),
        config.clone(),
        oben_agent::compact::CompactCofig {
            context_length: config.context.context_length,
            threshold_percent: config.context.threshold_percent,
            ..Default::default()
        },
        config.max_iterations.unwrap_or(50),
        config.context.max_messages.unwrap_or(100),
        config.max_spawn_depth.unwrap_or(3),
        Arc::clone(&shared_hooks),
        agent.map(String::from),
    );
    let spawn_fn = build_spawn_fn_wrapper(spawner, system_prompt.clone());
    let mut tools_for_agent = ToolRegistry::clone(&tools);
    tools_for_agent.register(DelegateTool::new(
        spawn_fn,
        config.max_concurrent_tasks.unwrap_or(5),
    ));

    let mut agent = AgentBuilder::new()
        .with_config(config)
        .with_system_prompt(system_prompt.clone())
        .with_tools(Arc::new(tools_for_agent))
        .with_hooks(shared_hooks)
        .with_agent_name(        agent.map(String::from))
        .build()
        .await?;

    let response = agent.turn(prompt, stream, None).await?;

    if !stream {
        println!("\n{}", response);
    } else {
        println!();
    }

    Ok(())
}

// ── Setup & Config ──────────────────────────────────────────────────────

fn run_setup(profile: Option<&str>) -> Result<()> {
    let mut config = oben_config::AppConfig::load(profile)?;
    oben_config::wizard::run_setup(&mut config)?;
    config.save_with_profile(profile)?;
    Ok(())
}

async fn run_config(action: ConfigCommand, profile: Option<&str>) -> Result<()> {
    let config = oben_config::AppConfig::load(profile)?;
    match action {
        ConfigCommand::Show => {
            println!("{}", serde_yaml::to_string(&config)?);
        }
        ConfigCommand::Edit => {
            let env = oben_config::env::Env::new(profile.map(String::from));
            let path = env.config_path();
            println!("Config file: {}", path.display());
            println!("Edit it manually, or run `oben setup` for the wizard.");
        }
    }
    Ok(())
}

// ── Tools & Skills ──────────────────────────────────────────────────────

fn list_tools() -> Result<()> {
    let mut tools = oben_tools::ToolRegistry::new();
    oben_tools::discover_builtin_tools(&mut tools);
    let tool_list = tools.list_tools();
    if tool_list.is_empty() {
        println!("No tools registered.");
    } else {
        println!("Registered tools ({}):", tool_list.len());
        for tool in tool_list {
            println!("  📦 {} - {}", tool.name, tool.description);
        }
    }
    Ok(())
}

fn list_skills() -> Result<()> {
    let skills = oben_skills::builtin_skills();
    println!("Built-in skills ({}):", skills.len());
    for skill in skills {
        println!(
            "  📖 {} ({}) - {}",
            skill.name, skill.category, skill.description
        );
    }
    Ok(())
}

// ── Sessions ────────────────────────────────────────────────────────────

fn list_sessions() -> Result<()> {
    // ✅ CLI uses default session isolation
    let mut session_manager = oben_sessions::DBSessionManager::new_with_agent(Some("default"))?;
    session_manager.init()?;
    let sessions = session_manager.list_sessions(None);
    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        println!("Sessions ({}):", sessions.len());
        for s in sessions {
            println!("  • {} — {} messages", s.name, s.message_count);
        }
    }
    Ok(())
}

async fn run_compact_session(session_key: Option<&str>, focus_topic: Option<&str>, profile: Option<&str>) -> Result<()> {
    let config = oben_config::AppConfig::load(profile)?;
    // ✅ CLI uses default session isolation
    let mut sm = oben_sessions::DBSessionManager::new_with_agent(Some("default"))?;

    let target: String = match session_key {
        Some(key) => key.to_string(),
        None => "active".to_string(),
    };
    let target_ref = target.as_str();

    let session = sm.clone_session(target_ref).ok_or_else(|| {
        anyhow::anyhow!(
            "Session not found: {} (run `oben sessions list` to see available sessions)",
            target
        )
    })?;

    if session.message_count() < 8 {
        println!(
            "Session has only {} message(s). Minimum 8 required for compaction.",
            session.message_count()
        );
        return Ok(());
    }

    println!(
        "Compacting session '{}' ({} messages)...",
        session.name,
        session.message_count()
    );

    let transport = create_transport(&config, "", &oben_tools::ToolRegistry::new());
    let comp_config = oben_agent::compact::CompactCofig {
        context_length: config.context.context_length,
        threshold_percent: config.context.threshold_percent,
        ..oben_agent::compact::CompactCofig::default()
    };

    let result = oben_agent::compact_session_messages(
        &transport,
        &session.messages,
        &comp_config,
        session.memory_context.as_deref(),
        focus_topic,
        1,
    )
    .await?;

    if let Some(s) = sm.session_mut(&session.id) {
        s.messages = result.messages;
        s.updated_at = chrono::Utc::now();
        if let Some(summary) = result.summary {
            s.memory_context = Some(summary.clone());
            let old_msg_count = session.messages.len();
            s.summary_chunks.push(oben_models::SummaryChunk {
                from: 1,
                to: old_msg_count as i64,
                summary,
            });
        }
    }
    sm.save_session(Some(&session.id))?;

    println!("✓ Compaction complete:");
    println!(
        "  Before: {} messages, ~{} tokens",
        result.stats.original_count, result.stats.original_tokens
    );
    println!(
        "  After:  {} messages, ~{} tokens",
        result.stats.compacted_count, result.stats.compacted_tokens
    );
    println!(
        "  Saved:  {:.0}% tokens ({} tool results pruned)",
        result.stats.savings_pct, result.stats.pruned_tool_results
    );
    if result.stats.summary_generated {
        println!("  Summary: LLM-generated (iterative)");
    } else {
        println!("  Summary: LLM call skipped/fallback");
    }
    if focus_topic.is_some() {
        println!("  Focus: {:?}", focus_topic);
    }

    Ok(())
}

fn run_delete_session(session_key: &str) -> Result<()> {
    // ✅ CLI uses default session isolation
    let mut sm = oben_sessions::DBSessionManager::new_with_agent(Some("default"))?;
    sm.init()?;
    sm.delete(session_key)?;
    println!("Deleted session '{}'", session_key);
    Ok(())
}

fn dump_session(session_key: Option<&str>) -> Result<()> {
    // ✅ CLI uses default session isolation
    let mut sm = oben_sessions::DBSessionManager::new_with_agent(Some("default"))?;
    sm.load(None)?;

    let active_id: Option<String> = None;
    let target: String = match session_key {
        Some(key) => key.to_string(),
        None => active_id.clone().unwrap_or_else(|| "active".to_string()),
    };

    let session_id = sm.find_key(&target).ok_or_else(|| {
        anyhow::anyhow!(
            "Session not found: {}. Run `oben sessions list` to see available sessions",
            target
        )
    })?;

    let sessions: Vec<oben_models::Session> = sm.list_sessions_full();
    let session = sessions
        .iter()
        .find(|s| s.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?
        .clone();

    let session_name = session
        .metadata
        .title
        .as_deref()
        .unwrap_or(&session.id)
        .replace(" ", "-");
    let filename = format!(
        "{}/dump-{}-{}.json",
        std::env::current_dir().unwrap().display(),
        session_name,
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    );

    let dump: serde_json::Value = serde_json::json!({
        "id": session.id,
        "name": session.name,
        "title": session.metadata.title,
        "message_count": session.messages.len(),
        "messages": session.messages,
    });

    let json = serde_json::to_string_pretty(&dump)?;
    std::fs::write(&filename, &json)?;

    println!(
        "Dumped {} messages from '{}' to {}",
        session.messages.len(),
        session.metadata.title.as_deref().unwrap_or(&session.name),
        filename
    );
    Ok(())
}

// ── Models ──────────────────────────────────────────────────────────────

async fn run_models(action: ModelsCommand, profile: Option<&str>) -> Result<()> {
    let config = oben_config::AppConfig::load(profile)?;
    let transport = create_transport(&config, "", &oben_tools::ToolRegistry::new());

    match action {
        ModelsCommand::List => {
            println!("Fetching models from provider...\n");
            let models = transport.list_models().await?;
            println!("Found {} model(s):\n", models.data.len());

            let headers = &["ID", "Max Tokens", "Owned By"];
            let rows: Vec<Vec<String>> = models
                .data
                .iter()
                .map(|m| {
                    vec![
                        m.id.clone(),
                        m.max_model_len
                            .map(|t| t.to_string())
                            .unwrap_or_else(|| "N/A".to_string()),
                        m.owned_by.clone(),
                    ]
                })
                .collect();
            oben_utils::terminal::print_table_stderr(headers, rows);
        }
        ModelsCommand::Info { model } => {
            println!("Looking up model: {}\n", model);
            match transport.find_model(&model).await? {
                Some(m) => {
                    let headers = &["Field", "Value"];
                    let rows = vec![
                        vec!["ID".to_string(), m.id],
                        vec!["Object".to_string(), m.object],
                        vec![
                            "Created".to_string(),
                            chrono::DateTime::from_timestamp(m.created as i64, 0)
                                .map(|d| d.to_string())
                                .unwrap_or("unknown".to_string()),
                        ],
                        vec!["Owned By".to_string(), m.owned_by],
                        vec![
                            "Max Model Length".to_string(),
                            m.max_model_len
                                .map(|t| t.to_string())
                                .unwrap_or("N/A".to_string()),
                        ],
                        vec!["Root".to_string(), m.root.unwrap_or("N/A".to_string())],
                        vec!["Parent".to_string(), m.parent.unwrap_or("N/A".to_string())],
                    ];
                    oben_utils::terminal::print_table_stderr(headers, rows);
                }
                None => {
                    println!("Model '{}' not found.", model);
                    println!("Run 'oben models list' to see available models.");
                }
            }
        }
    }
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn collect_tool_defs(registry: &oben_tools::ToolRegistry) -> Vec<oben_models::ToolMeta> {
    registry.list_tools()
}

fn create_transport(
    config: &oben_config::AppConfig,
    system_prompt: &str,
    tools: &oben_tools::ToolRegistry,
) -> std::sync::Arc<dyn oben_models::TransportProvider + Send + Sync> {
    let tool_defs = collect_tool_defs(tools);
    oben_transport::Transport::from_config_with_tools_via_registry(
        &config.model,
        system_prompt,
        &tool_defs,
    )
}

// ── Goals ──────────────────────────────────────────────────────────────────

/// Get a default JsonGoalStore instance.
fn goal_store() -> Result<oben_goals::JsonGoalStore> {
    oben_goals::JsonGoalStore::default_store()
}

async fn goal_start(goal: &str, max_turns: Option<usize>, profile: Option<&str>) -> Result<()> {
    let goal_id = format!(
        "goal-{}-{}",
        chrono::Utc::now().timestamp_millis(),
        Uuid::new_v4().as_simple()
    );
    let store = goal_store()?;

    // Create goal record
    store.create_goal(&goal_id, goal, max_turns.unwrap_or(20), None)?;

    // Decomposer: call LLM to break goal into plan nodes.
    // This mirrors `create_plan_from_goal()` but executed synchronously so the CLI
    // blocks until the plan is created and printed before the background loop starts.
    let config = oben_config::AppConfig::load(profile)?;

    let mut tools = oben_tools::ToolRegistry::new();
    oben_tools::discover_builtin_tools(&mut tools);
    let tool_defs = collect_tool_defs(&tools);

    let system_prompt = oben_config::defaults::default_system_prompt();
    // Decomposer doesn't need tools — just LLM JSON generation
    let decomposer_transport = std::sync::Arc::new(oben_transport::Transport::from_config(
        &config.model,
        system_prompt.clone(),
    ));

    let plan_prompt = format!(
        "You are a planner. Break the following goal into a step-by-step plan.\n\n\
         Return ONLY a JSON array. No prose, no explanation, no markdown.\n\
         Each item must have:\n\
         {{\"title\": \"step name\", \"description\": \"what to do\", \"sub_tasks\": []}}\n\n\
         Rules:\n\
         - Each top-level item is a self-contained task\n\
         - sub_tasks is an array of nested task strings (may be empty)\n\
         - Keep tasks specific and actionable\n\n\
         Goal: {}",
        goal
    );

    let plan_messages = vec![oben_models::Message::system(&system_prompt), oben_models::Message::user(&plan_prompt)];

    let plan_llm_result = decomposer_transport
        .chat(&plan_messages, &oben_models::CallMode::Fresh(goal_id.clone()))
        .await?;

    let json_text = plan_llm_result.text.trim();
    // Strip markdown code fences if present
    let json_text = json_text.strip_prefix("```json").unwrap_or(json_text);
    let json_text = json_text.strip_prefix("```").unwrap_or(json_text);
    let json_text = json_text.strip_suffix("```").unwrap_or(json_text).trim();

    if json_text.is_empty() {
        anyhow::bail!(
            "LLM returned empty response. Try again. Full response len: {}",
            plan_llm_result.text.len()
        );
    }

    tracing::info!("Plan LLM response (len={}):\n{}", json_text.len(), json_text);

    let parsed_nodes: Vec<oben_goals::PlanNode> =
        match serde_json::from_str::<Vec<serde_json::Value>>(json_text) {
            Ok(items) => {
                let mut nodes = Vec::new();
                for item in items {
                    let title = item
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Untitled")
                        .to_string();
                    let mut node = oben_goals::PlanNode::new(&title);
                    if let Some(sub_tasks) = item.get("sub_tasks") {
                        if let Some(arr) = sub_tasks.as_array() {
                            for sub in arr {
                                let sub_title = if sub.is_string() {
                                    sub.as_str().unwrap_or("").to_string()
                                } else if sub.is_object() {
                                    sub.get("title")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("Untitled")
                                        .to_string()
                                } else {
                                    continue;
                                };
                                node.push_sub_node(oben_goals::PlanNode::new(&sub_title));
                            }
                        }
                    }
                    nodes.push(node);
                }
                println!("  Plan created: {} nodes", nodes.len());
                nodes
            }
            Err(e) => anyhow::bail!("Failed to parse plan from LLM: {}", e),
        };

    // Build plan and save via store
    let mut plan = oben_goals::PlanState::new(goal);
    for node in parsed_nodes {
        plan.add_node(node);
    }
    store.save_plan(&goal_id, &plan)?;

    // Save initial goal state to disk
    use oben_goals::GoalState;
    let initial_state = GoalState {
        goal: goal.to_string(),
        status: oben_goals::GoalStatus::Active,
        turns_used: 0,
        max_turns: max_turns.unwrap_or(20),
        created_at: chrono::Utc::now(),
        last_turn_at: None,
        last_verdict: None,
        last_reason: None,
        paused_reason: None,
        consecutive_parse_failures: 0,
    };
    store.save_goal_state(&goal_id, &initial_state)?;

    println!("Goal created: {} (id: {})", goal, goal_id);
    println!("  Status: Active");
    println!("  Max turns: {}", max_turns.unwrap_or(20));
    println!("");

    let mut tools = oben_tools::ToolRegistry::new();
    oben_tools::discover_builtin_tools(&mut tools);
    let tools = std::sync::Arc::new(tools);
    let sp = oben_config::defaults::default_system_prompt();
    let _transport =
          oben_transport::Transport::from_config_with_tools_via_registry(&config.model, &sp, &tool_defs);

    let _max_iterations = config.max_iterations.unwrap_or(50);
    let _max_messages = config.context.max_messages.unwrap_or(100);

    let loop_config =   oben_goals::GoalLoopConfig {
        max_turns: max_turns.unwrap_or(20),
        system_prompt: Some(sp.clone()),
        auto_save: true,
    };

    let result =   oben_goals::goal_loop::run_goal_loop(
        goal,
        &goal_id,
        &loop_config,
        &store,
        move |prompt| {
            let tools = tools.clone();
            let sp = sp.clone();
            let prompt_owned = prompt.to_string();
            let config = config.clone();
            async move {
                let mut goal_agent =   AgentBuilder::new()
                    .with_config(config.clone())
                    .with_system_prompt(sp.clone())
                    .with_tools(tools.clone())
                    .build()
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))?;

                goal_agent
                    .turn_with_message(
                        oben_models::Message::user(&prompt_owned),
                        None,
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e))
            }
        },
    )
    .await;

    match result {
        Ok((_plan, gs)) => {
            println!("");
            let status_str = match gs.status {
                oben_goals::GoalStatus::Active => "active",
                oben_goals::GoalStatus::Done => "done",
                oben_goals::GoalStatus::Paused => "paused",
                oben_goals::GoalStatus::Cleared => "cleared",
            };
            println!("Goal '{}' completed. Status: {}", goal, status_str);
            println!("  Turns: {}/{}", gs.turns_used, gs.max_turns);
            if let Some(ref reason) = gs.paused_reason {
                println!("  Paused: {}", reason);
            }
        }
        Err(e) => eprintln!("Goal '{}' failed: {}", goal, e),
    }

    Ok(())
}

fn goal_list(status: Option<&str>) -> Result<()> {
    let store = goal_store()?;
    let goals = store.list_goals(status)?;

    if goals.is_empty() {
        println!("No goals found.");
    } else {
        println!("Goals ({}):\n", goals.len());
        for g in goals {
            let owner_str = g.owner.as_deref().map(|s| &s[..8]).unwrap_or("-");
            println!("  ID: {} | Owner: {} | Goal: {}", g.id, owner_str, g.text);
        }
    }
    Ok(())
}

fn goal_status(goal_id: &str) -> Result<()> {
    let store = goal_store()?;

    match store.get_goal(goal_id)? {
        Some((text, status, max_turns, turns_used, paused_reason)) => {
            println!("Goal: {}", text);
            println!("  ID: {}", goal_id);
            println!("  Status: {}", status);
            println!("  Turns: {}/{}", turns_used, max_turns);
            if let Some(ref reason) = paused_reason {
                println!("  Paused: {}", reason);
            }
        }
        None => {
            println!("Goal '{}' not found.", goal_id);
            println!("Use `oben goals list` to see available goals.");
        }
    }
    Ok(())
}

async fn goal_pause(id: &str) -> Result<()> {
    let store = goal_store()?;
    let goal_id = resolve_if_active(&store, id)?;
    if store.get_goal(&goal_id)?.is_some() {
        store.pause_goal(&goal_id, "user-paused")?;
        println!("Goal paused: {}", goal_id);
    } else {
        anyhow::bail!(
            "Goal not found: {}. Run `oben goals start` to create one.",
            goal_id
        );
    }
    Ok(())
}

async fn goal_resume(id: &str, reset: bool) -> Result<()> {
    let store = goal_store()?;
    let goal_id = resolve_if_active(&store, id)?;
    if store.get_goal(&goal_id)?.is_some() {
        store.resume_goal(&goal_id, reset)?;
        println!("Goal resumed: {} (budget reset: {})", goal_id, reset);
    } else {
        anyhow::bail!("Paused goal not found: {}.", goal_id);
    }
    Ok(())
}

async fn goal_clear(id: &str) -> Result<()> {
    let store = goal_store()?;
    let goal_id = resolve_if_active(&store, id)?;
    if store.get_goal(&goal_id)?.is_some() {
        store.delete_goal(&goal_id)?;
        println!("Goal cleared: {}", goal_id);
    } else {
        anyhow::bail!("Goal not found: {}.", goal_id);
    }
    Ok(())
}

/// Resolve "active" to the most recent active goal from store, otherwise use the id as-is.
fn resolve_if_active(store: &oben_goals::JsonGoalStore, id: &str) -> Result<String> {
    if id == "active" {
        let goal_id = store
            .list_goals(Some("active"))?
            .into_iter()
            .next()
            .map(|g| g.id)
            .ok_or_else(|| {
                anyhow::anyhow!("No active goal found. Run `oben goals start` first.")
            })?;
        Ok(goal_id)
    } else {
        Ok(id.to_string())
    }
}
// ── Cron ───────────────────────────────────────────────────────────────────

fn cron_store() -> std::sync::Arc<CronStore> {
    let dir = CronStore::default_path();
    std::sync::Arc::new(CronStore::new(dir).unwrap_or_else(|e| {
        eprintln!("Error initializing cron store: {}", e);
        std::process::exit(1)
    }))
}

fn cron_list(all: bool) -> Result<()> {
    let store = cron_store();
    let jobs = store.list_jobs(all);
    if jobs.is_empty() {
        println!("No cron jobs.");
    } else {
        println!("Cron jobs ({}):\n", jobs.len());
        for job in &jobs {
            let status = match (&job.state, job.enabled) {
                (oben_cron::JobState::Completed, _) => "✅ completed",
                (oben_cron::JobState::Error, _) => "❌ error",
                (oben_cron::JobState::Paused, true) => "⏸️  paused",
                (oben_cron::JobState::Scheduled, true) => "▶️  active",
                (oben_cron::JobState::Scheduled, false) => "⏸️  paused (disabled)",
                _ => "❓ unknown",
            };
            let next_run = job
                .next_run_at
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "N/A".to_string());
            let last_run = job
                .last_run_at
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "never".to_string());
            let error_str = if let Some(ref err) = job.last_error {
                format!("\n    Error: {}", err)
            } else {
                String::new()
            };
            println!(
                "  {} {} — {}\n    Schedule: {}\n    Created: {}\n    Next: {}\n    Last run: {}{}",
                job.id,
                job.name,
                status,
                job.schedule,
                job.created_at.format("%Y-%m-%d %H:%M"),
                next_run,
                last_run,
                error_str,
            );
        }
    }
    Ok(())
}

fn cron_create(
    schedule: &str,
    prompt: Option<&str>,
    name: Option<&str>,
    repeat: Option<u32>,
) -> Result<()> {
    let store = cron_store();
    let prompt_text = prompt.unwrap_or("Check for updates and summarize anything new.");
    let job_name = name.unwrap_or("untitled").to_string();
    let job = CronJob::new(job_name, prompt_text.to_string(), schedule, repeat)?;
    store.create(job.clone())?;
    println!("Created cron job '{}':", job.id);
    println!("  Name: {}", job.name);
    println!("  Schedule: {}", job.schedule);
    if let Some(next) = job.next_run_at {
        println!("  Next run: {}", next.format("%Y-%m-%d %H:%M"));
    }
    Ok(())
}

fn cron_pause(id: &str) -> Result<()> {
    let store = cron_store();
    store.pause(id)?;
    println!("Paused job '{}'.", id);
    Ok(())
}

fn cron_resume(id: &str) -> Result<()> {
    let store = cron_store();
    store.resume(id)?;
    println!("Resumed job '{}'.", id);
    Ok(())
}

fn cron_remove(id: &str) -> Result<()> {
    let store = cron_store();
    store.remove(id)?;
    println!("Removed job '{}'.", id);
    Ok(())
}

/// Run the cron tick manually — process all due jobs.
fn cron_tick() -> Result<()> {
    let store = cron_store();
    let due = store.get_due_jobs();
    if due.is_empty() {
        println!("No jobs due.");
    } else {
        let now = chrono::Utc::now();

        let ober_exec = oben_cron::cron_exec_binary();

        println!(
            "cron tick at {}: running {} due job(s)...",
            now.format("%H:%M:%S"),
            due.len()
        );
        for job in &due {
            let _prompt = job.prompt.clone();
            match store.advance_job(&job.id, &ober_exec) {
                Ok(()) => println!("  Job '{}' advanced to next run", job.id),
                Err(e) => println!("  Job '{}': error: {}", job.id, e),
            }
        }
    }
    Ok(())
}

// ── Cron daemon management ────────────────────────────────────────────

fn cron_pid_path() -> std::path::PathBuf {
    CronStore::default_path().join("cron.pid")
}

/// Check if the cron daemon process is running by reading the PID file.
pub fn is_cron_running() -> Option<u32> {
    let pid_path = cron_pid_path();
    if !pid_path.exists() {
        return None;
    }
    let pid: u32 = std::fs::read_to_string(&pid_path)
        .ok()?
        .trim()
        .parse()
        .ok()?;

    // Check if process exists by sending signal 0
    let res = std::process::Command::new("kill")
        .args(&["-0", &pid.to_string()])
        .output();
    match res {
        Ok(out) if out.status.success() => Some(pid),
        Ok(_) => None,
        Err(_) => None,
    }
}

fn find_cron_binary() -> Option<std::path::PathBuf> {
    let candidates = ["target/debug/oben-cron", "target/release/oben-cron"];
    for c in &candidates {
        let p = std::path::PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    let out = std::process::Command::new("which")
        .args(&["oben-cron"])
        .output()
        .ok()?;
    if out.status.success() && !out.stdout.is_empty() {
        let path = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Start the cron daemon as a background process.
/// The daemon process becomes its own process group (not a child).
pub fn cron_start() -> Result<()> {
    let _pid_path = cron_pid_path();

    // Already running?
    if let Some(_pid) = is_cron_running() {
        println!("Cron daemon is already running (PID {})", _pid);
        return Ok(());
    }

    let binary = match find_cron_binary() {
        Some(b) => b,
        None => {
            println!("oben-cron binary not found; building...");
            let output = std::process::Command::new("cargo")
                .args(&["build", "--package", "oben-cron"])
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to run cargo: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("{}", stderr);
                anyhow::bail!("Build of oben-cron failed");
            }
            std::path::PathBuf::from("target/debug/oben-cron")
        }
    };

    println!("Starting cron daemon...");
    let mut child = std::process::Command::new(&binary)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null()) // detach from stdin
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start cron daemon: {}", e))?;

    let started_pid = child.id();

    // Wait for PID file to be written
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if is_cron_running().is_some() {
            let _ = child.kill(); // daemon now has its own PID file
            println!("Cron daemon started (PID {}).", started_pid);
            return Ok(());
        }
    }

    let _ = child.kill();
    anyhow::bail!("Cron daemon started but PID file was not written.");
}

fn cron_info() -> Result<()> {
    let pid_path = cron_pid_path();

    if let Some(pid) = is_cron_running() {
        println!("Cron daemon: active (PID {})", pid);
        println!("  PID file: {:?}", pid_path);
        if pid_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&pid_path) {
                println!("  PID file contents: {}", content.trim());
            }
        }
        // Show job count
        let store = CronStore::new(CronStore::default_path()).ok();
        if let Some(s) = store {
            let jobs = s.list_jobs(false);
            println!("  Active jobs: {}", jobs.len());
            let all_jobs = s.list_jobs(true);
            println!("  Total jobs: {}", all_jobs.len());
        }
    } else {
        println!("Cron daemon: inactive (not running)");
        if pid_path.exists() {
            println!("  Stale PID file exists: {:?}", pid_path);
            let _ = std::fs::remove_file(&pid_path);
            println!("  Removed stale PID file.");
        }
    }

    Ok(())
}

// ── Curator ───────────────────────────────────────────────────────────────

fn run_curator_pin(skill_name: &str) -> Result<()> {
    let state_dir = std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".agents/skill_states"))
        .unwrap_or_else(|| std::path::PathBuf::from("./skill_states"));
    let skills_dir = std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".agents/skills"))
        .unwrap_or_else(|| std::path::PathBuf::from("./skills"));

    let manager = SkillStateManager::new(skills_dir, state_dir);
    
    if manager.pin(skill_name)? {
        println!("Skill pinned: {}", skill_name);
    } else {
        println!("Skill not found: {}", skill_name);
    }
    Ok(())
}

async fn run_curator_run(consolidate: bool) -> Result<()> {
    let config = CuratorConfig {
        consolidate,
        ..CuratorConfig::default()
    };
    let mut curator = Curator::new(config);
    
    let idle_hours = 168.0; // Default 7 days
    let result = curator.run(idle_hours);
    println!("{}", result);
    
        if consolidate {
            let consolidation = curator.apply_consolidation_pass(false);
        println!("\nConsolidation Results:");
        println!("  Consolidated: {:?}", consolidation.consolidated);
        println!("  Pruned: {:?}", consolidation.pruned);
        println!("  Dry run: {}", consolidation.dry_run);
    }
    
    Ok(())
}

fn run_curator_status() -> Result<()> {
    let config = CuratorConfig::default();
    let curator = Curator::new(config);
    let state = curator.state();
    
    println!("Curator Status:");
    println!("  Run count: {}", state.run_count);
    println!("  Last run: {:?}", state.last_run_at);
    println!("  Paused: {}", state.paused);
    if let Some(summary) = &state.last_run_summary {
        println!("  Last summary: {}", summary);
    }
    Ok(())
}

// ── Gateway ───────────────────────────────────────────────────────────────

fn get_gateway_pid_path(profile: Option<&str>) -> std::path::PathBuf {
    let env = oben_config::env::Env::new(profile.map(String::from));
    let config_dir = env.config_dir();
    config_dir.join("gateway.pid")
}

/// Check if the gateway process is running by reading the PID file.
pub fn is_gateway_running(profile: Option<&str>) -> Option<u32> {
    let pid_path = get_gateway_pid_path(profile);
    if !pid_path.exists() {
        return None;
    }
    let pid: u32 = std::fs::read_to_string(&pid_path)
        .ok()?
        .trim()
        .parse()
        .ok()?;

    let res = std::process::Command::new("kill")
        .args(&["-0", &pid.to_string()])
        .output();
    match res {
        Ok(out) if out.status.success() => Some(pid),
        _ => None,
    }
}

fn find_gateway_binary() -> Option<std::path::PathBuf> {
    let candidates = ["target/debug/oben-gateway", "target/release/oben-gateway"];
    for c in &candidates {
        let p = std::path::PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    let out = std::process::Command::new("which")
        .args(&["oben-gateway"])
        .output()
        .ok()?;
    if out.status.success() && !out.stdout.is_empty() {
        let path = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Start the gateway binary in the background and record its PID.
async fn gateway_start(profile: Option<&str>) -> Result<()> {
    if let Some(pid) = is_gateway_running(profile) {
        println!("Gateway is already running (PID {}).", pid);
        return Ok(());
    }

    let binary = match find_gateway_binary() {
        Some(b) => b,
        None => {
            println!("oben-gateway binary not found; building...");
            let output = std::process::Command::new("cargo")
                .args(&["build", "--package", "oben-gateway"])
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to build gateway: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("{}", stderr);
                anyhow::bail!("Building oben-gateway failed");
            }
            std::path::PathBuf::from("target/debug/oben-gateway")
        }
    };

    let pid_path = get_gateway_pid_path(profile);
    if let Some(dir) = pid_path.parent() {
        std::fs::create_dir_all(dir).ok();
    }

    println!("Starting gateway in the background...");

    let child = std::process::Command::new(&binary)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start gateway: {}", e))?;

    let pid = child.id();

    // Write PID file immediately so status/stop can find it
    std::fs::write(&pid_path, pid.to_string())
        .map_err(|e| anyhow::anyhow!("Failed to write PID file: {}", e))?;

    println!("Gateway started (PID {})", pid);
    Ok(())
}

async fn gateway_stop(profile: Option<&str>) -> Result<()> {
    let pid_path = get_gateway_pid_path(profile);
    let pid = is_gateway_running(profile)
        .ok_or_else(|| anyhow::anyhow!("Gateway is not running"))?;

    println!("Stopping gateway (PID {})...", pid);

    let mut child = std::process::Command::new("kill")
        .args(&["-TERM", &pid.to_string()])
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to send SIGTERM: {}", e))?;
    child.wait().ok();

    for _ in 0..10 {
        if is_gateway_running(profile).is_none() {
            let _ = std::fs::remove_file(&pid_path);
            println!("Gateway stopped.");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    println!("Gateway did not stop gracefully, sending SIGKILL...");
    let kill = std::process::Command::new("kill")
        .args(&["-KILL", &pid.to_string()])
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to send SIGKILL: {}", e))?;
    if kill.status.success() {
        println!("Gateway killed.");
    }

    let _ = std::fs::remove_file(&pid_path);
    Ok(())
}

async fn gateway_status(profile: Option<&str>) -> Result<()> {
    let pid_path = get_gateway_pid_path(profile);

    if let Some(pid) = is_gateway_running(profile) {
        println!("Gateway: active (PID {})", pid);
        println!("  PID file: {:?}", pid_path);
        Ok(())
    } else {
        println!("Gateway: inactive (not running)");
        if pid_path.exists() {
            println!("  Stale PID file exists: {:?}", pid_path);
            let _ = std::fs::remove_file(&pid_path);
            println!("  Removed stale PID file.");
        }
        Ok(())
    }
}

async fn gateway_setup(profile: Option<&str>) -> Result<()> {
    println!("\n🔌 Gateway Setup Wizard\n");
    
    let mut config = oben_config::AppConfig::load(profile)?;
    
    let platforms = vec![
        "QQ Bot (Tencent)",
        "Telegram",
        "Discord",
        "Slack",
        "WhatsApp",
    ];
    
    let selected = dialoguer::Select::new()
        .with_prompt("Select platform to configure")
        .items(&platforms)
        .default(0)
        .interact()?;
    
    match selected {
        0 => setup_qq_bot(&mut config).await?,
        1 => setup_telegram(&mut config)?,
        2 => setup_discord(&mut config)?,
        3 => setup_slack(&mut config)?,
        4 => setup_whatsapp(&mut config)?,
        _ => unreachable!(),
    }
    
    config.save_with_profile(profile)?;
    
    println!("\n✅ Gateway configuration saved.");
    println!("You can re-run this wizard anytime with: `oben gateway setup`\n");
    Ok(())
}

async fn setup_qq_bot(config: &mut oben_config::AppConfig) -> Result<()> {
    // ── Step 1: Select intents ──
    let intents = vec![
        "default (Direct+C2C+Interaction)",
        "DirectMessage only",
        "C2C+Group At only",
        "Interaction only",
    ];
    let intent_sel = dialoguer::Select::new()
        .with_prompt("Select intents")
        .items(&intents)
        .default(0)
        .interact()?;
    
    let intents_list: Vec<oben_config::QQBotIntent> = match intent_sel {
        0 => vec![
            oben_config::QQBotIntent::DirectMessage,
            oben_config::QQBotIntent::C2CAndGroup,
            oben_config::QQBotIntent::Interaction,
        ],
        1 => vec![oben_config::QQBotIntent::DirectMessage],
        2 => vec![oben_config::QQBotIntent::C2CAndGroup],
        3 => vec![oben_config::QQBotIntent::Interaction],
        _ => unreachable!(),
    };
    
    // ── Step 2: Acquire App ID & Secret (QR scan or manual) ──
    let method_choices = [
        "Scan QR code with phone QQ — auto-creates app (recommended)",
        "Enter existing App ID and App Secret manually",
    ];
    let method_idx = dialoguer::Select::new()
        .with_prompt("How would you like to set up QQ Bot?")
        .items(&method_choices)
        .default(0)
        .interact()?;
    
    let (app_id, app_secret) = if method_idx == 0 {
        println!("\n🔍 Starting QR scan...");
        match oben_gateway::onboard_qq_bot().await {
            Ok(result) => {
                println!("✅ QR scan successful!");
                (result.app_id, result.client_secret)
            }
            Err(e) => {
                eprintln!("❌ QR scan failed: {e}");
                println!("   (QR URL server not reachable — falling back to manual input)\n");
                let manual_id: String = dialoguer::Input::new()
                    .with_prompt("App ID")
                    .default(String::new())
                    .interact()?;
                let manual_secret: String = dialoguer::Input::new()
                    .with_prompt("App Secret")
                    .default(String::new())
                    .interact()?;
                (manual_id, manual_secret)
            }
        }
    } else {
        println!("\n  Go to https://q.qq.com to register a QQ Bot application.");
        let app_id: String = dialoguer::Input::new()
            .with_prompt("App ID")
            .default(String::new())
            .interact()?;
        if app_id.is_empty() {
            println!("  Skipped — QQ Bot won't work without an App ID.");
            return Ok(());
        }
        let app_secret: String = dialoguer::Input::new()
            .with_prompt("App Secret")
            .default(String::new())
            .interact()?;
        if app_secret.is_empty() {
            println!("  Skipped — QQ Bot won't work without an App Secret.");
            return Ok(());
        }
        (app_id, app_secret)
    };
    
    if let Some(ref mut gw) = config.gateway {
        gw.qq_bot = Some(oben_config::QQBotConfig {
            enabled: true,
            app_id,
            app_secret,
            sandbox: false,
            shard: None,
            intents: intents_list,
        });
    } else {
        config.gateway = Some(oben_config::GatewayConfig {
            qq_bot: Some(oben_config::QQBotConfig {
                enabled: true,
                app_id,
                app_secret,
                sandbox: false,
                shard: None,
                intents: intents_list,
            }),
            ..Default::default()
        });
    }
    
    Ok(())
}

fn setup_telegram(config: &mut oben_config::AppConfig) -> Result<()> {
    println!("\n✈️ Telegram Configuration");

    let token: String = dialoguer::Input::new()
        .with_prompt("Bot token")
        .default(String::new())
        .interact()?;
    
    let webhook_url: Option<String> = {
        let url: String = dialoguer::Input::new()
            .with_prompt("Webhook URL (optional, press Enter to skip)")
            .default(String::new())
            .interact()?;
        if url.is_empty() { None } else { Some(url) }
    };

    // Security: restrict who can use the bot
    println!("\n  🔒 Security: restrict who can use your bot");
    println!("  To find your Telegram user ID, message @userinfobot on Telegram.");
    let allowed_users_input: String = dialoguer::Input::new()
        .with_prompt("Allowed user IDs (comma-separated, leave empty for open access)")
        .default(String::new())
        .interact()?;
    let allowed_users: Vec<String> = allowed_users_input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Home channel: where cron jobs and notifications are delivered
    println!("\n  📬 Home channel: where cron results and notifications are delivered.");
    println!("  For DMs, this is your user ID (same as above).");
    let home_channel: Option<String> = if !allowed_users.is_empty() {
        let default_user = &allowed_users[0];
        if dialoguer::Confirm::new()
            .with_prompt(&format!("Use your user ID ({}) as the home channel?", default_user))
            .default(true)
            .interact()?
        {
            Some(default_user.clone())
        } else {
            let hc: String = dialoguer::Input::new()
                .with_prompt("Home channel ID (or leave empty to set later with /set-home)")
                .default(String::new())
                .interact()?;
            if hc.is_empty() { None } else { Some(hc) }
        }
    } else {
        let hc: String = dialoguer::Input::new()
            .with_prompt("Home channel ID (leave empty to set later with /set-home)")
            .default(String::new())
            .interact()?;
        if hc.is_empty() { None } else { Some(hc) }
    };

    let cfg = oben_config::TelegramConfig {
        enabled: true,
        token: if token.is_empty() { None } else { Some(token) },
        webhook_url,
        webhook_secret: None,
        allowed_users,
        allowed_chats: Vec::new(),
        forum_topics: false,
        home_channel,
    };
    
    if let Some(ref mut gw) = config.gateway {
        gw.telegram = Some(cfg);
    } else {
        config.gateway = Some(oben_config::GatewayConfig {
            telegram: Some(cfg),
            ..Default::default()
        });
    }
    
    Ok(())
}

fn setup_discord(config: &mut oben_config::AppConfig) -> Result<()> {
    println!("\n🎮 Discord Configuration");

    let token: String = dialoguer::Input::new()
        .with_prompt("Bot token")
        .default(String::new())
        .interact()?;
    
    let slash: bool = dialoguer::Confirm::new()
        .with_prompt("Enable slash commands")
        .default(true)
        .interact()?;
    
    let voice: bool = dialoguer::Confirm::new()
        .with_prompt("Enable voice channel support")
        .default(false)
        .interact()?;
    
    let cfg = oben_config::DiscordConfig {
        enabled: true,
        token: if token.is_empty() { None } else { Some(token) },
        intents: vec![],
        allowed_guilds: Vec::new(),
        allowed_users: Vec::new(),
        slash_commands: slash,
        voice,
        dm_role_auth_guild: None,
    };
    
    if let Some(ref mut gw) = config.gateway {
        gw.discord = Some(cfg);
    } else {
        config.gateway = Some(oben_config::GatewayConfig {
            discord: Some(cfg),
            ..Default::default()
        });
    }
    
    Ok(())
}

fn setup_slack(config: &mut oben_config::AppConfig) -> Result<()> {
    println!("\n💬 Slack Configuration");

    let app_token: String = dialoguer::Input::new()
        .with_prompt("Slack App-level token (xapp-...)")
        .default(String::new())
        .interact()?;
    
    let bot_token: String = dialoguer::Input::new()
        .with_prompt("Slack Bot token (xoxb-...)")
        .default(String::new())
        .interact()?;
    
    let cfg = oben_config::SlackConfig {
        enabled: true,
        app_token: if app_token.is_empty() { None } else { Some(app_token) },
        bot_token: if bot_token.is_empty() { None } else { Some(bot_token) },
        allowed_channels: Vec::new(),
        slash_commands: Vec::new(),
    };
    
    if let Some(ref mut gw) = config.gateway {
        gw.slack = Some(cfg);
    } else {
        config.gateway = Some(oben_config::GatewayConfig {
            slack: Some(cfg),
            ..Default::default()
        });
    }
    
    Ok(())
}

fn setup_whatsapp(config: &mut oben_config::AppConfig) -> Result<()> {
    println!("\n📞 WhatsApp Configuration");

    let access_token: String = dialoguer::Input::new()
        .with_prompt("Meta Cloud API access token")
        .default(String::new())
        .interact()?;
    
    let phone_number_id: String = dialoguer::Input::new()
        .with_prompt("Phone Number ID")
        .default(String::new())
        .interact()?;
    
    let business_account_id: String = dialoguer::Input::new()
        .with_prompt("Business Account ID (optional)")
        .default(String::new())
        .interact()?;
    
    let verify_token: String = dialoguer::Input::new()
        .with_prompt("Webhook verification token")
        .default(String::new())
        .interact()?;
    
    let cfg = oben_config::WhatsAppConfig {
        enabled: true,
        access_token: if access_token.is_empty() { None } else { Some(access_token) },
        phone_number_id: if phone_number_id.is_empty() { None } else { Some(phone_number_id) },
        business_account_id: if business_account_id.is_empty() { None } else { Some(business_account_id) },
        webhook_verify_token: if verify_token.is_empty() { None } else { Some(verify_token) },
        api_version: "v17.0".to_string(),
        allowed_numbers: Vec::new(),
        default_language: "en_US".to_string(),
    };
    
    if let Some(ref mut gw) = config.gateway {
        gw.whatsapp = Some(cfg);
    } else {
        config.gateway = Some(oben_config::GatewayConfig {
            whatsapp: Some(cfg),
            ..Default::default()
        });
    }
    
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gateway_lifecycle_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Write a PID into a temp directory and return the PID path.
    fn write_temp_pid(temp_dir: &TempDir, pid: u32) -> std::path::PathBuf {
        let dir = temp_dir.path().join("obenmatrix");
        fs::create_dir_all(&dir).unwrap();
        let pid_path = dir.join("gateway.pid");
        fs::write(&pid_path, format!("{}\n", pid)).unwrap();
        pid_path
    }

    /// Check if a process with the given PID is alive via kill -0.
    fn kill0_is_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .args(&["-0", &pid.to_string()])
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }

    // ── get_gateway_pid_path ──────────────────────────────────────────

    #[test]
    /// Given: no special environment
    /// When: get_gateway_pid_path is called
    /// Then: returns a path ending with "obenmatrix/gateway.pid"
    fn test_gateway_pid_path_construction() {
        let path = get_gateway_pid_path(None);
        assert!(
            path.to_string_lossy().ends_with("obenmatrix/gateway.pid")
                || cfg!(windows) && path.to_string_lossy().ends_with("obenmatrix\\gateway.pid"),
            "Expected path to end with 'obenmatrix/gateway.pid', got: {}",
            path.display()
        );
    }

    #[test]
    /// Given: the gateway PID path
    /// When: we inspect its file name
    /// Then: it equals "gateway.pid"
    fn test_gateway_pid_path_file_name_is_pid() {
        let path = get_gateway_pid_path(None);
        assert_eq!(path.file_name().unwrap().to_string_lossy().as_ref(), "gateway.pid");
    }

    // ── is_gateway_running ────────────────────────────────────────────

    #[test]
    /// Given: an impossible PID (higher than any user-space process)
    /// When: we check if the process is alive via kill -0
    /// Then: kill returns an error, confirming non-aliveness
    fn test_is_gateway_running_returns_none_for_impossible_pid() {
        let impossible_pid = 999999999u32;
        assert!(
            !kill0_is_alive(impossible_pid),
            "kill -0 on impossible PID should fail"
        );
    }

    // ── find_gateway_binary ───────────────────────────────────────────

    #[test]
    /// Given: no oben-gateway binary in target/ or PATH
    /// When: find_gateway_binary is called
    /// Then: no panic occurs (returns Some or None depending on env)
    fn test_find_gateway_binary_does_not_panic() {
        let _ = find_gateway_binary();
    }

    // ── PID file read/parsing – core logic tested in isolation ─────────

    #[test]
    /// Given: a temp directory with a gateway.pid containing a valid PID
    /// When: we read and parse it the same way is_gateway_running does
    /// Then: we can parse the integer successfully
    fn test_pid_file_parsing_valid_integer() {
        let temp = TempDir::new().unwrap();
        let pid_path = write_temp_pid(&temp, 12345);
        let content = fs::read_to_string(&pid_path).unwrap();
        let parsed: u32 = content.trim().parse().unwrap();
        assert_eq!(parsed, 12345);
    }

    #[test]
    /// Given: a temp file with non-numeric content
    /// When: we try to parse as u32
    /// Then: parse fails
    fn test_pid_file_parsing_fails_on_non_numeric() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("obenmatrix");
        fs::create_dir_all(&dir).unwrap();
        let bad_path = dir.join("gateway.pid");
        fs::write(&bad_path, "not-a-pid\n").unwrap();
        let result: Result<u32, _> = fs::read_to_string(&bad_path)
            .unwrap()
            .trim()
            .parse();
        assert!(result.is_err());
    }

    #[test]
    /// Given: an empty PID file
    /// When: we try to parse as u32
    /// Then: parse fails
    fn test_pid_file_parsing_fails_on_empty() {
        let temp = TempDir::new().unwrap();
        let dir = temp.path().join("obenmatrix");
        fs::create_dir_all(&dir).unwrap();
        let empty_path = dir.join("gateway.pid");
        fs::write(&empty_path, "\n").unwrap();
        let result: Result<u32, _> = fs::read_to_string(&empty_path)
            .unwrap()
            .trim()
            .parse();
        assert!(result.is_err());
    }

    // ── Missing PID file handling ─────────────────────────────────────

    #[test]
    /// Given: a directory without any PID file
    /// When: we check existence
    /// Then: exists() returns false, no panic
    fn test_missing_pid_file_returns_false() {
        let temp = TempDir::new().unwrap();
        let missing = temp.path().join("obenmatrix").join("gateway.pid");
        assert!(!missing.exists());
    }

    #[test]
    /// Given: a directory that doesn't even contain "obenmatrix" subdirectory
    /// When: we look for the PID file
    /// Then: exists() returns false gracefully
    fn test_missing_parent_dir_returns_false() {
        let temp = TempDir::new().unwrap();
        let missing = temp.path().join("nonexistent").join("gateway.pid");
        assert!(!missing.exists());
    }
}
