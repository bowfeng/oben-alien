//! Clap CLI definitions.
//!
//! All argument structs and subcommand enums are defined here.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "oben", version, about = "The self-improving AI agent")]
pub struct Cli {
    /// Enable verbose/debug output
    #[arg(short, long)]
    pub verbose: bool,

    /// Named profile to use (creates/loads a isolated config+data directory)
    #[arg(long)]
    pub profile: Option<String>,

    #[arg(long, help = "Enable Mixture of Agents (MoA) coordination mode")]
    pub moa: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start an interactive conversation
    Chat {
        /// Disable streaming
        #[arg(long)]
        no_stream: bool,
        /// Continue an existing session by ID or name.
        /// Without a value, continues the most recent session.
        #[arg(short, long = "continue", num_args=0..=1, default_missing_value="latest")]
        continue_session: Option<String>,
        /// Search provider to use: duckduckgo, brave, bing, google
        #[arg(long)]
        search_provider: Option<String>,
        /// Agent name for session isolation
        #[arg(long)]
        agent: Option<String>,
    },
    /// Run a one-shot prompt
    Run {
        /// The prompt/question
        #[arg(short, long)]
        prompt: String,
        /// Stream text output as it arrives
        #[arg(long)]
        stream: bool,
        /// Search provider to use: duckduckgo, brave, bing, google
        #[arg(long)]
        search_provider: Option<String>,
        /// Agent name for session isolation
        #[arg(long)]
        agent: Option<String>,
    },
    /// Setup/configure the agent
    Setup,
    /// Show current configuration
    Config {
        #[command(subcommand)]
        action: ConfigCommand,
    },
    /// List available tools
    Tools,
    /// List and manage skills
    Skills,
    /// List and manage sessions
    Sessions {
        #[command(subcommand)]
        action: Option<SessionsCommand>,
    },
    /// Discover models via LLM provider
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
    /// Start the terminal UI
    Tui {
        /// Session name or ID to load on startup
        #[arg(short, long)]
        session: Option<String>,
        /// Agent name for session isolation
        #[arg(long)]
        agent: Option<String>,
    },
    /// Manage scheduled cron tasks
    Cron {
        #[command(subcommand)]
        action: Option<CronCommand>,
    },
    /// Manage goals
    Goals {
        #[command(subcommand)]
        action: Option<GoalCommand>,
    },
    /// Manage the gateway process
    Gateway {
        #[command(subcommand)]
        #[command(name = "gateway")]
        action: Option<GatewayCommand>,
    },
    /// Skill curator - lifecycle management
    Curator {
        #[command(subcommand)]
        action: CuratorCommand,
    },
}

#[derive(Subcommand)]
pub enum CronCommand {
    /// List all scheduled jobs
    List {
        /// Show disabled/paused jobs too
        #[arg(long)]
        all: bool,
    },
    /// Create a new cron job
    Create {
        /// Schedule: "30m", "every 30m", "0 9 * * *", or ISO timestamp
        #[arg(short, long)]
        schedule: String,
        /// Prompt/question for the agent
        #[arg(short, long)]
        prompt: Option<String>,
        /// Job name
        #[arg(short, long)]
        name: Option<String>,
        /// Run N times (default: forever)
        #[arg(short, long)]
        repeat: Option<u32>,
    },
    /// Pause a cron job by ID
    Pause {
        /// Job ID
        id: String,
    },
    /// Resume a paused cron job by ID
    Resume {
        /// Job ID
        id: String,
    },
    /// Remove a cron job by ID
    Remove {
        /// Job ID
        id: String,
    },
    /// Run the cron tick manually
    Tick,
    /// Start the cron daemon process
    Start,
    /// Show daemon info
    Info,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Show current config
    Show,
    /// Edit config file
    Edit,
}

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// List available models from the LLM provider
    List,
    /// Show details for a specific model
    Info {
        /// Model ID to look up
        model: String,
    },
}

#[derive(Subcommand)]
pub enum SessionsCommand {
    /// List all sessions
    List,
    /// Compact (compress) a session using LLM summarization
    Compact {
        /// Session ID or name to compact
        #[arg(short, long)]
        session: Option<String>,
        /// Focus topic — prioritise preserving info related to this topic
        #[arg(short, long)]
        focus: Option<String>,
    },
    /// Delete a session
    Delete {
        /// Session ID or name to delete
        #[arg(short, long)]
        session: String,
    },
    /// Dump session messages to a JSON file
    Dump {
        /// Session ID or name (optional, defaults to active session)
        #[arg(short, long)]
        session: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum GoalCommand {
    /// Create a new goal
    Start {
        /// The goal statement
        goal: String,
        /// Maximum turns before auto-pause (default: 20)
        #[arg(long)]
        max_turns: Option<usize>,
    },
    /// List goals by status
    List {
        /// Filter by status: active, done, paused, failed
        #[arg(long)]
        status: Option<String>,
    },
    /// Show goal status
    Status {
        /// Goal ID
        goal_id: String,
    },
    /// Pause a goal
    Pause {
        /// Goal ID or "active" to pause the current active goal
        id: String,
    },
    /// Resume a paused goal
    Resume {
        /// Goal ID or "active" to resume the current active goal
        id: String,
        /// Reset the turn budget counter
        #[arg(long)]
        reset: bool,
    },
    /// Clear (delete) a goal
    Clear {
        /// Goal ID or "active" to clear the current active goal
        id: String,
    },
}

#[derive(Subcommand)]
pub enum GatewayCommand {
    Start,
    Stop,
    Status,
    Setup,
}

#[derive(Subcommand)]
pub enum CuratorCommand {
    /// Pin a skill (prevent auto-archival)
    Pin {
        /// Skill name to pin
        skill: String,
    },
    /// Run the curator review
    Run {
        /// Enable LLM consolidation pass (LLM-powered umbrella building)
        #[arg(long)]
        consolidate: bool,
    },
    /// Show curator status
    Status,
}
