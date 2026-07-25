use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use oben_models::ProviderConfig;

pub use oben_models::SessionStoreKind;
pub use crate::mixture_of_agents::MixtureOfAgentsConfig;

/// All application settings, stored in ~/.config/obenmatrix/config.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub model: ProviderConfig,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub max_iterations: Option<usize>,
    pub max_spawn_depth: Option<usize>,
    pub max_concurrent_tasks: Option<usize>,
    pub tools: ToolsConfig,
    pub skills: SkillsConfig,
    pub gateway: Option<GatewayConfig>,
    pub display: DisplayConfig,
    pub context: ContextConfig,
    pub voice: VoiceConfig,
    pub providers: Vec<ProviderConfig>,
    pub custom_providers: Vec<String>,
    pub vision: VisionConfig,
    pub session_store: SessionStoreKind,
    pub retry: RetryConfig,
    pub concurrency: ConcurrencyConfig,
    pub hooks: HooksConfig,
    pub fallback_models: Vec<FallbackConfig>,
    pub agent: AgentConfig,
    pub agents: Vec<NamedAgentConfig>,
    pub events: EventsConfig,
    pub compaction: CompactionConfig,
    pub memory: MemoryConfig,
    pub search: SearchConfig,
    pub mixture_of_agents: MixtureOfAgentsConfig,
}

/// Configuration for session context compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactionConfig {
    /// Number of recent turns to keep verbatim during compaction. Default: 2.
    pub tail_turns: usize,
    /// Token budget for recent preservation. If None, computed as 25% of usable input, clamped to [2000, 8000]. Default: None.
    pub preserve_recent_tokens: Option<usize>,
    /// Whether to inject an auto-continue prompt after compaction. Default: false.
    pub auto_continue: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            tail_turns: 2,
            preserve_recent_tokens: None,
            auto_continue: false,
        }
    }
}

/// Configuration for the memory provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Whether the memory provider is enabled. Default: true.
    pub enabled: bool,
    /// Provider identifier (e.g. "builtin", "sqlite"). Default: "builtin".
    #[serde(default = "default_memory_provider")]
    pub provider: String,
}

fn default_memory_provider() -> String {
    "builtin".to_string()
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: default_memory_provider(),
        }
    }
}

/// Search provider selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchProviderKind {
    Disabled,
    DuckDuckGo,
    Brave,
    Bing,
    Google,
}

impl Default for SearchProviderKind {
    fn default() -> Self {
        SearchProviderKind::DuckDuckGo
    }
}

impl SearchProviderKind {
    pub fn is_disabled(&self) -> bool {
        matches!(self, SearchProviderKind::Disabled)
    }
}

/// Configuration for search tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Search provider to use. Default: "duckduckgo".
    #[serde(default = "default_search_provider")]
    pub provider: SearchProviderKind,
    /// API key for paid providers (Brave, Bing, Google). Default: None.
    pub api_key: Option<String>,
}

fn default_search_provider() -> SearchProviderKind {
    SearchProviderKind::DuckDuckGo
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            provider: default_search_provider(),
            api_key: None,
        }
    }
}

/// Configuration for vision/image analysis.
/// When set, vision tools call the specified API with downloaded images.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// API provider: "openai" or "anthropic".
    #[serde(default = "default_vision_provider")]
    pub provider: String,
    /// Base URL for the API (use OpenRouter-compatible URL for multi-provider).
    pub base_url: Option<String>,
    /// API key for the vision provider.
    pub api_key: Option<String>,
    /// Model name for vision analysis.
    pub model: Option<String>,
    /// Max tokens for the analysis response.
    pub max_tokens: Option<usize>,
}

fn default_vision_provider() -> String {
    "openai".to_string()
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            provider: default_vision_provider(),
            base_url: None,
            api_key: None,
            model: None,
            max_tokens: Some(1024),
        }
    }
}

/// Configuration for voice (STT + TTS) tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceConfig {
    /// STT (speech-to-text) configuration.
    pub stt: SttConfig,
    /// TTS (text-to-speech) configuration.
    pub tts: TtsConfig,
}

/// STT provider selection.
/// Returns the default provider name ("whisper-rs").
fn default_stt_provider() -> String {
    String::from("whisper-rs")
}

/// STT configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    /// Provider: "whisper-rs" (local GGML), "openai", "groq", "mistral", "xai", "elevenlabs".
    #[serde(default = "default_stt_provider")]
    pub provider: String,
    /// Model name (provider-specific). Defaults vary by provider.
    pub model: Option<String>,
    /// Language override (ISO 639-1 code, e.g. "en", "zh", "de"). Auto-detect if None.
    pub language: Option<String>,
    // Provider-specific sub-configs
    /// OpenAI-compatible STT config (used as base URL for OpenAI/Groq/Mistral/xAI/ElevenLabs).
    #[serde(flatten)]
    pub openai_like: OpenAiAudioConfig,
    /// Whisper local model path (relative to HOME or absolute).
    /// If None, uses the default "base" model downloaded from official source.
    pub model_path: Option<String>,
}

/// OpenAI-compatible audio API configuration.
/// All OpenAI-compatible providers (OpenAI itself, Groq, Mistral, xAI, ElevenLabs)
/// share the same `/v1/audio/transcriptions` API shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiAudioConfig {
    /// Base URL for the transcription API. Defaults to "https://api.openai.com/v1".
    pub base_url: Option<String>,
}

fn default_openai_audio_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

impl Default for OpenAiAudioConfig {
    fn default() -> Self {
        Self {
            base_url: Some(default_openai_audio_base_url()),
        }
    }
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            provider: default_stt_provider(),
            model: None,
            language: None,
            openai_like: OpenAiAudioConfig::default(),
            model_path: None,
        }
    }
}

/// TTS provider selection.
/// Returns the default provider name ("edge").
fn default_tts_provider() -> String {
    String::from("edge")
}

/// TTS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    /// Provider: "edge" (default, free), "openai", "elevenlabs", "gemini", "xai", "mistral".
    #[serde(default = "default_tts_provider")]
    pub provider: String,
    /// Voice name/ID for the provider.
    pub voice: Option<String>,
    /// Speed multiplier (e.g. 1.0 = normal, 0.5 = slow, 2.0 = fast).
    pub speed: Option<f64>,
    /// Output format: .mp3 (default), .ogg (Telegram voice bubbles), .wav.
    #[serde(default = "default_output_format")]
    pub output_format: String,
    /// Base URL for OpenAI-compatible providers (OpenAI, xAI). Not used by Edge/Mistral/ElevenLabs/Gemini.
    pub base_url: Option<String>,
    /// Model name for OpenAI provider.
    pub model: Option<String>,
    /// Custom command TTS configuration for local engines (Piper, KittenTTS, etc.).
    #[serde(default)]
    pub command: Option<CommandTtsConfig>,
}

fn default_output_format() -> String {
    "mp3".to_string()
}

/// Custom command TTS configuration for local engines (Piper, KittenTTS, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CommandTtsConfig {
    /// Shell command template. Hermes writes text to a temp file and runs:
    ///   {command} with {input_path}, {text_path}, {output_path}, {format} placeholders.
    pub command: String,
    /// Output format: mp3, wav, ogg, flac.
    #[serde(default = "default_cmd_tts_format")]
    pub output_format: String,
    /// If true, output is treated as voice-compatible (for Telegram voice bubbles).
    #[serde(default)]
    pub voice_compatible: bool,
}

fn default_cmd_tts_format() -> String {
    "mp3".to_string()
}

impl Default for CommandTtsConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            output_format: default_cmd_tts_format(),
            voice_compatible: false,
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            provider: default_tts_provider(),
            voice: None,
            speed: None,
            output_format: default_output_format(),
            base_url: None,          // Providers use their own defaults
            model: None,             // Providers use their own defaults
            command: None,
        }
    }
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    pub enabled: Vec<String>,
    /// Auto-enable tools by category.
    #[serde(default)]
    pub auto_detect: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: vec![],
            auto_detect: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsConfig {
    pub enabled: Vec<String>,
    pub auto_use: Vec<String>,
    /// Skill directories to scan (default: ["skills"]).
    #[serde(default = "default_skills_dirs")]
    pub dirs: Vec<String>,
}

fn default_skills_dirs() -> Vec<String> {
    vec!["skills".to_string()]
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enabled: vec![],
            auto_use: vec![],
            dirs: default_skills_dirs(),
        }
    }
}

/// QQ Bot intents as user-facing enum variants.
/// The serde rename + aliases ensure backward compatibility with old Hermes-style
/// `g_build_in_*` intent names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QQBotIntent {
    /// Direct message create/delete (1<<12)
    #[serde(rename = "direct_message", alias = "direct_message_create", alias = "g_build_in_direct_message", alias = "g_build_in_c2c_message")]
    DirectMessage,
    /// C2C and group add/remove, robot add/remove, etc. (1<<25)
    #[serde(rename = "c2c_and_group", alias = "c_c_and_group", alias = "group_and_c2c", alias = "g_build_in_guilds", alias = "g_build_in_group_at_message")]
    C2CAndGroup,
    /// Interactive message create/delete (1<<26)
    #[serde(rename = "interaction", alias = "interaction_create", alias = "i_interaction")]
    Interaction,
}

/// QQ Bot gateway configuration.
///
/// Connects to QQ Open Platform via WebSocket Gateway (wss://gw.open.q.qq.com)
/// and routes bi-directional messages through the agent system.
///
/// YAML example:
/// ```yaml
/// gateway:
///   qq_bot:
///     enabled: true
///     app_id: "123456"
///     app_secret: "secret_key"
///     intents:
///       - g_build_in_guilds
///       - g_build_in_c2c_message
///       - g_build_in_group_at_message
///     sandbox: true
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QQBotConfig {
    pub enabled: bool,
    /// QQ Open Platform AppID (bot application ID).
    pub app_id: String,
    /// QQ Open Platform AppSecret (bot application secret).
    pub app_secret: String,
    /// Event intents to subscribe to (comma-separated bitflags internally).
    #[serde(default)]
    pub intents: Vec<QQBotIntent>,
    /// Shard info: `shard_id` and `num_shards`. `None` means single-instance mode.
    #[serde(default)]
    pub shard: Option<[usize; 2]>,
    /// Use sandbox (testing) endpoints. Default: false.
    #[serde(default)]
    pub sandbox: bool,
}

// ---------------------------------------------------------------------------
// Gateway platform-specific config structs
// ---------------------------------------------------------------------------

/// Telegram platform configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub enabled: bool,
    /// Telegram bot token.
    pub token: Option<String>,
    /// Webhook URL (for webhook mode). None = use long polling.
    pub webhook_url: Option<String>,
    /// Webhook secret (for webhook mode).
    pub webhook_secret: Option<String>,
    /// Allowed user IDs (empty = all).
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Allowed chat IDs (empty = all).
    #[serde(default)]
    pub allowed_chats: Vec<String>,
    /// Forum topics support (auto-create topics for DMs on Telegram).
    #[serde(default)]
    pub forum_topics: bool,
    /// Home channel for cron job and notification delivery.
    #[serde(default)]
    pub home_channel: Option<String>,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: None,
            webhook_url: None,
            webhook_secret: None,
            allowed_users: Vec::new(),
            allowed_chats: Vec::new(),
            forum_topics: false,
            home_channel: None,
        }
    }
}

/// Discord intents that can be enabled for gateway connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscordIntent {
    #[serde(rename = "guild_messages")]
    GuildMessages,
    #[serde(rename = "guild_message_typing")]
    GuildMessageTyping,
    #[serde(rename = "direct_messages")]
    DirectMessages,
    #[serde(rename = "direct_message_typing")]
    DirectMessageTyping,
    #[serde(rename = "guild_messages_reactions")]
    GuildMessagesReactions,
    #[serde(rename = "message_content")]
    MessageContent,
}

/// Discord platform configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    pub enabled: bool,
    /// Discord bot token (from Discord Developer Portal).
    pub token: Option<String>,
    /// Gateway intents to enable.
    #[serde(default)]
    pub intents: Vec<DiscordIntent>,
    /// Guild IDs to restrict to (empty = all).
    #[serde(default)]
    pub allowed_guilds: Vec<String>,
    /// User IDs allowed to interact (empty = all).
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// Slash commands to register.
    #[serde(default)]
    pub slash_commands: bool,
    /// Voice channel support.
    #[serde(default)]
    pub voice: bool,
    /// DM role auth gateway guild ID.
    pub dm_role_auth_guild: Option<String>,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: None,
            intents: Vec::new(),
            allowed_guilds: Vec::new(),
            allowed_users: Vec::new(),
            slash_commands: false,
            voice: false,
            dm_role_auth_guild: None,
        }
    }
}

/// Slack platform configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    pub enabled: bool,
    /// Slack App-level token (xapp-...) for Socket Mode.
    pub app_token: Option<String>,
    /// Slack Bot token (xoxb-...) for REST API calls.
    pub bot_token: Option<String>,
    /// Allowed channel IDs (empty = all channels).
    #[serde(default)]
    pub allowed_channels: Vec<String>,
    /// Slash command prefixes to recognize.
    #[serde(default)]
    pub slash_commands: Vec<String>,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            app_token: None,
            bot_token: None,
            allowed_channels: Vec::new(),
            slash_commands: Vec::new(),
        }
    }
}

/// WhatsApp platform configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatsAppConfig {
    pub enabled: bool,
    /// Meta Cloud API access token.
    pub access_token: Option<String>,
    /// Phone number ID from Meta dashboard.
    pub phone_number_id: Option<String>,
    /// WhatsApp Business Account ID.
    pub business_account_id: Option<String>,
    /// Webhook verification token for validating incoming webhooks.
    pub webhook_verify_token: Option<String>,
    /// API version (default: "v17.0").
    #[serde(default = "default_api_version")]
    pub api_version: String,
    /// Allowed phone numbers (empty = all).
    #[serde(default)]
    pub allowed_numbers: Vec<String>,
    /// Template message default language.
    #[serde(default = "default_lang")]
    pub default_language: String,
}

fn default_api_version() -> String {
    "v17.0".to_string()
}

fn default_lang() -> String {
    "en_US".to_string()
}

impl Default for WhatsAppConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            access_token: None,
            phone_number_id: None,
            business_account_id: None,
            webhook_verify_token: None,
            api_version: default_api_version(),
            allowed_numbers: Vec::new(),
            default_language: default_lang(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GatewayConfig {
    pub telegram: Option<TelegramConfig>,
    pub discord: Option<DiscordConfig>,
    pub slack: Option<SlackConfig>,
    pub whatsapp: Option<WhatsAppConfig>,
    pub qq_bot: Option<QQBotConfig>,
    /// Directory containing WASM platform plugins (.wasm files).
    /// If not set, defaults to ~/.obenmatrix/plugins/wasm.
    #[serde(default)]
    pub plugin_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    pub enabled: bool,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    pub spinner_style: String,
    pub theme: String,
    pub code_block_language: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            spinner_style: "dots".to_string(),
            theme: "dark".to_string(),
            code_block_language: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Max messages to keep in context before compression.
    #[serde(default)]
    pub max_messages: Option<usize>,
    /// Compression method: "summary", "token_count", "none".
    #[serde(default = "default_compression")]
    pub compression: String,
    /// Total context window size in tokens (default: 128000).
    #[serde(default = "default_context_length")]
    pub context_length: usize,
    /// Token threshold as percentage of context_length for compaction (default: 0.75 = 75%).
    #[serde(default = "default_threshold_percent")]
    pub threshold_percent: f64,
    /// Project context file names to discover (walk to git root for first entry).
    pub files: Vec<String>,
    /// Max chars to read from context files.
    #[serde(default = "default_context_max_chars")]
    pub max_chars: usize,
}

fn default_context_max_chars() -> usize {
    20_000
}

fn default_compression() -> String {
    "summary".to_string()
}

fn default_context_length() -> usize {
    128_000
}

fn default_threshold_percent() -> f64 {
    0.75
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_messages: Some(100),
            compression: "summary".to_string(),
            context_length: default_context_length(),
            threshold_percent: default_threshold_percent(),
            files: vec![".obenmatrix.md".to_string(), "OBEN.md".to_string(), "AGENTS.md".to_string(), "CLAUDE.md".to_string(), ".cursorrules".to_string()],
            max_chars: default_context_max_chars(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetryConfig {
    #[serde(default = "default_retry_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_retry_base_delay_ms")]
    pub base_delay_ms: u64,
    #[serde(default = "default_retry_max_delay_ms")]
    pub max_delay_ms: u64,
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f64,
    pub retryable_codes: Vec<u16>,
}

fn default_retry_max_retries() -> u32 {
    3
}

fn default_retry_base_delay_ms() -> u64 {
    500
}

fn default_retry_max_delay_ms() -> u64 {
    60_000
}

fn default_retry_jitter_factor() -> f64 {
    0.5
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_retry_max_retries(),
            base_delay_ms: default_retry_base_delay_ms(),
            max_delay_ms: default_retry_max_delay_ms(),
            jitter_factor: default_retry_jitter_factor(),
            retryable_codes: vec![429, 500, 502, 503, 504],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConcurrencyConfig {
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default)]
    pub serial_only_tools: Vec<String>,
    pub destructive_tools: Vec<String>,
}

fn default_max_concurrency() -> usize {
    8
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrency: default_max_concurrency(),
            serial_only_tools: vec![],
            destructive_tools: vec![
                "write_file".to_string(),
                "patch".to_string(),
                "create_dir".to_string(),
                "delete_file".to_string(),
                "shell".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HooksConfig {
    /// Hook types enabled for this agent (e.g. ["nudge"]).
    /// Each type must have a corresponding entry in `configs`.
    #[serde(default = "default_hooks_enabled")]
    pub enabled: Vec<String>,
    /// Per-hook-type configuration keyed by hook type name.
    #[serde(default)]
    pub configs: std::collections::BTreeMap<String, serde_yaml::Value>,
}

fn default_hooks_enabled() -> Vec<String> {
    vec!["nudge".to_string()]
}

impl Default for HooksConfig {
    fn default() -> Self {
        let mut map = std::collections::BTreeMap::new();
        map.insert(
            "nudge".to_string(),
            serde_yaml::Value::Mapping(Default::default()),
        );
        Self {
            enabled: default_hooks_enabled(),
            configs: map,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_discipline: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            identity: None,
            execution_discipline: None,
        }
    }
}

impl Default for NamedAgentConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            role: String::new(),
            model: String::new(),
            tools: Vec::new(),
            execution_discipline: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NamedAgentConfig {
    pub name: String,
    pub role: String,
    pub model: String,
    pub tools: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_discipline: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EventsConfig {
    #[serde(default)]
    pub tool_events: EventFilters,
    #[serde(default)]
    pub thought_events: EventFilters,
    #[serde(default)]
    pub status_events: EventFilters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EventFilters {
    #[serde(default = "default_true")]
    pub progress: bool,
    #[serde(default = "default_true")]
    pub start_complete: bool,
    #[serde(default = "default_true")]
    pub thinking: bool,
    #[serde(default = "default_true")]
    pub reasoning: bool,
    #[serde(default = "default_true")]
    pub lifecycle: bool,
    #[serde(default = "default_true")]
    pub fallback: bool,
}

fn default_true() -> bool {
    true
}

impl Default for EventFilters {
    fn default() -> Self {
        Self {
            progress: true,
            start_complete: true,
            thinking: true,
            reasoning: true,
            lifecycle: true,
            fallback: true,
        }
    }
}

impl Default for EventsConfig {
    fn default() -> Self {
        Self {
            tool_events: EventFilters::default(),
            thought_events: EventFilters::default(),
            status_events: EventFilters::default(),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model: ProviderConfig::new(
                oben_models::ProviderKind::OpenRouter,
                "qwen/qwen3-235b:free",
            ),
            temperature: Some(0.7),
            max_tokens: Some(8192),
            max_iterations: Some(50),
            max_spawn_depth: Some(3),
            max_concurrent_tasks: Some(5),
            tools: ToolsConfig {
                enabled: vec![],
                auto_detect: true,
            },
            skills: SkillsConfig {
                enabled: vec![],
                auto_use: vec![],
                dirs: default_skills_dirs(),
            },
            gateway: None,
            display: DisplayConfig {
                spinner_style: "dots".to_string(),
                theme: "dark".to_string(),
                code_block_language: true,
            },
            context: ContextConfig {
                max_messages: Some(100),
                compression: "summary".to_string(),
                context_length: 128_000,
                threshold_percent: 0.75,
                files: vec![".obenmatrix.md".to_string(), "OBEN.md".to_string(), "AGENTS.md".to_string(), "CLAUDE.md".to_string(), ".cursorrules".to_string()],
                max_chars: default_context_max_chars(),
            },
            providers: Vec::new(),
            custom_providers: Vec::new(),
            vision: VisionConfig::default(),
            voice: VoiceConfig::default(),
            session_store: SessionStoreKind::Database,
            retry: RetryConfig::default(),
            concurrency: ConcurrencyConfig::default(),
            hooks: HooksConfig::default(),
            fallback_models: Vec::new(),
            agent: AgentConfig::default(),
            agents: Vec::new(),
             events: EventsConfig::default(),
              compaction: CompactionConfig::default(),
              memory: MemoryConfig::default(),
              search: SearchConfig::default(),
              mixture_of_agents: MixtureOfAgentsConfig::default(),
          }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults;
    use oben_models::ProviderKind;
    use std::fs;

    #[test]
    fn test_default_model_is_openrouter_qwen() {
        let config = AppConfig::default();
        assert_eq!(config.model.kind, ProviderKind::OpenRouter);
        assert_eq!(config.model.model, "qwen/qwen3-235b:free");
        assert_eq!(config.model.default_model, "qwen/qwen3-235b:free");
        assert!(config.model.api_key.is_none());
    }

    #[test]
    fn test_default_settings() {
        let config = AppConfig::default();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(8192));
        assert_eq!(config.max_iterations, Some(50));
        assert!(config.tools.auto_detect);
        assert_eq!(config.display.theme, "dark");
        assert_eq!(config.context.compression, "summary");
        assert_eq!(config.context.max_messages, Some(100));
    }

    #[test]
    fn test_default_system_prompt_not_empty() {
        let prompt = defaults::default_system_prompt();
        assert!(!prompt.trim().is_empty());
        assert!(prompt.contains("AI agent"));
        assert!(prompt.contains("tools"));
    }

    #[test]
    fn test_config_yaml_roundtrip() {
        let config = AppConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();
        // Compare core fields
        assert_eq!(restored.model.kind, config.model.kind);
        assert_eq!(restored.model.model, config.model.model);
        assert_eq!(restored.temperature, config.temperature);
        assert_eq!(restored.max_tokens, config.max_tokens);
        assert_eq!(restored.max_iterations, config.max_iterations);
        assert_eq!(restored.tools.auto_detect, config.tools.auto_detect);
        assert_eq!(restored.display.theme, config.display.theme);
        assert_eq!(restored.context.compression, config.context.compression);
    }

    #[test]
    fn test_config_yaml_roundtrip_with_gateway() {
        let mut config = AppConfig::default();
        config.gateway = Some(GatewayConfig {
            telegram: Some(TelegramConfig {
                enabled: true,
                token: Some("tg-secret-token".to_string()),
                ..Default::default()
            }),
            discord: None,
            slack: None,
            whatsapp: None,
            qq_bot: None,
            plugin_dir: None,
        });
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.gateway.is_some());
        let gw = restored.gateway.unwrap();
        let tg = gw.telegram.unwrap();
        assert!(tg.enabled);
        assert_eq!(tg.token, Some("tg-secret-token".to_string()));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.yaml");

        let config = AppConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        fs::write(&config_path, &yaml).unwrap();

        let content = fs::read_to_string(&config_path).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&content).unwrap();
        assert_eq!(restored.model.kind, config.model.kind);
        assert_eq!(restored.temperature, config.temperature);
    }

    #[test]
    fn test_minimal_config_deserialize() {
        // User's actual config only has model settings.
        // All other fields should use defaults.
        let yaml = r#"model:
  kind: Custom
  base_url: "http://10.0.0.177:8000/v1"
  model: "qwen35-local"
  default_model: "qwen35-local"
  api_key: "sk-local"
"#;
        let config: AppConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model.kind, oben_models::ProviderKind::Custom);
        assert_eq!(config.model.model, "qwen35-local");
        assert_eq!(
            config.model.base_url.as_deref(),
            Some("http://10.0.0.177:8000/v1")
        );
        assert_eq!(config.tools.enabled, Vec::<String>::new());
        assert!(config.tools.auto_detect);
        assert_eq!(config.display.theme, "dark");
        assert_eq!(config.context.compression, "summary");
    }

    #[test]
    fn test_providers_field_serializes_empty() {
        let config = AppConfig::default();
        assert!(config.providers.is_empty());
        assert!(config.custom_providers.is_empty());
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.providers.is_empty());
        assert!(restored.custom_providers.is_empty());
    }

    #[test]
    fn test_gateway_config_roundtrip_qq_bot() {
        let mut config = AppConfig::default();
        config.gateway = Some(GatewayConfig {
            telegram: Some(TelegramConfig {
                enabled: true,
                token: Some("telegram-token-123".to_string()),
                ..Default::default()
            }),
            discord: Some(DiscordConfig {
                enabled: true,
                token: Some("discord-token-456".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        });

        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();

        assert!(restored.gateway.is_some());
        let gw = restored.gateway.unwrap();

        let tg = gw.telegram.unwrap();
        assert!(tg.enabled);
        assert_eq!(tg.token, Some("telegram-token-123".to_string()));

        let dc = gw.discord.unwrap();
        assert!(dc.enabled);
        assert_eq!(dc.token, Some("discord-token-456".to_string()));
    }

    #[test]
    fn test_gateway_config_qq_bot_serialization() {
        let mut config = AppConfig::default();
        config.gateway = Some(GatewayConfig {
            telegram: None,
            discord: None,
            slack: None,
            whatsapp: None,
            plugin_dir: None,
            qq_bot: Some(QQBotConfig {
                enabled: true,
                app_id: "12345".to_string(),
                app_secret: "super-secret-key".to_string(),
                sandbox: true,
                shard: None,
                intents: vec![
                    QQBotIntent::DirectMessage,
                    QQBotIntent::C2CAndGroup,
                    QQBotIntent::Interaction,
                ],
            }),
        });

        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();

        assert!(restored.gateway.is_some());
        let gw = restored.gateway.unwrap();

        let qq = gw.qq_bot.unwrap();
        assert!(qq.enabled);
        assert_eq!(qq.app_id, "12345");
        assert_eq!(qq.app_secret, "super-secret-key");
        assert!(qq.sandbox);
        assert!(qq.intents.iter().any(|i| matches!(i, QQBotIntent::DirectMessage)));
        assert!(qq.intents.iter().any(|i| matches!(i, QQBotIntent::C2CAndGroup)));
    }
}

impl AppConfig {
    /// Get config directory for a profile (delegates to Env).
    pub fn config_dir_from_profile(profile: Option<&str>) -> PathBuf {
        let env = crate::env::Env::new(profile.map(String::from));
        env.config_dir().clone()
    }

    /// Get config file path for a profile (delegates to Env).
    pub fn config_path_from_profile(profile: Option<&str>) -> PathBuf {
        let env = crate::env::Env::new(profile.map(String::from));
        env.config_path()
    }

    /// Read from `~/.config/obenmatrix/config.yaml` (legacy/standard path).
    pub fn config_dir_legacy() -> PathBuf {
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("~"));
        home.join(".config/obenmatrix")
    }

    /// Read from `~/.config/obenmatrix/config.yaml` (legacy/standard path).
    pub fn config_path_legacy() -> PathBuf {
        Self::config_dir_legacy().join("config.yaml")
    }

    /// Return true if the config file does not exist on disk.
    pub fn is_config_missing(profile: Option<&str>) -> bool {
        let env = crate::env::Env::new(profile.map(String::from));
        !env.config_path().exists()
    }

    /// Load config from the profile-specific (or default) config file.
    pub fn load(profile: Option<&str>) -> anyhow::Result<Self> {
        let env = crate::env::Env::new(profile.map(String::from));
        let path = env.config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path)?;
        let mut config: Self = serde_yaml::from_str(&content)?;

        if let Some(t) = config.temperature {
            config.model.temperature = Some(t);
        }
        if let Some(m) = config.max_tokens {
            config.model.max_tokens = Some(m);
        }

        if config
            .model
            .api_key
            .as_ref()
            .map_or(true, |v| v.trim().is_empty())
        {
            if let Some(key) =
                oben_models::provider_registry::resolve_api_key_from_env(config.model.kind.as_str())
            {
                config.model.api_key = Some(key);
            }
        }

        if config
            .model
            .base_url
            .as_ref()
            .map_or(true, |v| v.trim().is_empty())
        {
            if let Some(env_var) = config.model.kind.base_url_env_var() {
                if let Ok(val) = std::env::var(env_var) {
                    let val = val.trim().to_string();
                    if !val.is_empty() {
                        config.model.base_url = Some(val);
                    }
                }
            }
        }

        Ok(config)
    }

    /// Save config to the current profile's config dir.
    pub fn save(&self) -> anyhow::Result<()> {
        let env = crate::env::Env::new(None);
        let dir = env.config_dir();
        std::fs::create_dir_all(dir)?;
        let content = serde_yaml::to_string(self)?;
        std::fs::write(dir.join("config.yaml"), content)?;
        Ok(())
    }

    /// Save config to the given profile's config dir.
    pub fn save_with_profile(&self, profile: Option<&str>) -> anyhow::Result<()> {
        let env = crate::env::Env::new(profile.map(String::from));
        let dir = env.config_dir();
        std::fs::create_dir_all(dir)?;
        let content = serde_yaml::to_string(self)?;
        std::fs::write(dir.join("config.yaml"), content)?;
        Ok(())
    }
}

#[cfg(test)]
mod multi_agent_tests {
    use super::*;

    #[test]
    fn test_named_agent_config_default() {
        let config = NamedAgentConfig::default();
        assert!(config.name.is_empty());
        assert!(config.role.is_empty());
        assert!(config.model.is_empty());
        assert!(config.tools.is_empty());
        assert!(config.execution_discipline.is_none());
    }

    #[test]
    fn test_agents_field_serializes_empty() {
        let config = AppConfig::default();
        assert!(config.agents.is_empty());
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(restored.agents.is_empty());
    }

    #[test]
    fn test_multi_agent_config_yaml_roundtrip() {
        let mut config = AppConfig::default();
        config.agents = vec![
            NamedAgentConfig {
                name: "researcher".to_string(),
                role: "Research specialist - finds information".to_string(),
                model: "openai/gpt-4o".to_string(),
                tools: vec!["web_search".to_string(), "http_get".to_string()],
                execution_discipline: None,
            },
            NamedAgentConfig {
                name: "writer".to_string(),
                role: "Content writer - creates documents".to_string(),
                model: "anthropic/claude-3-5-sonnet".to_string(),
                tools: vec!["write_file".to_string(), "read_file".to_string()],
                execution_discipline: Some("strict".to_string()),
            },
        ];

        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(restored.agents.len(), 2);
        assert_eq!(restored.agents[0].name, "researcher");
        assert_eq!(restored.agents[0].model, "openai/gpt-4o");
        assert_eq!(restored.agents[1].name, "writer");
        assert_eq!(restored.agents[1].execution_discipline, Some("strict".to_string()));
    }

    #[test]
    fn test_backward_compatibility_single_agent_only() {
        // Config with only agent field (no agents)
        let yaml = r#"agent:
  identity: "You are a helpful assistant"
  execution_discipline: "Be concise"
"#;
        let config: AppConfig = serde_yaml::from_str(yaml).unwrap();
        
        assert_eq!(config.agent.identity, Some("You are a helpful assistant".to_string()));
        assert_eq!(config.agents.is_empty(), true);
    }

    #[test]
    fn test_multi_agent_only() {
        // Config with only agents field (no agent)
        let yaml = r#"agents:
  - name: "researcher"
    role: "Research specialist"
    model: "openai/gpt-4o"
    tools: ["web_search"]
"#;
        let config: AppConfig = serde_yaml::from_str(yaml).unwrap();
        
        assert_eq!(config.agents.len(), 1);
        assert_eq!(config.agents[0].name, "researcher");
        assert!(config.agent.identity.is_none());
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.provider, SearchProviderKind::DuckDuckGo);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_search_config_duckduckgo() {
        let config = SearchConfig {
            provider: SearchProviderKind::DuckDuckGo,
            api_key: None,
        };
        assert_eq!(config.provider, SearchProviderKind::DuckDuckGo);
    }

    #[test]
    fn test_search_config_brave_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Brave,
            api_key: Some("brave-key".to_string()),
        };
        assert_eq!(config.provider, SearchProviderKind::Brave);
        assert_eq!(config.api_key, Some("brave-key".to_string()));
    }

    #[test]
    fn test_search_config_bing_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Bing,
            api_key: Some("bing-key".to_string()),
        };
        assert_eq!(config.provider, SearchProviderKind::Bing);
        assert_eq!(config.api_key, Some("bing-key".to_string()));
    }

    #[test]
    fn test_search_config_google_with_key() {
        let config = SearchConfig {
            provider: SearchProviderKind::Google,
            api_key: Some("google-key".to_string()),
        };
        assert_eq!(config.provider, SearchProviderKind::Google);
        assert_eq!(config.api_key, Some("google-key".to_string()));
    }

    #[test]
    fn test_search_provider_serialization() {
        let config = SearchConfig {
            provider: SearchProviderKind::Brave,
            api_key: Some("test-key".to_string()),
        };
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: SearchConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(restored.provider, SearchProviderKind::Brave);
        assert_eq!(restored.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_search_provider_disabled() {
        let config = SearchConfig {
            provider: SearchProviderKind::Disabled,
            api_key: None,
        };
        assert!(config.provider.is_disabled());
    }

    #[test]
    fn test_app_config_search_field_roundtrip() {
        let mut config = AppConfig::default();
        config.search.provider = SearchProviderKind::Brave;
        config.search.api_key = Some("custom-key".to_string());

        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(restored.search.provider, SearchProviderKind::Brave);
        assert_eq!(restored.search.api_key, Some("custom-key".to_string()));
    }

    #[test]
    fn test_app_config_default_search() {
        let config = AppConfig::default();
        assert_eq!(config.search.provider, SearchProviderKind::DuckDuckGo);
        assert!(config.search.api_key.is_none());
    }
    
    #[test]
    fn test_moa_config_default() {
        let config = MixtureOfAgentsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.num_layers, 0);
        assert!(config.agents_per_layer.is_empty());
    }
    
    #[test]
    fn test_moa_config_validation() {
        let mut config = MixtureOfAgentsConfig::default();
        config.enabled = true;
        config.num_layers = 3;
        config.agents_per_layer = vec![2, 2, 1];
        
        assert!(config.validate());
        assert_eq!(config.total_agents(), 5);
    }
    
    #[test]
    fn test_moa_config_serialization() {
        let config = MixtureOfAgentsConfig {
            enabled: true,
            num_layers: 2,
            agents_per_layer: vec![3, 2],
        };
        
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: MixtureOfAgentsConfig = serde_yaml::from_str(&yaml).unwrap();
        
        assert!(restored.enabled);
        assert_eq!(restored.num_layers, 2);
        assert_eq!(restored.agents_per_layer, vec![3, 2]);
    }
    
    #[test]
    fn test_app_config_with_moa() {
        let mut config = AppConfig::default();
        config.mixture_of_agents = MixtureOfAgentsConfig {
            enabled: true,
            num_layers: 3,
            agents_per_layer: vec![2, 2, 1],
        };
        
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AppConfig = serde_yaml::from_str(&yaml).unwrap();
        
        assert!(restored.mixture_of_agents.enabled);
        assert_eq!(restored.mixture_of_agents.num_layers, 3);
        assert_eq!(restored.mixture_of_agents.total_agents(), 5);
    }
}
