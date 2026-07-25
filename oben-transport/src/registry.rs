/// Transport registry — dynamic discovery, registration, and dispatch.
///
/// Maps to `agent/transports/__init__.py` from Hermes-Agent.
///
/// Architecture:
///
/// ```ignore
/// TransportRegistry
/// ├── register_transport(name, factory)
/// ├── get_transport(name, config, system_prompt) -> Option<Arc<dyn TransportProvider>>  (lazy discovery)
/// ├── unregister_transport(name) -> bool
/// └── list_transport_names() -> Vec<String>
/// ```
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use oben_models::providers::{ProviderConfig, TransportProvider};

use super::{
    anthropic_messages::AnthropicMessagesTransport, chat_completions::ChatCompletionsTransport,
    gemini::GeminiMessagesTransport,
};

/// Transport factory function: builds an Arc<dyn TransportProvider> from config + system prompt.
pub type TransportFactory =
    Box<dyn Fn(&ProviderConfig, &str) -> Arc<dyn TransportProvider + Send + Sync> + Send + Sync>;

/// The global transport registry.
/// All built-in transports are registered on first call to `get_transport()`.
static REGISTRY: LazyLock<Mutex<TransportRegistry>> =
    LazyLock::new(|| Mutex::new(TransportRegistry::new()));

/// Where a transport was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransportSource {
    Builtin,
    Plugin,
}

/// Transport registry — a map of name-to-factory.
struct TransportRegistry {
    transports: HashMap<String, RegistryEntry>,
}

struct RegistryEntry {
    factory: TransportFactory,
}

impl TransportRegistry {
    fn new() -> Self {
        Self {
            transports: HashMap::new(),
        }
    }

    fn register(
        &mut self,
        name: impl Into<String>,
        factory: TransportFactory,
        _source: TransportSource,
    ) {
        self.transports
            .insert(name.into(), RegistryEntry { factory });
    }

    fn unregister(&mut self, name: &str) -> bool {
        self.transports.remove(name).is_some()
    }

    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.transports.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Built-in transport discovery ─────────────────────────────────────────────

/// Discover and register all built-in transports.
///
/// Called lazily on first `get_transport()` call.
fn discover_builtin_transports(reg: &mut TransportRegistry) {
    reg.register(
        "chat_completions",
        Box::new(|config: &ProviderConfig, system_prompt: &str| {
            let tools: Vec<oben_models::ToolMeta> = config
                .tools_json
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            Arc::new(ChatCompletionsTransport::from_config_with_tools(
                config,
                system_prompt,
                tools,
            )) as Arc<dyn TransportProvider + Send + Sync>
        }),
        TransportSource::Builtin,
    );
    reg.register(
        "anthropic_messages",
        Box::new(|config: &ProviderConfig, system_prompt: &str| {
            let tools: Vec<oben_models::ToolMeta> = config
                .tools_json
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            Arc::new(AnthropicMessagesTransport::from_config_with_tools(
                config,
                system_prompt,
                tools,
            )) as Arc<dyn TransportProvider + Send + Sync>
        }),
        TransportSource::Builtin,
    );
    // Future: "bedrock_converse", "codex_responses"
    reg.register(
        "gemini_native",
        Box::new(|config: &ProviderConfig, _system_prompt: &str| {
            let base_url = config
                .base_url
                .as_deref()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
            let model = config.model.to_string();
            let tools: Vec<oben_models::ToolMeta> = config
                .tools_json
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let api_key = oben_models::resolve_api_key_from_env("google-gemini")
                .or_else(|| oben_models::resolve_api_key_from_env("google-gemini-cli"))
                .unwrap_or(String::new());
            Arc::new(GeminiMessagesTransport::new(api_key, base_url, model).with_tools(tools))
                as Arc<dyn TransportProvider + Send + Sync>
        }),
        TransportSource::Builtin,
    );
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Register a transport factory for the given API mode.
///
/// Plugins should use this function to register dynamic transports.
/// Built-in transports are registered automatically on first `get_transport()` call.
pub fn register_transport(name: impl Into<String>, factory: TransportFactory) {
    let mut reg = REGISTRY.lock().unwrap();
    reg.register(name, factory, TransportSource::Plugin);
}

/// Get a transport instance for the given API mode.
///
/// On first call, lazily discovers all built-in transports. Returns an instance
/// created by the registered factory with the provided config and system prompt.
///
/// Returns `None` if no transport is registered for this mode.
///
/// # Example
/// ```ignore
/// if let Some(t) = get_transport("anthropic_messages", &config, &system_prompt) {
///     let resp = t.chat(&messages, &mode).await?;
/// }
/// ```
pub fn get_transport(
    name: &str,
    config: &ProviderConfig,
    system_prompt: &str,
) -> Option<Arc<dyn TransportProvider + Send + Sync>> {
    let mut reg = REGISTRY.lock().unwrap();
    if reg.transports.is_empty() {
        discover_builtin_transports(&mut reg);
    }
    reg.transports
        .get(name)
        .map(|entry| (entry.factory)(config, system_prompt))
}

/// Unregister a transport by name.
///
/// Returns `true` if a transport was found and removed, `false` otherwise.
pub fn unregister_transport(name: &str) -> bool {
    let mut reg = REGISTRY.lock().unwrap();
    reg.unregister(name)
}

/// List all registered transport names, sorted alphabetically.
pub fn list_transport_names() -> Vec<String> {
    let mut reg = REGISTRY.lock().unwrap();
    if reg.transports.is_empty() {
        discover_builtin_transports(&mut reg);
    }
    reg.names()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use oben_models::providers::{ProviderConfig, TransportProvider, TransportResponse};
    use oben_models::ProviderKind;
    use oben_models::{CallMode, Message, MessageContent, MessageRole, StreamDeltaCallback};

    // ── Simple test transport ──────────────────────────────────────────────────

    struct TestTransport {
        response_text: String,
        name: String,
    }

    impl TestTransport {
        fn new(response_text: &str) -> Self {
            Self {
                response_text: response_text.to_string(),
                name: "test".to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl TransportProvider for TestTransport {
        fn name(&self) -> &str {
            &self.name
        }

        async fn chat(
            &self,
            _messages: &[Message],
            _mode: &CallMode,
        ) -> Result<TransportResponse, anyhow::Error> {
            Ok(TransportResponse {
                text: self.response_text.clone(),
                tool_calls: vec![],
                tokens_used: Some(42),
                reasoning: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: &[Message],
            _mode: &CallMode,
            _delta_callback: StreamDeltaCallback,
            _reasoning_callback: Option<oben_models::StreamReasoningCallback>,
        ) -> Result<TransportResponse, anyhow::Error> {
            Ok(TransportResponse {
                text: self.response_text.clone(),
                tool_calls: vec![],
                tokens_used: Some(42),
                reasoning: None,
            })
        }
    }

    // ── Registry struct tests ──────────────────────────────────────────────────

    #[test]
    fn fresh_registry_is_empty() {
        let reg = TransportRegistry::new();
        assert_eq!(reg.names().len(), 0);
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = TransportRegistry::new();
        let text = "hello".to_string();
        reg.register(
            "foo",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&text)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        assert!(reg.transports.get("foo").is_some());
    }

    #[test]
    fn lookup_unknown_returns_none() {
        let reg = TransportRegistry::new();
        assert!(reg.transports.get("not_found").is_none());
    }

    #[test]
    fn unregister_removes_entry() {
        let mut reg = TransportRegistry::new();
        let text = "x".to_string();
        reg.register(
            "remove_me",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&text)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        assert!(reg.unregister("remove_me"));
        assert!(reg.transports.get("remove_me").is_none());
    }

    #[test]
    fn unregister_nonexistent_returns_false() {
        let mut reg = TransportRegistry::new();
        assert!(!reg.unregister("no_such_transport"));
    }

    #[test]
    fn names_sorted() {
        let mut reg = TransportRegistry::new();
        let z = "z".to_string();
        let a = "a".to_string();
        let m = "m".to_string();
        reg.register(
            "zebra",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&z)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        reg.register(
            "alpha",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&a)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        reg.register(
            "mango",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&m)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        let names = reg.names();
        assert_eq!(names, vec!["alpha", "mango", "zebra"]);
    }

    #[test]
    fn duplicate_registration_overwrites_factory() {
        let mut reg = TransportRegistry::new();
        let f1 = "first".to_string();
        reg.register(
            "dup",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&f1)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        assert_eq!(reg.transports.len(), 1);

        let f2 = "second".to_string();
        reg.register(
            "dup",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&f2)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        assert_eq!(reg.transports.len(), 1);
        assert!(reg.transports.get("dup").is_some());
    }

    #[test]
    fn builtin_discovery() {
        let mut reg = TransportRegistry::new();
        discover_builtin_transports(&mut reg);
        let names = reg.names();
        assert!(names.contains(&"anthropic_messages".to_string()));
        assert!(names.contains(&"chat_completions".to_string()));
        assert!(names.contains(&"gemini_native".to_string()));
        assert_eq!(names.len(), 3);
    }

    // ── Live integration test with test transport ──────────────────────────────

    fn lock_registry() -> std::sync::MutexGuard<'static, TransportRegistry> {
        REGISTRY.lock().unwrap()
    }

    #[test]
    fn live_test_transport_is_callable() {
        // given: a TestTransport registered under "live_test_transport"
        // when: get_transport is called with that name
        // then: returns an Arc<dyn TransportProvider> with the expected text
        let text = "hello from test".to_string();
        let mut guard = lock_registry();
        guard.register(
            "live_test_transport",
            Box::new(move |_config, _sys| {
                Arc::new(TestTransport::new(&text)) as Arc<dyn TransportProvider + Send + Sync>
            }),
            TransportSource::Plugin,
        );
        drop(guard);

        let config = ProviderConfig::new(ProviderKind::Custom, "test");
        let transport = get_transport("live_test_transport", &config, "system prompt")
            .expect("test transport should be registered");

        assert_eq!(transport.name(), "test");

        // Use it
        let messages = vec![Message {
            role: MessageRole::User,
            content: MessageContent::Text("hello".to_string()),
            id: None,
            tool_call_ids: vec![],
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }];
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(transport.chat(&messages, &CallMode::Fresh("s1".into())))
            .expect("chat should work");
        assert_eq!(rt.text, "hello from test");
    }
}
