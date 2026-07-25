use anyhow::Result;
use crate::BuiltinTools;
use blake3::Hasher;
use oben_models::ToolResult;
use serde_json::Value;
/// Tool registry — stores and dispatches tool calls.
///
/// Maps to `tools/registry.py`.
///
/// The `Tool` trait is the deep interface at the tool seam. Callers create
/// structs that implement the trait, and the registry stores them as `Box<dyn
/// Tool>`. Universal pre-checks (validation before dispatch) apply across all
/// tools in `execute()`.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{info, warn};
use parking_lot::Mutex;

pub type ToolCallHash = String;

// ---------------------------------------------------------------------------
// Tool trait — deep interface at the tool seam
// ---------------------------------------------------------------------------

/// A deep tool module. The interface is small (4 methods) but hides
/// the full implementation (parsing, validation, execution, error handling).
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn validate(&self, _call: &ToolCall) -> Result<()> {
        Ok(())
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult;
    /// Clone this tool into a new Box<dyn Tool>.
    fn clone_tool(&self) -> Box<dyn Tool>;
}

// ---------------------------------------------------------------------------
// ToolCall — extracted from raw args at the registry seam
// ---------------------------------------------------------------------------

/// Wrapper around raw tool call arguments with ergonomic accessors.
///
/// Created by the registry from the raw JSON input. Provides safe,
/// consistent extraction of call_id and strongly-typed field accessors
/// that every tool can use.
pub struct ToolCall<'a> {
    /// The call_id from the LLM response, used for correlation.
    pub call_id: String,
    /// The raw argument object reference.
    pub args: &'a Value,
}

impl<'a> ToolCall<'a> {
    pub fn new(_tool_name: &'a str, args: &'a Value) -> Self {
        let call_id = args
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Self { call_id, args }
    }

    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    /// Get a required string argument, returning an error message if missing.
    pub fn required_str(&self, key: &str) -> Result<&'a str> {
        self.args
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required argument: '{}'", key))
    }

    /// Get an optional string argument.
    pub fn optional_str(&self, key: &str) -> Option<&'a str> {
        self.args.get(key).and_then(|v| v.as_str())
    }

    /// Get an optional string argument with a default value.
    pub fn optional_str_with_default(&self, key: &str, default: &'a str) -> &'a str {
        self.optional_str(key).unwrap_or(default)
    }

    /// Get an optional bool argument.
    pub fn optional_bool(&self, key: &str) -> Option<bool> {
        self.args.get(key).and_then(|v| v.as_bool())
    }

    /// Get an optional u64 argument with a default fallback.
    pub fn optional_u64(&self, key: &str, default: u64) -> u64 {
        self.args
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(default)
    }

    /// Get an optional array argument.
    pub fn optional_array(&self, key: &str) -> Option<&'a Vec<Value>> {
        self.args.get(key).and_then(|v| v.as_array())
    }

    /// Get a nested object argument.
    pub fn optional_object(&self, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
        self.args.get(key).and_then(|v| v.as_object())
    }

    /// Get an optional string from a nested object key.
    pub fn nested_str(&self, parent: &str, key: &str) -> Option<&'a str> {
        self.args
            .get(parent)?
            .get(key)?
            .as_str()
    }
}

// ---------------------------------------------------------------------------
// Spawn function for subagent delegation
// ---------------------------------------------------------------------------

/// Spawn function for subagent delegation.
/// Maps to `delegate_task_handler` from hermes-agent.
///
/// Parameters:
/// - `parent_session_id`: session ID of the delegating agent
/// - `goal`: task description for the child
/// - `agent_depth`: current nesting depth (0 = parent, increments on each delegate call)
/// - `role`: "leaf" (default, cannot delegate further) or "orchestrator" (can delegate further if depth < max)
///
/// Creates a child session in the shared database, then spawns a child agent.
/// Returns a `JoinHandle` for the child's async execution.
pub type SpawnFn = Arc<
    dyn Fn(String, String, usize, &str) -> tokio::task::JoinHandle<Result<SubagentResult>>
        + Send
        + Sync,
>;

/// Result of executing a child agent delegation run.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SubagentResult {
    /// Whether the child completed successfully.
    pub status: String,
    /// The child's final response (truncated to 500 chars).
    pub summary: String,
    /// Number of API calls the child made.
    pub api_calls: u32,
    /// How long the child ran (monotonic seconds).
    pub duration_seconds: f64,
    /// The model the child used.
    pub model: Option<String>,
    /// The child's session ID in the shared database.
    pub session_id: String,
    /// The parent session ID that spawned this child.
    pub parent_session_id: String,
    /// Role: "leaf" or "orchestrator".
    pub role: Option<String>,
    /// Depth of this subagent in the delegate tree.
    pub depth: usize,
    /// Optional exit reason (for parity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    tool_defs: HashMap<String, oben_models::ToolMeta>,
    call_cache: Mutex<HashMap<ToolCallHash, ToolResult>>,
    in_progress: Mutex<HashSet<ToolCallHash>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            tool_defs: HashMap::new(),
            call_cache: Mutex::new(HashMap::new()),
            in_progress: Mutex::new(HashSet::new()),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        info!("Registering tool: {}", tool.name());
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn register_with_def(&mut self, tool: Box<dyn Tool>, def: oben_models::ToolMeta) {
        let name = tool.name().to_string();
        self.tools.insert(name.clone(), tool);
        self.tool_defs.insert(name, def);
    }

    pub fn list_tools(&self) -> Vec<oben_models::ToolMeta> {
        // Return stored tool definitions with full parameter specs.
        // Fall back to empty params for any tool registered without a def.
        let mut defs: Vec<oben_models::ToolMeta> = self.tool_defs.values().cloned().collect();
        // If there are tools without definitions, fill them in with empty params
        for name in self.tools.keys() {
            if !defs.iter().any(|d| d.name == *name) {
                let tool = &self.tools[name];
                defs.push(oben_models::ToolMeta {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: oben_models::ToolParameters::Flat(vec![]),
                });
            }
        }
        defs
    }

    fn compute_call_hash(tool_name: &str, arguments: &Value) -> String {
        let mut hasher = Hasher::new();
        hasher.update(tool_name.as_bytes());
        let args_str = arguments.to_string();
        hasher.update(args_str.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    pub async fn execute(&self, tool_name: &str, arguments: &Value) -> ToolResult {
        let call_hash = Self::compute_call_hash(tool_name, arguments);
        
        if self.in_progress.lock().contains(&call_hash) {
            warn!("Tool call {:?} is already in progress, skipping duplicate", tool_name);
            return ToolResult {
                call_id: arguments.get("call_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                output: String::new(),
                error: Some(format!("Duplicate call in progress: {}", tool_name)),
            };
        }
        
        if let Some(cached) = self.call_cache.lock().get(&call_hash) {
            return cached.clone();
        }
        
        self.in_progress.lock().insert(call_hash.clone());
        
        let result = self.execute_inner(tool_name, arguments).await;
        
        self.in_progress.lock().remove(&call_hash);
        if result.error.is_none() {
            self.call_cache.lock().insert(call_hash, result.clone());
        }
        
        result
    }

    async fn execute_inner(&self, tool_name: &str, arguments: &Value) -> ToolResult {
        info!("Executing tool: {} with args...", tool_name);
        let call = ToolCall::new(tool_name, arguments);
        match self.tools.get(tool_name) {
            Some(tool) => {
                if let Err(e) = tool.validate(&call) {
                    warn!("Tool {} validation failed: {}", tool_name, e);
                    return ToolResult {
                        call_id: call.call_id.clone(),
                        output: String::new(),
                        error: Some(format!("Validation: {}", e)),
                    };
                }
                let result = tool.execute(&call).await;
                if result.error.is_none() {
                    info!("Tool {} succeeded", tool_name);
                } else {
                    warn!("Tool {} failed: {:?}", tool_name, result.error);
                }
                result
            }
            None => {
                warn!("Unknown tool: {}", tool_name);
                ToolResult {
                    call_id: call.call_id.clone(),
                    output: String::new(),
                    error: Some(format!("Unknown tool: {}", tool_name)),
                }
            }
        }
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Create a new registry containing only the tools NOT in the `blocked` list.
    /// Used by `build_child_toolset` to restrict subagent tool access.
    pub fn filtered_clone(&self, blocked: &[&str]) -> Self {
        let blocked_set: std::collections::HashSet<&str> = blocked.iter().copied().collect();
        let mut filtered = Self::new();
        for name in self.tools.keys() {
            if blocked_set.contains(name.as_str()) {
                continue;
            }
            if let Some(tool) = self.tools.get(name) {
                // Clone via re-registering the boxed dyn Tool
                filtered.register(tool.as_ref().clone_tool());
                if let Some(def) = self.tool_defs.get(name) {
                    filtered.tool_defs.insert(name.clone(), def.clone());
                }
            }
        }
        filtered
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        let mut cloned = Self::new();
        for (name, tool) in &self.tools {
            cloned.tools.insert(name.clone(), tool.clone_tool());
        }
        for (name, def) in &self.tool_defs {
            cloned.tool_defs.insert(name.clone(), def.clone());
        }
        cloned
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn discover_builtin_tools(registry: &mut ToolRegistry) {
    BuiltinTools::register_all(registry);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;
    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str { "echo-test" }
        fn description(&self) -> &str { "Test echo" }
        async fn execute(&self, call: &ToolCall) -> ToolResult {
            let msg = call
                .optional_str("message")
                .unwrap_or("no-msg");
            ToolResult {
                call_id: call.call_id.clone(),
                output: format!("echo: {}", msg),
                error: None,
            }
        }
        fn clone_tool(&self) -> Box<dyn Tool> { Box::new(Self) }
    }

    struct FailTool;
    #[async_trait::async_trait]
    impl Tool for FailTool {
        fn name(&self) -> &str { "fail-test" }
        fn description(&self) -> &str { "Test fail" }
        async fn execute(&self, call: &ToolCall) -> ToolResult {
            ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some("boom".into()),
            }
        }
        fn clone_tool(&self) -> Box<dyn Tool> { Box::new(Self) }
    }

    struct ValidatingTool;
    #[async_trait::async_trait]
    impl Tool for ValidatingTool {
        fn name(&self) -> &str { "val-tool" }
        fn description(&self) -> &str { "Validates args" }
        fn validate(&self, call: &ToolCall) -> Result<()> {
            if call.optional_bool("block") == Some(true) {
                Err(anyhow::anyhow!("Blocked by validation"))
            } else {
                Ok(())
            }
        }
        async fn execute(&self, call: &ToolCall) -> ToolResult {
            ToolResult {
                call_id: call.call_id.clone(),
                output: "ok".into(),
                error: None,
            }
        }
        fn clone_tool(&self) -> Box<dyn Tool> {
            Box::new(ValidatingTool)
        }
    }

    #[tokio::test]
    async fn test_empty() {
        let r = ToolRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[tokio::test]
    async fn test_register_list_has() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(EchoTool));
        assert_eq!(r.len(), 1);
        assert!(r.has_tool("echo-test"));
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let r = ToolRegistry::new();
        let res = r.execute("nope", &serde_json::json!({})).await;
        assert!(res.error.is_some());
        assert!(res.error.unwrap().contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_handler_error() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(FailTool));
        let res = r.execute("fail-test", &serde_json::json!({"call_id":"x"})).await;
        assert_eq!(res.error.as_ref().unwrap(), "boom");
    }

    #[tokio::test]
    async fn test_validation_blocks() {
        let mut r = ToolRegistry::new();
        r.register(Box::new(ValidatingTool));
        let res = r
            .execute("val-tool", &serde_json::json!({"block": true}))
            .await;
        assert!(res.error.is_some());
        assert!(res.error.unwrap().contains("Validation"));
    }
}
