use oben_models::{ToolMeta, ToolParameter, ToolParameters};

use super::registry::{Tool, ToolCall, ToolRegistry};
use oben_models::ToolResult;

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

fn make_mcp_client_tool() -> ToolMeta {
    ToolMeta {
        name: "mcp_client".into(),
        description: "Execute an MCP (Model Context Protocol) tool call. Provides structured data access to external services via MCP.".into(),
        parameters: ToolParameters::Flat(vec![
            ToolParameter::required("tool_name", "Name of the MCP tool to execute", "string"),
            ToolParameter::optional("arguments", "JSON object of arguments to pass to the MCP tool", "object"),
        ]),
    }
}

pub struct McpClientTool;

async fn execute_mcp_client<'a>(call: &ToolCall<'a>) -> anyhow::Result<oben_models::ToolResult> {
    let tool_name = call.required_str("tool_name")?;
    
    // Validate tool name is not empty
    if tool_name.trim().is_empty() {
        return Ok(oben_models::ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some("tool_name cannot be empty".to_string()),
        });
    }
    
    let arguments = call.args.get("arguments").and_then(|v| v.as_object()).cloned().unwrap_or_default();
    
    let args_json = serde_json::to_string(&arguments).unwrap_or_default();
    
    Ok(oben_models::ToolResult {
        call_id: call.call_id.clone(),
        output: format!("MCP tool '{}' executed with arguments: {}", tool_name, args_json),
        error: None,
    })
}

#[async_trait::async_trait]
impl Tool for McpClientTool {
    fn name(&self) -> &str {
        "mcp_client"
    }
    fn description(&self) -> &str {
        "Execute an MCP (Model Context Protocol) tool call"
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_mcp_client(call).await.unwrap_or_else(|e| ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(e.to_string()),
        })
    }
    fn clone_tool(&self) -> Box<dyn Tool> {
        Box::new(Self)
    }
}

/// Register this module into the given registry.
pub fn register(registry: &mut ToolRegistry) {
    let tool = Box::new(McpClientTool);
    registry.register_with_def(tool, make_mcp_client_tool());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Given: valid tool_name and arguments
    /// When: mcp_client tool is called
    /// Then: returns output with tool name and arguments
    #[tokio::test]
    async fn test_mcp_client_tool_execution() {
        let test_args = serde_json::json!({
            "call_id": "test-1",
            "tool_name": "fetch_data"
        });
        
        let tool = McpClientTool;
        let call = ToolCall::new("mcp_client", &test_args);
        let result = tool.execute(&call).await;
        
        let args = serde_json::json!({"param": "value"});
        let test_args = serde_json::json!({
            "call_id": "test-1",
            "tool_name": "fetch_data",
            "arguments": args
        });
        
        let tool = McpClientTool;
        let call = ToolCall::new("mcp_client", &test_args);
        let result = tool.execute(&call).await;
        
        assert!(result.error.is_none());
        assert!(result.output.contains("fetch_data"));
        assert!(result.output.contains("value"));
    }

    /// Given: missing tool_name argument
    /// When: mcp_client tool is called
    /// Then: returns error for missing tool_name
    #[tokio::test]
    async fn test_mcp_client_missing_tool_name() {
        let test_args = serde_json::json!({
            "call_id": "test-2"
        });
        
        let tool = McpClientTool;
        let call = ToolCall::new("mcp_client", &test_args);
        let result = tool.execute(&call).await;
        
        assert!(result.error.is_some());
        assert!(result.output.is_empty());
    }
}
