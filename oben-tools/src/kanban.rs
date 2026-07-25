use oben_models::{ToolMeta, ToolParameter, ToolParameters};

use super::registry::{Tool, ToolCall, ToolRegistry};
use oben_models::ToolResult;

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

fn make_kanban_tool() -> ToolMeta {
    ToolMeta {
        name: "kanban".into(),
        description: "Manage tasks on a Kanban board. Supports creating, updating, moving, and deleting tasks across columns (To Do, In Progress, Done).".into(),
        parameters: ToolParameters::Flat(vec![
            ToolParameter::required("action", "Action: 'create', 'update', 'move', or 'delete'", "string"),
            ToolParameter::optional("task_id", "Task ID for update/move/delete actions", "string"),
            ToolParameter::optional("title", "Task title for create/update actions", "string"),
            ToolParameter::optional("description", "Task description", "string"),
            ToolParameter::optional("column", "Target column (To Do, In Progress, Done) for move/create actions", "string"),
            ToolParameter::optional("priority", "Task priority (Low, Medium, High, Critical)", "string"),
        ]),
    }
}

pub struct KanbanTool;

async fn execute_kanban<'a>(call: &ToolCall<'a>) -> anyhow::Result<oben_models::ToolResult> {
    let action = call.required_str("action")?;
    
    let valid_actions = ["create", "update", "move", "delete"];
    if !valid_actions.contains(&action) {
        return Ok(oben_models::ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(format!("Invalid action: '{}'. Must be one of: create, update, move, delete", action)),
        });
    }
    
    let task_id = call.args.get("task_id").and_then(|v| v.as_str()).map(|s| s.to_string());
    let title = call.args.get("title").and_then(|v| v.as_str()).map(|s| s.to_string());
    let description = call.args.get("description").and_then(|v| v.as_str()).map(|s| s.to_string());
    let column = call.args.get("column").and_then(|v| v.as_str()).map(|s| s.to_string());
    let priority = call.args.get("priority").and_then(|v| v.as_str()).map(|s| s.to_string());
    
    Ok(oben_models::ToolResult {
        call_id: call.call_id.clone(),
        output: format!(
            "Kanban action '{}': task_id={:?}, title={:?}, description={:?}, column={:?}, priority={:?}",
            action, task_id, title, description, column, priority
        ),
        error: None,
    })
}

#[async_trait::async_trait]
impl Tool for KanbanTool {
    fn name(&self) -> &str {
        "kanban"
    }
    fn description(&self) -> &str {
        "Manage tasks on a Kanban board"
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_kanban(call).await.unwrap_or_else(|e| ToolResult {
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
    let tool = Box::new(KanbanTool);
    registry.register_with_def(tool, make_kanban_tool());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Given: valid create action with title and column
    /// When: kanban tool is called with create
    /// Then: returns output with task details
    #[tokio::test]
    async fn test_kanban_create() {
        let test_args = serde_json::json!({
            "call_id": "test-1",
            "action": "create",
            "title": "Implement feature",
            "column": "To Do"
        });
        
        let tool = KanbanTool;
        let call = ToolCall::new("kanban", &test_args);
        let result = tool.execute(&call).await;
        
        assert!(result.error.is_none());
        assert!(result.output.contains("create"));
        assert!(result.output.contains("Implement feature"));
    }

    /// Given: missing action argument
    /// When: kanban tool is called
    /// Then: returns error for missing action
    #[tokio::test]
    async fn test_kanban_missing_action() {
        let test_args = serde_json::json!({
            "call_id": "test-2"
        });
        
        let tool = KanbanTool;
        let call = ToolCall::new("kanban", &test_args);
        let result = tool.execute(&call).await;
        
        assert!(result.error.is_some());
        assert!(result.output.is_empty());
    }
}
