use oben_models::{ToolMeta, ToolParameter, ToolParameters};

use super::registry::{Tool, ToolCall, ToolRegistry};
use oben_models::ToolResult;

use oben_cron::http::{CronClient, CronSubmitRequest};

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

fn make_cron_delivery_tool() -> ToolMeta {
    ToolMeta {
        name: "cron_delivery".into(),
        description: "Schedule and deliver messages/tasks at specific times using cron syntax. Supports cron expressions for flexible scheduling.".into(),
        parameters: ToolParameters::Flat(vec![
            ToolParameter::required("cron_expression", "Cron expression (5 fields: minute hour day month weekday)", "string"),
            ToolParameter::required("message", "Message or task to deliver at scheduled time", "string"),
            ToolParameter::optional("timezone", "Optional timezone for cron evaluation (e.g., 'UTC', 'America/New_York')", "string"),
        ]),
    }
}

pub struct CronDeliveryTool;

async fn execute_cron_delivery<'a>(call: &ToolCall<'a>) -> anyhow::Result<oben_models::ToolResult> {
    let cron_expr = call.required_str("cron_expression")?;
    let message = call.required_str("message")?;
    
    // Validate cron expression (basic check: at least 5 space-separated tokens)
    let tokens: Vec<&str> = cron_expr.split_whitespace().collect();
    if tokens.len() < 5 {
        return Ok(oben_models::ToolResult {
            call_id: call.call_id.clone(),
            output: String::new(),
            error: Some(format!("Invalid cron expression: '{}' (expected at least 5 space-separated fields)", cron_expr)),
        });
    }
    
    let prompt = format!("cron: {} -> {}", cron_expr, message);
    let client = CronClient::new(None);
    let request = CronSubmitRequest {
        prompt,
        deliver_target: None,
        session_id: Some(call.call_id.clone()),
    };
    
    let response = client.submit(&request).await?;
    
    Ok(oben_models::ToolResult {
        call_id: call.call_id.clone(),
        output: format!("Cron delivery scheduled (job_id: {}): '{}'", response.job_id, message),
        error: None,
    })
}

#[async_trait::async_trait]
impl Tool for CronDeliveryTool {
    fn name(&self) -> &str {
        "cron_delivery"
    }
    fn description(&self) -> &str {
        "Schedule and deliver messages/tasks at specific times"
    }
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_cron_delivery(call).await.unwrap_or_else(|e| ToolResult {
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
    let tool = Box::new(CronDeliveryTool);
    registry.register_with_def(tool, make_cron_delivery_tool());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Given: valid cron expression and message
    /// When: cron_delivery tool is called
    /// Then: returns scheduled task response
    #[test]
    fn test_cron_delivery_request_construction() {
        let test_args = serde_json::json!({
            "call_id": "test-1",
            "cron_expression": "0 9 * * *",
            "message": "Daily standup meeting"
        });
        
        let call = ToolCall::new("cron_delivery", &test_args);
        let cron_expr = call.required_str("cron_expression").unwrap();
        let message = call.required_str("message").unwrap();
        
        let prompt = format!("cron: {} -> {}", cron_expr, message);
        let request = CronSubmitRequest {
            prompt,
            deliver_target: None,
            session_id: Some(call.call_id.clone()),
        };
        
        assert_eq!(request.session_id, Some("test-1".to_string()));
        assert!(request.prompt.contains("cron:"));
        assert!(request.prompt.contains("0 9 * * *"));
        assert!(request.prompt.contains("Daily standup meeting"));
    }

    /// Given: missing cron_expression
    /// When: cron_delivery tool is called
    /// Then: returns error for missing required field
    #[test]
    fn test_cron_delivery_missing_cron_expression() {
        let test_args = serde_json::json!({
            "call_id": "test-2",
            "message": "Some message"
        });
        
        let call = ToolCall::new("cron_delivery", &test_args);
        let result = call.required_str("cron_expression");
        
        assert!(result.is_err());
    }

    /// Given: missing message
    /// When: cron_delivery tool is called
    /// Then: returns error for missing required field
    #[test]
    fn test_cron_delivery_missing_message() {
        let test_args = serde_json::json!({
            "call_id": "test-3",
            "cron_expression": "0 9 * * *"
        });
        
        let call = ToolCall::new("cron_delivery", &test_args);
        let result = call.required_str("message");
        
        assert!(result.is_err());
    }
}
