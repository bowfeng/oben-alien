//! Browser automation tool using agent-browser CLI.
//!
//! Provides browser automation capabilities including:
//! - Page navigation (browser_navigate)
//! - DOM snapshot capture (browser_snapshot)
//! - Element interaction (browser_click, browser_type)
//! - Form filling (browser_fill)
//! - Screenshot capture (browser_screenshot)
//! - Page evaluation (browser_eval)
//!
//! Uses agent-browser CLI for browser automation. Requires:
//! - Node.js and npx available in PATH
//! - agent-browser installed: `npm install -g agent-browser`

use anyhow::Result;
use serde_json::Value;
use std::process::{Command, Stdio};
use tracing::debug;
use url::Url;

use oben_models::{ToolMeta, ToolParameters, ToolResult};

use super::registry::{Tool, ToolCall, ToolRegistry};

// ===========================================================================
// Schema Definitions
// ===========================================================================

fn browser_navigate_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to (required)"
            },
            "wait_until": {
                "type": "string",
                "enum": ["domcontentloaded", "load", "networkidle"],
                "description": "When to consider navigation successful. Default: 'networkidle'."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30."
            }
        },
        "required": ["url"]
    })
}

fn browser_snapshot_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page. Default: false."
            },
            "include_images": {
                "type": "boolean",
                "description": "Include image elements in the snapshot. Default: false."
            },
            "max_elements": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "Maximum number of elements to include. Default: 5000."
            }
        }
    })
}

fn browser_click_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector or element ref (e.g., @e5) to click"
            },
            "element_index": {
                "type": "integer",
                "description": "Element index from snapshot (1-based) as alternative to selector"
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button to use. Default: 'left'."
            },
            "modifiers": {
                "type": "array",
                "items": {"type": "string", "enum": ["shift", "ctrl", "alt", "meta"]},
                "description": "Modifier keys to hold during click"
            }
        },
        "required": ["selector"]
    })
}

fn browser_type_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector for the input element"
            },
            "text": {
                "type": "string",
                "description": "Text to type into the element"
            },
            "delay": {
                "type": "integer",
                "minimum": 0,
                "description": "Delay between keystrokes in milliseconds. Default: 0."
            },
            "clear": {
                "type": "boolean",
                "description": "Clear the element before typing. Default: true."
            }
        },
        "required": ["selector", "text"]
    })
}

fn browser_fill_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector for the input/textarea/select element"
            },
            "value": {
                "type": "string",
                "description": "Value to set"
            }
        },
        "required": ["selector", "value"]
    })
}

fn browser_screenshot_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to save screenshot (optional)"
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page. Default: false."
            },
            "selector": {
                "type": "string",
                "description": "Capture only the element matching this selector"
            },
            "quality": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "JPEG quality. Default: 80."
            }
        }
    })
}

fn browser_eval_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate"
            },
            "return_by_value": {
                "type": "boolean",
                "description": "Return primitive values instead of object references. Default: true."
            }
        },
        "required": ["expression"]
    })
}

fn browser_scroll_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction"
            },
            "amount": {
                "type": "integer",
                "description": "Amount to scroll. Default: 100."
            }
        },
        "required": ["direction"]
    })
}

// ===========================================================================
// Safety Gates
// ===========================================================================

fn is_safe_url(url: &str) -> bool {
    let lower = url.trim().to_lowercase();
    if lower.starts_with("file://") || lower.starts_with("javascript:")
        || lower.starts_with("data:") || lower.starts_with("blob:")
    {
        return false;
    }
    true
}

fn check_url_syntax(url: &str) -> Result<Url, String> {
    Url::parse(url).map_err(|e| format!("Invalid URL syntax: {} (try http_get for raw text fetching)", e))
}

fn check_url_safety(url: &str) -> Option<String> {
    if !is_safe_url(url) {
        Some(format!("Blocked dangerous URL scheme: {}", url))
    } else {
        None
    }
}

// ===========================================================================
// Agent-Browser CLI Integration
// ===========================================================================

fn run_agent_browser_command(action: &str, args: &Value) -> Result<(String, Vec<String>)> {
    let mut cmd = Command::new("npx");
    cmd.arg("agent-browser")
        .arg(action)
        .arg(args.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    
    debug!("agent-browser command: {:?}", cmd);
    
    let output = cmd.output()?;
    
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow::anyhow!("agent-browser failed: {}", err));
    }
    
    let output_str = String::from_utf8_lossy(&output.stdout);
    Ok(extract_result(&output_str))
}

fn extract_result(raw: &str) -> (String, Vec<String>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    
    if let Ok(obj) = serde_json::from_str::<Value>(raw) {
        if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                text_parts.push(text.to_string());
            }
        }
        if let Some(img_data) = obj.get("data").and_then(|v| v.as_str()) {
            if !img_data.is_empty() {
                images.push(img_data.to_string());
            }
        }
        if let Some(arr) = obj.get("images").and_then(|v| v.as_array()) {
            for img in arr {
                if let Some(s) = img.as_str() {
                    images.push(s.to_string());
                }
            }
        }
    } else {
        if !raw.trim().is_empty() {
            text_parts.push(raw.trim().to_string());
        }
    }
    
    let text = text_parts.join("\n").trim().to_string();
    (text, images)
}

fn format_images(images: &[String]) -> String {
    if images.is_empty() {
        String::new()
    } else {
        format!("\n[{} images]", images.len())
    }
}

// ===========================================================================
// Tool Handlers
// ===========================================================================

async fn handle_browser_navigate(args: &Value, call_id: String) -> Result<ToolResult> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("url parameter is required"))?;
    
    if let Err(e) = check_url_syntax(url) {
        return Ok(ToolResult {
            call_id,
            output: String::new(),
            error: Some(e),
        });
    }
    
    if let Some(blocked) = check_url_safety(url) {
        return Ok(ToolResult {
            call_id,
            output: String::new(),
            error: Some(blocked),
        });
    }
    
    let wait_until = args.get("wait_until").and_then(|v| v.as_str()).unwrap_or("networkidle");
    let timeout = args.get("timeout").and_then(|v| v.as_i64()).unwrap_or(30);
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("url".into(), url.into());
    cmd_args.insert("waitUntil".into(), wait_until.into());
    cmd_args.insert("timeout".into(), (timeout * 1000).into());
    
    let (text, _images) = run_agent_browser_command("open", &Value::Object(cmd_args))?;
    
    let output = format!("Navigated to {}\n{}", url, text);
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_snapshot(args: &Value, call_id: String) -> Result<ToolResult> {
    let full_page = args.get("full_page").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_elements = args.get("max_elements").and_then(|v| v.as_i64()).unwrap_or(5000) as usize;
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("fullPage".into(), full_page.into());
    cmd_args.insert("maxElements".into(), max_elements.into());
    
    let (text, images) = run_agent_browser_command("snapshot", &Value::Object(cmd_args))?;
    
    if text.trim().is_empty() && images.is_empty() {
        return Ok(ToolResult {
            call_id,
            output: String::new(),
            error: Some("No DOM elements found on page".to_string()),
        });
    }
    
    let output = format!("{}{}", text, format_images(&images));
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_click(args: &Value, call_id: String) -> Result<ToolResult> {
    let selector = args
        .get("selector")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("selector parameter is required"))?;
    
    let button = args.get("button").and_then(|v| v.as_str()).unwrap_or("left");
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("selector".into(), selector.into());
    cmd_args.insert("button".into(), button.into());
    
    let (text, _images) = run_agent_browser_command("click", &Value::Object(cmd_args))?;
    
    let output = format!("Clicked {}\n{}", selector, text);
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_type(args: &Value, call_id: String) -> Result<ToolResult> {
    let selector = args
        .get("selector")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("selector parameter is required"))?;
    
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("text parameter is required"))?;
    
    let delay = args.get("delay").and_then(|v| v.as_i64()).unwrap_or(0) as u64;
    let clear = args.get("clear").and_then(|v| v.as_bool()).unwrap_or(true);
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("selector".into(), selector.into());
    cmd_args.insert("text".into(), text.into());
    cmd_args.insert("delay".into(), delay.into());
    cmd_args.insert("clear".into(), clear.into());
    
    let (text_out, _images) = run_agent_browser_command("type", &Value::Object(cmd_args))?;
    
    let output = format!("Typed '{}' into {}\n{}", text, selector, text_out);
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_fill(args: &Value, call_id: String) -> Result<ToolResult> {
    let selector = args
        .get("selector")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("selector parameter is required"))?;
    
    let value = args
        .get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("value parameter is required"))?;
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("selector".into(), selector.into());
    cmd_args.insert("value".into(), value.into());
    
    let (text_out, _images) = run_agent_browser_command("fill", &Value::Object(cmd_args))?;
    
    let output = format!("Filled '{}' into {}\n{}", value, selector, text_out);
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_screenshot(args: &Value, call_id: String) -> Result<ToolResult> {
    let full_page = args.get("full_page").and_then(|v| v.as_bool()).unwrap_or(false);
    let quality = args.get("quality").and_then(|v| v.as_i64()).unwrap_or(80);
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("fullPage".into(), full_page.into());
    cmd_args.insert("quality".into(), quality.into());
    
    if let Some(selector) = args.get("selector").and_then(|v| v.as_str()) {
        cmd_args.insert("selector".into(), selector.into());
    }
    
    let (text, images) = run_agent_browser_command("screenshot", &Value::Object(cmd_args))?;
    
    let mut output = format!("Screenshot taken{}", text);
    if !images.is_empty() {
        output = format!("{}\n{}\n[1 image captured]", output, images[0]);
     }
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_eval(args: &Value, call_id: String) -> Result<ToolResult> {
    let expression = args
        .get("expression")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("expression parameter is required"))?;
    
    let return_by_value = args.get("return_by_value").and_then(|v| v.as_bool()).unwrap_or(true);
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("expression".into(), expression.into());
    cmd_args.insert("returnByValue".into(), return_by_value.into());
    
    let (text_out, images) = run_agent_browser_command("eval", &Value::Object(cmd_args))?;
    
    let output = format!("{}\n{}", text_out, format_images(&images));
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

async fn handle_browser_scroll(args: &Value, call_id: String) -> Result<ToolResult> {
    let direction = args
        .get("direction")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("direction parameter is required"))?;
    
    let amount = args.get("amount").and_then(|v| v.as_i64()).unwrap_or(100);
    
    let mut cmd_args = serde_json::Map::new();
    cmd_args.insert("direction".into(), direction.into());
    cmd_args.insert("amount".into(), amount.into());
    
    let (text_out, _images) = run_agent_browser_command("scroll", &Value::Object(cmd_args))?;
    
    let output = format!("Scrolled {} by {} pixels\n{}", direction, amount, text_out);
    
    Ok(ToolResult {
        call_id,
        output,
        error: None,
    })
}

/// Browser tool with specific action.
pub struct BrowserTool {
    action: String,
}

impl BrowserTool {
    pub fn new(action: &str) -> Self {
        BrowserTool {
            action: action.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Tool for BrowserTool {
    fn name(&self) -> &str {
        match self.action.as_str() {
            "navigate" => "browser_navigate",
            "snapshot" => "browser_snapshot",
            "click" => "browser_click",
            "type" => "browser_type",
            "fill" => "browser_fill",
            "screenshot" => "browser_screenshot",
            "eval" => "browser_eval",
            "scroll" => "browser_scroll",
            _ => "browser",
        }
    }
    
    fn description(&self) -> &str {
        match self.action.as_str() {
            "navigate" => "Navigate the browser to a URL and return a snapshot of the page content.",
            "snapshot" => "Capture a snapshot of the current page's accessibility tree.",
            "click" => "Click an element on the page using a CSS selector.",
            "type" => "Type text into an input or textarea element.",
            "fill" => "Fill an input, textarea, or select element with a value.",
            "screenshot" => "Capture a screenshot of the current page or a specific element.",
            "eval" => "Execute JavaScript in the browser context.",
            "scroll" => "Scroll the page in the specified direction.",
            _ => "Browser automation tool using agent-browser CLI",
        }
    }
    
    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let result = match self.action.as_str() {
            "navigate" => handle_browser_navigate(call.args, call.call_id.clone()).await,
            "snapshot" => handle_browser_snapshot(call.args, call.call_id.clone()).await,
            "click" => handle_browser_click(call.args, call.call_id.clone()).await,
            "type" => handle_browser_type(call.args, call.call_id.clone()).await,
            "fill" => handle_browser_fill(call.args, call.call_id.clone()).await,
            "screenshot" => handle_browser_screenshot(call.args, call.call_id.clone()).await,
            "eval" => handle_browser_eval(call.args, call.call_id.clone()).await,
            "scroll" => handle_browser_scroll(call.args, call.call_id.clone()).await,
            _ => Ok(ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some(format!("Unknown action: {}", self.action)),
            }),
        };
        
        match result {
            Ok(tr) => tr,
            Err(e) => ToolResult {
                call_id: call.call_id.clone(),
                output: String::new(),
                error: Some(e.to_string()),
            },
        }
    }
    
    fn clone_tool(&self) -> Box<dyn Tool> {
        Box::new(Self {
            action: self.action.clone(),
        })
    }
}


// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_url_check() {
        assert!(is_safe_url("https://example.com"));
        assert!(is_safe_url("http://localhost:3000"));
        assert!(!is_safe_url("file:///etc/passwd"));
        assert!(!is_safe_url("javascript:alert(1)"));
        assert!(!is_safe_url("data:text/html,<script>alert(1)</script>"));
    }
    
    #[test]
    fn test_url_safety_blocking() {
        assert!(check_url_safety("https://example.com").is_none());
        assert!(check_url_safety("file:///etc/passwd").is_some());
    }
    
    #[test]
    fn test_browser_tool_action_navigate() {
        let tool = BrowserTool::new("navigate");
        assert_eq!(tool.name(), "browser_navigate");
    }
    
    #[test]
    fn test_browser_tool_action_snapshot() {
        let tool = BrowserTool::new("snapshot");
        assert_eq!(tool.name(), "browser_snapshot");
    }
    
    #[test]
    fn test_browser_tool_action_click() {
        let tool = BrowserTool::new("click");
        assert_eq!(tool.name(), "browser_click");
    }
    
    #[test]
    fn test_browser_tool_action_type() {
        let tool = BrowserTool::new("type");
        assert_eq!(tool.name(), "browser_type");
    }
    
    #[test]
    fn test_browser_tool_action_fill() {
        let tool = BrowserTool::new("fill");
        assert_eq!(tool.name(), "browser_fill");
    }
    
    #[test]
    fn test_browser_tool_action_screenshot() {
        let tool = BrowserTool::new("screenshot");
        assert_eq!(tool.name(), "browser_screenshot");
    }
    
    #[test]
    fn test_browser_tool_action_eval() {
        let tool = BrowserTool::new("eval");
        assert_eq!(tool.name(), "browser_eval");
    }
    
    #[test]
    fn test_browser_tool_action_scroll() {
        let tool = BrowserTool::new("scroll");
        assert_eq!(tool.name(), "browser_scroll");
    }
}

// Registration

pub fn register(registry: &mut ToolRegistry) {
    registry.register_with_def(
        Box::new(BrowserTool::new("navigate")),
        ToolMeta {
            name: "browser_navigate".into(),
            description: "Navigate the browser to a URL and return a snapshot of the page content.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_navigate_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("snapshot")),
        ToolMeta {
            name: "browser_snapshot".into(),
            description: "Capture a snapshot of the current page's accessibility tree.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_snapshot_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("click")),
        ToolMeta {
            name: "browser_click".into(),
            description: "Click an element on the page using a CSS selector.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_click_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("type")),
        ToolMeta {
            name: "browser_type".into(),
            description: "Type text into an input or textarea element.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_type_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("fill")),
        ToolMeta {
            name: "browser_fill".into(),
            description: "Fill an input, textarea, or select element with a value.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_fill_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("screenshot")),
        ToolMeta {
            name: "browser_screenshot".into(),
            description: "Capture a screenshot of the current page or a specific element.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_screenshot_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("eval")),
        ToolMeta {
            name: "browser_eval".into(),
            description: "Execute JavaScript in the browser context.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_eval_schema(),
            },
        },
    );
    registry.register_with_def(
        Box::new(BrowserTool::new("scroll")),
        ToolMeta {
            name: "browser_scroll".into(),
            description: "Scroll the page in the specified direction.".into(),
            parameters: ToolParameters::JsonSchema {
                schema: browser_scroll_schema(),
            },
        },
    );
}
