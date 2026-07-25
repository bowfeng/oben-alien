/// Prompt building — assembles system prompt + conversation history for the API.
use anyhow::Result;
use oben_models::Message;

pub struct PromptBuilder {
    system_prompt: String,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self {
            system_prompt: oben_config::defaults::default_system_prompt(),
        }
    }

    /// Set a custom system prompt.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Build the messages array for the LLM API from a slice of messages.
    pub fn build_api_messages(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let mut api_messages = vec![Message::system(self.system_prompt.clone())];
        api_messages.extend(
            messages
                .iter()
                .filter(|m| m.include_in_prompt)
                .cloned()
        );
        Ok(api_messages)
    }

    /// Get current system prompt.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}
