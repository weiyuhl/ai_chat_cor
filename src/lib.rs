use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::collections::HashMap;
use std::sync::Mutex;

pub mod agent;
pub mod providers;
pub use agent::{Agent, AgentConfig, AgentResult};
pub use providers::*;

#[derive(Error, Debug)]
pub enum ChatError {
    #[error("Network request failed: {0}")]
    NetworkError(String),
    
    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },
    
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
    
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        // 使用 providers.rs 中定义的供应商 API 地址
        let default_api_url = format!(
            "{}/v1/chat/completions",
            providers::LLMProvider::SiliconFlow.default_base_url()
        );
        Self {
            api_url: default_api_url,
            api_key: String::new(),
            model: String::new(),
            max_tokens: Some(1024),
            temperature: Some(0.7),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    choices: Vec<Choice>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

pub struct ChatClient {
    config: ChatConfig,
    client: reqwest::blocking::Client,
    history: Vec<Message>,
}

impl ChatClient {
    pub fn new(config: ChatConfig) -> Self {
        // 构建带有超时和宽松证书验证的客户端
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(30))
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        
        Self {
            config,
            client,
            history: Vec::new(),
        }
    }
    
    pub fn with_system_message(mut self, content: impl Into<String>) -> Self {
        self.history.push(Message::system(content));
        self
    }
    
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    
    pub fn get_history(&self) -> &[Message] {
        &self.history
    }
    
    pub fn send(&mut self, user_input: &str) -> Result<String, ChatError> {
        self.history.push(Message::user(user_input));
        
        let request_body = serde_json::json!({
            "model": self.config.model,
            "messages": self.history,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });
        
        let response = self.client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("certificate") || err_str.contains("ssl") {
                    ChatError::NetworkError(format!("SSL证书验证失败: {}", e))
                } else if err_str.contains("dns") || err_str.contains("resolve") {
                    ChatError::NetworkError(format!("DNS解析失败: {}", e))
                } else if err_str.contains("timeout") {
                    ChatError::NetworkError(format!("请求超时: {}", e))
                } else if err_str.contains("connection") {
                    ChatError::NetworkError(format!("连接失败: {}", e))
                } else {
                    ChatError::NetworkError(e.to_string())
                }
            })?;
        
        let status = response.status().as_u16();
        let api_response: ApiResponse = response
            .json()
            .map_err(|e| ChatError::NetworkError(e.to_string()))?;
        
        if let Some(error) = api_response.error {
            return Err(ChatError::ApiError {
                status,
                message: error.message,
            });
        }
        
        let reply = api_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ChatError::InvalidResponse("No choices in response".into()))?;
        
        self.history.push(Message::assistant(reply.clone()));
        Ok(reply)
    }
    
    pub fn chat(&mut self, user_input: &str) -> Result<String, ChatError> {
        self.send(user_input)
    }
}

// JNI 接口层
mod ffi {
    use super::*;
    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::jstring;

    lazy_static::lazy_static! {
        static ref CLIENTS: Mutex<HashMap<i64, ChatClient>> = Mutex::new(HashMap::new());
        static ref CLIENT_COUNTER: Mutex<i64> = Mutex::new(0);
    }

    // 安全地将 jstring 转换为 Rust String
    fn jstring_to_string(env: &mut JNIEnv, s: jstring) -> String {
        let jstr = unsafe { JString::from_raw(s) };
        env.get_string(&jstr).unwrap().into()
    }

    // 安全地将 Rust String 转换为 jstring
    fn string_to_jstring(env: &mut JNIEnv, s: &str) -> jstring {
        env.new_string(s).unwrap().into_raw()
    }

    /// 创建聊天客户端，返回客户端 ID 字符串
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeCreate(
        mut env: JNIEnv,
        _class: JClass,
        api_url: jstring,
        api_key: jstring,
        model: jstring,
        max_tokens: i32,
        temperature: f32,
    ) -> jstring {
        let api_url = jstring_to_string(&mut env, api_url);
        let api_key = jstring_to_string(&mut env, api_key);
        let model = jstring_to_string(&mut env, model);

        let config = ChatConfig {
            api_url,
            api_key,
            model,
            max_tokens: if max_tokens > 0 { Some(max_tokens as u32) } else { None },
            temperature: Some(temperature),
        };

        let client = ChatClient::new(config);

        let mut clients = CLIENTS.lock().unwrap();
        let mut counter = CLIENT_COUNTER.lock().unwrap();
        *counter += 1;
        let id = *counter;
        clients.insert(id, client);

        string_to_jstring(&mut env, &id.to_string())
    }

    /// 发送消息，返回 AI 回复
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeSend(
        mut env: JNIEnv,
        _class: JClass,
        client_id: jstring,
        message: jstring,
    ) -> jstring {
        let id_str = jstring_to_string(&mut env, client_id);
        let client_id: i64 = match id_str.parse() {
            Ok(id) => id,
            Err(_) => return string_to_jstring(&mut env, "Error: Invalid client ID"),
        };

        let message = jstring_to_string(&mut env, message);

        let mut clients = CLIENTS.lock().unwrap();
        let client = match clients.get_mut(&client_id) {
            Some(c) => c,
            None => return string_to_jstring(&mut env, "Error: Client not found"),
        };

        match client.chat(&message) {
            Ok(reply) => string_to_jstring(&mut env, &reply),
            Err(e) => string_to_jstring(&mut env, &format!("Error: {}", e)),
        }
    }

    /// 清除聊天历史
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeClearHistory(
        mut env: JNIEnv,
        _class: JClass,
        client_id: jstring,
    ) {
        let id_str = jstring_to_string(&mut env, client_id);
        if let Ok(id) = id_str.parse::<i64>() {
            if let Some(client) = CLIENTS.lock().unwrap().get_mut(&id) {
                client.clear_history();
            }
        }
    }

    /// 销毁客户端
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeDestroy(
        mut env: JNIEnv,
        _class: JClass,
        client_id: jstring,
    ) {
        let id_str = jstring_to_string(&mut env, client_id);
        if let Ok(id) = id_str.parse::<i64>() {
            CLIENTS.lock().unwrap().remove(&id);
        }
    }

    /// 获取供应商默认 API 地址（从 lib.rs 的 providers.rs）
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeGetDefaultConfig(
        mut env: JNIEnv,
        _class: JClass,
    ) -> jstring {
        use providers::LLMProvider;
        
        let config = ChatConfig::default();
        // 返回供应商列表和对应的 API 地址
        let providers_list = r#"["SiliconFlow","OpenRouter","DeepSeek","Custom"]"#;
        let urls_json = format!(
            r#"{{"SiliconFlow":"{}","OpenRouter":"{}","DeepSeek":"{}","Custom":""}}"#,
            LLMProvider::SiliconFlow.default_base_url(),
            LLMProvider::OpenRouter.default_base_url(),
            LLMProvider::DeepSeek.default_base_url()
        );
        let json = format!(
            r#"{{"api_url":"{}","model":"{}","max_tokens":{},"temperature":{:.1},"providers":{},"provider_urls":{}}}"#,
            config.api_url,
            config.model,
            config.max_tokens.unwrap_or(0),
            config.temperature.unwrap_or(0.7),
            providers_list,
            urls_json
        );
        string_to_jstring(&mut env, &json)
    }

    /// 获取指定供应商的 API 地址
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_LLMProvider_00024Companion_nativeGetProviderBaseUrl(
        mut env: JNIEnv,
        _class: JClass,
        provider_name: jstring,
    ) -> jstring {
        use providers::LLMProvider;
        
        let name = jstring_to_string(&mut env, provider_name);
        let url = match name.as_str() {
            "SiliconFlow" => LLMProvider::SiliconFlow.default_base_url(),
            "OpenRouter" => LLMProvider::OpenRouter.default_base_url(),
            "DeepSeek" => LLMProvider::DeepSeek.default_base_url(),
            _ => "",
        };
        string_to_jstring(&mut env, url)
    }

    /// Agent 执行任务
    #[no_mangle]
    pub extern "system" fn Java_com_example_myapplication_AiChatClient_nativeAgentRun(
        mut env: JNIEnv,
        _class: JClass,
        client_id: jstring,
        task: jstring,
        max_iterations: i32,
    ) -> jstring {
        let id_str = jstring_to_string(&mut env, client_id);
        let client_id: i64 = match id_str.parse() {
            Ok(id) => id,
            Err(_) => return string_to_jstring(&mut env, "Error: Invalid client ID"),
        };

        let task = jstring_to_string(&mut env, task);
        let max_iter = if max_iterations > 0 { max_iterations as usize } else { 10 };

        let mut clients = CLIENTS.lock().unwrap();
        let client = match clients.get_mut(&client_id) {
            Some(c) => c,
            None => return string_to_jstring(&mut env, "Error: Client not found"),
        };

        let config = AgentConfig {
            max_iterations: max_iter,
        };

        let agent = Agent::new(config);
        let result = agent.run(client, &task);

        // 格式化为 JSON 返回
        let steps_json: Vec<String> = result.steps.iter().map(|s| {
            format!(r#"{{"type":"{}","content":{}}}"#, s.step_type, escape_json(&s.content))
        }).collect();

        let json = format!(
            r#"{{"success":{},"answer":{},"steps":[{}]}}"#,
            result.success,
            escape_json(&result.answer),
            steps_json.join(",")
        );

        string_to_jstring(&mut env, &json)
    }
}

fn escape_json(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }
    
    #[test]
    fn test_default_config() {
        let config = ChatConfig::default();
        assert_eq!(config.model, "");
        assert_eq!(config.max_tokens, Some(1024));
    }
}
