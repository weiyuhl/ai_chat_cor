use serde::{Deserialize, Serialize};

// ===== 通用错误类型 =====

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String, code: Option<String> },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Auth error: {0}")]
    Auth(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),
}

// ===== 通用数据模型 =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDef {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCallDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDef {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub content: String,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct BalanceDetail {
    pub currency: String,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
}

#[derive(Debug, Clone)]
pub struct BalanceInfo {
    pub is_available: bool,
    pub balances: Vec<BalanceDetail>,
}

// ===== 重试配置 =====

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub connect_timeout_ms: u64,
    pub socket_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub retry_on_status: Vec<u16>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            connect_timeout_ms: 30000,
            socket_timeout_ms: 60000,
            request_timeout_ms: 120000,
            retry_on_status: vec![429, 500, 502, 503, 504],
        }
    }
}

// ===== 供应商枚举 =====

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LLMProvider {
    SiliconFlow,
    OpenRouter,
    OpenAI,
    DeepSeek,
    Generic,
}

impl LLMProvider {
    pub fn name(&self) -> &str {
        match self {
            LLMProvider::SiliconFlow => "SiliconFlow",
            LLMProvider::OpenRouter => "OpenRouter",
            LLMProvider::OpenAI => "OpenAI",
            LLMProvider::DeepSeek => "DeepSeek",
            LLMProvider::Generic => "Generic",
        }
    }

    /// 获取供应商默认 API 地址
    pub fn default_base_url(&self) -> &str {
        match self {
            LLMProvider::SiliconFlow => "https://api.siliconflow.cn",
            LLMProvider::OpenRouter => "https://openrouter.ai/api",
            LLMProvider::OpenAI => "https://api.openai.com",
            LLMProvider::DeepSeek => "https://api.deepseek.com",
            LLMProvider::Generic => "",
        }
    }

    /// 获取供应商默认模型名称
    pub fn default_model(&self) -> &str {
        match self {
            LLMProvider::SiliconFlow => "Qwen/Qwen2.5-7B-Instruct",
            LLMProvider::OpenRouter => "openai/gpt-4o",
            LLMProvider::OpenAI => "gpt-4o",
            LLMProvider::DeepSeek => "deepseek-chat",
            LLMProvider::Generic => "",
        }
    }
}

// ===== OpenAI 兼容基础接口 =====

/// OpenAI 兼容 API的基础客户端接口
/// 硅基流动、OpenRouter、DeepSeek 等都实现此接口
pub trait OpenAICompatClient: Send + Sync {
    /// 返回供应商信息
    fn provider(&self) -> LLMProvider;

    /// 非流式聊天
    fn chat(
        &mut self,
        messages: &[ChatMessage],
        model: &str,
        max_tokens: Option<u32>,
        sampling_params: Option<SamplingParams>,
        tools: Option<&[ToolDescriptor]>,
    ) -> Result<String, ClientError>;

    /// 带 Usage 的聊天
    fn chat_with_usage(
        &mut self,
        messages: &[ChatMessage],
        model: &str,
        max_tokens: Option<u32>,
        sampling_params: Option<SamplingParams>,
        tools: Option<&[ToolDescriptor]>,
    ) -> Result<ChatResult, ClientError>;

    /// 流式聊天回调
    /// on_chunk: 每个文本 chunk
    /// on_thinking: 推理过程 chunk (如 DeepSeek-R1 的 thought)
    fn chat_stream(
        &mut self,
        messages: &[ChatMessage],
        model: &str,
        max_tokens: Option<u32>,
        sampling_params: Option<SamplingParams>,
        tools: Option<&[ToolDescriptor]>,
        on_chunk: &mut dyn FnMut(&str) -> Result<(), ClientError>,
        on_thinking: &mut dyn FnMut(&str) -> Result<(), ClientError>,
    ) -> Result<ChatResult, ClientError>;

    /// 获取模型列表
    fn list_models(&self) -> Result<Vec<ModelInfo>, ClientError>;

    /// 关闭连接，释放资源
    fn close(&mut self);
}

// ===== 硅基流动专属接口 =====

/// 硅基流动客户端扩展接口
pub trait SiliconFlowExt: OpenAICompatClient {
    /// 查询账户余额
    /// GET /v1/user/info
    fn get_balance(&self) -> Result<BalanceInfo, ClientError>;

    /// 获取模型列表（带筛选）
    /// GET /v1/models?type=text&sub_type=chat
    fn list_models_filtered(
        &self,
        model_type: Option<&str>,
        sub_type: Option<&str>,
    ) -> Result<Vec<ModelInfo>, ClientError>;
}

// ===== OpenRouter 专属接口 =====

/// OpenRouter 模型端点信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterEndpoint {
    pub name: String,
    pub model_id: String,
    pub model_name: String,
    pub context_length: i64,
    pub pricing: OpenRouterPricing,
    pub provider_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterPricing {
    pub prompt: String,
    pub completion: String,
    pub request: String,
    pub image: String,
}

/// OpenRouter 模型端点响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelEndpoints {
    pub data: OpenRouterModelEndpointData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelEndpointData {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub endpoints: Vec<OpenRouterEndpoint>,
}

/// OpenRouter 生成信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterGeneration {
    pub data: OpenRouterGenerationData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterGenerationData {
    pub id: String,
    pub model: String,
    pub total_cost: f64,
    pub tokens_prompt: i64,
    pub tokens_completion: i64,
    pub finish_reason: Option<String>,
    pub provider_name: Option<String>,
    pub latency: Option<i64>,
    pub created_at: String,
}

/// OpenRouter 密钥信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterKeyInfo {
    pub data: OpenRouterKeyInfoData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterKeyInfoData {
    pub label: Option<String>,
    pub usage: Option<f64>,
    pub limit: Option<f64>,
    pub limit_remaining: Option<f64>,
    pub is_free_tier: Option<bool>,
    pub expires_at: Option<String>,
}

/// OpenRouter 客户端扩展接口
pub trait OpenRouterExt: OpenAICompatClient {
    /// 获取指定模型的端点信息
    /// GET /v1/models/:author/:slug/endpoints
    fn get_model_endpoints(
        &self,
        author: &str,
        slug: &str,
    ) -> Result<OpenRouterModelEndpoints, ClientError>;

    /// 获取生成请求的使用元数据
    /// GET /v1/generation?id=xxx
    fn get_generation(&self, generation_id: &str) -> Result<OpenRouterGeneration, ClientError>;

    /// 检查 API Key 的速率限制和剩余额度
    /// GET /v1/key
    fn get_key_info(&self) -> Result<OpenRouterKeyInfo, ClientError>;

    /// 获取原始模型列表 JSON（含完整定价信息）
    /// GET /v1/models?output_modalities=xxx&supported_parameters=xxx
    fn list_models_raw(
        &self,
        output_modalities: Option<&str>,
        supported_parameters: Option<&str>,
    ) -> Result<String, ClientError>;
}
