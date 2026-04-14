use serde::{Deserialize, Serialize};

use crate::{ChatClient, Message};

// ===== Agent =====

/// Agent 循环中的单步
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub step_type: String,
    pub content: String,
}

/// Agent 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub success: bool,
    pub answer: String,
    pub steps: Vec<AgentStep>,
    pub error: Option<String>,
}

/// Agent 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_iterations: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
        }
    }
}

/// Agent 实例
pub struct Agent {
    config: AgentConfig,
    system_prompt: String,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        let system_prompt = r#"你是一个智能助手，请仔细思考并回答问题。

## 响应格式
请严格按照以下格式响应，使用 XML 标签标记每个部分：

<thought>你的思考过程</thought>
<final_answer>你的最终答案</final_answer>

## 规则
1. 保持思考过程简洁但有逻辑
2. 最终答案要清晰明确"#.into();

        Self {
            config,
            system_prompt,
        }
    }

    /// 执行 Agent Loop
    pub fn run(
        &self,
        client: &mut ChatClient,
        task: &str,
    ) -> AgentResult {
        let mut steps = Vec::new();
        let mut messages = vec![
            Message::system(&self.system_prompt),
            Message::user(format!("任务: {}", task)),
        ];

        for _iteration in 0..self.config.max_iterations {
            let request_body = serde_json::json!({
                "model": client.config.model,
                "messages": messages,
                "max_tokens": client.config.max_tokens,
                "temperature": client.config.temperature.unwrap_or(0.7),
            });

            let response = match client.client
                .post(&client.config.api_url)
                .header("Authorization", format!("Bearer {}", client.config.api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
            {
                Ok(resp) => resp,
                Err(e) => {
                    return AgentResult {
                        success: false,
                        answer: format!("请求失败: {}", e),
                        steps,
                        error: Some(e.to_string()),
                    };
                }
            };

            if response.status().as_u16() != 200 {
                return AgentResult {
                    success: false,
                    answer: format!("API 返回错误 (状态码: {})", response.status()),
                    steps,
                    error: Some(format!("HTTP {}", response.status())),
                };
            }

            let api_response: crate::ApiResponse = match response.json() {
                Ok(resp) => resp,
                Err(e) => {
                    return AgentResult {
                        success: false,
                        answer: format!("解析响应失败: {}", e),
                        steps,
                        error: Some(e.to_string()),
                    };
                }
            };

            let assistant_msg = match api_response.choices.first() {
                Some(c) => c.message.content.clone(),
                None => {
                    return AgentResult {
                        success: false,
                        answer: "AI 未返回有效回复".into(),
                        steps,
                        error: Some("空响应".into()),
                    };
                }
            };

            messages.push(Message::assistant(assistant_msg.clone()));

            // 查找 final_answer
            if let Some(final_answer) = Self::extract_tag(&assistant_msg, "final_answer") {
                steps.push(AgentStep {
                    step_type: "final_answer".into(),
                    content: final_answer.clone(),
                });
                return AgentResult {
                    success: true,
                    answer: final_answer,
                    steps,
                    error: None,
                };
            }

            // 提取 thought
            if let Some(thought) = Self::extract_tag(&assistant_msg, "thought") {
                steps.push(AgentStep {
                    step_type: "thought".into(),
                    content: thought,
                });
            }

            // 直接回复
            steps.push(AgentStep {
                step_type: "final_answer".into(),
                content: assistant_msg.clone(),
            });
            return AgentResult {
                success: true,
                answer: assistant_msg,
                steps,
                error: None,
            };
        }

        AgentResult {
            success: false,
            answer: "超过最大迭代次数限制".into(),
            steps,
            error: Some(format!("已达到 {} 次迭代上限", self.config.max_iterations)),
        }
    }

    fn extract_tag(content: &str, tag: &str) -> Option<String> {
        let open = format!("<{}>", tag);
        let close = format!("</{}>", tag);
        
        if let Some(start) = content.find(&open) {
            let start = start + open.len();
            if let Some(end) = content[start..].find(&close) {
                return Some(content[start..start + end].trim().to_string());
            }
        }
        None
    }
}
