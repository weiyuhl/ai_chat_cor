use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{ChatClient, Message};

// ===== 工具定义 =====

/// 工具参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: String,
    pub required: bool,
}

/// 工具定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParam>,
}

/// 工具调用请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: HashMap<String, String>,
}

/// 工具执行结果
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_name: String,
    pub output: String,
    pub success: bool,
}

/// Agent 循环中的单步
#[derive(Debug, Clone)]
pub struct AgentStep {
    pub step_type: String, // "thought" | "action" | "observation" | "final_answer"
    pub content: String,
}

/// Agent 执行结果
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub success: bool,
    pub answer: String,
    pub steps: Vec<AgentStep>,
    pub error: Option<String>,
}

// ===== 内置工具 =====

/// 内置工具枚举
#[derive(Debug, Clone)]
pub enum BuiltinTool {
    Calculator,
    CurrentTime,
    Echo,
}

impl BuiltinTool {
    pub fn name(&self) -> &str {
        match self {
            BuiltinTool::Calculator => "calculator",
            BuiltinTool::CurrentTime => "current_time",
            BuiltinTool::Echo => "echo",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            BuiltinTool::Calculator => "执行数学计算，如: calculator(2 + 3 * 4)",
            BuiltinTool::CurrentTime => "获取当前日期和时间",
            BuiltinTool::Echo => "回显输入内容，用于测试",
        }
    }

    pub fn execute(&self, args: &HashMap<String, String>) -> String {
        match self {
            BuiltinTool::Calculator => {
                let expr = args.get("expression")
                    .map(|s| s.as_str())
                    .unwrap_or("0");
                Self::calc_expression(expr)
            }
            BuiltinTool::CurrentTime => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = now.as_secs();
                let days = secs / 86400;
                let hours = (secs % 86400) / 3600;
                let mins = (secs % 3600) / 60;
                let secs = secs % 60;
                format!("Unix时间戳: {} ({}天 {}:{:02}:{:02} UTC)", secs, days, hours, mins, secs)
            }
            BuiltinTool::Echo => {
                args.get("text")
                    .map(|s| s.clone())
                    .unwrap_or_else(|| "(空)".into())
            }
        }
    }

    fn calc_expression(expr: &str) -> String {
        // 简单的表达式计算器
        let expr = expr.trim();
        if expr.is_empty() {
            return "错误: 表达式为空".into();
        }

        // 尝试解析并计算
        match Self::eval(expr) {
            Ok(result) => format!("{} = {}", expr, result),
            Err(e) => format!("计算错误: {}", e),
        }
    }

    fn eval(expr: &str) -> Result<f64, String> {
        let expr = expr.trim();
        // 支持简单的加减乘除
        if expr.contains('+') {
            let parts: Vec<&str> = expr.splitn(2, '+').collect();
            if parts.len() == 2 {
                let left = Self::eval(parts[0].trim())?;
                let right = Self::eval(parts[1].trim())?;
                return Ok(left + right);
            }
        }
        if expr.contains('-') {
            let parts: Vec<&str> = expr.splitn(2, '-').collect();
            if parts.len() == 2 && !parts[0].trim().is_empty() {
                let left = Self::eval(parts[0].trim())?;
                let right = Self::eval(parts[1].trim())?;
                return Ok(left - right);
            }
        }
        if expr.contains('*') {
            let parts: Vec<&str> = expr.splitn(2, '*').collect();
            if parts.len() == 2 {
                let left = Self::eval(parts[0].trim())?;
                let right = Self::eval(parts[1].trim())?;
                return Ok(left * right);
            }
        }
        if expr.contains('/') {
            let parts: Vec<&str> = expr.splitn(2, '/').collect();
            if parts.len() == 2 {
                let left = Self::eval(parts[0].trim())?;
                let right = Self::eval(parts[1].trim())?;
                if right == 0.0 {
                    return Err("除以零".into());
                }
                return Ok(left / right);
            }
        }
        // 尝试解析为数字
        expr.parse::<f64>().map_err(|_| format!("无法解析: {}", expr))
    }

    pub fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            name: self.name().into(),
            description: self.description().into(),
            parameters: match self {
                BuiltinTool::Calculator => vec![ToolParam {
                    name: "expression".into(),
                    param_type: "string".into(),
                    description: "数学表达式，如: 2 + 3 * 4".into(),
                    required: true,
                }],
                BuiltinTool::CurrentTime => vec![],
                BuiltinTool::Echo => vec![ToolParam {
                    name: "text".into(),
                    param_type: "string".into(),
                    description: "要回显的文本".into(),
                    required: true,
                }],
            },
        }
    }
}

// ===== Agent =====

/// Agent 配置
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_iterations: usize,
    pub tools: Vec<BuiltinTool>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tools: vec![
                BuiltinTool::Calculator,
                BuiltinTool::CurrentTime,
                BuiltinTool::Echo,
            ],
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
        let tools_desc: Vec<String> = config.tools.iter()
            .map(|t| {
                let params: Vec<String> = t.to_tool_def().parameters.iter()
                    .map(|p| format!("    - {}: {} ({})", p.name, p.param_type, p.description))
                    .collect();
                let params_str = if params.is_empty() {
                    "  无参数".into()
                } else {
                    params.join("\n")
                };
                format!("### {}\n{}\n{}", t.name(), t.description(), params_str)
            })
            .collect();

        let system_prompt = format!(
            r#"你是一个智能助手，可以使用工具来完成任务。

## 可用工具
{}

## 响应格式
请严格按照以下格式响应，使用 XML 标签标记每个部分：

<thought>你的思考过程</thought>
<action>工具名称</action>
<action_input>工具参数，格式: key=value, key2=value2</action_input>

当你得到最终答案时，使用:
<final_answer>你的最终答案</final_answer>

## 规则
1. 每次只能调用一个工具
2. 等待工具返回结果后再继续
3. 如果可以直接回答用户问题，使用 final_answer
4. 保持思考过程简洁但有逻辑"#,
            tools_desc.join("\n\n")
        );

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
            // 构建请求体
            let request_body = serde_json::json!({
                "model": client.config.model,
                "messages": messages,
                "max_tokens": client.config.max_tokens,
                "temperature": client.config.temperature.unwrap_or(0.7),
            });

            // 发送请求
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
                Some(c) => &c.message.content,
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

            // 解析 AI 回复
            if let Some(final_answer) = Self::extract_tag(assistant_msg, "final_answer") {
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
            if let Some(thought) = Self::extract_tag(assistant_msg, "thought") {
                steps.push(AgentStep {
                    step_type: "thought".into(),
                    content: thought,
                });
            }

            // 提取 action
            if let Some(action) = Self::extract_tag(assistant_msg, "action") {
                let action_input = Self::extract_tag(assistant_msg, "action_input")
                    .unwrap_or_default();

                steps.push(AgentStep {
                    step_type: "action".into(),
                    content: format!("{}({})", action, action_input),
                });

                // 执行工具
                let result = self.execute_tool(&action, &action_input);
                steps.push(AgentStep {
                    step_type: "observation".into(),
                    content: result.clone(),
                });

                // 将工具结果添加到对话历史
                messages.push(Message::user(format!(
                    "工具 {} 执行结果:\n{}",
                    action, result
                )));
            } else {
                // 没有 action 也没有 final_answer，可能是直接回复
                if !steps.iter().any(|s| s.step_type == "final_answer") {
                    steps.push(AgentStep {
                        step_type: "final_answer".into(),
                        content: assistant_msg.clone(),
                    });
                    return AgentResult {
                        success: true,
                        answer: assistant_msg.clone(),
                        steps,
                        error: None,
                    };
                }
            }
        }

        // 超过最大迭代次数
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

    fn execute_tool(&self, tool_name: &str, args_str: &str) -> String {
        // 解析参数
        let mut args = HashMap::new();
        for pair in args_str.split(',') {
            let pair = pair.trim();
            if let Some(eq_pos) = pair.find('=') {
                let key = pair[..eq_pos].trim().to_string();
                let value = pair[eq_pos + 1..].trim().to_string();
                args.insert(key, value);
            }
        }

        // 查找并执行工具
        for tool in &self.config.tools {
            if tool.name() == tool_name {
                return tool.execute(&args);
            }
        }

        format!("错误: 未知工具 '{}'", tool_name)
    }
}
