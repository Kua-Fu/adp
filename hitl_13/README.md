# hitl_13

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google ADK](https://img.shields.io/badge/Google-ADK-4285F4?logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个 Human-in-the-Loop（HITL）技术支持智能体示例，演示：

- 技术支持 Agent 如何调用排障、建单、转人工工具；
- `before_model_callback` 如何把客户画像信息注入模型请求；
- 模型策略：OpenAI-compatible 优先，Gemini 回退；
- callback/tool 调用日志如何用于链路排查。

## 模块文件说明

- `main.py`：HITL 技术支持智能体主程序。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型配置。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少配置一套可用凭据（OpenAI-compatible 或 Gemini）。

本项目当前指定环境：

```bash
.venv/bin/python -V
```

安装依赖：

```bash
.venv/bin/pip install -r adp/hitl_13/requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `python-dotenv`：加载 `.env` 环境变量。
- `google-adk[extensions]`：ADK Agent、Runner、工具调用能力。
- `google-genai`：`google.genai.types` 消息类型支持。
- `litellm`：OpenAI-compatible 适配能力。

## .env 变量说明

在 `hitl_13/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus

# Gemini（当 OpenAI-compatible 不可用时回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash
```

## 运行方式

```bash
cd adp/hitl_13
.venv/bin/python main.py
```

## 实际运行输出（main.py）

下面内容来自你本地真实运行结果（2026-04-20，已原样整理）：

```text
python main.py
.venv/lib/python3.12/site-packages/google/adk/features/_feature_decorator.py:72: UserWarning: [EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled.
  check_feature_enabled()
模型后端：OpenAI-compatible（优先） -> openai/qwen3-coder-plus
请输入你遇到的技术问题：gpt 是如何构造的呢
[CALLBACK] 收到事件，author=technical_support_specialist

智能体输出：
您好，Alex Chen！感谢您联系我们的技术支持团队。我是您的技术支援专员。

我注意到您询问的是关于GPT的构造问题。作为我们尊贵的高级客户，您最近购买了Smart TV X1和Soundbar Pro产品。不过，您提出的问题似乎不是关于这些产品的技术故障。

如果您是想了解我们公司产品的AI功能或者智能特性，请告诉我具体是哪款设备，我很乐意为您解释相关的技术细节。如果您遇到了具体的设备使用问题，我也很愿意为您提供帮助。

请问您是否有关于Smart TV X1或Soundbar Pro的技术问题需要解决呢？

python main.py
.venv/lib/python3.12/site-packages/google/adk/features/_feature_decorator.py:72: UserWarning: [EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled.
  check_feature_enabled()
模型后端：OpenAI-compatible（优先） -> openai/qwen3-coder-plus
请输入你遇到的技术问题：Soundbar Pro产品不能使用了
[CALLBACK] 收到事件，author=technical_support_specialist
[CALLBACK] function_call -> name=troubleshoot_issue, args={'issue': 'Soundbar Pro not working'}
[CALLBACK] 收到事件，author=technical_support_specialist
[CALLBACK] function_response -> name=troubleshoot_issue, response={'status': 'success', 'report': 'Troubleshooting steps for Soundbar Pro not working.'}
[CALLBACK] 收到事件，author=technical_support_specialist

智能体输出：
感谢您联系我们，Alex Chen。我已经了解到您遇到的Soundbar Pro无法使用的问题。

根据我们初步的故障诊断，这里有一些基本的排查步骤：

1. 首先检查电源连接：
   - 确保电源线已牢固插入Soundbar和电源插座
   - 尝试更换插座以排除电源问题
   - 检查电源指示灯是否亮起

2. 检查所有音频连接线：
   - 确认HDMI/光纤/蓝牙连接正常
   - 检查线缆是否有损坏
   - 尝试重新插拔所有连接线

3. 尝试重置设备：
   - 断开电源30秒后重新连接
   - 查看是否有固件更新可用

如果这些基本步骤无法解决问题，您能详细描述一下具体的故障现象吗？例如，设备是否完全无反应，还是有声音输出问题等？这将帮助我们进一步确定解决方案。
```

## 说明

- 当问题描述不够明确时，模型可能先追问，不一定立即调用工具。
- 当问题明确是故障（如“不能使用了”）时，通常会先触发 `troubleshoot_issue`。
- 如果用户明确要求“转人工”，可进一步触发 `escalate_to_human`。
