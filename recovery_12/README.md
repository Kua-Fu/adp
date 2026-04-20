# recovery_12

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Google ADK](https://img.shields.io/badge/Google-ADK-4285F4?logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“恢复链路（Recovery Chain）”示例模块，演示：

- `primary_handler` 优先做精确定位；
- `fallback_handler` 在主路径失败时做区域级回退；
- `response_agent` 统一汇总并输出结果；
- 支持 callback/tool 调用日志，方便观察代理链执行轨迹。

当前模型策略：
- 若配置了 OpenAI-compatible（`OPENAI_API_KEY + OPENAI_MODEL`），固定使用 OpenAI-compatible；
- 仅当未配置 OpenAI-compatible 时，才使用 Gemini。

## 模块文件说明

- `main.py`：顺序代理主程序（含 runner、callback 日志、交互式输入）。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型配置。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少配置一套可用凭据。

本项目当前指定环境：

```bash
.venv/bin/python -V
```

安装依赖：

```bash
.venv/bin/pip install -r adp/recovery_12/requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `python-dotenv`：加载 `.env` 环境变量。
- `google-adk[extensions]`：ADK 代理、Runner、工具调用能力。
- `google-genai`：`google.genai.types` 消息对象类型支持。
- `litellm`：OpenAI-compatible 适配能力（由 ADK 扩展链路使用）。

## .env 变量说明

在 `recovery_12/.env` 中可配置：

```env
# OpenAI-compatible（若配置则固定优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus

# Gemini（仅在未配置 OpenAI-compatible 时使用）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash
```

## 运行方式

```bash
cd adp/recovery_12
.venv/bin/python main.py
```

运行后输入地址，例如：`不存在的地址`。

## 运行输出（本地实测）

运行结果（本地实测，2026-04-18）：

```text
python main.py
.venv/lib/python3.12/site-packages/google/adk/features/_feature_decorator.py:72: UserWarning: [EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled.
  check_feature_enabled()
模型后端：OpenAI-compatible（固定） -> openai/qwen3-coder-plus
请输入要查询的地址：不存在的地址
已收到地址：不存在的地址
[CALLBACK] 收到事件，author=primary_handler
[CALLBACK] function_call -> name=get_precise_location_info, args={"address": "不存在的地址"}
[TOOL] 调用 get_precise_location_info(address='不存在的地址')
[TOOL] get_precise_location_info 返回: [PRECISE_FAIL] 未检索到精确门牌级结果。建议提取城市信息后执行区域级检索。
[CALLBACK] 收到事件，author=primary_handler
[CALLBACK] function_response -> name=get_precise_location_info, response={"result": "[PRECISE_FAIL] 未检索到精确门牌级结果。建议提取城市信息后执行区域级检索。"}
[CALLBACK] 收到事件，author=primary_handler
[CALLBACK] 收到事件，author=fallback_handler
[CALLBACK] function_call -> name=get_general_area_info, args={"city": "北京"}
[TOOL] 调用 get_general_area_info(city='北京')
[TOOL] get_general_area_info 返回: 已回退到区域级定位：北京。当前返回该城市的中心区域与通用地理信息（示例结果）。
[CALLBACK] 收到事件，author=fallback_handler
[CALLBACK] function_response -> name=get_general_area_info, response={"result": "已回退到区域级定位：北京。当前返回该城市的中心区域与通用地理信息（示例结果）。"}
[CALLBACK] 收到事件，author=fallback_handler
[CALLBACK] 收到事件，author=response_agent
智能体输出：
根据系统检索结果，您提供的地址"不存在的地址"无法精确定位到具体的门牌信息。系统已为您回退到区域级定位，当前位置为：**北京**。

由于原始地址信息不够具体，目前返回的是北京市的中心区域与通用地理信息。如需更精确的位置信息，建议您提供更详细的地址描述。
```

## 常见问题

- 如果出现 `OpenAIException - Connection error`，通常是 OpenAI-compatible 网关连通性问题；
- 当前 `main.py` 已支持 callback/tool 级别日志，可快速定位是模型调用失败还是代理链逻辑问题。
