# mcp_10

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Server%20%26%20Client-0B5FFF)](https://gofastmcp.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Tools%20Agent-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“MCP 工具调用智能体”示例模块，演示：

- 使用 `FastMCP` 创建并运行 MCP Server（`fastmcp_server.py`）。
- MCP Server 暴露 `greet` 工具，返回个性化问候语。
- 智能体（`main.py`）通过 MCP Client 连接 Server 并调用工具。
- 模型后端策略：OpenAI-compatible 优先，Gemini 回退。
- 交互式对话：可持续多轮输入，`exit/quit/q` 退出。

## 模块文件说明

- `fastmcp_server.py`：MCP 服务端，注册 `greet(name: str) -> str` 工具。
- `main.py`：交互式智能体入口，负责模型调用、工具决策与 MCP 转发。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型与服务配置。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址（OpenAI-compatible 或 Gemini）。
- 鉴权：至少准备一套可用凭证（OpenAI-compatible 或 Gemini）。

本仓库当前已使用隔离虚拟环境（推荐）：

```bash
/Users/yz/work/env/adp/.venv_mcp_10/bin/python -V
```

若需安装依赖：

```bash
/Users/yz/work/env/adp/.venv_mcp_10/bin/pip install -r /Users/yz/work/github/adp/mcp_10/requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `fastmcp`：MCP Server / Client 能力。
- `python-dotenv`：加载 `.env` 环境变量。
- `langchain-core`：消息对象与工具抽象。
- `langchain-openai`：OpenAI-compatible 模型封装。
- `langchain-google-genai`：Gemini 模型封装（回退路径）。

## .env 变量说明

在 `mcp_10/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0

# Gemini（OpenAI-compatible 调用失败时回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0

# 可选：MCP 服务端地址（默认 http://127.0.0.1:8000/mcp）
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
```

## 运行步骤

需要两个终端。

1. 终端 A：启动 MCP Server

```bash
cd /Users/yz/work/github/adp/mcp_10
/Users/yz/work/env/adp/.venv_mcp_10/bin/python fastmcp_server.py
```

2. 终端 B：启动交互式智能体

```bash
cd /Users/yz/work/github/adp/mcp_10
/Users/yz/work/env/adp/.venv_mcp_10/bin/python main.py
```

3. 在智能体终端输入问题，示例：

```text
请通过工具向 Alice 打个招呼，并用中文说明你做了什么。
```

4. 退出交互：输入 `exit` / `quit` / `q`。

## 运行输出（本地实测）

运行结果（本地实测，2026-04-18）：

```text
## Running mcp_10 interactive agent ##
模型初始化状态：仅 OpenAI-compatible 就绪：qwen3-coder-plus
MCP Server URL: http://127.0.0.1:8000/mcp
MCP 工具列表：['greet']

已进入交互模式。
输入 `exit` / `quit` / `q` 可退出。
示例：请通过工具向 Alice 打个招呼，并用中文说明你做了什么。

用户: [INFO] 后端：OpenAI-compatible（已执行工具调用）
助手: 我已经通过工具向 Alice 打了个招呼，发送了"Hello, Alice! Nice to meet you."这条问候信息。

用户: 会话已结束。
```

## 常见问题

1. 提示无法连接 MCP Server

- 先确认 `fastmcp_server.py` 是否已在另一个终端运行。
- 确认 `MCP_SERVER_URL` 与服务监听地址一致（默认 `http://127.0.0.1:8000/mcp`）。

2. 没触发工具调用

- 在输入中明确“打招呼/问候某人”的意图，更容易触发 `mcp_greet`。
- 若输入是普通闲聊，模型可能直接回答，不一定调用 MCP 工具。

3. OpenAI-compatible 调用失败

- 程序会自动尝试 Gemini（前提是配置了 `GOOGLE_API_KEY`）。
- 若两者都不可用，程序会输出明确错误原因。
