# routing_2

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-0B5FFF)](https://www.langchain.com/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-AI_Platform-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)

这是一个基于 LangChain 的“路由协调器”示例：  
先让模型判断请求属于 `booker / info / unclear`，再分发到对应处理器。

## Requirements（依赖清单）

本模块依赖如下：

- `langchain`：LCEL 链式表达与核心运行能力。
- `langgraph`：图式工作流相关能力。
- `google-cloud-aiplatform`：Google Cloud Vertex AI SDK。
- `langchain-google-genai`：LangChain 对 Gemini 的封装接入。
- `google-adk`：Google Agent Development Kit。
- `deprecated`：兼容某些依赖的废弃标记功能。
- `pydantic`：数据模型与类型校验。
- `python-dotenv`：自动加载 `.env` 环境变量。

依赖文件：

- `requirements.txt`

安装方式：

```bash
cd /Users/yz/work/github/adp/routing_2
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## .env 变量说明

在 `routing_2/.env` 中配置以下变量：

```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

变量解释：

- `GOOGLE_API_KEY`（必填）：Gemini API Key。
- `GOOGLE_MODEL`（可选）：模型名，默认 `gemini-2.5-flash`。

`main.py` 启动时会自动读取同目录 `.env`，无需手动 `source .env`。

## 运行方式

```bash
cd /Users/yz/work/github/adp/routing_2
source .venv/bin/activate
python main.py
```

## 实际运行结果（本地）

本地一次实际运行输出如下：

```text
Language model initialized: gemini-2.5-flash

--- Running with a booking request ---

--- DELEGATING TO BOOKING HANDLER ---
Final Result A: Booking handler processed request: 'Book me a flight to London.'. Result: Simulated booking action.

--- Running with an info request ---

--- DELEGATING TO INFO HANDLER ---
Final Result B: Info handler processed request: 'What is the capital of Italy?'. Result: Simulated information retrieval.

--- Running with an unclear request ---

--- DELEGATING TO INFO HANDLER ---
Final Result C: Info handler processed request: 'Tell me about quantum physics.'. Result: Simulated information retrieval.
```

结果解读：

- `request_a`（订票）命中 `booker` 分支。
- `request_b`（常规问答）命中 `info` 分支。
- `request_c` 当前也命中 `info` 分支，说明当前提示词下模型倾向把此问题归入通用信息类。
