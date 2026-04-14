# routing_2

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-0B5FFF)](https://www.langchain.com/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-AI_Platform-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)

这是一个“路由协调器”示例模块，包含两套实现：

- `main.py`：基于 LangChain LCEL 的轻量路由实现。
- `main_google_adk.py`：基于 Google ADK 的多代理协作实现。

两套实现都支持双后端选择逻辑（优先 OpenAI-compatible，回退 Gemini）。

## 环境依赖

运行本模块前，建议先确认以下前置条件：

- Python：建议 `3.10 - 3.13`（当前在 `3.14` 下可运行，但会看到部分三方库 warning）。
- 包管理：`pip` 可用。
- 网络：可访问你配置的模型服务地址（OpenAI-compatible 网关或 Gemini API）。
- 鉴权：至少准备一套可用凭证（OpenAI-compatible 或 Gemini）。

从零创建环境（推荐）：

```bash
cd /Users/yz/work/github/adp/routing_2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `langchain`：LCEL 链式表达与核心运行能力。
- `langchain-openai`：LangChain 对 OpenAI-compatible 模型的封装接入。
- `langgraph`：图式工作流相关能力。
- `google-cloud-aiplatform`：Google Cloud Vertex AI SDK。
- `langchain-google-genai`：LangChain 对 Gemini 的封装接入。
- `google-adk`：Google Agent Development Kit。
- `litellm`：让 ADK 可接 OpenAI-compatible 模型。
- `deprecated`：兼容某些依赖的废弃标记功能。
- `pydantic`：数据模型与类型校验。
- `python-dotenv`：自动加载 `.env` 环境变量。

依赖文件：

- `requirements.txt`

## .env 变量说明

在 `routing_2/.env` 中可配置两套变量（`main.py` 与 `main_google_adk.py` 都会优先使用 OpenAI-compatible）：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Gemini（当未配置 OPENAI_API_KEY+OPENAI_MODEL 时使用）
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

变量解释：

- `OPENAI_API_KEY`（可选）：OpenAI-compatible Key。
- `OPENAI_BASE_URL`（可选）：OpenAI-compatible 网关地址。
- `OPENAI_MODEL`（可选）：OpenAI-compatible 模型名（例如 `qwen3-coder-plus`）。
- `GOOGLE_API_KEY`（可选）：Gemini API Key。
- `GOOGLE_MODEL`（可选）：Gemini 模型名，默认 `gemini-2.5-flash`。

`main.py` 启动时会自动读取同目录 `.env`，无需手动 `source .env`。

## 脚本功能说明

- `main.py`：
  - 使用 LangChain LCEL + `RunnableBranch` 实现轻量路由链。
  - 路由目标是 3 类：`booker` / `info` / `unclear`。
  - 子处理器是本地模拟函数，适合快速验证“路由策略是否正确”。
  - 模型后端支持双模式：优先 OpenAI-compatible，缺失时回退 Gemini。
  - 适用场景建议：希望代码简洁、便于教学演示、快速调试路由规则时优先使用。

- `main_google_adk.py`：
  - 使用 Google ADK 的 `Agent + Runner + SessionService` 实现多代理协作。
  - 协调器（coordinator）按规则把请求委派给 `booking_agent` / `info_agent` / `unclear_agent`。
  - 通过 `FunctionTool` 绑定工具函数，展示 ADK 事件流与会话执行流程。
  - 同样支持双后端：优先 OpenAI-compatible，缺失时回退 Gemini。
  - 适用场景建议：需要会话管理、事件流、可扩展多代理编排时优先使用。

## （1）运行 `main.py`

推荐：先跑这个版本确认模型与环境配置正确，再切换到 ADK 版本。

```bash
cd /Users/yz/work/github/adp/routing_2
source .venv/bin/activate
python main.py
```

运行结果（本地实测）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

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

## （2）运行 `main_google_adk.py`

推荐：当 `main.py` 验证通过后，再用这个版本接入更真实的代理协作流程。

```bash
cd /Users/yz/work/github/adp/routing_2
source .venv/bin/activate
python main_google_adk.py
```

运行结果（本地实测）：

```text
=== Running Google ADK routing demo ===
Using OpenAI-compatible backend: openai/qwen3-coder-plus

--- Query: Book me a flight to London.
Result A: I'd be happy to help you book a flight to London. To proceed with your booking, I'll need some additional information:

1. Where will you be flying from?
2. What dates would you like to travel?
3. Are there any specific preferences (e.g., direct flights only, preferred airline)?

Could you please provide these details so I can search for available flights?

--- Query: What is the capital of Italy?
Result B: The capital of Italy is Rome.

--- Query: Tell me about quantum physics.
Result C: Quantum physics, also known as quantum mechanics, is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles. Here are some key concepts:

1. Wave-particle duality: Particles exhibit both wave-like and particle-like properties depending on how they are observed.
2. Uncertainty principle: Formulated by Heisenberg, it states that you cannot simultaneously know both the exact position and momentum of a particle.
3. Superposition: A quantum system can exist in multiple states at once until it is measured, at which point it collapses into one state.
4. Entanglement: Particles can become correlated in such a way that the state of one instantly influences the state of another, regardless of distance.
5. Quantization: Energy, angular momentum, and other quantities are often restricted to discrete values (quanta).
6. Schrödinger equation: This mathematical equation describes how the quantum state of a system evolves over time.
```
