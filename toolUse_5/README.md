# toolUse_5

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Tool%20Calling-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agent%20Orchestration-111827)](https://www.crewai.com/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“工具调用（Tool Use）”示例模块，包含两套实现：

- `main.py`：基于 LangChain Agent 的工具调用示例。
- `main_crew_ai.py`：基于 CrewAI 的 Agent + Task + Crew 编排示例。

两套实现都支持后端选择逻辑：优先 OpenAI-compatible，回退 Gemini。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少准备一套可用凭证（OpenAI-compatible 或 Gemini）。

推荐安装：

```bash
cd adp/toolUse_5
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `langchain`：LangChain Agent 与工具调用主框架。
- `langchain-openai`：OpenAI-compatible 模型封装。
- `langchain-google-genai`：Gemini 模型封装。
- `crewai`：CrewAI 多智能体编排框架（`main_crew_ai.py` 使用）。
- `python-dotenv`：加载 `.env` 环境变量。
- `nest-asyncio`：兼容异步事件循环（`main.py` 使用）。

## .env 变量说明

在 `toolUse_5/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus
OPENAI_TEMPERATURE=0

# Gemini（当未配置 OPENAI_API_KEY 时使用）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash
GOOGLE_TEMPERATURE=0
```

## 两个主函数说明

- `main.py` 中的 `main()`：
  - 并发发起多条查询（首都、天气、常识问答）。
  - 通过工具调用 Agent 自动判断是否调用 `search_information`。
  - 展示“工具调用 + 最终回答”的完整流程。

- `main_crew_ai.py` 中的 `main_crew_ai()`：
  - 构建一个金融分析 Crew（Agent + Task + Crew）。
  - 调用 `Stock Price Lookup Tool` 获取模拟股价。
  - 由 Agent 产出最终一句话结论。

## （1）运行 `main.py`

```bash
cd adp/toolUse_5
python main.py
```

运行结果（本地实测）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

--- Running Agent with Query: 'What is the capital of France?' ---

--- Running Agent with Query: 'What's the weather like in London?' ---

--- Running Agent with Query: 'Tell me something about dogs.' ---

--- TOOL CALLED: search_information(query='capital of france') ---
--- TOOL RESULT: The capital of France is Paris. ---

--- TOOL CALLED: search_information(query='weather in london') ---
--- TOOL RESULT: The weather in London is currently cloudy with a temperature of 15C. ---

--- Final Agent Response ---
The capital of France is Paris.

--- Final Agent Response ---
The weather in London is currently cloudy with a temperature of 15°C.

--- Final Agent Response ---
Dogs are one of the most popular pets worldwide and are known for their loyalty, companionship, and diverse range of breeds. They come in various sizes, from tiny Chihuahuas to large Great Danes, and serve many roles, including family companions, working animals, and service animals. Dogs are highly social creatures and often form strong bonds with humans and other animals. They also possess a keen sense of smell and hearing, making them excellent at tasks such as search and rescue, detecting substances, and guarding.
```

## （2）运行 `main_crew_ai.py`

```bash
cd adp/toolUse_5
python main_crew_ai.py
```

运行结果（本地实测）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

Starting the Financial Crew...
-----------------------------------
...
Tool call: get_stock_price for ticker 'AAPL'
...
Final Answer:
The simulated stock price for AAPL is $178.15.
...
Crew execution finished.
Final Result:
The simulated stock price for AAPL is $178.15.
```

