# parallelization_3

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-0B5FFF)](https://www.langchain.com/langgraph)
[![Google ADK](https://img.shields.io/badge/Google_ADK-Agent_Framework-4285F4?logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-OpenAI--Compatible-111827)](https://github.com/BerriAI/litellm)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“并行化处理（Parallelization）”示例模块，包含两套实现：

- `main.py`：基于 LangChain `RunnableParallel` 的并行链实现。
- `main_google_adk.py`：基于 Google ADK 的多代理并行研究 + 串行综合实现。

两套实现都支持双后端选择逻辑：优先 OpenAI-compatible，回退 Gemini。

## 环境依赖

推荐环境：

- Python：`3.10 - 3.13`（`3.14` 也可运行，但可能出现三方库 warning）。
- pip：可用。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少配置一套可用凭证（OpenAI-compatible 或 Gemini）。

创建环境：

```bash
cd /Users/yz/work/github/adp/parallelization_3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python 依赖清单

以下内容与 `requirements.txt` 完全一致：

- `langchain`
- `langchain-openai`
- `langgraph`
- `langchain-google-genai`
- `google-cloud-aiplatform`
- `google-adk`
- `litellm`
- `deprecated`
- `pydantic`
- `python-dotenv`

依赖文件：

- `requirements.txt`

## .env 变量说明

在 `parallelization_3/.env` 中配置（两脚本通用）：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# Gemini（当未配置 OPENAI_API_KEY 时回退）
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0.7
```

变量说明：

- `OPENAI_API_KEY`：OpenAI-compatible Key。
- `OPENAI_BASE_URL`：OpenAI-compatible 网关地址（可选）。
- `OPENAI_MODEL`：OpenAI-compatible 模型名。
- `OPENAI_TEMPERATURE`：`main.py` 使用的温度参数（可选）。
- `GOOGLE_API_KEY`：Gemini API Key。
- `GOOGLE_MODEL`：Gemini 模型名（可选）。
- `GOOGLE_TEMPERATURE`：`main.py` 的 Gemini 温度参数（可选）。

## 脚本功能说明

- `main.py`：
  - 用 `RunnableParallel` 并行执行 3 个子任务：摘要、问题生成、关键词提取。
  - 将并行结果喂给综合提示词，产出最终回答。
  - 使用 `ainvoke` 展示异步并行链调用方式。

- `main_google_adk.py`：
  - 3 个研究代理并行执行：`RenewableEnergyResearcher`、`EVResearcher`、`CarbonCaptureResearcher`。
  - 每个研究代理写入 `output_key` 到状态。
  - `FinalSynthesisAgent` 串行收口，融合三个结果输出最终总结。
  - 后端是 Gemini 时使用 `google_search`；OpenAI-compatible 时自动回退到本地 `FunctionTool`，避免工具兼容问题。

## （1）运行 `main.py`

```bash
cd /Users/yz/work/github/adp/parallelization_3
source .venv/bin/activate
python main.py
```

运行结果（本地实测，节选）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

--- Running Parallel LangChain Example for Topic: 'The history of space exploration' ---

--- Final Response ---
# The History of Space Exploration: A Journey from Competition to Collaboration
...
```

## （2）运行 `main_google_adk.py`

```bash
cd /Users/yz/work/github/adp/parallelization_3
source .venv/bin/activate
python main_google_adk.py
```

运行结果（本地实测，节选）：

```text
=== Running Google ADK Parallelization Demo ===
Using OpenAI-compatible backend: openai/qwen3-coder-plus

--- Query 1: Provide a comprehensive analysis of sustainable energy advancements and EV technology trends in recent years, including carbon capture progress.
Result A: # Final Synthesis: Sustainable Energy, EV Technology, and Carbon Capture Integration
...
```
