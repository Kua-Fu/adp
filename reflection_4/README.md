# reflection_4

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible_API-111827)](https://platform.openai.com/docs/api-reference)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Google ADK](https://img.shields.io/badge/Google_ADK-Sequential_Agent-4285F4?logo=google&logoColor=white)](https://google.github.io/adk-docs/)

这是一个“反思迭代（Reflection Loop）”示例模块，包含两个脚本：

- `main.py`：可运行脚本，展示“生成 -> 审查 -> 改进”的代码反思循环。
- `main_google_adk.py`：按截图抄录的 ADK 顺序代理示例片段（用于讲解结构，不是完整可运行程序）。

`main.py` 支持双后端选择逻辑：优先 OpenAI-compatible，回退 Gemini。

## 环境依赖

运行本模块前建议确认：

- Python：建议 `3.10 - 3.13`（`3.14` 可运行，但可能出现部分三方 warning）。
- 包管理：`pip` 可用。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少配置一套可用凭证（OpenAI-compatible 或 Gemini）。

创建环境（推荐）：

```bash
cd adp/reflection_4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python 依赖清单

以下依赖与 `requirements.txt` 保持一致：

- `langchain`：LangChain 核心能力。
- `langchain-openai`：OpenAI-compatible 接入。
- `langchain-google-genai`：Gemini 接入。
- `python-dotenv`：自动加载 `.env`。

依赖文件：

- `requirements.txt`

## .env 变量说明

在 `reflection_4/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# Gemini（当未配置 OPENAI_API_KEY 时回退）
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0.1

# 反思循环最大轮数（可选）
REFLECTION_MAX_ITERATIONS=3
```

变量说明：

- `OPENAI_API_KEY`：OpenAI-compatible Key。
- `OPENAI_BASE_URL`：OpenAI-compatible 网关地址（可选）。
- `OPENAI_MODEL`：OpenAI-compatible 模型名（可选，默认 `gpt-4o`）。
- `OPENAI_TEMPERATURE`：OpenAI-compatible 温度参数（可选）。
- `GOOGLE_API_KEY`：Gemini API Key。
- `GOOGLE_MODEL`：Gemini 模型名（可选）。
- `GOOGLE_TEMPERATURE`：Gemini 温度参数（可选）。
- `REFLECTION_MAX_ITERATIONS`：反思循环最大迭代轮数（可选，默认 `3`）。

## 脚本功能说明

- `main.py`：
  - 首轮让模型根据任务生成一版 Python 代码。
  - 第二阶段让模型切换“代码审查员”角色，输出 critique。
  - 如果审查结果为 `CODE_IS_PERFECT` 则提前停止，否则继续迭代改进。
  - 适合演示“自我反馈优化”工作流。

- `main_google_adk.py`：
  - 内容按截图抄录，展示 `LlmAgent + SequentialAgent` 的写作-审查流水线思路。
  - 这是教学片段，不是完整可执行程序（按需求未补全运行入口与上下文）。

## （1）运行 `main.py`

```bash
cd adp/reflection_4
source .venv/bin/activate
python main.py
```

运行结果（本地实测，节选）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

========================= REFLECTION LOOP: ITERATION 1 =========================

>>> STAGE 1: GENERATING initial code...
--- Generated Code (v1) ---
def calculate_factorial(n):
    ...

>>> STAGE 2: REFLECTING on the generated code...
--- Critique ---
No further critiques found. The code is satisfactory.

============================== FINAL RESULT ==============================
Final refined code after the reflection process:
def calculate_factorial(n):
    ...
```

## （2）查看 `main_google_adk.py`

该文件用于展示 ADK 顺序编排结构（`generator -> reviewer`），可直接阅读其注释理解执行链路：

- `generator` 将草稿写入 `state['draft_text']`
- `reviewer` 读取草稿并将结构化审查结果写入 `state['review_output']`

说明：该文件当前是示例片段，不作为本模块运行入口。
