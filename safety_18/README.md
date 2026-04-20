# safety_18

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Safety Guardrail](https://img.shields.io/badge/Guardrail-Input%20%2B%20Output-DC2626)](https://platform.openai.com/docs/guides/safety-best-practices)
[![OpenAI-compatible](https://img.shields.io/badge/Backend-OpenAI--compatible-0EA5E9)](https://platform.openai.com/docs/api-reference)
[![Gemini Fallback](https://img.shields.io/badge/Fallback-Gemini-34A853)](https://ai.google.dev/)

这是一个“**对话安全护栏（输入审核 + 输出复检）**”示例模块，特点如下：

- 在用户输入进入主模型前先做一次安全分类；
- 若输入不安全，直接拒绝并给出安全替代建议；
- 对模型输出再做一次安全复检，降低越狱/误答风险；
- 模型策略：**OpenAI-compatible 优先，Gemini 回退**；
- 支持通过 `.env` 配置检测模型与回答模型（避免写死不可用模型）。

## 模块文件说明

- `main.py`：主程序（环境加载、模型初始化、安全检测、回答生成、输出复检、交互循环）。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型与网关配置。

## 环境与安装

建议 Python 版本：`3.10 - 3.13`。

项目当前使用的 Python 环境：

```bash
.venv/bin/python -V
```

安装依赖：

```bash
.venv/bin/pip install -r adp/safety_18/requirements.txt
```

## requirements 说明

`requirements.txt` 当前内容：

- `python-dotenv`：加载 `.env` 环境变量；
- `openai`：OpenAI-compatible 调用；
- `google-generativeai`：Gemini 回退调用。

## `.env` 配置示例

`safety_18/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus

# 可选：按职责拆分模型；不配置则默认回落到 OPENAI_MODEL
OPENAI_GUARD_MODEL=qwen3-coder-plus
OPENAI_ANSWER_MODEL=qwen3-coder-plus

# Gemini（回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash
```

说明：
- `OPENAI_GUARD_MODEL` 用于输入/输出安全检测；
- `OPENAI_ANSWER_MODEL` 用于正常回答；
- 若 OpenAI-compatible 调用失败，且配置了 `GOOGLE_API_KEY`，会自动回退 Gemini；
- 若两者都未配置，程序会在启动时直接报错提示。

## 使用流程（详细）

1. 进入模块目录：

```bash
cd adp/safety_18
```

2. 配置 `.env`：
- 优先配置 OpenAI-compatible（`OPENAI_API_KEY` + `OPENAI_MODEL`）；
- 如需回退，补充 `GOOGLE_API_KEY`。

3. 运行程序：

```bash
.venv/bin/python main.py
```

4. 程序每轮执行顺序：

- 输入审核：`evaluate_safety(..., stage="input")`
- 回答生成：`generate_answer(...)`（仅在输入安全时执行）
- 输出复检：`evaluate_safety(..., stage="output")`
- 输出最终结果：安全则返回回答，不安全则返回替代提示

## 安全策略说明

当前策略会重点拦截以下风险类型：

- 自残/他伤、暴力、武器与爆炸物制作；
- 违法活动教程（入侵、诈骗、盗刷、制毒、洗钱等）；
- 仇恨/骚扰/歧视内容；
- 涉及未成年人的性相关内容；
- 提示词注入、绕过规则、越权与机密泄露诱导。

对于普通学习、科普、非操作性讨论，通常判定为可放行。

## 实际运行输出（示例）

```text
模型策略：OpenAI-compatible 优先，Gemini 回退
安全链路：输入审核 -> 回答生成 -> 输出复检
输入问题开始对话，输入 exit / quit / q 结束。

请输入问题：请帮我写一个学习 Python 的 7 天计划
[输入审核后端] OpenAI-compatible/qwen3-coder-plus
[输入审核结果] safe=True | category=safe | reason=普通学习需求，无明显风险。
[回答后端] OpenAI-compatible/qwen3-coder-plus
[输出复检后端] OpenAI-compatible/qwen3-coder-plus
[输出复检结果] safe=True | category=safe | reason=回答内容安全。
[最终回复]
（这里输出模型生成的学习计划）
```

## 常见问题

- 如果出现 503 或“模型无可用渠道”：
  - 请确认 `.env` 中 `OPENAI_MODEL` 是当前网关真实可用模型；
  - 或配置 `GOOGLE_API_KEY` 以启用 Gemini 回退。

- 如果输出被复检拦截：
  - 说明回答文本被判断为高风险；
  - 程序会自动返回更安全的替代回复，这是预期行为。

- 如果 Gemini 回退时报缺少依赖：
  - 安装 `google-generativeai`：
    `.venv/bin/pip install google-generativeai`
