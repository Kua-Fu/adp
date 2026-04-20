# resource_16

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI-compatible](https://img.shields.io/badge/Backend-OpenAI--compatible-0EA5E9)](https://platform.openai.com/docs/api-reference)
[![Gemini Fallback](https://img.shields.io/badge/Fallback-Gemini-34A853)](https://ai.google.dev/)
[![Google CSE](https://img.shields.io/badge/Search-Google%20Custom%20Search-4285F4)](https://developers.google.com/custom-search/v1/overview)

这是一个“**问题分类 + 可选联网检索 + 回答生成**”示例模块，特点如下：

- 先将问题分类为 `simple / reasoning / internet_search`；
- 当分类为 `internet_search` 时，调用 Google Custom Search；
- 模型策略：**OpenAI-compatible 优先，Gemini 回退**；
- 模型选择优先读取 `.env`，避免调用不存在模型；
- 关键流程均有详细中文注释，方便学习与二次改造。

## 模块文件说明

- `main.py`：主程序（环境加载、分类、检索、回答生成、交互循环）。
- `.env`：模型与检索配置。

## 环境与安装

建议 Python 版本：`3.10 - 3.13`。

项目当前使用的 Python 环境：

```bash
.venv/bin/python -V
```

安装依赖（如未安装）：

```bash
.venv/bin/pip install python-dotenv requests openai google-generativeai
```

## `.env` 配置说明

`resource_16/.env` 当前可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus

# 可选：按阶段覆盖模型；不配置时默认回落到 OPENAI_MODEL
OPENAI_CLASSIFIER_MODEL=qwen3-coder-plus
OPENAI_SIMPLE_MODEL=qwen3-coder-plus
OPENAI_REASONING_MODEL=qwen3-coder-plus
OPENAI_SEARCH_MODEL=qwen3-coder-plus

# Gemini（回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash

# Google Custom Search（联网检索）
CSE_API_KEY=your_google_cse_api_key
CSE_ID=your_google_cse_id
```

说明：
- `CSE_API_KEY` 与 `CSE_ID` 仅用于 `internet_search` 分类；
- 若未配置 CSE，程序仍可运行，但联网检索会返回配置缺失提示；
- 若 OpenAI-compatible 调用失败，且配置了 `GOOGLE_API_KEY`，会自动回退 Gemini。

## 使用流程（详细）

1. 进入模块目录：

```bash
cd adp/resource_16
```

2. 配置 `.env`：
- 优先配置 OpenAI-compatible（`OPENAI_API_KEY` + `OPENAI_MODEL`）；
- 如需回退，补充 `GOOGLE_API_KEY`；
- 如需联网检索，补充 `CSE_API_KEY` 与 `CSE_ID`。

3. 运行程序：

```bash
.venv/bin/python main.py
```

4. 交互使用：
- 输入自然语言问题；
- 输入 `exit` / `quit` / `q` 退出。

## 程序执行逻辑

每轮问答按以下顺序执行：

1. `classify_prompt()`：先做问题分类；
2. `internet_search` 才会调用 `google_search()`；
3. `generate_response()` 根据分类选择模型生成答案；
4. 若 OpenAI-compatible 报错，自动尝试 Gemini 回退。

## 实际运行输出（示例）

```text
模型策略：OpenAI-compatible 优先，Gemini 回退
输入问题开始对话，输入 exit / quit / q 结束。

请输入问题：今天星期几
[分类] simple | 后端: OpenAI-compatible/qwen3-coder-plus
[分类原因] 这是一个直接事实问题，不需要联网检索。
[回答后端] OpenAI-compatible/qwen3-coder-plus
[回答]
今天是星期一。
```

## 常见问题

- 如果出现模型不可用（例如 503）：
  - 请先确认 `.env` 中 `OPENAI_MODEL` 是否是当前网关可用模型；
  - 或配置 `GOOGLE_API_KEY` 启用 Gemini 回退。

- 如果联网检索无结果或报配置错误：
  - 检查 `CSE_API_KEY` 与 `CSE_ID` 是否有效；
  - 确认 Google Custom Search API 已开通，搜索引擎范围配置正确。
