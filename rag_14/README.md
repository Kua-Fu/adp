# rag_14

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-0B4F6C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1F6FEB)](https://www.langchain.com/langgraph)
[![Weaviate](https://img.shields.io/badge/Weaviate-Embedded-00C9A7)](https://weaviate.io/)

这是一个基于 **LangChain + LangGraph + Weaviate Embedded** 的 RAG 示例模块，特点如下：

- 使用本地语料 `state_of_the_union.txt`；
- 工作流为 `retrieve -> generate` 两节点；
- 模型策略：**OpenAI-compatible 优先，Gemini 回退**；
- 已适配 `weaviate-client 4.x`；
- 增强了可读日志（启动/关闭进度条、检索摘要、最终回答）。

## 模块文件说明

- `main.py`：RAG 主程序（含模型初始化、向量化入库、LangGraph 执行、日志输出）。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型配置。
- `state_of_the_union.txt`：本地语料文件。
- `.weaviate_bin/`：Weaviate Embedded 下载/解压后的二进制目录。
- `.weaviate_data/`：Weaviate Embedded 数据目录（索引、分片、状态等）。

## 环境与安装

建议 Python 版本：`3.10 - 3.13`。

项目当前使用的 Python 环境：

```bash
/Users/yz/work/env/adp/.venv/bin/python -V
```

安装依赖：

```bash
/Users/yz/work/env/adp/.venv/bin/pip install -r /Users/yz/work/github/adp/rag_14/requirements.txt
```

## requirements 说明

`requirements.txt` 当前内容：

- `python-dotenv`：加载 `.env` 环境变量。
- `langchain-core`：LangChain 核心接口。
- `langchain-community`：`TextLoader` 等社区组件。
- `langchain-text-splitters`：文本切分器。
- `langgraph`：状态图工作流。
- `langchain-openai`：OpenAI-compatible 聊天模型接入。
- `langchain-google-genai`：Gemini 聊天/嵌入回退接入。
- `langchain-weaviate`：Weaviate v4 向量库封装。
- `weaviate-client`：Weaviate Python 客户端。
- `openai`：OpenAI SDK（用于兼容网关 embedding 适配）。

## `.env` 配置示例

`rag_14/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus
# 可选：若网关 embedding 模型有差异，建议显式设置
OPENAI_EMBEDDING_MODEL=text-embedding-v3

# Gemini（回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.0-flash
GOOGLE_EMBEDDING_MODEL=models/text-embedding-004
```

## 使用流程（详细）

1. 进入模块目录：

```bash
cd /Users/yz/work/github/adp/rag_14
```

2. 确认 `state_of_the_union.txt` 存在。

3. 配置 `.env`：优先填 OpenAI-compatible；如需回退，补充 Gemini 配置。

4. 运行程序：

```bash
/Users/yz/work/env/adp/.venv/bin/python main.py
```

5. 程序执行顺序：

- 启动 RAG 初始化（读取语料、模型初始化）；
- 启动 Weaviate Embedded；
- 文档切分并向量化写入 Weaviate；
- 执行 Query #1、Query #2；
- 关闭 Weaviate 并释放资源。

## Weaviate 运行产物说明

运行后通常会生成以下目录：

- `.weaviate_bin/`
  - 作用：存放 Weaviate Embedded 二进制。
  - 首次运行会自动下载/准备。
  - 可删除：可以。删除后下次运行会重新准备。

- `.weaviate_data/`
  - 作用：存放向量库数据（collection、shard、状态文件等）。
  - 与索引持久化有关，删除后会丢失已有索引数据。
  - 可删除：可以，但等价于“重建库”。

建议：
- 开发调试可保留，减少重复初始化；
- 如需完全重建，删除 `.weaviate_data` 后再运行 `main.py`。

## 实际运行输出（main.py）

以下内容为你提供的实测输出，已原样收录：

```text
(.venv) ➜  rag_14 git:(main) ✗ python main.py

========================================================================
[RAG-14] 启动 RAG 初始化
========================================================================
[进度] [#######---------------------]  25% | 读取并切分本地语料
[进度] [##############--------------]  50% | 模型初始化完成：模型后端：OpenAI-compatible（优先） | chat=qwen3-coder-plus | embedding=text-embedding-v3
模型后端：OpenAI-compatible（优先） | chat=qwen3-coder-plus | embedding=text-embedding-v3

========================================================================
[RAG-14] Weaviate 启动中
========================================================================
[进度] [#####################-------]  75% | Weaviate embedded 已连接
[进度] [############################] 100% | 向量库写入完成，RAG 可用

========================================================================
[RAG-14] 启动完成
========================================================================

========================================================================
[RAG-14] 问题 #1
========================================================================
[用户问题] What did the president say about Justice Breyer?
[检索结果] 原始 4 段，去重后 1 段
  1. 来源=state_of_the_union.txt | 摘要=Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar...
[最终回答]
The president honored Justice Breyer as someone who has dedicated his life to serving the country, calling him an Army veteran, Constitutional scholar, and retiring Supreme Court Justice. The president thanked Justice Breyer for his service and noted that one of the most serious constitutional responsibilities of a president is nominating someone to serve on the Supreme Court.

========================================================================
[RAG-14] 问题 #2
========================================================================
[用户问题] What did the president say about the economy?
[检索结果] 原始 4 段，去重后 1 段
  1. 来源=state_of_the_union.txt | 摘要=And it worked. It created jobs. Lots of jobs. In fact—our economy created over 6.5 Million new jobs just last year, more jobs created in one year t...
[最终回答]
The president said the economy created over 6.5 million new jobs last year, which is more than ever before in American history. The economy grew at a rate of 5.7%, representing the strongest growth in nearly 40 years. The president characterized this as fundamental change for an economy that hadn't worked well for working people for too long.

========================================================================
[RAG-14] Weaviate 关闭中
========================================================================
[进度] [#########-------------------]  33% | 发送关闭请求
{"build_git_commit":"62dcafac32","build_go_version":"go1.24.3","build_image_tag":"HEAD","build_wv_version":"1.30.5","error":"cannot find peer","level":"error","msg":"transferring leadership","time":"2026-04-20T10:58:55+08:00"}
[进度] [##################----------]  66% | 释放连接资源
[进度] [############################] 100% | 关闭完成

========================================================================
[RAG-14] 流程结束
========================================================================
```

## 常见问题

- 如果出现 embedding 503：
  - 通常是网关未开通默认 embedding 模型；
  - 建议在 `.env` 显式设置 `OPENAI_EMBEDDING_MODEL`（如 `text-embedding-v3`）。

- 如果首次启动较慢：
  - Weaviate Embedded 需要准备二进制并初始化本地数据目录。

- 如果要清理本地状态：
  - 删除 `.weaviate_data/` 可重建索引；
  - 删除 `.weaviate_bin/` 会触发下次重新准备二进制。
