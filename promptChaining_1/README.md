# promptChaining_1

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible_API-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-0B5FFF)](https://www.langchain.com/langgraph)

这个示例使用 LangChain 的 LCEL（`|` 管道）实现一个两段式链路：

1. 从原始文本中提取技术规格（Extraction）
2. 将提取结果转换为 JSON（Transform）

入口文件是 `main.py`。

## Requirements（依赖清单）

本项目 Python 依赖如下：

- `langchain`：LCEL 链式表达和核心运行能力。
- `langchain-community`：社区扩展能力（工具、集成等）。
- `langchain-openai`：OpenAI / OpenAI-compatible 模型接入。
- `langgraph`：图式工作流能力（本示例环境中一并安装）。
- `python-dotenv`：从 `.env` 加载环境变量。

依赖文件：

- `requirements.txt`

安装方式：

```bash
cd adp/promptChaining_1
source .venv/bin/activate
pip install -r requirements.txt
```

## .env 变量说明

在 `promptChaining_1/.env` 中配置以下变量：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=qwen3-coder-plus
OPENAI_TEMPERATURE=0
```

变量解释：

- `OPENAI_API_KEY`（必填）：调用模型服务的 API Key。
- `OPENAI_BASE_URL`（可选）：OpenAI 兼容接口地址。  
  不填时，`main.py` 默认使用 `https://api.openai.com/v1`。
- `OPENAI_MODEL`（可选）：模型名。  
  不填时，`main.py` 默认使用 `gpt-5.4-mini`。
- `OPENAI_TEMPERATURE`（可选）：采样温度，`0` 更稳定，值越高越发散。  
  不填时，`main.py` 默认使用 `0`。

## 运行方式

```bash
cd adp/promptChaining_1
source .venv/bin/activate
python main.py
```

## 运行结果（示例）

本地一次成功运行输出如下：

```text
--- Final JSON Output ---
{
  "cpu": "3.5 GHz octa-core",
  "memory": "16GB RAM",
  "storage": "1TB NVMe SSD"
}
```

说明：最终结果是一个 JSON 字符串，包含 `cpu`、`memory`、`storage` 三个字段。
