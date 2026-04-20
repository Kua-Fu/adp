# mm_8

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Agents%20%26%20Tools-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Memory%20Workflow-0B5FFF)](https://www.langchain.com/langgraph)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“内存管理（Memory Management）”示例模块，基于 LangGraph 演示：

- 短期记忆（`MemorySaver`）：同一 `thread_id` 内保持会话状态。
- 长期记忆（`InMemoryStore`）：同一 `user_id` 跨会话保存用户事实。
- 工具写入记忆（`save_recall_memory`）：模型可主动把重要信息持久化。
- 后端策略：OpenAI-compatible 优先，Gemini 回退。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少准备一套可用凭证（OpenAI-compatible 或 Gemini）。

推荐安装：

```bash
cd adp/mm_8
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果你复用统一环境（本仓库当前做法）：

```bash
.venv/bin/pip install -r adp/mm_8/requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `python-dotenv`：加载 `.env` 环境变量。
- `langchain`：模型与工具封装。
- `langgraph`：状态图、短期记忆、工具节点编排。
- `langchain-openai`：OpenAI-compatible 模型封装。
- `langchain-google-genai`：Gemini 模型封装。

## .env 变量说明

在 `mm_8/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0

# Gemini（当未配置 OPENAI_API_KEY 时回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0

# 可选：演示里用于区分“同一用户”的 user_id
MM_USER_ID=user-001
```

## 主函数说明

- `main.py` 中的 `main()`：
  - 初始化 LLM（OpenAI-compatible 优先，Gemini 回退）。
  - 构建 LangGraph：`START -> call_model -> tools -> call_model`。
  - 连续执行三轮对话，分别演示：
    - 在 `thread-A` 写入偏好记忆；
    - 在 `thread-A` 继续提问验证会话内记忆；
    - 切换到 `thread-B` 验证跨会话行为。

## 运行 `main.py`

```bash
cd adp/mm_8
.venv/bin/python main.py
```

运行结果（本地实测，2026-04-15）：

```text
## Running mm_8 memory management demo ##
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus

[thread=thread-A user=user-001] 用户: 你好，我叫小王，我喜欢川菜和徒步。
助手: 你好，小王！很高兴认识你。川菜以其麻辣鲜香著称，而徒步则是一项很好的户外运动，既能锻炼身体又能欣赏自然风光。你有特别喜欢的川菜菜品或者徒步路线吗？

根据我们的对话，我已经记录了你喜欢川菜和徒步的信息，以便未来能更好地为你提供相关建议和服务。如果你还有其他兴趣爱好或需要帮助的地方，请随时告诉我。

[thread=thread-A user=user-001] 用户: 你记得我的饮食和爱好吗？
助手: 是的，我记得你的饮食和爱好。你特别喜欢川菜，这是一种以麻辣口味著称的中国地方菜系；此外，你还热爱徒步这项户外活动，它能够让你在大自然中享受运动的乐趣。这些信息已经存储在我的记忆中，方便我在未来的交流中更好地了解你的偏好并提供相应的建议。如果你有任何新的兴趣或需求，也可以继续与我分享。

[thread=thread-B user=user-001] 用户: 我们换了一个新会话，你还记得我喜欢什么吗？
助手: 您好！我们确实换了一个新会话，我这里暂时没有您之前的记忆记录。不过没关系，您可以告诉我您喜欢什么，我会认真记住的！

如果您愿意分享的话，我可以将您的喜好、偏好的信息保存到长期记忆中，这样下次我们聊天时就能继续使用这些信息了。
```
