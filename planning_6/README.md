# planning_6

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agent%20Orchestration-111827)](https://www.crewai.com/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“规划 + 写作（Planning + Writing）”示例模块，基于 CrewAI 完成单 Agent 的结构化输出流程：

- 先生成 `Plan`（要点提纲）；
- 再生成 `Summary`（约 200 词摘要）。

模型后端支持：OpenAI-compatible 优先，Gemini 回退。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`
- 网络：可访问你配置的模型服务地址
- 鉴权：至少准备一套可用凭证（OpenAI-compatible 或 Gemini）

安装方式：

```bash
cd /Users/yz/work/github/adp/planning_6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Python 依赖清单

与 `requirements.txt` 保持一致：

- `crewai`
- `python-dotenv`

## .env 变量说明

在 `planning_6/.env` 中可配置：

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

## 运行 main.py

```bash
cd /Users/yz/work/github/adp/planning_6
python main.py
```

运行结果（本地实测）：

```text
Language model initialized. Using OpenAI-compatible backend: qwen3-coder-plus
## Running the planning and writing task ##
...
---
## Task Result ##
---

### Plan
• Define reinforcement learning as a type of machine learning where agents learn through trial-and-error interactions with environments
• Explain the core concept of reward/penalty systems that guide agent behavior toward optimal outcomes
• Highlight key applications across robotics, gaming, autonomous vehicles, and recommendation systems
• Discuss how RL enables machines to make sequential decisions and adapt to dynamic environments
• Address the significance of RL in achieving artificial general intelligence goals
• Mention major breakthroughs like AlphaGo and their impact on AI advancement
• Cover challenges including sample efficiency, safety concerns, and computational requirements
• Conclude with future potential and growing importance in real-world AI deployments

### Summary
Reinforcement Learning (RL) stands as a cornerstone of modern artificial intelligence, enabling agents to learn optimal behaviors through environmental interaction and feedback mechanisms. Unlike supervised learning, RL operates on reward-penalty systems, allowing machines to discover effective strategies through trial and error. This approach proves particularly valuable for sequential decision-making tasks, making it essential for applications ranging from game-playing algorithms like AlphaGo to autonomous vehicles and personalized recommendation systems. RL's ability to adapt to dynamic environments and optimize long-term outcomes sets it apart from other machine learning paradigms. Despite challenges including high computational demands and safety considerations, RL continues driving breakthroughs toward artificial general intelligence. As real-world deployment increases across industries, reinforcement learning remains critical for developing adaptive, intelligent systems capable of operating in complex, unpredictable environments.
```

