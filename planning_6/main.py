"""
planning_6: CrewAI 规划 + 写作示例

核心目标：
1) 先让 Agent 生成结构化提纲（Plan）；
2) 再基于提纲输出简洁摘要（Summary）；
3) 模型后端支持 OpenAI-compatible 优先、Gemini 回退。
"""

import logging
import os
from typing import Optional, Tuple

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task


# ------------------------- 环境加载 -------------------------
# 固定从脚本所在目录读取 .env，避免在不同 cwd 下运行时读取不到配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


# ------------------------- 日志配置 -------------------------
# 使用 INFO 级别便于观察执行链路（模型初始化、任务执行、错误定位等）。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _init_crew_llm() -> Tuple[Optional[object], str]:
    """
    初始化 CrewAI 使用的 LLM（OpenAI-compatible 优先，Gemini 回退）。

    返回：
        (llm, status)
        - llm: 可传给 Agent 的模型对象；失败时返回 None
        - status: 初始化状态文本，便于排查配置问题
    """
    # ---------- 1) OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if openai_api_key:
        try:
            kwargs = {
                "model": openai_model,
                "api_key": openai_api_key,
                "temperature": openai_temperature,
            }
            # 若使用兼容网关（非官方 OpenAI 域名），可通过 base_url 指定。
            if openai_base_url:
                kwargs["base_url"] = openai_base_url

            llm = LLM(**kwargs)
            return llm, f"Using OpenAI-compatible backend: {openai_model}"
        except Exception as e:
            # OpenAI-compatible 初始化失败后，不立即终止，继续尝试 Gemini。
            logging.warning("OpenAI-compatible LLM 初始化失败，将尝试 Gemini 回退: %s", e)

    # ---------- 2) Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0"))

    if google_api_key:
        try:
            # 在 CrewAI/LiteLLM 体系中，Gemini 通常写为 gemini/<model_name>。
            llm = LLM(
                model=f"gemini/{google_model}",
                api_key=google_api_key,
                temperature=google_temperature,
            )
            return llm, f"Using Gemini backend: {google_model}"
        except Exception as e:
            return None, f"Gemini LLM 初始化失败: {e}"

    return (
        None,
        "No valid model credentials found. "
        "Please set OPENAI_API_KEY (preferred) or GOOGLE_API_KEY in planning_6/.env.",
    )


def _build_planning_crew(llm: object, topic: str) -> Crew:
    """
    基于给定 llm 和主题，构建“先规划、后写作”的 Crew。

    参数：
        llm: 已初始化完成的模型对象
        topic: 目标主题（例如强化学习的重要性）

    返回：
        Crew: 可直接 kickoff() 执行的 Crew 实例
    """
    # ---------- 1) 定义 Agent ----------
    # Agent 角色采用“规划 + 写作”二合一，聚焦可执行输出，避免空泛内容。
    planner_writer_agent = Agent(
        role="Article Planner and Writer",
        goal=(
            "Plan and then write a concise, engaging summary on a specified topic."
        ),
        backstory=(
            "You are an expert technical writer and content strategist. "
            "Your strength lies in creating a clear, actionable plan before writing, "
            "ensuring the final summary is both informative and easy to digest."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ---------- 2) 定义 Task ----------
    # 任务要求明确分两步：
    # - 先输出要点式提纲；
    # - 再输出约 200 词摘要。
    # expected_output 强制结果结构统一，便于评估与复用。
    high_level_task = Task(
        description=(
            f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\\n"
            "2. Write the summary based on your plan, keeping it around 200 words."
        ),
        expected_output=(
            "A final report containing two distinct sections:\\n\\n"
            "### Plan\\n"
            "- A bulleted list outlining the main points of the summary.\\n\\n"
            "### Summary\\n"
            "- A concise and well-structured summary of the topic."
        ),
        agent=planner_writer_agent,
    )

    # ---------- 3) 组装 Crew ----------
    # 使用顺序流程（Process.sequential），保证先完成任务规划再输出写作结果。
    crew = Crew(
        agents=[planner_writer_agent],
        tasks=[high_level_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def main() -> None:
    """
    脚本入口：
    1) 初始化 LLM（OpenAI-compatible 优先、Gemini 回退）；
    2) 构建 Crew 并执行；
    3) 打印最终结果。
    """
    llm, llm_status = _init_crew_llm()
    if not llm:
        print(f"Error initializing language model: {llm_status}")
        return

    topic = "The importance of Reinforcement Learning in AI"

    print(f"Language model initialized. {llm_status}")
    print("## Running the planning and writing task ##")

    try:
        crew = _build_planning_crew(llm=llm, topic=topic)
        result = crew.kickoff()

        print("\n---\n## Task Result ##\n---")
        print(result)
    except Exception as e:
        print(f"\nAn error occurred during crew execution: {e}")


if __name__ == "__main__":
    main()
