"""
CrewAI Tool Use 示例（带多模型回退）：
1) 使用 CrewAI 的 Agent / Task / Crew 组织执行流程；
2) 通过 @tool 定义一个可调用的股票价格查询工具；
3) 参考 main.py：优先使用 OpenAI-compatible，失败时回退 Gemini；
4) 使用详细中文注释，方便教学与二次开发。
"""

import logging
import os
from typing import Optional, Tuple

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool


# ------------------------- 环境加载 -------------------------
# 固定从当前脚本目录加载 .env，避免在不同 cwd 下执行时找不到配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


# ------------------------- 日志配置 -------------------------
# 统一设置日志格式：
# - level=INFO：输出关键运行信息，便于观察工具是否被调用；
# - format 包含时间与级别，方便定位问题。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _init_crew_llm() -> Tuple[Optional[object], str]:
    """
    初始化 CrewAI 使用的 LLM（OpenAI-compatible 优先，Gemini 回退）。

    返回：
        (llm, status)
        - llm: 可传给 CrewAI Agent 的模型对象；失败时为 None
        - status: 初始化状态说明，便于排查配置问题

    说明：
        这里使用 CrewAI 官方的 LLM 包装器 `crewai.LLM`。
        这样可以统一处理不同后端，并保持和 Agent 的对接方式一致。
    """
    # ---------- OpenAI-compatible（优先） ----------
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
            # 对 OpenAI-compatible 服务（如自建网关/第三方兼容接口）可选传 base_url。
            if openai_base_url:
                kwargs["base_url"] = openai_base_url

            llm = LLM(**kwargs)
            return llm, f"Using OpenAI-compatible backend: {openai_model}"
        except Exception as e:
            # 不中断，继续尝试 Gemini 回退。
            logging.warning("OpenAI-compatible LLM 初始化失败，将尝试 Gemini 回退: %s", e)

    # ---------- Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0"))

    if google_api_key:
        try:
            # 在 CrewAI/LiteLLM 体系中，Gemini 常用 "gemini/<model_name>" 形式。
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
        "Please set OPENAI_API_KEY (preferred) or GOOGLE_API_KEY in toolUse_5/.env.",
    )


# ------------------------- 工具定义 -------------------------
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    根据股票代码返回“模拟”股价（float）。

    设计意图：
    1) 返回 float 而不是拼接好的句子，让 Agent 自己组织最终答案；
    2) 当股票代码不存在时，抛出 ValueError，强制 Agent 处理异常分支，
       这比“悄悄返回默认字符串”更适合演示真实工具调用的鲁棒性。

    参数：
        ticker: 股票代码，例如 AAPL / GOOGL / MSFT

    返回：
        float: 模拟股价

    异常：
        ValueError: ticker 不在模拟数据中
    """
    logging.info("Tool call: get_stock_price for ticker '%s'", ticker)

    # 这里使用内置字典模拟外部行情接口。
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }

    # 统一大写，避免用户输入 aapl / Aapl 导致匹配失败。
    price = simulated_prices.get(ticker.upper())

    if price is not None:
        return price

    # 抛出明确异常，提示 Agent 说明“未找到该股票”。
    raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")


def _build_financial_crew(llm: object) -> Crew:
    """
    基于指定 llm 构建金融分析 Crew。

    参数：
        llm: 已初始化的模型对象（OpenAI-compatible 或 Gemini）

    返回：
        Crew: 可执行 kickoff() 的 Crew 实例
    """
    # ------------------------- 1) 定义 Agent -------------------------
    # 核心职责：使用工具查询并给出简洁、直接的结论。
    financial_analyst_agent = Agent(
        role="Senior Financial Analyst",
        goal="Analyze stock data using provided tools and report key prices.",
        backstory=(
            "You are an experienced financial analyst adept at using data sources "
            "to find stock information. You provide clear, direct answers."
        ),
        verbose=True,
        tools=[get_stock_price],
        allow_delegation=False,
        llm=llm,
    )

    # ------------------------- 2) 定义 Task -------------------------
    # 任务要求 Agent：
    # - 必须调用工具获取 AAPL 模拟价格；
    # - 若找不到，明确说明无法检索该价格。
    analyze_aapl_task = Task(
        description=(
            "What is the current simulated stock price for AAPL? (ticker: AAPL). "
            "Use the 'Stock Price Lookup Tool' to find it. "
            "If the ticker is not found, report that you were unable to retrieve the price."
        ),
        expected_output=(
            "A single, clear sentence stating the simulated stock price for AAPL "
            "in this format: 'The simulated stock price for AAPL is $178.15.' "
            "If not found, clearly state that."
        ),
        agent=financial_analyst_agent,
    )

    # ------------------------- 3) 组装 Crew -------------------------
    # Crew 负责调度 agent + task，并统一执行。
    financial_crew = Crew(
        agents=[financial_analyst_agent],
        tasks=[analyze_aapl_task],
        verbose=True,
    )
    return financial_crew


def main_crew_ai() -> None:
    """
    启动 CrewAI 示例流程。

    与截图中的 main 类似，但已按你的要求：
    1) 函数名改为 main_crew_ai；
    2) 自动支持 OpenAI-compatible 优先、Gemini 回退；
    3) 对初始化失败场景给出明确提示。
    """
    llm, llm_status = _init_crew_llm()
    if not llm:
        print(f"Error initializing language model: {llm_status}")
        return

    print(f"Language model initialized. {llm_status}")
    print("\nStarting the Financial Crew...")
    print("-----------------------------------")

    try:
        financial_crew = _build_financial_crew(llm)
        # kickoff() 会触发完整执行：Agent 思考 -> 工具调用 -> 输出结果。
        result = financial_crew.kickoff()

        print("\nCrew execution finished.")
        print(f"Final Result:\n{result}")
    except Exception as e:
        print(f"\nAn error occurred during crew execution: {e}")


if __name__ == "__main__":
    main_crew_ai()
