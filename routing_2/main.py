# Copyright (c) 2025 Marco Fago
# https://www.linkedin.com/in/marco-fago/
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.

"""
Routing 示例（LangChain 版“协调器 + 子代理”）：

目标：
1. 先由“协调器（Router）”判断用户请求类型：booker / info / unclear
2. 再把请求委派给对应的“子处理器（Sub-Agent Handler）”

这个脚本等价于一个最小化的“多代理路由”模式，重点展示：
- 使用 LLM 做文本分类路由
- 使用 RunnableBranch 做条件分支
- 使用 LCEL 管道（|）把整个流程串起来
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough


# ------------------------- 初始化模型 -------------------------
# 启动时自动加载当前脚本目录下的 .env，避免每次手动 source。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

def _init_llm():
    """
    初始化 LLM，优先级：
    1) OpenAI-compatible（OPENAI_API_KEY + OPENAI_MODEL）
    2) Gemini（GOOGLE_API_KEY）
    """
    # ---------- OpenAI-compatible ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if openai_api_key and openai_model:
        try:
            # 延迟导入，避免未安装 langchain-openai 时在模块导入阶段直接崩溃。
            from langchain_openai import ChatOpenAI
        except ImportError:
            return (
                None,
                "OpenAI-compatible configuration detected, but `langchain-openai` is not installed. "
                "Install it with: pip install langchain-openai",
            )

        kwargs = {
            "model": openai_model,
            "temperature": openai_temperature,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        llm = ChatOpenAI(**kwargs)
        return llm, f"Using OpenAI-compatible backend: {openai_model}"

    # ---------- Gemini ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if google_api_key:
        llm = ChatGoogleGenerativeAI(model=google_model, temperature=0)
        return llm, f"Using Gemini backend: {google_model}"

    return (
        None,
        "No valid model credentials found. "
        "Set OPENAI_API_KEY+OPENAI_MODEL (optionally OPENAI_BASE_URL), "
        "or set GOOGLE_API_KEY in routing_2/.env.",
    )


llm = None
llm_status = ""
try:
    llm, llm_status = _init_llm()
    if llm:
        # 对 ChatOpenAI / ChatGoogleGenerativeAI 统一输出后端状态，便于排查配置来源。
        print(f"Language model initialized. {llm_status}")
    else:
        print(f"Error initializing language model: {llm_status}")
except Exception as e:
    # 初始化失败时不直接抛出异常，保留可读提示并在 main() 中安全退出。
    print(f"Error initializing language model: {e}")
    llm = None


# ------------------------- 子处理器（模拟子代理） -------------------------
# 为了便于演示，这里用普通 Python 函数模拟不同子代理。
# 在真实项目中你可以替换成真实工具调用/数据库查询/外部 API 请求等。


def booking_handler(request: str) -> str:
    """模拟“订票代理”处理逻辑。"""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking handler processed request: '{request}'. Result: Simulated booking action."


def info_handler(request: str) -> str:
    """模拟“信息检索代理”处理逻辑。"""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info handler processed request: '{request}'. Result: Simulated information retrieval."


def unclear_handler(request: str) -> str:
    """处理无法明确归类的请求，给出兜底回复。"""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


# ------------------------- 协调器路由链 -------------------------
# 这条链负责把用户请求分类为 3 类之一：
# - booker：与机票/酒店/预订相关
# - info：一般信息问答
# - unclear：无法清楚归类
#
# 注意：为了后续分支判断稳定，提示词要求“只输出一个词”。
coordinator_router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Analyze the user's request and determine which specialized handler should process it.\n"
            "- If the request is related to booking flights or hotels, output 'booker'.\n"
            "- For all other general information questions, output 'info'.\n"
            "- If the request is unclear or doesn't fit either category, output 'unclear'.\n"
            "ONLY output one word: 'booker', 'info', or 'unclear'.",
        ),
        ("user", "{request}"),
    ]
)

if llm:
    # 路由链输出是纯字符串，例如："booker"
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()


# ------------------------- 分支委派逻辑 -------------------------
# RunnableBranch 会根据 decision 字段将数据路由到不同分支。
# 各分支通过 RunnablePassthrough.assign 增加 output 字段，保留原始输入字段。
branches = {
    "booker": RunnablePassthrough.assign(
        output=lambda x: booking_handler(x["request"])
    ),
    "info": RunnablePassthrough.assign(output=lambda x: info_handler(x["request"])),
    "unclear": RunnablePassthrough.assign(
        output=lambda x: unclear_handler(x["request"])
    ),
}

delegation_branch = RunnableBranch(
    # 当 decision 为 "booker" 时走订票分支
    (lambda x: x["decision"].strip().lower() == "booker", branches["booker"]),
    # 当 decision 为 "info" 时走信息分支
    (lambda x: x["decision"].strip().lower() == "info", branches["info"]),
    # 默认兜底分支（包括 "unclear" 及其他意外输出）
    branches["unclear"],
)


# ------------------------- 组装总链 -------------------------
# 1) 先计算 decision（由路由链得到）
# 2) 同时保留原始 request
# 3) 基于 decision 进入对应分支并生成 output
# 4) 提取最终 output 作为用户可见结果
if llm:
    coordinator_agent = (
        {
            "decision": coordinator_router_chain,
            "request": lambda x: x["request"],
        }
        | delegation_branch
        | (lambda x: x["output"])
    )


# ------------------------- 示例运行 -------------------------
def main() -> None:
    # 启动保护：
    # 如果模型初始化失败（例如没有配置 GOOGLE_API_KEY、网络问题、模型名错误），
    # 则直接退出，避免后续 coordinator_agent.invoke 抛出连锁异常。
    if not llm:
        if llm_status:
            print(f"LLM status: {llm_status}")
        print("\nSkipping execution due to LLM initialization failure.")
        return

    # 用例 A：明确的“预订”意图
    # 期望路由结果：booker 分支（订票处理器）。
    print("\n--- Running with a booking request ---")
    request_a = "Book me a flight to London."
    # invoke 输入是一个字典，键名必须与上游 prompt / 链中约定一致（这里是 request）。
    result_a = coordinator_agent.invoke({"request": request_a})
    # 打印最终聚合结果（只展示分支处理器返回的 output 字段）。
    print(f"Final Result A: {result_a}")

    # 用例 B：一般知识问答
    # 期望路由结果：info 分支（信息处理器）。
    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result B: {result_b}")

    # 用例 C：边界场景（可能被判定为 info，也可能是 unclear）
    # 用来观察路由器在“非订票”请求上的分类倾向。
    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about quantum physics."
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result C: {result_c}")


if __name__ == "__main__":
    main()
