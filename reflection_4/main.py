"""
Reflection Loop 示例：
通过“生成 -> 反思 -> 改进”的循环，让模型持续优化一段 Python 代码。

本脚本的模型选择策略：
1) 优先 OpenAI-compatible（OPENAI_API_KEY + OPENAI_MODEL）
2) 回退 Gemini（GOOGLE_API_KEY + GOOGLE_MODEL）
"""

import os
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------- 环境加载 -------------------------
# 固定从当前脚本目录读取 .env，避免“从不同工作目录执行”时读取失败。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


def _init_llm() -> Tuple[Optional[object], str]:
    """
    初始化 LLM（OpenAI-compatible 优先，Gemini 回退）。

    Returns:
        Tuple[Optional[object], str]:
            - llm: 模型实例，失败时为 None
            - status: 状态说明（用于终端排错）
    """
    # ---------- OpenAI-compatible ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    if openai_api_key:
        try:
            # 延迟导入，避免未安装 langchain-openai 时在导入阶段报错。
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None, "Detected OPENAI_API_KEY, but `langchain-openai` is not installed."

        kwargs = {
            "model": openai_model,
            "temperature": openai_temperature,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        return ChatOpenAI(**kwargs), f"Using OpenAI-compatible backend: {openai_model}"

    # ---------- Gemini ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.1"))
    if google_api_key:
        llm = ChatGoogleGenerativeAI(model=google_model, temperature=google_temperature)
        return llm, f"Using Gemini backend: {google_model}"

    return (
        None,
        "No valid credentials found. "
        "Please set OPENAI_API_KEY (preferred) or GOOGLE_API_KEY in reflection_4/.env.",
    )


def _message_to_text(message) -> str:
    """
    把 LLM 返回消息统一转换成纯文本。

    不同后端/版本返回的 content 结构可能不同（str 或 list），
    这里做兼容处理，避免后续字符串拼接报错。
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)


def run_reflection_loop() -> None:
    """
    执行“反思循环”示例：
    1) 第一次先生成初版代码
    2) 然后让模型扮演审查者给出 critique
    3) 再将 critique 反馈给模型继续改进
    4) 满足停止条件后退出循环
    """
    llm, llm_status = _init_llm()
    if not llm:
        print(f"Error initializing language model: {llm_status}")
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print(f"Language model initialized. {llm_status}")

    # ------------------------- 核心任务描述 -------------------------
    # 这段任务会在第一轮直接发送给模型，用于生成“初始版本代码”。
    task_prompt = """
Your task is to create a Python function named `calculate_factorial`.
This function should:
1. Accept a single integer `n` as input.
2. Calculate its factorial (`n!`).
3. Include a clear docstring explaining what the function does.
4. Handle edge case: factorial of 0 is 1.
5. Handle invalid input: raise ValueError if the input is negative.
Only output valid Python code.
""".strip()

    # 最大迭代次数（可通过环境变量覆盖）
    max_iterations = int(os.getenv("REFLECTION_MAX_ITERATIONS", "3"))

    # current_code 用于保存当前轮“最新代码版本”。
    current_code = ""

    # message_history 保存“代码生成器”的上下文历史：
    # 初始只有任务；后续会不断追加“上版代码 + 审查意见”。
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print(f"\n{'=' * 25} REFLECTION LOOP: ITERATION {i + 1} {'=' * 25}")

        # ------------------------- 阶段 1：生成 / 改进 -------------------------
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            response = llm.invoke(message_history)
            current_code = _message_to_text(response)
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            # 在后续迭代中，给模型明确任务：根据上轮 critique 改进代码。
            message_history.append(
                HumanMessage(content="Please refine the code using the critiques provided.")
            )
            response = llm.invoke(message_history)
            current_code = _message_to_text(response)

        print(f"\n--- Generated Code (v{i + 1}) ---\n{current_code}")

        # 把生成结果作为历史上下文的一部分，供后续反思和下一轮改进使用。
        message_history.append(response)

        # ------------------------- 阶段 2：反思 / 审查 -------------------------
        print("\n>>> STAGE 2: REFLECTING on the generated code...")

        # 让模型切换到“资深代码审查员”角色，仅输出批判性意见。
        # 若代码已满足要求，必须返回固定短语 CODE_IS_PERFECT。
        reflector_prompt = [
            SystemMessage(
                content=(
                    "You are a senior software engineer and code reviewer.\n"
                    "Critically evaluate the provided Python code based on the original task requirements.\n"
                    "Focus on bugs, edge cases, missing validations, readability, and style.\n"
                    "If the code is perfect and meets all requirements, respond ONLY with: CODE_IS_PERFECT\n"
                    "Otherwise, provide concise, actionable critiques."
                )
            ),
            HumanMessage(
                content=f"Original task:\n{task_prompt}\n\nCode to review:\n{current_code}"
            ),
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = _message_to_text(critique_response).strip()

        # ------------------------- 阶段 3：停止条件 -------------------------
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break

        print(f"\n--- Critique ---\n{critique}")

        # 将 critique 回灌到 message_history，驱动下一轮改进。
        message_history.append(
            HumanMessage(content=f"Critique of the previous code:\n{critique}")
        )

    # ------------------------- 最终输出 -------------------------
    print(f"\n{'=' * 30} FINAL RESULT {'=' * 30}")
    print("Final refined code after the reflection process:\n")
    print(current_code)


if __name__ == "__main__":
    run_reflection_loop()
