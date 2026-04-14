"""
Parallelization 示例（LangChain RunnableParallel）

本示例演示“并行子任务 + 汇总生成”的经典模式：
1) 对同一主题并行执行三个子链：摘要、问题生成、关键词提取
2) 将并行结果合并后再交给一个综合提示词做统一回答

后端模型初始化策略：
- 优先使用 OpenAI-compatible（OPENAI_API_KEY + OPENAI_MODEL）
- 若未配置，则回退 Gemini（GOOGLE_API_KEY + GOOGLE_MODEL）
"""

import os
import asyncio
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------------- 配置加载 -------------------------
# 自动加载 parallelization_3/.env，避免手动 source 环境变量。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


def _init_llm() -> Tuple[Optional[object], str]:
    """
    初始化 LLM，优先级如下：
    1) OpenAI-compatible：OPENAI_API_KEY (+ OPENAI_MODEL, 可选 OPENAI_BASE_URL)
    2) Gemini：GOOGLE_API_KEY (+ GOOGLE_MODEL)

    Returns:
        (llm, message)
        llm: 模型对象，失败时为 None
        message: 初始化状态说明，便于排错
    """
    # ---------- OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    if openai_api_key:
        try:
            # 延迟导入：只有真的使用 OpenAI-compatible 才需要这个包。
            from langchain_openai import ChatOpenAI
        except ImportError:
            return (
                None,
                "Detected OPENAI_API_KEY, but `langchain-openai` is not installed.",
            )

        kwargs = {
            "model": openai_model,
            "temperature": openai_temperature,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        return ChatOpenAI(**kwargs), f"Using OpenAI-compatible backend: {openai_model}"

    # ---------- Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.7"))

    if google_api_key:
        llm = ChatGoogleGenerativeAI(model=google_model, temperature=google_temperature)
        return llm, f"Using Gemini backend: {google_model}"

    return (
        None,
        "No valid credentials found. "
        "Please set OPENAI_API_KEY (recommended) or GOOGLE_API_KEY in parallelization_3/.env.",
    )


llm = None
llm_status = ""
full_parallel_chain = None

try:
    llm, llm_status = _init_llm()
    if llm:
        print(f"Language model initialized. {llm_status}")
    else:
        print(f"Error initializing language model: {llm_status}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None


if llm:
    # ------------------------- 定义并行子链 -------------------------
    # 子链 1：生成简明摘要
    summarize_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize the following topic concisely:"),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # 子链 2：围绕主题生成 3 个值得继续追问的问题
    questions_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Generate three interesting questions about the following topic:"),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # 子链 3：提取 5-10 个关键词（逗号分隔）
    terms_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Identify 5-10 key terms from the following topic, separated by commas:",
                ),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # ------------------------- 并行映射层 -------------------------
    # RunnableParallel 会“并发触发”各个子链。
    # 这里输出 4 个字段：
    # - summary / questions / key_terms：三个子链的结果
    # - topic：原始输入（通过 RunnablePassthrough 透传），供后续汇总提示词引用
    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),
        }
    )

    # ------------------------- 汇总合成层 -------------------------
    # 将并行结果拼装成一个更完整的答案。
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the following information:\n"
                "Summary: {summary}\n"
                "Related Questions: {questions}\n"
                "Key Terms: {key_terms}\n"
                "Synthesize a comprehensive answer.",
            ),
            ("user", "Original topic: {topic}"),
        ]
    )

    # 完整链路：
    # 输入 topic(字符串) -> 并行处理 -> 汇总提示词 -> LLM -> 字符串输出
    full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()


async def run_parallel_example(topic: str) -> None:
    """
    异步执行并行链并打印结果。

    详细说明：
    1. 入参 topic 是单个字符串；
    2. 该字符串会被 RunnableParallel 同时送入三个子链（摘要/问题/关键词）；
    3. 并行结果会和原始 topic 一起进入 synthesis_prompt；
    4. 最终由模型生成“综合回答”，并通过 StrOutputParser 转成纯文本。
    """
    if not llm or not full_parallel_chain:
        print("LLM not initialized. Cannot run example.")
        return

    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
    try:
        # 注意：这里 ainvoke 的入参是“单个主题字符串”，
        # 因为 map_chain 中各子链的 prompt 都使用了 {topic} 变量，且 topic 由输入值直接透传。
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")


def main() -> None:
    """
    程序入口：
    - 先检查模型是否初始化成功；
    - 成功后使用一个示例主题触发并行链；
    - 通过 asyncio.run 执行异步函数。
    """
    if not llm:
        if llm_status:
            print(f"LLM status: {llm_status}")
        print("\nSkipping execution due to LLM initialization failure.")
        return

    test_topic = "The history of space exploration"
    asyncio.run(run_parallel_example(test_topic))


if __name__ == "__main__":
    main()
