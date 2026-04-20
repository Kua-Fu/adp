"""
rag_14: 基于 LangChain + LangGraph + Weaviate 的 RAG 示例

整体流程（从上到下）：
1) 读取本地文本语料（state_of_the_union.txt）；
2) 按固定 chunk 规则切分文档；
3) 将 chunk 向量化后写入 Weaviate；
4) 把“检索 + 生成”封装为 LangGraph 两节点工作流；
5) 使用示例问题触发工作流并输出中间状态。

后端策略：
- 聊天模型与向量模型统一采用「OpenAI-compatible 优先，Gemini 回退」；
- 对 OpenAI-compatible embedding 增加了网关兼容适配，降低 400/503 风险。
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END

import weaviate

try:
    # v4 推荐向量库封装（兼容 weaviate-client 4.x）
    from langchain_weaviate import WeaviateVectorStore
except ImportError:
    # 回退到旧封装（仅在旧环境需要）
    from langchain_community.vectorstores import Weaviate as WeaviateVectorStore

# ------------------------- 环境加载 -------------------------
# 固定从当前模块目录加载 .env：
# - 避免在不同 cwd 下运行时读到错误配置；
# - 保证“脚本双击运行 / 终端运行 / IDE 运行”行为一致。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 将关键环境变量显式写回进程环境，便于底层 SDK 自动读取。
# 说明：
# - ChatOpenAI / OpenAI SDK 既支持显式参数也支持环境变量；
# - 这里两种都保留，减少后续迁移时的心智负担。
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
if os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")


def _log_banner(title: str) -> None:
    """打印显著分隔日志，方便在长日志中快速定位阶段。"""
    print("\n" + "=" * 72)
    print(f"[RAG-14] {title}")
    print("=" * 72)


def _print_progress(step: int, total: int, label: str) -> None:
    """打印简单文本进度条（无额外依赖，兼容终端/日志文件）。"""
    width = 28
    filled = int(width * step / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * step / total)
    print(f"[进度] [{bar}] {percent:>3}% | {label}")


def _shorten(text: str, max_len: int = 140) -> str:
    """把长文本压缩成单行预览，避免日志刷屏。"""
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


class CompatibleOpenAIEmbeddings(Embeddings):
    """
    兼容 OpenAI-compatible 网关的 Embeddings 适配器。

    使用场景与背景：
    - 某些网关对 langchain_openai.OpenAIEmbeddings 的请求格式兼容性较弱；
    - 这里改用 openai SDK 原生 `embeddings.create`，以字符串输入直连；
    - 该实现遵循 LangChain `Embeddings` 接口，可无缝注入 vectorstore。
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        # 延迟导入：只有实际走 OpenAI-compatible embedding 路径时才依赖 openai 包。
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 逐条调用能获得最大网关兼容性；
        # 如需更高吞吐，可在你确认网关支持后改为批量 input=list[str]。
        vectors: List[List[float]] = []
        for text in texts:
            resp = self.client.embeddings.create(model=self.model, input=text)
            vectors.append(resp.data[0].embedding)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        # 单条查询向量（用于相似度检索时的 query embedding）。
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


# ------------------------- 1) 模型与嵌入初始化 -------------------------
def _init_llm_and_embeddings() -> tuple[Any, Any, str]:
    """
    初始化 LLM 与 Embeddings。

    优先级策略：
    1. OpenAI-compatible（优先）
       - 需要 OPENAI_API_KEY
       - OPENAI_MODEL 可选（默认 gpt-4o-mini）
       - OPENAI_BASE_URL 可选（兼容网关常用）
    2. Gemini（回退）
       - 需要 GOOGLE_API_KEY
       - GOOGLE_MODEL / GOOGLE_EMBEDDING_MODEL 可选

    返回值说明：
    - llm: 聊天模型对象（ChatOpenAI 或 ChatGoogleGenerativeAI）
    - embeddings: 向量模型对象（CompatibleOpenAIEmbeddings 或 GoogleGenerativeAIEmbeddings）
    - backend_status: 可直接打印的后端状态文本（用于排障）
    """
    # ---------- OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if openai_api_key:
        # 只要检测到 OPENAI_API_KEY，就优先尝试 OpenAI-compatible 路径。
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "检测到 OPENAI_API_KEY，但未安装 langchain-openai。"
            ) from exc

        # 统一通过 kwargs 控制，便于在有/无 base_url 场景下复用。
        llm_kwargs: Dict[str, Any] = {
            "model": openai_model,
            "temperature": 0,
            "api_key": openai_api_key,
        }
        # 当接入 OpenAI-compatible 网关时，base_url 一般必须显式传入。
        if openai_base_url:
            llm_kwargs["base_url"] = openai_base_url

        llm = ChatOpenAI(**llm_kwargs)

        # 某些网关只开放了部分 embedding 模型，因此做“候选探测”。
        # 规则：
        # 1) 若用户显式给了 OPENAI_EMBEDDING_MODEL，先试该模型；
        # 2) 再按内置候选顺序逐个探测；
        # 3) 探测失败时记录最后一次错误，便于最终报错定位。
        embedding_candidates: List[str] = []
        if os.getenv("OPENAI_EMBEDDING_MODEL"):
            embedding_candidates.append(openai_embedding_model)
        embedding_candidates.extend(
            [m for m in ("text-embedding-v3", "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002") if m not in embedding_candidates]
        )

        last_openai_embedding_error: Exception | None = None
        for emb_model in embedding_candidates:
            try:
                candidate = CompatibleOpenAIEmbeddings(
                    model=emb_model,
                    api_key=openai_api_key,
                    base_url=openai_base_url,
                )
                # 轻量探测：启动阶段先 embed 一次，尽早暴露不兼容问题。
                candidate.embed_query("ping")
                backend_status = (
                    "模型后端：OpenAI-compatible（优先）"
                    f" | chat={openai_model} | embedding={emb_model}"
                )
                return llm, candidate, backend_status
            except Exception as emb_exc:
                last_openai_embedding_error = emb_exc

    # ---------- Gemini（回退） ----------
    # 仅在 OpenAI-compatible 不可用时进入此分支。
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    google_embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")

    if google_api_key:
        try:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )
        except ImportError as exc:
            raise RuntimeError(
                "检测到 GOOGLE_API_KEY，但未安装 langchain-google-genai。"
            ) from exc

        llm = ChatGoogleGenerativeAI(
            model=google_model,
            temperature=0,
            google_api_key=google_api_key,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=google_embedding_model,
            google_api_key=google_api_key,
        )
        backend_status = (
            "模型后端：Gemini（回退）"
            f" | chat={google_model} | embedding={google_embedding_model}"
        )
        return llm, embeddings, backend_status

    # 如果走到这里，说明：
    # 1) OpenAI 通道存在，但 embedding 候选全部失败；
    # 2) Gemini 回退不可用（通常是没配 GOOGLE_API_KEY）。
    if openai_api_key:
        raise RuntimeError(
            "OpenAI-compatible 已配置，但 embedding 模型均不可用。"
            "请在 rag_14/.env 显式设置 OPENAI_EMBEDDING_MODEL 为网关支持的模型；"
            "或配置 GOOGLE_API_KEY 启用 Gemini 回退。"
            f"最后一次 embedding 错误：{last_openai_embedding_error}"
        )

    raise RuntimeError(
        "未检测到可用模型配置。请在 rag_14/.env 中配置 OPENAI_API_KEY（优先）或 GOOGLE_API_KEY（回退）。"
    )


def _init_weaviate_client() -> Any:
    """
    初始化 Weaviate 客户端（embedded 模式）。

    兼容策略：
    - 优先走 weaviate-client v4 的 `connect_to_embedded`；
    - 若环境为旧版本，则尝试 v3 `Client(embedded_options=...)`；
    - 二者都不可用则抛错。

    返回：
        已连接的 Weaviate 客户端实例。
    """
    # ---------- v4 推荐路径 ----------
    if hasattr(weaviate, "connect_to_embedded"):
        # 持久化目录与二进制目录都固定在模块内：
        # - 避免写入用户 HOME（某些沙箱会拒绝）；
        # - 保证项目可迁移与可清理。
        persist_path = os.path.join(BASE_DIR, ".weaviate_data")
        binary_path = os.path.join(BASE_DIR, ".weaviate_bin")
        os.makedirs(persist_path, exist_ok=True)
        os.makedirs(binary_path, exist_ok=True)
        return weaviate.connect_to_embedded(
            persistence_data_path=persist_path,
            binary_path=binary_path,
            environment_variables={"LOG_LEVEL": "error"},
        )

    # ---------- v3 兼容路径 ----------
    # 仅在旧版 client 可用时使用（新版本通常不会走到这里）。
    if hasattr(weaviate, "Client"):
        from weaviate.embedded import EmbeddedOptions  # type: ignore

        return weaviate.Client(embedded_options=EmbeddedOptions())

    raise RuntimeError("当前 weaviate-client 版本不支持 embedded 模式。")


# ------------------------- 2) 数据准备（Data Preparation） -------------------------
# 直接使用模块内本地语料（state_of_the_union.txt）：
# - 不依赖 curl / requests 访问外网；
# - 更适合离线环境与受限网络环境。
text_file_path = os.path.join(BASE_DIR, "state_of_the_union.txt")
if not os.path.exists(text_file_path):
    raise FileNotFoundError(
        "未找到本地语料文件：rag_14/state_of_the_union.txt。"
        "请先将该文件放到当前模块目录。"
    )

# 读取文本为 Document 列表（通常是 1 个长文档）。
loader = TextLoader(text_file_path, encoding="utf-8")
documents = loader.load()

# 对长文档进行切分：
# - chunk_size 控制每段长度；
# - chunk_overlap 保留上下文重叠，减少跨段语义断裂。
# 可按业务调整：
# - 若答案常依赖跨段上下文，可增大 overlap；
# - 若希望更快检索，可适当减小 chunk_size。
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


# ------------------------- 3) 向量化并写入 Weaviate -------------------------
_log_banner("启动 RAG 初始化")
_print_progress(1, 4, "读取并切分本地语料")

# 初始化 LLM + Embedding（优先 OpenAI-compatible，失败/缺失再回退 Gemini）。
llm, embeddings, backend_status = _init_llm_and_embeddings()
_print_progress(2, 4, f"模型初始化完成：{backend_status}")
print(backend_status)

# 使用 Embedded Weaviate 本地模式，无需单独部署远端服务（适合示例与本地调试）。
# 这里优先走 v4 连接方式，并在旧版环境自动回退。
_log_banner("Weaviate 启动中")
client = _init_weaviate_client()
_print_progress(3, 4, "Weaviate embedded 已连接")

# 将切分后的文档写入向量库：
# - index_name 固定命名，便于复查与调试；
# - text_key 指定“正文”字段名，检索时会从该字段回填 page_content。
try:
    vectorstore = WeaviateVectorStore.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        index_name="StateOfTheUnionRAG",
        text_key="text",
    )
except Exception:
    # 入库失败时主动关闭客户端，避免 embedded 子进程和 socket 资源泄漏。
    if hasattr(client, "close"):
        client.close()
    raise
_print_progress(4, 4, "向量库写入完成，RAG 可用")
_log_banner("启动完成")

# 定义检索器：
# - k=4 表示每次召回 4 个最相似片段；
# - 可按精度/速度权衡调大或调小。
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ------------------------- 4) 定义状态（State） -------------------------
class RAGGraphState(TypedDict):
    """
    LangGraph 全局状态定义。

    字段语义：
    - question: 用户问题（入口输入）
    - documents: 检索召回结果（中间状态）
    - generation: 最终生成文本（出口输出）
    """

    question: str
    documents: List[Document]
    generation: str


# ------------------------- 5) 定义节点（Node Functions） -------------------------
def retrieve_documents_node(state: RAGGraphState) -> RAGGraphState:
    """
    检索节点：
    - 输入问题；
    - 在向量库中召回相关文档；
    - 将结果写回状态，供后续生成节点使用。
    """
    # 这里不修改问题本身，直接用原始问题做向量检索。
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "question": question,
        "documents": documents,
        "generation": "",
    }


def generate_response_node(state: RAGGraphState) -> RAGGraphState:
    """
    生成节点：
    - 读取检索到的上下文；
    - 拼接提示词并调用模型；
    - 生成最终回答。
    """
    question = state["question"]
    documents = state["documents"]

    # 这里沿用你截图里的提示模板，并加入“简洁回答”约束。
    # 生产环境可把模板抽到独立 prompt 文件做版本管理。
    template = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )

    prompt = ChatPromptTemplate.from_template(template)

    # 将多个文档片段拼接为单一上下文字符串，供提示词注入。
    context = "\n\n".join([doc.page_content for doc in documents])

    # 使用 RunnablePassthrough 让链路结构更清晰：
    # 1) 先把 context/question 组装成输入字典；
    # 2) 进入 prompt；
    # 3) 调用 llm；
    # 4) 解析为纯文本输出。
    # RunnablePassthrough 使输入字典原样透传给 prompt 的占位符。
    rag_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
    }


# ------------------------- 6) 组装 LangGraph 流程图 -------------------------
workflow = StateGraph(RAGGraphState)

# 添加节点（节点名将出现在 stream 事件中，尽量语义化）。
workflow.add_node("retrieve", retrieve_documents_node)
workflow.add_node("generate", generate_response_node)

# 设定入口：先检索
workflow.set_entry_point("retrieve")

# 添加边（流转关系）
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 编译后得到可执行 app，可用 `invoke/stream/astream` 等接口触发。
app = workflow.compile()


def _run_query_with_pretty_logs(query: str, query_no: int) -> None:
    """
    执行单个查询并输出可读日志：
    1) 打印问题；
    2) 打印检索摘要（文档数、来源、预览）；
    3) 打印最终回答。
    """
    _log_banner(f"问题 #{query_no}")
    print(f"[用户问题] {query}")

    inputs = {"question": query}
    for event in app.stream(inputs):
        if "retrieve" in event:
            state = event["retrieve"]
            docs = state.get("documents", [])

            # 仅用于日志展示去重：按内容预览键去重，避免重复文档刷屏。
            seen = set()
            dedup_docs = []
            for doc in docs:
                key = _shorten(doc.page_content, 120)
                if key in seen:
                    continue
                seen.add(key)
                dedup_docs.append(doc)

            print(f"[检索结果] 原始 {len(docs)} 段，去重后 {len(dedup_docs)} 段")
            for i, doc in enumerate(dedup_docs[:4], start=1):
                source = os.path.basename(str(doc.metadata.get("source", "unknown")))
                preview = _shorten(doc.page_content, 150)
                print(f"  {i}. 来源={source} | 摘要={preview}")

        if "generate" in event:
            answer = event["generate"].get("generation", "").strip()
            print("[最终回答]")
            print(answer if answer else "(空回答)")


# ------------------------- 7) 运行示例 -------------------------
if __name__ == "__main__":
    # 这里用 stream 打印每个节点产出的中间状态，便于教学与排错。
    # 若用于生产服务，可改为 app.invoke(...) 并仅返回最终结果。
    try:
        _run_query_with_pretty_logs(
            query="What did the president say about Justice Breyer?",
            query_no=1,
        )
        _run_query_with_pretty_logs(
            query="What did the president say about the economy?",
            query_no=2,
        )
    finally:
        # 无论成功或失败都关闭连接，防止 embedded 进程残留与资源告警。
        _log_banner("Weaviate 关闭中")
        _print_progress(1, 3, "发送关闭请求")
        if hasattr(client, "close"):
            client.close()
        _print_progress(2, 3, "释放连接资源")
        _print_progress(3, 3, "关闭完成")
        _log_banner("流程结束")
