import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 从当前脚本所在目录加载 .env。
# 这样无论你从哪个工作目录执行 `python main.py`，都能稳定读取到同目录下的配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# 提前检查 API Key，避免在真正调用模型时才抛出不直观的报错。
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "未检测到 OPENAI_API_KEY。请先在环境变量或 .env 文件中设置后再运行。"
    )

# 从环境变量读取模型参数，使用“ChatGPT/OpenAI 风格”的命名：
# - OPENAI_API_KEY：必填，OpenAI API 密钥
# - OPENAI_BASE_URL：可选，默认官方地址；如果是兼容 OpenAI 的代理服务可改这里
# - OPENAI_MODEL：可选，默认 gpt-4o-mini
# - OPENAI_TEMPERATURE：可选，默认 0（稳定输出）
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# 初始化聊天模型。
# 显式传入参数便于排查问题，也让配置来源更清晰（全部来自 .env / 环境变量）。
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=OPENAI_TEMPERATURE,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=OPENAI_BASE_URL,
)

# Prompt 1：从原始文本中提取“技术规格”信息。
# 这里的 {text_input} 是占位符，调用 invoke 时会被实际输入替换。
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

# Prompt 2：把上一步提取出来的规格，重组为 JSON 结构。
# {specifications} 占位符将接收 extraction_chain 的输出结果。
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object with "
    "'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

# 第一段链：提取链（Extraction Chain）
# 数据流：text_input -> prompt_extract -> llm -> StrOutputParser -> 字符串规格文本
# StrOutputParser 的作用是把模型返回的消息对象转成纯字符串，便于后续继续传递。
extraction_chain = prompt_extract | llm | StrOutputParser()

# 完整链：先执行提取，再执行结构化转换。
# {"specifications": extraction_chain} 这段是 LCEL 的“映射”写法：
# - 它会先运行 extraction_chain
# - 然后把结果放进一个字典里，键名叫 specifications
# - 这个字典会喂给 prompt_transform，用于替换 {specifications}
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

# 示例输入：一段非结构化产品描述。
input_text = (
    "The new laptop model features a 3.5 GHz octa-core processor, "
    "16GB of RAM, and a 1TB NVMe SSD."
)

# 触发整条链执行。
# 这里传入的是字典，因为 prompt_extract 定义了变量名 {text_input}。
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- Final JSON Output ---")
print(final_result)
