# fastmcp_server.py
# 本脚本演示如何使用 FastMCP 快速创建一个最小可用的 MCP 服务端。
# 它仅暴露一个工具（tool）：greet，用于生成个性化问候语。

# 运行前请先安装依赖：
# pip install fastmcp
from fastmcp import FastMCP, Client

# 初始化 FastMCP 服务实例。
# 后续通过装饰器注册的函数，都会自动挂载到这个服务对象上。
mcp_server = FastMCP()


# 定义一个简单工具函数。
# `@mcp_server.tool` 装饰器会把该 Python 函数注册为 MCP 工具。
# 函数签名（参数名、类型）与文档字符串会被 MCP/LLM 用来推断工具用途与输入输出结构。
@mcp_server.tool
def greet(name: str) -> str:
    """
    生成一个个性化问候语。

    Args:
        name: 要问候的人的名字。

    Returns:
        一个问候字符串。
    """
    # 使用 f-string 拼接问候文本，返回给调用方。
    return f"Hello, {name}! Nice to meet you."


# 允许直接通过 `python fastmcp_server.py` 启动服务。
# 这里采用 HTTP 传输，监听本机 127.0.0.1:8000。
# 如果你要让局域网设备访问，可按需调整 host（例如 0.0.0.0）与 port。
if __name__ == "__main__":
    mcp_server.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
    )
