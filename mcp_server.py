from agent.tools.agent_tools import (
    build_cleaning_environment_advice,
    build_weather_text,
    fetch_external_data,
    fetch_latest_external_data,
    get_available_months,
    get_user_profile,
    rag,
    resolve_user_location,
)
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("洁境智顾 Agent MCP")


@mcp.tool()
def get_user_location() -> str:
    """获取当前用户所在城市。"""
    return resolve_user_location()


@mcp.tool()
def get_weather(city: str = "") -> str:
    """获取指定城市实时天气；city为空时使用当前用户所在城市。"""
    return build_weather_text(city)


@mcp.tool()
def get_cleaning_environment_advice(city: str = "") -> str:
    """根据实时天气生成扫地/扫拖机器人的环境清扫建议。"""
    return build_cleaning_environment_advice(city)


@mcp.tool()
def rag_search(query: str) -> str:
    """从智能清洁设备知识库中检索并总结答案。"""
    return rag.rag_summarize(query)


@mcp.tool()
def get_available_usage_months(user_id: str) -> str:
    """获取指定用户可查询的使用记录月份。"""
    return get_available_months.invoke({"user_id": user_id})


@mcp.tool()
def get_latest_usage_record(user_id: str) -> str:
    """获取指定用户最近一期使用记录。"""
    return fetch_latest_external_data.invoke({"user_id": user_id})


@mcp.tool()
def get_usage_record(user_id: str, month: str) -> str:
    """获取指定用户在指定月份的使用记录。"""
    return fetch_external_data.invoke({"user_id": user_id, "month": month})


@mcp.tool()
def get_usage_profile(user_id: str) -> str:
    """获取指定用户画像摘要与趋势信息。"""
    return get_user_profile.invoke({"user_id": user_id})


if __name__ == "__main__":
    mcp.run()
