from langchain.agents import create_agent

from agent.tools.agent_tools import (
    fetch_external_data,
    fetch_latest_external_data,
    fill_context_for_report,
    get_available_months,
    get_cleaning_environment_advice,
    get_current_month,
    get_user_id,
    get_user_location,
    get_user_profile,
    get_weather,
    rag_summarize,
)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                get_cleaning_environment_advice,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
                get_user_profile,
                get_available_months,
                fetch_latest_external_data,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        normalized = []
        for msg in messages:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _extract_text(content) -> str:
        if content is None:
            return ""

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
            return "".join(parts)

        return str(content)

    def execute_stream(self, messages: list[dict]):
        input_messages = self._normalize_messages(messages)
        input_dict = {"messages": input_messages}

        has_output = False
        last_content = ""

        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                latest_message = chunk["messages"][-1]
                current = self._extract_text(getattr(latest_message, "content", ""))
                if not current:
                    continue

                delta = current
                if current.startswith(last_content):
                    delta = current[len(last_content):]
                last_content = current

                if delta:
                    has_output = True
                    yield delta
        except Exception as e:
            logger.error(f"[execute_stream]智能体执行失败：{str(e)}", exc_info=True)
            yield "抱歉，我暂时无法完成这次请求。请稍后重试，或换一种问法。"
            return

        if not has_output:
            logger.warning("[execute_stream]智能体未返回有效内容")
            yield "抱歉，我暂时没有检索到有效信息，请换一种问法试试。"


if __name__ == "__main__":
    agent = ReactAgent()

    for chunk in agent.execute_stream([{"role": "user", "content": "给我生成我的使用报告"}]):
        print(chunk, end="", flush=True)
