import json
import re
from typing import Sequence, List, Dict
from langchain_core.messages import ToolCall
from langchain.schema import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

def parse_llm_output(text_output) -> AIMessage:
    tool_calls: List[ToolCall] = []

    decoder = json.JSONDecoder()
    
    try:
        start_index = 0
        while start_index < len(text_output):
            json_obj, end_index = decoder.raw_decode(text_output[start_index:])
            name = json_obj.get("name")
            tool_call = ToolCall(
                name = name,
                args = json_obj.get("parameters"),
                id = f"tool_{name}_{len(tool_calls)}"
            )
            tool_calls.append(tool_call)
            start_index += end_index
            start_index = len(text_output) - len(text_output[start_index:].lstrip())
        return AIMessage(content = "", tool_calls = tool_calls)

    except Exception:
        return AIMessage(content = text_output)


def _normalize_content(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
        return "".join(texts)

    return str(content)


def convert_messages_to_llama3_messages(
    messages: Sequence[BaseMessage],
    tools_list=None
) -> List[Dict]:

    converted = []

    for message in messages:

        content = _normalize_content(message.content)

        # system
        if message.type == "system":
            converted.append({
                "role": "system",
                "content": content
            })

        # user
        elif message.type == "human":
            converted.append({
                "role": "user",
                "content": content
            })

        # assistant
        elif message.type == "ai":

            if getattr(message, "tool_calls", None):

                tool_calls = []

                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_call.get("name"),
                            "arguments": json.dumps(tool_call.get("args", {}), ensure_ascii=False)
                        }
                    })

                converted.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })

            else:
                converted.append({
                    "role": "assistant",
                    "content": content
                })

        # tool output
        elif message.type == "tool":

            content = _normalize_content(message.content)

            converted.append({
                "role": "assistant",
                "content": f"""다음은 검색된 참고 자료입니다.

                {content}

                위 정보를 참고해서 사용자의 질문에 답하세요."""
            })

    return converted