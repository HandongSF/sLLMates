import json
import re
from typing import List
from langchain_core.messages import ToolCall
from langchain.schema import AIMessage

def parse_llm_output(response_data: dict) -> AIMessage:
    text_output = response_data['choices'][0]['text'].strip()

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