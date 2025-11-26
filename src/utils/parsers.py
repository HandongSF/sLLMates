import json
import re
from typing import List
from langchain_core.messages import ToolCall
from langchain.schema import AIMessage


def makeJSONToToolCall(content_str : str):
    """
    LLM이 생성한 문자열에서 <tool_call> 태그를 찾아 JSON으로 파싱
    """
    tool_calls: List[ToolCall] = []

    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, content_str, re.DOTALL)

    for match in matches:
        json_str = match.strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            fixed_json_str = None
            
            # 닫는 중괄호가 너무 많은 경우
            if json_str.endswith("}}"):
                fixed_json_str = json_str[:-1]

            # 닫는 중괄호가 없는 경우
            elif json_str.startswith("{") and not json_str.endswith("}"):
                fixed_json_str = json_str + "}"

            if fixed_json_str:
                try:
                    data = json.loads(fixed_json_str)
                except json.JSONDecodeError as e2:
                    print(f"ERROR: Failed to parse even after fixing. Block: {json_str}\nError: {e2}")
            else:
                # 위 조건에 해당하지 않는 다른 JSON 오류일 경우
                print(f"ERROR: Cannot fix JSON with simple heuristics. Block: {json_str}\nError: {e}")

        if data:
            name = data.get("name")
            args = data.get("arguments")

            if name and args is not None:
                tool_call = ToolCall(
                    name=name,
                    args=args,
                    id=f"tool_{name}_{len(tool_calls)}"
                )
                tool_calls.append(tool_call)
        else:
            tool_call = ToolCall(
                name="retrieve",
                args={"query": "__NONE__"},
                id=f"tool_fallback_{len(tool_calls)}"
            )
            tool_calls.append(tool_call)

    return tool_calls


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