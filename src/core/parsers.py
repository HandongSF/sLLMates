import json
import re
from typing import Sequence, List, Dict
from langchain_core.messages import ToolCall
from langchain.schema import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

def parse_llm_output(text_output) -> AIMessage:
    tool_calls: List[ToolCall] = []

    # tool_call tag 추출
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text_output, re.DOTALL)

    if matches:
        for i, match in enumerate(matches):
            try:
                json_obj = json.loads(match)

                name = json_obj.get("name")

                tool_call = ToolCall(
                    name=name,
                    args=json_obj.get("arguments"),
                    id=f"tool_{name}_{i}"
                )

                tool_calls.append(tool_call)

            except Exception:
                continue

        return AIMessage(content="", tool_calls=tool_calls)

    # tool call이 없으면 일반 답변
    return AIMessage(content=text_output.strip())


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

def parse_bio_with_importance(text: str) -> List[Dict[str, any]]:
        bio_blocks = re.findall(r"<bio>(.*?)</bio>", text, flags=re.DOTALL)
        
        bio_list = []
        for block in bio_blocks:
            content_match = re.search(r"content:\s*(.*)", block)
            importance_match = re.search(r"importance:\s*(\d+)", block)
            is_core_match = re.search(r"is_core:\s*(true|false)", block, re.IGNORECASE)
            
            if not content_match:
                continue
                
            content = content_match.group(1).strip()
            
            # importance 파싱 및 제한 (기본값 5)
            raw_importance = int(importance_match.group(1)) if importance_match else 5
            importance_value = max(1, min(raw_importance, 10))
            
            # is_core 파싱 (기본값 False)
            is_core = is_core_match.group(1).lower() == "true" if is_core_match else False
            
            bio_list.append({
                "content": content,
                "importance": importance_value,
                "is_core": is_core
            })
            
        return bio_list
