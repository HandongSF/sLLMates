import json
from typing import Sequence
from langchain.schema import BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def convert_messages_to_text_format_llama3(messages : Sequence[BaseMessage], tools_list = None) -> str:
    context = "" # <|begin_of_text|> 토큰은 llm 시작 시 자동 선언됨

    if tools_list is not None:
        tools_dict = [convert_to_openai_tool(tool) for tool in tools_list]

    for message in messages:
        if message.type == 'system':
            context += f"<|start_header_id|>system<|end_header_id|>\n{message.content}"
            if tools_list is not None:
                context += '\n\nYou have access to the following functions. To call a function, please respond with JSON for a function call.\nRespond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\nDo not use variables.\n\n'
                for dictionary in tools_dict:
                    context += json.dumps(dictionary, indent=4)
                    context += "\n\n"
                context += "<|eot_id|>"
            else:
                context += "<|eot_id|>"

        elif message.type == 'human':
            context += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"

        elif message.type == 'ai':
            if message.tool_calls:
                context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                for tool_call in message.tool_calls:
                    function_name = tool_call.get('name')
                    function_args = json.dumps(tool_call.get('args'))
                    context += "{\"name\": \"" + function_name + "\", \"parameters\": " + function_args + "}<|eot_id|>"
            else:
                context += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"

        elif message.type == 'tool':
            context += "<|start_header_id|>ipython<|end_header_id|>\n\n<|python_tag|>{\n    \"tool_call_id\": \"" + message.tool_call_id + "\"\n    \"output\": \"" + message.content + "\"\n}<|eot_id|>"

    context += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    print("모델 최종 프롬프트: " + context)

    return context