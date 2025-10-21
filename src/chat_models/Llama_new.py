from typing import List
from llama_cpp import Llama
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


class Llama_new(Llama):

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        total_tokens = 0
        for message in messages:
            content_str = ""

            if isinstance(message.content, str):
                content_str = message.content
            elif isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_str += part.get("text", "")

            if content_str:
                try:
                    message_bytes = content_str.encode("utf-8")
                    # llama-cpp-python의 네이티브 tokenize 메서드 사용
                    tokens = self.tokenize(message_bytes)
                    total_tokens += len(tokens)
                except Exception as e:
                    print(f"Warning: Could not tokenize message content: {e}")
                    total_tokens += len(content_str) // 4
        
        return total_tokens