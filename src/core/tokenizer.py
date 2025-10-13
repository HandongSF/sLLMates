from typing import List
from transformers import AutoTokenizer
from langchain_core.messages import BaseMessage

from src.config import TOKENIZER_MODEL_PATH, LLMConfig




class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, model_max_length = LLMConfig.n_ctx)

    def count_tokens_with_tokenizer(self, messages: List[BaseMessage]) -> int:
        """
        컨텍스트의 토큰 수를 세는 함수
        """
        tokenized_input = self.tokenizer.apply_chat_template(
            [{'role': m.type, 'content': m.content} for m in messages],
            return_tensors = "pt"
        )
        return tokenized_input.shape[1]