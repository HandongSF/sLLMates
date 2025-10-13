from langchain_core.messages import trim_messages

from src.config import TrimmerConfig
from src.core.tokenizer import Tokenizer




class Trimmer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.trimmer = trim_messages(
            max_tokens = TrimmerConfig.max_tokens,
            strategy = TrimmerConfig.strategy,
            token_counter = self.tokenizer.count_tokens_with_tokenizer,
            include_system = TrimmerConfig.include_system,
            allow_partial = TrimmerConfig.allow_partial,
            start_on = TrimmerConfig.start_on,
        )