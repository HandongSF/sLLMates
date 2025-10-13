from langchain_core.messages import trim_messages

from src.config import TrimmerConfig




class Trimmer:
    def get_trimmer():
        return trim_messages(
            max_tokens = TrimmerConfig.max_tokens,
            strategy = TrimmerConfig.strategy,
            token_counter = TrimmerConfig.token_counter,
            include_system = TrimmerConfig.include_system,
            allow_partial = TrimmerConfig.allow_partial,
            start_on = TrimmerConfig.start_on,
        )