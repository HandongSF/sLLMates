from langchain_community.chat_models import ChatLlamaCpp

from src.config import LLMConfig




class LLM:
    def get_llm():
        return ChatLlamaCpp(
            model_path = LLMConfig.model_path,
            n_gpu_layers = LLMConfig.n_gpu_layers,
            main_gpu = LLMConfig.main_gpu,
            tensor_split = LLMConfig.tensor_split,
            use_mmap = LLMConfig.use_mmap,
            use_mlock = LLMConfig.use_mlock,
            n_batch = LLMConfig.n_batch,
            n_ctx = LLMConfig.n_ctx,
            max_tokens = LLMConfig.max_tokens,
            f16_kv = LLMConfig.f16_kv,
            verbose = LLMConfig.verbose,
            temperature = LLMConfig.temperature,
            top_p = LLMConfig.top_p,
            top_k = LLMConfig.top_k,
            min_p = LLMConfig.min_p,
            stop = LLMConfig.stop,
        )