from src import config
from typing import List
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import trim_messages, BaseMessage




def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )


def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        config.TOKENIZER_MODEL_PATH, 
        model_max_length=config.MODEL_CONTEXT_SIZE
    )


def count_tokens_with_tokenizer(messages: List[BaseMessage]) -> int:
    tokenized_input = tokenizer.apply_chat_template(
        [{'role': m.type, 'content': m.content} for m in messages],
        return_tensors="pt"
    )
    return tokenized_input.shape[1]


def get_model():
    return ChatLlamaCpp(
        model_path=config.LLM_MODEL_PATH,
        n_gpu_layers=-1,
        tensor_split=[1, 1],
        use_mmap=True,
        use_mlock=False,
        n_batch=512,
        n_ctx=config.MODEL_CONTEXT_SIZE,
        max_tokens=-1,
        f16_kv=True,
        verbose=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        stop=["<|endoftext|>", "<|im_end|>"],
    )


def get_trimmer():
    return trim_messages(
        max_tokens=config.TRIMMER_MAX_TOKENS,
        strategy="last",
        token_counter=count_tokens_with_tokenizer,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )