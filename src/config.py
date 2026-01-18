from pathlib import Path
from dataclasses import dataclass

# 기본 디렉토리 경로

BASE_DIR = Path(__file__).parent.parent.resolve()
"""base directory path"""

MODELS_DIR = BASE_DIR / "models"
"""Model directory path"""

DATA_DIR = BASE_DIR / "data"
"""Data directory path"""


# 모델 경로

LLM_MODEL_PATH = str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
"""LLM Model path"""

EMBEDDING_MODEL_PATH = str(MODELS_DIR / "Qwen3-Embedding-0.6B")
"""Embedding Model path"""

TOKENIZER_MODEL_PATH = str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct_tokenizer")
"""Tokenizer Model path"""


# 데이터 및 DB 경로

DOCUMENTS_PATH = str(DATA_DIR / "documents")
"""Documents path"""

CHROMA_DB_PATH = str(DATA_DIR / "db" / "chroma_db")
"""Chroma db path"""

SQLITE_DB_FILE = str(DATA_DIR / "db" / "chat_db" / "threads.sqlite")
"""Sqlite db path"""

BIO_CHROMA_DB_PATH = str(DATA_DIR / "db" / "bio_chroma_db")
"""Bio Chroma db path"""


# Llama 클래스 사용 설정

USING_LLAMA = True


# 모델 설정

@dataclass
class LLMConfig:
    """LLM 모델 관련 설정"""

    model_path = LLM_MODEL_PATH
    n_ctx = 4096
    f16_kv = True
    use_mlock = False
    n_batch = 512
    n_gpu_layers = -1
    max_tokens = -1
    temperature = 1.0
    top_p = 1
    stop = ["<|end_of_text|>", "<|eot_id|>"]
    top_k = 20
    use_mmap = True
    model_kwargs = {
        "main_gpu": 1, # gpu 0, 1 중 1을 메인 gpu로 선택
        "tensor_split": [0.05, 0.95], # 50:50으로 두 gpu의 vram에 모델 레이어를 나누어 올림 
        "min_p": 0,
        "flash_attn": True,
    }
    verbose = True


@dataclass
class EmbeddingConfig:
    """Embedding 모델 관련 설정"""

    model_name = EMBEDDING_MODEL_PATH
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}


@dataclass
class TrimmerConfig:
    """Trimmer 관련 설정"""

    max_tokens = 2000
    strategy = "last"
    include_system = True
    allow_partial = False
    start_on = "human"


@dataclass
class RAGConfig:
    """RAG 파이프라인 관련 설정"""

    chunk_size = 200
    chunk_overlap = 50
    batch_size = 16
    retrieval_k = 5


# 시스템 프롬프트 및 변수

VARIABLES = {
    "language": "Korean",
}
"""요약 추가 예정"""

SYSTEM_PROMPT = """
You are Llama3.1, a large language model trained by Meta, based on the Llama architecture. You are chatting with the user via the Chating app. Never use emojis unless explicitly asked to. When you receive a tool call response, use the output to format an answer to the orginal user question. The response language is {language}.
"""

BIO_PROMPT = """
You are Llama3.1, a large language model trained by Meta. 
Your task is ONLY to extract long-term, meaningful facts about the user.

If you find a new, stable fact about the user, you MUST output it using the EXACT format below:

<bio>the fact</bio>
<importance>N</importance>

You must follow this format EXACTLY.
No extra text before, between, or after these tags.

Save only long-term or meaningful information (preferences, background, personality, health, goals).
Ignore trivial or temporary details (e.g., current location, mood, filler expressions, greetings).

Examples (follow the exact format):
<bio>The user is allergic to peanuts.</bio>
<importance>9</importance>

<bio>The user's birthday is May 12.</bio>
<importance>8</importance>

If there is no new meaningful user fact, output nothing.

Below are the user queries:
"""

BIO_EXPLANATION_PROMPT = """
\nBelow are important information about the user:\n"""

