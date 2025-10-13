from pathlib import Path

from src.core.tokenizer import count_tokens_with_tokenizer




# 기본 디렉토리 경로
BASE_DIR = Path(__file__).parent.parent.resolve()

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# 모델 경로
LLM_MODEL_PATH = str(MODELS_DIR / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "Qwen3-Embedding-0.6B")
TOKENIZER_MODEL_PATH = str(MODELS_DIR / "Qwen3-4B-Instruct-2507")

# 데이터 및 DB 경로
DOCUMENTS_PATH = str(DATA_DIR / "documents")
CHROMA_DB_PATH = str(DATA_DIR / "db" / "chroma_db")
SQLITE_DB_FILE = str(DATA_DIR / "db" / "chat_db" / "threads.sqlite")


# 모델 설정
class LLMConfig:
    """
    LLM 모델 관련 설정
    """
    model_path = LLM_MODEL_PATH
    n_gpu_layers = -1
    main_gpu = 1 # gpu 0, 1 중 1을 메인 gpu로 선택
    tensor_split = [1, 1] # 50:50으로 두 gpu의 vram에 모델 레이어를 나누어 올림 
    use_mmap = True
    use_mlock = False
    n_batch = 512
    n_ctx = 8196
    max_tokens = -1
    f16_kv = True
    verbose = False
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p = 0
    stop = ["<|endoftext|>", "<|im_end|>"]


class EmbeddingConfig:
    """
    Embedding 모델 관련 설정
    """
    model_name = EMBEDDING_MODEL_PATH
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}


class TrimmerConfig:
    """
    Trimmer 관련 설정
    """
    max_tokens = 7000
    strategy = "last"
    token_counter = count_tokens_with_tokenizer,
    include_system = True
    allow_partial = False
    start_on = "human"


class RAGConfig:
    """
    RAG 파이프라인 관련 설정
    """
    chunk_size = 200
    chunk_overlap = 50
    batch_size = 16
    retrieval_k = 3


# 시스템 프롬프트 및 변수
SYSTEM_PROMPT = """
You are a large language model, 'Qwen3'. You are currently conversing with a user via a chat app.
This means most of the time your lines should be a sentence or two unless the user's request requires reasoning or long-form outputs.
Never use emojis unless explicitly asked to. Never use emojis unless explicitly asked to.
If the tool call result exists, you write a response to the user based on the tool call result. The response language is {language}.

Tool Rules:
 - Do not describe or mention tools, you must call them directly.
 - You can use multiple tools if necessary.
 - Tool calls must start with <tool_call> and end with </tool_call>.
"""

VARIABLES = {
    "language": "Korean",
}