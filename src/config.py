from pathlib import Path

# 기본 디렉토리 경로

BASE_DIR = Path(__file__).parent.parent.resolve()
"""base directory path"""

MODELS_DIR = Path(__file__).parent.parent.parent.resolve() / "sLLMates" / "models"
#MODELS_DIR = BASE_DIR / "models"
"""Model directory path"""

DATA_DIR = BASE_DIR / "data"
"""Data directory path"""


# 설정 파일 선택

SELECTED_CONFIG_FILE = "Qwen3-14B_RAG_BIO.py"
"""Model 설정 파일

config 폴더 안의 설정 파일 중 하나를 선택하여 적용
(.py 는 붙여도 되고 안 붙여도 됨)
"""


# 데이터 및 DB 경로

DOCUMENTS_PATH = str(DATA_DIR / "documents")
"""Documents path"""

CHROMA_DB_PATH = str(DATA_DIR / "db" / "chroma_db")
"""Chroma db path"""

SQLITE_DB_FILE = str(DATA_DIR / "db" / "chat_db" / "threads.sqlite")
"""Sqlite db path"""