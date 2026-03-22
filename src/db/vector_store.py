import sys
import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DOCUMENTS_PATH, CHROMA_DB_PATH

class ChromaDBManager:
    def __init__(self, config):
        self.config = config
        self.db_path = CHROMA_DB_PATH

        # initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(
            model_name = config["EMBEDDING_MODEL_CONFIG"].get("model_name", ""),
            model_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("model_kwargs", {'device': 'cuda'}),
            encode_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("encode_kwargs", {'normalize_embeddings': True}),
        )

        # CLI feature to re-embed documents from the beginning
        reembedding_confirmed = self._ask_reembedding()

        if reembedding_confirmed:
            print("[ChromaDB] 기존 'documents' 컬렉션을 초기화하고 새로 임베딩을 시작합니다...")
            self.doc_store = self._create_new_doc_collection()
        else:
            print("[ChromaDB] 기존 저장된 'documents' 데이터를 로드합니다.")
            self.doc_store = Chroma(
                collection_name="documents",
                persist_directory=self.db_path,
                embedding_function=self.embedding_function
            )

        # bio_memory collection is managed separately
        self.bio_store = Chroma(
            collection_name="bio_memory",
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )    

    # function to ask user whether to re-embed documents, with input validation
    def _ask_reembedding(self):
        if not os.path.exists(self.db_path):
            return True
            
        print("\n" + "="*50)
        print(" 기존 Document Vector Store 데이터를 비우고 새로 임베딩하시겠습니까? (y/n)")
        print("="*50)
        
        # 표준 출력을 강제로 비워 프롬프트가 먼저 보이게 함
        sys.stdout.flush() 
        
        # 입력을 받을 때까지 루프 (잘못된 입력 방지)
        while True:
            choice = input(" >> 선택: ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no', '']:
                return False
            else:
                print(" 'y' 또는 'n'을 입력해주세요.")

    def _create_new_doc_collection(self):
        """기존 컬렉션을 삭제(초기화)하고 문서를 새로 로드하여 저장"""
        temp_db = Chroma(
            collection_name="documents",
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        temp_db.delete_collection() # 해당 컬렉션 삭제

        # 문서 로드
        loader = DirectoryLoader(
            DOCUMENTS_PATH,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()

        # 기존 200은 문맥이 너무 끊길 수 있어 600 정도로 상향 조정
        new_chunk_size = self.config["RAG_CONFIG"].get("chunk_size", 600) 
        new_chunk_overlap = self.config["RAG_CONFIG"].get("chunk_overlap", 100)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=new_chunk_size, 
            chunk_overlap=new_chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)

        # 새 컬렉션 생성 및 데이터 추가
        vector_store = Chroma(
            collection_name="documents",
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )

        batch_size = self.config["RAG_CONFIG"].get("batch_size", 16)
        print(f"[ChromaDB] 총 {len(all_splits)}개의 청크를 임베딩 중...")
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i : i + batch_size]
            vector_store.add_documents(batch)
        
        print(f"[ChromaDB] 'documents' 컬렉션 재생성 완료.")
        return vector_store
    
    def get_doc_store(self):
        return self.doc_store

    def get_bio_store(self):
        return self.bio_store
    