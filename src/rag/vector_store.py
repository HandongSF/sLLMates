import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHROMA_DB_PATH, RAGConfig
from src.core.embedding_model import EmbeddingModel




class ChromaDBVectorStore:
    def __init__(self):
        if not os.path.exists(CHROMA_DB_PATH):
            documents_path = os.path.join(DATA_DIR, "documents")
            loader = DirectoryLoader(
                documents_path,
                glob = "**/*.txt",
                loader_cls = TextLoader,
                loader_kwargs = {"encoding": "utf-8"}
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = RAGConfig.chunk_size, chunk_overlap = RAGConfig.chunk_overlap)
            all_splits = text_splitter.split_documents(docs)

            self.vector_store = Chroma(
                persist_directory = CHROMA_DB_PATH,
                embedding_function = EmbeddingModel.get_embedding_model()
            )

            batch_size = RAGConfig.batch_size
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i : i + batch_size]
                self.vector_store.add_documents(batch)
                self.vector_store.persist()  
        else:
            self.vector_store = Chroma(
                persist_directory = CHROMA_DB_PATH,
                embedding_function = EmbeddingModel.get_embedding_model()
            )