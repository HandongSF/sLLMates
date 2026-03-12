import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DOCUMENTS_PATH, CHROMA_DB_PATH, BIO_CHROMA_DB_PATH

class ChromaDBVectorStore:

    vector_store: Chroma

    def __init__(self, config):
        if not os.path.exists(CHROMA_DB_PATH):
            loader = DirectoryLoader(
                DOCUMENTS_PATH,
                glob = "**/*.txt",
                loader_cls = TextLoader,
                loader_kwargs = {"encoding": "utf-8"}
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = config["RAG_CONFIG"].get("chunk_size", 200), chunk_overlap = config["RAG_CONFIG"].get("chunk_overlap", 50))
            all_splits = text_splitter.split_documents(docs)

            self.vector_store = Chroma(
                persist_directory = CHROMA_DB_PATH,
                embedding_function = HuggingFaceEmbeddings(
                    model_name = config["EMBEDDING_MODEL_CONFIG"].get("model_name", ""),
                    model_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("model_kwargs", {'device': 'cuda'}),
                    encode_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("encode_kwargs", {'normalize_embeddings': True}),
                )
            )

            batch_size = config["RAG_CONFIG"].get("batch_size", 16)
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i : i + batch_size]
                self.vector_store.add_documents(batch)
                self.vector_store.persist()  
        else:
            self.vector_store = Chroma(
                persist_directory = CHROMA_DB_PATH,
                embedding_function = HuggingFaceEmbeddings(
                    model_name = config["EMBEDDING_MODEL_CONFIG"].get("model_name", ""),
                    model_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("model_kwargs", {'device': 'cuda'}),
                    encode_kwargs = config["EMBEDDING_MODEL_CONFIG"].get("encode_kwargs", {'normalize_embeddings': True}),
                )
            )


class BioChromaDBVectorStore:

    vector_store: Chroma
    collection_name: str = "bio_memory"

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory = BIO_CHROMA_DB_PATH,
            embedding_function = HuggingFaceEmbeddings(
                    model_name = EmbeddingConfig.model_name,
                    model_kwargs = EmbeddingConfig.model_kwargs,
                    encode_kwargs = EmbeddingConfig.encode_kwargs,
                ) 
        )

    def embed_text(self, text: str):
        embedding_function = self.vector_store._embedding_function
        if embedding_function:
            return embedding_function.embed_query(text)
        return None

    def get_collection_name(self):
        return self.collection_name
    
    def get_bio_vector_store(self):
        return self.vector_store
    
    def get_bio_collection(self):
        return self.vector_store._collection