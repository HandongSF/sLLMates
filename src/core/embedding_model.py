from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EmbeddingConfig




class EmbeddingModel:
    def get_embedding_model():
        return HuggingFaceEmbeddings(
            model_name = EmbeddingConfig.model_name,
            model_kwargs = EmbeddingConfig.model_kwargs,
            encode_kwargs = EmbeddingConfig.encode_kwargs,
        )