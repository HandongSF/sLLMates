from langchain_core.tools import tool
from typing import Annotated

from src.config import RAGConfig
from src.db.vector_store import ChromaDBVectorStore


chroma_db_vector_store = ChromaDBVectorStore()
bio_chroma_db_vector_store = ChromaDBVectorStore()


@tool(response_format="content_and_artifact")
def retrieve(
    query: Annotated[str, "A search query composed of the essential keywords from the user's question. For example: 'Tell me the name of the largest bird' -> 'the largest bird'"]
):
    """"You have the tool `retrieve`. Use `retrieve` in the following circumstances:\n - User is asking about some term you are totally unfamiliar with (it might be new).\n - User explicitly asks you to browse or provide links to references.\n\n Given a query that requires retrieval, you call the retrieve function to get a list of results."
    """

    if query == "__NONE__":
        return "No results found.", []
    
    retrieved_docs = chroma_db_vector_store.vector_store.similarity_search(query, k = RAGConfig.retrieval_k)

    # results_with_scores = chroma_db_vector_store.vector_store.similarity_search_with_score(
    #     query, k = RAGConfig.retrieval_k
    # )

    # if not results_with_scores:
    #     return "No results found.", []
    
    # DISTANCE_THRESHOLD = 0.5

    # filtered_docs = [
    #     doc for doc, score in results_with_scores 
    #     if score < DISTANCE_THRESHOLD
    # ]

    if not retrieved_docs:
        return "No results found.", []
    
    serialized = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs


TOOL_LIST = [retrieve, ]