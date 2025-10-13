from langchain_core.tools import tool

from src.rag.vector_store import vector_store_instance




@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve information related to a query.

    You have the tool `retrieve`. Use `retrieve` in the following circumstances:
        - User is asking about some term you are totally unfamiliar with (it might be new).
        - User explicitly asks you to browse or provide links to references.

    Given a query that requires retrieval, you call the search function to get a list of results.
    """
    if query == "__NONE__":
        return "No results found.", []

    retrieved_docs = vector_store.similarity_search(query, k = 3)

    if not retrieved_docs:
        return "No results found.", []

    serialized = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs




# 사용할 도구 리스트
TOOLS_LIST = [retrieve, ]