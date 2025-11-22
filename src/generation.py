from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from src.utils import load_env_vars

load_env_vars()

# Configure Gemini LLM
Settings.llm = Gemini(model_name="models/gemini-flash-latest")

def get_query_engine(index, similarity_top_k=5):
    """
    Create a query engine from the index.
    
    Args:
        index: The VectorStoreIndex.
        similarity_top_k: Number of documents to retrieve.
        
    Returns:
        BaseQueryEngine: The query engine.
    """
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact"
    )

def generate_response(query_engine, query_str):
    """
    Generate a response for a query.
    
    Args:
        query_engine: The query engine.
        query_str: The user's question.
        
    Returns:
        Response: The LlamaIndex response object.
    """
    response = query_engine.query(query_str)
    return response
