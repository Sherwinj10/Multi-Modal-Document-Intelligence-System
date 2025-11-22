import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
from src.utils import load_env_vars

load_env_vars()

# Configure Embeddings (Switching to Local HuggingFace to avoid API Quota limits)
# Using a small, efficient model: BAAI/bge-small-en-v1.5
print("Loading local embedding model (this may take a moment first time)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_vector_store(persist_dir="chroma_db"):
    """Initialize ChromaDB vector store."""
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("multimodal_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def build_index(documents, persist_dir="storage"):
    """
    Build and persist a vector index from documents.
    
    Args:
        documents: List of LlamaIndex Document objects.
        persist_dir: Directory to save the index.
        
    Returns:
        VectorStoreIndex: The built index.
    """
    print("Building vector index...")
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Use MultiModalVectorStoreIndex if we have image nodes, otherwise VectorStoreIndex
    # For now, assuming text-heavy documents from LlamaParse
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index built and persisted to {persist_dir}")
    return index

def load_index(persist_dir="storage"):
    """Load the index from storage."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No index found at {persist_dir}. Build it first.")
        
    print(f"Loading index from {persist_dir}...")
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
    index = load_index_from_storage(storage_context)
    return index

def get_retriever(index, similarity_top_k=5):
    """Get a retriever from the index."""
    return index.as_retriever(similarity_top_k=similarity_top_k)
