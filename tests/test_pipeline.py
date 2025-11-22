import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_documents
from src.retrieval import build_index, load_index
from src.generation import get_query_engine, generate_response

def test_pipeline():
    print("Testing RAG Pipeline...")
    
    # Check for data
    data_dir = "data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("No data found in 'data' directory. Please add a PDF.")
        return

    # 1. Ingestion
    try:
        documents = load_documents(data_dir)
        print(f"Ingestion successful: {len(documents)} documents loaded.")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    # 2. Indexing
    try:
        index = build_index(documents, persist_dir="test_storage")
        print("Indexing successful.")
    except Exception as e:
        print(f"Indexing failed: {e}")
        return

    # 3. Retrieval & Generation
    try:
        query_engine = get_query_engine(index)
        response = generate_response(query_engine, "What is the summary of this document?")
        print(f"Response: {response.response}")
        print("Generation successful.")
    except Exception as e:
        print(f"Generation failed: {e}")
        return

    print("Pipeline test passed!")

if __name__ == "__main__":
    test_pipeline()
