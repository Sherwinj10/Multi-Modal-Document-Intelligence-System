import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from src.utils import load_env_vars

load_env_vars()

def get_parser():
    """Initialize LlamaParse with multi-modal support."""
    return LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=4,
    )

def load_documents(data_dir: str):
    """
    Load documents from the specified directory using LlamaParse.
    
    Args:
        data_dir (str): Path to the directory containing documents.
        
    Returns:
        List[Document]: List of loaded documents.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
        
    print(f"Loading documents from {data_dir}...")
    
    # Initialize LlamaParse
    parser = get_parser()
    
    # Use SimpleDirectoryReader with LlamaParse
    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        file_extractor=file_extractor,
        recursive=True
    )
    
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")
    return documents

if __name__ == "__main__":
    # Test ingestion
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created {data_path} directory. Please add some PDFs there.")
    else:
        try:
            docs = load_documents(data_path)
            print(f"Successfully loaded {len(docs)} documents.")
        except Exception as e:
            print(f"Error loading documents: {e}")
