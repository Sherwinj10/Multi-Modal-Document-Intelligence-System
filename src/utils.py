import os
from dotenv import load_dotenv

def load_env_vars():
    """Load environment variables from .env file."""
    load_dotenv()
    
    required_vars = ["GOOGLE_API_KEY", "LLAMA_CLOUD_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file.")
