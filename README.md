# Multi-Modal Document Intelligence (RAG)

A Multi-Modal Retrieval-Augmented Generation (RAG) system capable of ingesting, retrieving, and answering questions from complex documents containing text, tables, and images.

## Features
- **Multi-modal Ingestion**: Handles text, tables, and images using LlamaParse.
- **Vector Search**: Uses ChromaDB for efficient retrieval.
- **Embeddings**: Uses local HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) to avoid API rate limits.
- **QA Chatbot**: Interactive Streamlit interface powered by Gemini Flash (`gemini-flash-latest`).
- **Citations**: Provides source attribution for answers.

## Setup

1. **Clone the repository**
2. **Prerequisites**: Ensure you have **Python 3.10+** installed (Python 3.12 recommended).
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up Environment Variables**:
   - Copy `.env.example` to `.env`.
   - Add your `GOOGLE_API_KEY` and `LLAMA_CLOUD_API_KEY`.

## Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
2. **Upload Documents**: Use the sidebar to upload PDFs.
3. **Process**: Click "Process Documents" to ingest and index.
4. **Chat**: Ask questions about your documents.

## Architecture
- **Ingestion**: `src/ingestion.py` uses LlamaParse to extract text and structure from PDFs.
- **Retrieval**: `src/retrieval.py` builds a ChromaDB index with local HuggingFace embeddings.
- **Generation**: `src/generation.py` uses Gemini Flash to generate answers.
- **UI**: `app.py` provides the user interface.
