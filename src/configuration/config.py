import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """Centralized application configuration class."""
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "chroma_db"))
    LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(BASE_DIR, "logs"))
    
    # Embeddings and Vector DB Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))

    # LLM Configuration
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Frankfurter API Settings
    FRANKFURTER_API_URL = "https://api.frankfurter.app"