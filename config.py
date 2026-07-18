import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / ".qdrant_db"
DOWNLOAD_DIR = BASE_DIR / "downloaded_papers"

# Ensure directories exist
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Configurable global variables
# Change this variable to swap between Groq models (e.g. llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768)
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding Provider: "huggingface" or "gemini"
# Defaults to "gemini" if GOOGLE_API_KEY is found, otherwise "huggingface"
if GOOGLE_API_KEY:
    EMBEDDING_PROVIDER = "gemini"
else:
    EMBEDDING_PROVIDER = "huggingface"

# Local huggingface embedding model path/name
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
