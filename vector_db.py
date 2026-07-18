import os
import json
import requests
import datetime
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import config

MANIFEST_PATH = config.PERSIST_DIR / "indexed_papers.json"

def get_embeddings(provider: str = None, google_api_key: str = None):
    """
    Get the embedding model based on provider and API key settings.
    Falls back to config settings if parameters are not provided.
    """
    prov = provider or config.EMBEDDING_PROVIDER
    gkey = google_api_key or config.GOOGLE_API_KEY
    
    if prov == "gemini" and gkey:
        from langchain_google_genai import GoogleGenAIEmbeddings
        return GoogleGenAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gkey
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.HF_EMBEDDING_MODEL
        )

def load_manifest() -> dict:
    """Load the manifest of indexed papers."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_manifest(manifest: dict):
    """Save the manifest of indexed papers."""
    # Ensure manifest parent directory exists (which is config.PERSIST_DIR)
    config.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)

def is_paper_indexed(arxiv_id: str) -> bool:
    """Check if a paper is already indexed."""
    manifest = load_manifest()
    return arxiv_id in manifest

def get_indexed_papers() -> list[dict]:
    """Get list of all indexed papers from manifest."""
    manifest = load_manifest()
    return list(manifest.values())

def download_pdf(arxiv_id: str, pdf_url: str) -> Path:
    """Download PDF from ArXiv URL and save it locally."""
    pdf_path = config.DOWNLOAD_DIR / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        return pdf_path
        
    print(f"Downloading PDF for {arxiv_id} from {pdf_url}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(pdf_url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(pdf_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return pdf_path

def index_paper(paper_metadata: dict, provider: str = None, google_api_key: str = None) -> bool:
    """
    Downloads paper PDF, parses it, chunks the text, 
    embeds chunks using selected embedding provider, and indexes to Qdrant local DB.
    """
    arxiv_id = paper_metadata["arxiv_id"]
    
    if is_paper_indexed(arxiv_id):
        print(f"Paper {arxiv_id} is already indexed.")
        return True
        
    try:
        # 1. Download PDF
        pdf_path = download_pdf(arxiv_id, paper_metadata["pdf_url"])
        
        # 2. Parse PDF
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # 3. Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Enrich metadata for each chunk
        for chunk in chunks:
            chunk.metadata.update({
                "arxiv_id": arxiv_id,
                "title": paper_metadata["title"],
                "authors": ", ".join(paper_metadata["authors"]),
                "source": str(pdf_path)
            })
            
        # 4. Embed and save to Qdrant (stored locally on disk)
        embeddings = get_embeddings(provider, google_api_key)
        
        # Initialize local QdrantClient
        client = QdrantClient(path=str(config.PERSIST_DIR))
        
        # Ensure collection exists
        collection_name = "arxiv_papers"
        if not client.collection_exists(collection_name=collection_name):
            sample_emb = embeddings.embed_query("sample")
            dimension = len(sample_emb)
            from qdrant_client.http import models
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            )
            
        # Initialize and populate vector store
        vector_db = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        vector_db.add_documents(chunks)
        
        # 5. Update manifest
        manifest = load_manifest()
        manifest[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": paper_metadata["title"],
            "authors": paper_metadata["authors"],
            "published": paper_metadata["published"],
            "pdf_url": paper_metadata["pdf_url"],
            "pdf_path": str(pdf_path),
            "indexed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_manifest(manifest)
        
        print(f"Successfully indexed paper to Qdrant: {arxiv_id}")
        return True
        
    except Exception as e:
        print(f"Error indexing paper {arxiv_id} to Qdrant: {e}")
        return False

def query_paper(arxiv_id: str, query: str, k: int = 4, provider: str = None, google_api_key: str = None) -> list:
    """
    Query the Qdrant local vector store for relevant chunks matching a specific paper.
    """
    try:
        embeddings = get_embeddings(provider, google_api_key)
        
        # Initialize local QdrantClient
        client = QdrantClient(path=str(config.PERSIST_DIR))
        
        # If collection does not exist, return empty results
        if not client.collection_exists(collection_name="arxiv_papers"):
            return []
            
        vector_db = QdrantVectorStore(
            client=client,
            collection_name="arxiv_papers",
            embedding=embeddings
        )
        
        # Construct proper Qdrant filter
        from qdrant_client.http import models as qdrant_models
        filter_condition = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.arxiv_id",
                    match=qdrant_models.MatchValue(value=arxiv_id)
                )
            ]
        )
        
        # Query Qdrant filtered by payload arxiv_id
        results = vector_db.similarity_search(
            query,
            k=k,
            filter=filter_condition
        )
        return results
    except Exception as e:
        print(f"Error querying Qdrant for paper {arxiv_id}: {e}")
        return []
