from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
import os

def init_vectorstore():
    try:
        # Try the preferred lightweight embeddings first
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except ImportError:
        # Fallback to default embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    # Rest of your ChromaDB initialization code...
