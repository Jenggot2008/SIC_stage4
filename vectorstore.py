from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def init_vectorstore():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set Chroma persistence directory
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Initialize Chroma
    vector_store = Chroma(
        collection_name="sampah_collection",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store.as_retriever()
