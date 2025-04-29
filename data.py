import os
import streamlit as st
import google.generativeai as genai
from data import init_db, get_sampah
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize
st.set_page_config(page_title="Chatbot Sampah", page_icon="♻️")
init_db()

# Config
os.environ["GOOGLE_API_KEY"] = "AIzaSyAjtK5VMleflxeMmJsRRZHVgGK2tJPKGDA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
PERSIST_DIRECTORY = "./chroma_db"

@st.cache_resource
def init_vectorstore():
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Clean existing data
    if os.path.exists(PERSIST_DIRECTORY):
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)
    
    # Create documents
    documents = []
    for item in get_sampah():
        documents.append(Document(
            page_content=f"Nama: {item['nama']}\nJenis: {item['jenis']}\nDeskripsi: {item['deskripsi']}",
            metadata={"source": item['nama'], "jenis": item['jenis']}
        ))
    
    # Add general knowledge
    documents.extend([
        Document(
            page_content="Sampah organik: berasal dari makhluk hidup, dapat terurai alami",
            metadata={"source": "definisi_organik", "jenis": "informasi_umum"}
        ),
        Document(
            page_content="Sampah non-organik: tidak terurai alami, biasanya sintetis",
            metadata={"source": "definisi_nonorganik", "jenis": "informasi_umum"}
        ),
        Document(
            page_content="Sampah B3 (Bahan Berbahaya Beracun): zat berbahaya untuk kesehatan dan lingkungan",
            metadata={"source": "definisi_b3", "jenis": "informasi_umum"}
        )
    ])
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_gemini(question):
    vectorstore = init_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """Anda adalah ahli pengelolaan sampah. Jawablah pertanyaan berdasarkan konteks berikut:
    
    {context}
    
    Pertanyaan: {question}
    
    Format jawaban:
    1. Identifikasi jenis sampah
    2. Kategori (Organik/Non-Organik/B3)
    3. Penjelasan singkat
    4. Sumber informasi
    
    Jika tidak tahu, jawab: "Maaf, saya tidak memiliki informasi tentang itu."
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | StrOutputParser()
    )
    
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}"

# UI
st.title("♻️ Chatbot Kategori Sampah")
st.markdown("""
**Contoh pertanyaan:**
- Botol plastik termasuk sampah apa?
- Bagaimana dengan daun kering?
- Apa itu sampah B3?
""")

question = st.text_input("Tanya tentang jenis sampah:")
if question:
    with st.spinner("Mencari jawaban..."):
        answer = ask_gemini(question)
        st.success(answer)
