import os
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from db import get_sampah
from config import PERSIST_DIRECTORY

def init_vectorstore():
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            os.remove(os.path.join(PERSIST_DIRECTORY, file))

    vector_store = Chroma(collection_name="sampah_collection", embedding_function=embeddings)

    sampah_list = get_sampah()
    documents = [
        Document(
            page_content=f"Nama: {item['nama']}\nJenis: {item['jenis']}\nDeskripsi: {item['deskripsi']}",
            metadata={"source": item['nama'], "jenis": item['jenis']}
        )
        for item in sampah_list
    ]

    documents.extend([
        Document(
            page_content="Sampah organik adalah sampah yang berasal dari makhluk hidup dan dapat terurai secara alami",
            metadata={"source": "definisi_organik", "jenis": "informasi_umum"}
        ),
        Document(
            page_content="Sampah non-organik adalah sampah yang tidak dapat terurai secara alami dan biasanya berasal dari bahan sintetis",
            metadata={"source": "definisi_nonorganik", "jenis": "informasi_umum"}
        ),
        Document(
            page_content="Sampah B3 (Bahan Berbahaya dan Beracun) adalah sampah yang mengandung zat berbahaya bagi kesehatan dan lingkungan",
            metadata={"source": "definisi_b3", "jenis": "informasi_umum"}
        )
    ])

    vector_store.add_documents(documents)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
