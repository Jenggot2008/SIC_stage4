import os
import sqlite3
import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.schema import Document
import time

#__import__('pysqlite3')
#import sys

#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit UI configuration
st.set_page_config(page_title="Chatbot Sampah", page_icon="♻️")

# Sidebar (Navbar kiri)
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih halaman:", ["Chatbot", "Daur Ulang", "Tentang kami"])

# Konfigurasi API Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyAjtK5VMleflxeMmJsRRZHVgGK2tJPKGDA"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Path dan direktori
DB_PATH = "database.db"
PERSIST_DIRECTORY = "./chroma_langchain_newest"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Inisialisasi Database dengan data lebih lengkap
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sampah (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nama TEXT,
                    jenis TEXT,
                    deskripsi TEXT)''')  # Tambahkan kolom deskripsi
    c.execute("SELECT COUNT(*) FROM sampah")
    if c.fetchone()[0] == 0:
        sampah_data = [
            ("Daun", "Organik", "Daun-daunan yang gugur dari tanaman"),
            ("Sisa Makanan", "Organik", "Makanan sisa atau bahan makanan yang sudah basi"),
            ("Botol Plastik", "Non-Organik", "Botol minuman berbahan plastik"),
            ("Kertas", "Non-Organik", "Berbagai jenis kertas bekas"),
            ("Kayu", "Organik", "Potongan kayu atau serbuk kayu"),
            ("Kain", "Non-Organik", "Pakaian bekas atau potongan kain"),
            ("Kaleng", "Non-Organik", "Kaleng minuman atau makanan"),
            ("Karet", "Non-Organik", "Barang-barang berbahan karet"),
            ("Kardus", "Non-Organik", "Kardus bekas kemasan"),
            ("Baterai", "B3", "Baterai bekas termasuk limbah berbahaya"),
            ("Elektronik", "B3", "Barang elektronik rusak"),
            ("Kaca", "Non-Organik", "Botol kaca atau pecahan kaca")
        ]
        c.executemany("INSERT INTO sampah (nama, jenis, deskripsi) VALUES (?, ?, ?)", sampah_data)
    conn.commit()
    conn.close()

init_db()

# Ambil data dari database dengan deskripsi
def get_sampah():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nama, jenis, deskripsi FROM sampah")
    data = [{"id": row[0], "nama": row[1], "jenis": row[2], "deskripsi": row[3]} for row in c.fetchall()]
    conn.close()
    return data

# Inisialisasi Vector Store dengan dokumen lebih deskriptif
@st.cache_resource
def init_vectorstore():
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Bersihkan direktori jika sudah ada data sebelumnya
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        for file in os.listdir(PERSIST_DIRECTORY):
            os.remove(os.path.join(PERSIST_DIRECTORY, file))
    
    vector_store = Chroma(collection_name="sampah_collection", 
                         embedding_function=embeddings
                         )
    
    # Buat dokumen lebih rinci untuk setiap item sampah
    sampah_list = get_sampah()
    documents = []
    
    for item in sampah_list:
        doc = Document(
            page_content=f"Nama: {item['nama']}\nJenis: {item['jenis']}\nDeskripsi: {item['deskripsi']}",
            metadata={"source": item['nama'], "jenis": item['jenis']}
        )
        documents.append(doc)
    
    # Tambahkan juga dokumen umum tentang kategori
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

retriever = init_vectorstore()

# Fungsi untuk mengambil informasi dengan logging
def retrieve_info(question):
    try:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return None, "Tidak menemukan informasi yang relevan di database."
        
        context = "\n\nInformasi yang ditemukan:\n"
        for i, doc in enumerate(docs, 1):
            context += f"\n[{i}] {doc.page_content}\n(Sumber: {doc.metadata.get('source', 'tidak diketahui')})"
        
        return context, None
    except Exception as e:
        return None, f"Terjadi kesalahan saat mengambil informasi: {str(e)}"

# Fungsi untuk bertanya ke Gemini dengan prompt lebih baik
def ask_gemini(question):
    context, error = retrieve_info(question)
    if error:
        return error

    full_prompt = f"""
    Anda adalah ahli pengelolaan sampah. Berikut adalah beberapa informasi tentang berbagai jenis sampah:

    {context}

    Pertanyaan: "{question}"

    Tugas Anda:
    1. Identifikasi jenis sampah yang ditanyakan
    2. Kategorikan ke dalam Organik, Non-Organik, atau B3
    3. Jika tidak yakin, katakan tidak tahu

    Format jawaban yang HARUS diikuti:
    [Nama Sampah] termasuk dalam kategori [Kategori] karena [Alasan singkat].

    Contoh:
    Botol plastik termasuk dalam kategori Non-Organik karena terbuat dari bahan sintetis yang tidak dapat terurai secara alami.

    Jika tidak ada informasi yang cukup, jawab:
    Maaf, saya tidak memiliki informasi cukup tentang [kata kunci pertanyaan]. Mohon berikan deskripsi lebih detail.
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"

# Streamlit UI dengan contoh pertanyaan
st.title("♻️ Chatbot Kategori Sampah")

# Tambahkan contoh pertanyaan
st.markdown("""
**Contoh pertanyaan:**
- Botol plastik termasuk sampah apa?
- Bagaimana dengan daun kering?
- Apa yang dimaksud dengan sampah B3?
- Termasuk kategori apa baterai bekas?
""")

pertanyaan = st.text_input("Masukkan pertanyaan Anda tentang jenis sampah:")

if pertanyaan:
    with st.spinner("Mencari informasi..."):
        jawaban = ask_gemini(pertanyaan)
        if "tidak memiliki informasi cukup" in jawaban or "tidak tahu" in jawaban:
            st.warning(jawaban)
            st.info("""
            Tips: Coba gunakan nama yang lebih spesifik, seperti:
            - 'Botol minuman plastik' daripada 'botol'
            - 'Sisa sayuran' daripada 'sisa makanan'
            """)
        elif "Terjadi kesalahan" in jawaban:
            st.error(jawaban)
        else:
            st.success(jawaban)
            st.balloons()

if halaman == "Daur Ulang":
    st.title("♻️ Selamat Datang di Chatbot Kategori Sampah")
    st.markdown("Gunakan menu di kiri untuk mulai.")
    
elif halaman == "Tentang Sampah":
    st.title("🗑️ Informasi Kategori Sampah")
    st.markdown("""
    - **Organik:** Sampah yang bisa terurai secara alami (misalnya daun, sisa makanan).
    - **Non-Organik:** Sampah buatan manusia yang tidak mudah terurai (misalnya plastik, kaleng).
    - **B3:** Limbah berbahaya seperti baterai dan elektronik rusak.
    """)