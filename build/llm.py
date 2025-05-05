import google.generativeai as genai
from retrieval import retrieve_info
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(question, retriever):
    context, error = retrieve_info(question, retriever)
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

    Format jawaban:
    [Nama Sampah] termasuk dalam kategori [Kategori] karena [Alasan singkat].
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
