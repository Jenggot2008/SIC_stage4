def retrieve_info(question, retriever):
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
