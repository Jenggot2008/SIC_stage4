import sqlite3
import os

DB_PATH = "database.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS sampah (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nama TEXT,
                jenis TEXT,
                deskripsi TEXT)''')
    
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

def get_sampah():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nama, jenis, deskripsi FROM sampah")
    data = [{"id": row[0], "nama": row[1], "jenis": row[2], "deskripsi": row[3]} for row in c.fetchall()]
    conn.close()
    return data
