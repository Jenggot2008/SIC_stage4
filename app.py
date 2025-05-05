import streamlit as st
from db import init_db
from vectorstore import init_vectorstore
from llm import ask_gemini
import folium
from streamlit_folium import folium_static
import geocoder
import requests
import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import tempfile

# Inisialisasi konfigurasi halaman
st.set_page_config(page_icon="♻️")

# Inisialisasi database
init_db()

# Inisialisasi Session State
def init_session_state():
    if 'users' not in st.session_state:
        st.session_state.users = {}
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.logged_user = None
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

init_session_state()

# Sidebar (Navbar kiri)
st.sidebar.image("rgf.png", width=200)
st.sidebar.title("Navigasi")
halaman = st.sidebar.radio("Pilih halaman:", ["Beranda", "User", "Driver", "Tanya Chatbot"])

# Halaman Beranda
def halaman_beranda():
    st.title("♻️ Waste Flow")
    st.subheader("Sistem Pemantauan dan Klasifikasi Sampah Berbasis AI")
    st.write("Selamat datang di aplikasi Waste Flow!")
    st.write("""
    WasteFlow adalah sistem berbasis AI dan IoT yang memanfaatkan kamera dan teknologi pengenalan objek 
    untuk memilah sampah serta memantau jumlahnya secara real-time. Sistem ini secara otomatis mengirim notifikasi 
    kepada petugas kebersihan dan menyajikan analisis data kepada masyarakat guna meningkatkan kesadaran serta 
    partisipasi dalam pengelolaan sampah berkelanjutan. Selain itu, WasteFlow mendukung perumusan kebijakan 
    lingkungan yang lebih tepat melalui data yang akurat dan up-to-date.
    """)

# Ambil data dari Ubidots
def get_ubidots_value(variable):
    UBIDOTS_TOKEN = "BBUS-Y5s7CObAKhExn20yRnu9kKoGYTXnBB"
    DEVICE_LABEL = "RGF_BAKSO"
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}/{variable}/lv"
    headers = {"X-Auth-Token": UBIDOTS_TOKEN}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return float(response.text)
        else:
            st.error(f"Gagal mengambil data {variable}. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Halaman User
def halaman_user():
    st.title("♻️ Monitoring Sampah")
    
    # Mendapatkan data sampah dari Ubidots
    organik = get_ubidots_value("sampah_organik")
    anorganik = get_ubidots_value("sampah_anorganik")
    b3 = get_ubidots_value("sampah_b3")

    if organik is not None and anorganik is not None and b3 is not None:
        st.metric("Sampah Organik", f"{int(organik)} item")
        st.metric("Sampah Anorganik", f"{int(anorganik)} item")
        st.metric("Sampah B3", f"{int(b3)} item")

    st.subheader("📷 Deteksi Kamera (YOLO)")
    
    # URL kamera yang digunakan untuk mengambil gambar
    URL = "http://192.168.153.238/cam-mid.jpg"  # Sesuaikan IP dengan kamera yang digunakan
    
    # Try using a smaller YOLO model that's available by default
    try:
        # Try loading a pretrained YOLOv8n model (smallest version)
        model = YOLO('yolov8n.pt')  # This will download the model if not found locally
        
        if st.button("🔍 Jalankan Deteksi"):
            try:
                # Ambil gambar dari URL
                img_resp = urllib.request.urlopen(URL, timeout=2)
                img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                img = cv2.flip(img, 0)  # Membalikkan gambar jika perlu
                
                # Deteksi objek menggunakan YOLO
                results = model(img, stream=True)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        
                        # Menambahkan bounding box dan label ke gambar
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Menyimpan gambar hasil deteksi ke sementara file
                _, output_img = tempfile.mkstemp(suffix=".jpg")
                cv2.imwrite(output_img, img)
                
                # Menampilkan gambar hasil deteksi di Streamlit
                st.image(output_img, caption="Deteksi Kamera (YOLO)", use_column_width=True)
            except Exception as e:
                st.error(f"❌ Gagal mengambil atau memproses gambar: {e}")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        st.info("Pastikan model YOLO tersedia atau gunakan model yang dapat diunduh otomatis.")

# Halaman Driver
def halaman_driver():
    st.title("♻️ Halaman Driver Sampah")
    st.write("Silakan daftar dan login untuk menggunakan layanan driver sampah.")

    # Form Daftar dan Login
    if not st.session_state.logged_in:
        if st.button("Belum punya akun? Daftar di sini"):
            st.session_state.show_signup = not st.session_state.show_signup

        if st.session_state.show_signup:
            st.markdown("### 📝 Form Pendaftaran Driver")
            phone = st.text_input("Nomor HP", key="signup_phone")
            password = st.text_input("Password", type="password", key="signup_password")
            repassword = st.text_input("Ulangi Password", type="password", key="signup_repassword")
            if st.button("Daftar"):
                if phone in st.session_state.users:
                    st.warning("⚠️ Nomor HP sudah terdaftar.")
                elif password != repassword:
                    st.error("❌ Password tidak cocok.")
                elif not phone or not password:
                    st.error("❌ Semua kolom harus diisi.")
                else:
                    st.session_state.users[phone] = password
                    st.success("✅ Pendaftaran berhasil. Silakan login.")
                    st.session_state.show_signup = False

        # Form Login
        st.markdown("### 🔐 Login Driver")
        phone_login = st.text_input("Nomor HP", key="login_phone")
        password_login = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if phone_login in st.session_state.users and st.session_state.users[phone_login] == password_login:
                st.session_state.logged_in = True
                st.session_state.logged_user = phone_login
                st.success("✅ Login berhasil!")
            else:
                st.error("❌ Nomor HP atau password salah.")
    
    if st.session_state.logged_in:
        st.markdown("---")
        st.subheader("📍 Lokasi Anda & Tong Sampah")

        lokasi_driver = geocoder.ip('me').latlng or [-6.200, 106.816]

        lokasi_tong_list = [
            {"nama": "Tong Sampah MAN 9", "lokasi": [-6.240920384479476, 106.91067582361595]},
            {"nama": "Tong Sampah SMA 71", "lokasi": [-6.241870872587285, 106.9117994757494]},
            {"nama": "Tong Sampah Puskesmas Duren Sawit", "lokasi": [-6.241985021106573, 106.91109136189029]},
            {"nama": "Tong Sampah Masjid Daarul 'Ilmi", "lokasi": [-6.240678193236319, 106.90698684039081]},
            {"nama": "Tong Sampah OYO 842 Arafuru Residence", "lokasi": [-6.23982503696325, 106.90968580798639]},
        ]

        m = folium.Map(location=lokasi_driver, zoom_start=15)
        folium.Marker(lokasi_driver, popup="Lokasi Anda", icon=folium.Icon(color='green')).add_to(m)

        for tong in lokasi_tong_list:
            folium.Marker(tong["lokasi"], popup=tong["nama"], icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine([lokasi_driver, tong["lokasi"]], color='blue', weight=2, opacity=0.6).add_to(m)

        folium_static(m)

# Halaman Tanya Chatbot
def halaman_chatbot():
    retriever = init_vectorstore()
    st.title("♻️ Chatbot Kategori Sampah")
    st.markdown("""
    **Contoh pertanyaan:**
    - Botol plastik termasuk sampah apa?
    - Bagaimana dengan daun kering?
    - Apa yang dimaksud dengan sampah B3?
    - Termasuk kategori apa baterai bekas?
    """)
    pertanyaan = st.text_input("Tanya tentang sampah", key="chatbot_input")
    if pertanyaan:
        answer = ask_gemini(pertanyaan, retriever)
        st.write(answer)

# Menampilkan halaman sesuai pilihan
if halaman == "Beranda":
    halaman_beranda()
elif halaman == "User":
    halaman_user()
elif halaman == "Driver":
    halaman_driver()
elif halaman == "Tanya Chatbot":
    halaman_chatbot()
