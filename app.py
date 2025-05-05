import streamlit as st
from db import init_db
from vectorstore import init_vectorstore
from llm import ask_gemini
import folium
from streamlit_folium import folium_static
import streamlit.components.v1 as components
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
    # Set page title and header
    st.title("♻️ Smart Waste Monitoring System")
    
    # Section 1: Waste Metrics Dashboard
    st.header("📊 Real-time Waste Metrics")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    try:
        # Get waste data with error handling
        organik = get_ubidots_value("sampah_organik")
        anorganik = get_ubidots_value("sampah_anorganik")
        b3 = get_ubidots_value("sampah_b3")
        
        # Display metrics in columns
        with col1:
            st.metric(
                label="Organic Waste", 
                value=f"{int(organik)} items" if organik is not None else "N/A",
                help="Biodegradable waste like food scraps"
            )
        
        with col2:
            st.metric(
                label="Inorganic Waste", 
                value=f"{int(anorganik)} items" if anorganik is not None else "N/A",
                help="Recyclable waste like plastic and paper"
            )
        
        with col3:
            st.metric(
                label="Hazardous Waste (B3)", 
                value=f"{int(b3)} items" if b3 is not None else "N/A",
                help="Dangerous materials like batteries and chemicals"
            )
            
    except Exception as e:
        st.error(f"⚠️ Failed to retrieve waste data: {str(e)}")
    
    # Section 2: Camera Detection
    st.header("📷 Waste Detection Camera")
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px 24px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Configuration
    CAMERA_URL = "http://192.168.153.238/cam-mid.jpg"
    MODEL_PATH = "yolo11n.pt"
    
    # Initialize YOLO model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {str(e)}")
        return
    
    # Detection button with spinner
    if st.button("🔍 Run Waste Detection", key="detect_button"):
        with st.spinner("Detecting waste items..."):
            try:
                # Fetch camera image
                img_resp = urllib.request.urlopen(CAMERA_URL, timeout=5)
                img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                img = cv2.flip(img, 0)
                
                # Run detection
                results = model(img, stream=True)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        
                        # Draw bounding box and label
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img, 
                            f"{label} {conf:.2f}", 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            color, 
                            2
                        )
                
                # Convert BGR to RGB for Streamlit
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.image(
                    img_rgb, 
                    caption="Waste Detection Results", 
                    use_column_width=True,
                    channels="RGB"
                )
                
                # Show success message
                st.success("✅ Detection completed successfully!")
                
            except urllib.error.URLError as e:
                st.error(f"📡 Camera connection error: Please check the camera URL and network connection")
            except Exception as e:
                st.error(f"❌ Detection failed: {str(e)}")
    
    # Help section
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
            - **Real-time Metrics**: Shows current waste levels in your area
            - **Camera Detection**: 
                1. Click the detection button to capture an image
                2. The system will identify different types of waste
                3. Results show with bounding boxes and confidence levels
            - For best results:
                - Ensure proper lighting
                - Position waste items clearly in camera view
                - Keep camera lens clean
        """)

# Halaman Driver
def halaman_driver():
    # Custom CSS for styling
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .header {
            color: #2e7d32;
        }
        .map-container {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .form-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .location-status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .location-active {
            background-color: #e8f5e9;
            border-left: 5px solid #4CAF50;
        }
        .location-inactive {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .task-card {
            transition: all 0.3s ease;
            margin-bottom: 10px;
        }
        .task-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("♻️ WasteFlow - Driver Dashboard")
    
    if not st.session_state.logged_in:
        st.subheader("Selamat datang di Layanan Driver Sampah")
        st.write("Silakan login atau daftar untuk mengakses fitur lengkap")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container(border=True):
                st.markdown("### 🔐 Login Driver")
                phone_login = st.text_input("Nomor HP", key="login_phone")
                password_login = st.text_input("Password", type="password", key="login_password")
                if st.button("Login", key="login_btn"):
                    if phone_login in st.session_state.users and st.session_state.users[phone_login] == password_login:
                        st.session_state.logged_in = True
                        st.session_state.logged_user = phone_login
                        st.success("✅ Login berhasil!")
                        st.rerun()
                    else:
                        st.error("❌ Nomor HP atau password salah")
        
        with col2:
            if st.session_state.show_signup:
                with st.container(border=True):
                    st.markdown("### 📝 Pendaftaran Driver Baru")
                    phone = st.text_input("Nomor HP", key="signup_phone")
                    password = st.text_input("Password", type="password", key="signup_password")
                    repassword = st.text_input("Ulangi Password", type="password", key="signup_repassword")
                    
                    if st.button("Daftar Akun", key="signup_btn"):
                        if phone in st.session_state.users:
                            st.warning("⚠️ Nomor HP sudah terdaftar")
                        elif password != repassword:
                            st.error("❌ Password tidak cocok")
                        elif not phone or not password:
                            st.error("❌ Semua kolom harus diisi")
                        else:
                            st.session_state.users[phone] = password
                            st.success("✅ Pendaftaran berhasil! Silakan login")
                            st.session_state.show_signup = False
                            st.rerun()
            
            if not st.session_state.show_signup:
                with st.container(border=True):
                    st.markdown("### 🆕 Belum punya akun?")
                    if st.button("Daftar Sekarang", key="show_signup_btn"):
                        st.session_state.show_signup = True
                        st.rerun()
    
    if st.session_state.logged_in:
        st.success(f"Selamat datang, Driver {st.session_state.logged_user}!")
        
        # Initialize location tracking in session state
        if 'location_active' not in st.session_state:
            st.session_state.location_active = False
        if 'driver_location' not in st.session_state:
            st.session_state.driver_location = None
        
        # Location Tracking Section
        st.subheader("📍 Pelacakan Lokasi")
        
        # Location status indicator
        location_status = st.empty()
        
        # Toggle location button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔵 Aktifkan Pelacakan" if not st.session_state.location_active else "🔴 Matikan Pelacakan"):
                st.session_state.location_active = not st.session_state.location_active
                if st.session_state.location_active:
                    st.toast("Pelacakan lokasi diaktifkan", icon="✅")
                else:
                    st.toast("Pelacakan lokasi dimatikan", icon="⛔")
                st.rerun()
        
        with col2:
            if st.session_state.location_active:
                try:
                    # Get current location
                    g = geocoder.ip('me')
                    if g.latlng:
                        st.session_state.driver_location = g.latlng
                        st.success(f"Lokasi terbaru: {g.latlng[0]:.4f}, {g.latlng[1]:.4f}")
                    else:
                        st.warning("Tidak dapat mendapatkan lokasi saat ini")
                except Exception as e:
                    st.error(f"Error mendapatkan lokasi: {str(e)}")
                    st.session_state.location_active = False
        
        # Update location status display
        if st.session_state.location_active:
            location_status.markdown("""
            <div class="location-status location-active">
                <h4>🟢 PELACAKAN AKTIF</h4>
                <p>Lokasi Anda sedang dipantau secara real-time</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            location_status.markdown("""
            <div class="location-status location-inactive">
                <h4>🔴 PELACAKAN NON-AKTIF</h4>
                <p>Nyalakan pelacakan untuk memulai pemantauan</p>
            </div>
            """, unsafe_allow_html=True)

def get_location():
    loc_js = """
    <script>
    navigator.geolocation.getCurrentPosition(
        position => {
            const loc = position.coords.latitude + "," + position.coords.longitude;
            window.parent.document.getElementById('location').value = loc;
        }
    );
    </script>
    """
    components.html(loc_js, height=0)
    return st.text_input("Location", key="location")
    
        # Dashboard Overview
        st.subheader("📊 Dashboard Pemantauan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tugas Hari Ini", "8", "+2 dari kemarin")
        with col2:
            st.metric("Sampah Terkumpul", "142 kg", "12% ↑")
        with col3:
            st.metric("Rating Driver", "⭐ 4.8", "0.2 ↑")
        
        # Map Section with real-time location
        st.subheader("🗺️ Peta Lokasi Real-time")
        
        with st.container(border=True):
            # Use either current location or default
            map_center = st.session_state.driver_location if st.session_state.driver_location else [-6.200, 106.816]
            
            m = folium.Map(location=map_center, zoom_start=15)
            
            # Add driver location if available
            if st.session_state.driver_location:
                folium.Marker(
                    location=st.session_state.driver_location,
                    popup="Lokasi Anda Saat Ini",
                    icon=folium.Icon(color='green', icon='user', prefix='fa')
                ).add_to(m)
                
                # Add moving marker for better visualization
                folium.CircleMarker(
                    location=st.session_state.driver_location,
                    radius=8,
                    color='green',
                    fill=True,
                    fill_color='green',
                    popup="Posisi Terkini"
                ).add_to(m)
            
            # Add trash bins
            lokasi_tong_list = [
                {"nama": "Tong Sampah MAN 9", "lokasi": [-6.240920384479476, 106.91067582361595], "status": "Penuh", "warna": "red"},
                {"nama": "Tong Sampah SMA 71", "lokasi": [-6.241870872587285, 106.9117994757494], "status": "Sedang", "warna": "orange"},
                {"nama": "Tong Sampah Puskesmas", "lokasi": [-6.241985021106573, 106.91109136189029], "status": "Kosong", "warna": "green"},
                {"nama": "Tong Sampah Masjid", "lokasi": [-6.240678193236319, 106.90698684039081], "status": "Penuh", "warna": "red"},
                {"nama": "Tong Sampah Residence", "lokasi": [-6.23982503696325, 106.90968580798639], "status": "Sedang", "warna": "orange"},
            ]
            
            for tong in lokasi_tong_list:
                folium.Marker(
                    location=tong["lokasi"],
                    popup=f"{tong['nama']} - Status: {tong['status']}",
                    icon=folium.Icon(color=tong["warna"], icon='trash')
                ).add_to(m)
                
                if st.session_state.driver_location:
                    folium.PolyLine(
                        [st.session_state.driver_location, tong["lokasi"]],
                        color='blue',
                        weight=2,
                        opacity=0.6,
                        tooltip=f"Jarak: {np.random.randint(500, 2000)} meter"
                    ).add_to(m)
            
            folium_static(m, height=400)
        
        # Task Management
        st.subheader("📋 Daftar Tugas")
        
        tasks = [
            {"lokasi": "MAN 9", "waktu": "08:00 - 09:00", "status": "Belum"},
            {"lokasi": "SMA 71", "waktu": "10:00 - 11:00", "status": "Proses"},
            {"lokasi": "Puskesmas", "waktu": "13:00 - 14:00", "status": "Belum"}
        ]
        
        for task in tasks:
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 1])
                cols[0].write(f"**{task['lokasi']}**")
                cols[1].write(f"⏰ {task['waktu']}")
                cols[2].write(f"📌 {task['status']}")
                if cols[3].button("Mulai", key=f"btn_{task['lokasi']}"):
                    st.success(f"Tugas di {task['lokasi']} dimulai!")
        
        # Auto-refresh when location is active
        if st.session_state.location_active:
            st.experimental_rerun()
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.logged_user = None
            st.session_state.location_active = False
            st.success("Anda telah logout")
            st.rerun()

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
