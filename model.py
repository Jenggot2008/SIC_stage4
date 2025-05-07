import streamlit as st

# Upload file model
uploaded_file = st.file_uploader("model.py", type=["pt"])

if uploaded_file is not None:
    with open("yolov8n.pt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Model berhasil di-upload!")
    # Load model
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
