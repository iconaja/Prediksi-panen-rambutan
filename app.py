
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_rambutan.pkl")

st.title("🌳 Prediksi Hasil Panen Rambutan")

st.write("""
Sistem prediksi hasil panen rambutan menggunakan
algoritma Random Forest Regression.
""")

st.subheader("Masukkan Data Kebun")

umur = st.number_input(
    "Umur Pohon (tahun)",
    min_value=4,
    max_value=15,
    value=8
)

jumlah = st.number_input(
    "Jumlah Pohon",
    min_value=5,
    max_value=10,
    value=7
)

curah_hujan = st.number_input(
    "Curah Hujan (mm/periode)",
    min_value=0.0,
    value=220.0
)

pemupukan = st.number_input(
    "Frekuensi Pemupukan (kali/periode)",
    min_value=0,
    value=3
)

penyiraman = st.number_input(
    "Frekuensi Penyiraman (kali/periode)",
    min_value=0,
    value=12
)

hama = st.number_input(
    "Intensitas Serangan Hama (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0
)

if st.button("Prediksi Hasil Panen"):

    data_input = pd.DataFrame([{
        'Umur Pohon (tahun)': umur,
        'Jumlah Pohon': jumlah,
        'Curah Hujan (mm/periode)': curah_hujan,
        'Frekuensi Pemupukan (kali/periode)': pemupukan,
        'Frekuensi Penyiraman (kali/periode)': penyiraman,
        'Intensitas Serangan Hama (%)': hama
    }])

    prediksi = model.predict(data_input)

    st.success(
        f"Prediksi Hasil Panen: {prediksi[0]:.2f} kg"
    )
