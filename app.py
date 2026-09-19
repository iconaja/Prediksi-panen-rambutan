import streamlit as st
import pandas as pd
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Panen Rambutan",
    page_icon="🌳",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("model_rambutan.pkl")

model = load_model()

# =========================================================
# HEADER
# =========================================================
st.title("🌳 Sistem Prediksi Hasil Panen Rambutan")

st.caption(
    "Implementasi Random Forest Regression pada "
    "perkebunan rambutan skala rumahan"
)

st.divider()

# =========================================================
# MENU
# =========================================================
menu = st.sidebar.radio(
    "Menu Utama",
    [
        "🌳 Prediksi Panen",
        "📊 Informasi Model",
        "ℹ️ Tentang Sistem"
    ]
)

# =========================================================
# HALAMAN PREDIKSI
# =========================================================
if menu == "🌳 Prediksi Panen":

    st.header("Prediksi Hasil Panen")

    st.write(
        "Masukkan kondisi perkebunan rambutan untuk "
        "memperoleh estimasi hasil panen."
    )

    st.info(
        "Umur pohon merupakan rata-rata umur pohon "
        "rambutan pada satu rumah/kebun."
    )

    # -------------------------
    # INPUT BARIS 1
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input(
            "Umur Rata-rata Pohon (tahun)",
            min_value=4,
            max_value=15,
            value=8,
            step=1
        )

    with col2:
        jumlah = st.number_input(
            "Jumlah Pohon",
            min_value=5,
            max_value=10,
            value=7,
            step=1
        )

    # -------------------------
    # INPUT BARIS 2
    # -------------------------

    col3, col4 = st.columns(2)

    with col3:
        curah_hujan = st.number_input(
            "Curah Hujan (mm/periode)",
            min_value=0.0,
            value=220.0,
            step=10.0
        )

    with col4:
        pemupukan = st.number_input(
            "Frekuensi Pemupukan (kali/periode)",
            min_value=0,
            value=3,
            step=1
        )

    # -------------------------
    # INPUT BARIS 3
    # -------------------------

    col5, col6 = st.columns(2)

    with col5:
        penyiraman = st.number_input(
            "Frekuensi Penyiraman (kali/periode)",
            min_value=0,
            value=12,
            step=1
        )

    with col6:
        hama = st.number_input(
            "Intensitas Serangan Hama (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0
        )

    st.write("")

    # =====================================================
    # PREDIKSI
    # =====================================================

    if st.button(
        "🔍 Prediksi Hasil Panen",
        type="primary",
        use_container_width=True
    ):

        data_input = pd.DataFrame([{
            "Umur Pohon (tahun)": umur,
            "Jumlah Pohon": jumlah,
            "Curah Hujan (mm/periode)": curah_hujan,
            "Frekuensi Pemupukan (kali/periode)": pemupukan,
            "Frekuensi Penyiraman (kali/periode)": penyiraman,
            "Intensitas Serangan Hama (%)": hama
        }])

        prediksi = model.predict(data_input)[0]

        rata_rata_pohon = prediksi / jumlah

        st.divider()

        st.subheader("📋 Hasil Prediksi")

        hasil1, hasil2 = st.columns(2)

        with hasil1:
            st.metric(
                "Prediksi Total Hasil Panen",
                f"{prediksi:.2f} kg"
            )

        with hasil2:
            st.metric(
                "Estimasi Rata-rata per Pohon",
                f"{rata_rata_pohon:.2f} kg/pohon"
            )

        st.success(
            f"Berdasarkan kondisi kebun yang dimasukkan, "
            f"model memperkirakan total hasil panen rambutan "
            f"sebesar **{prediksi:.2f} kg per periode**."
        )

        # Tampilkan data yang dimasukkan
        with st.expander("Lihat data yang digunakan"):
            st.dataframe(
                data_input,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    st.caption(
        "Catatan: Sistem saat ini merupakan prototipe penelitian "
        "yang dikembangkan menggunakan data sintetis 100 rumah. "
        "Hasil prediksi belum ditujukan sebagai rekomendasi "
        "produksi aktual sebelum divalidasi menggunakan data lapangan."
    )


# =========================================================
# HALAMAN INFORMASI MODEL
# =========================================================
elif menu == "📊 Informasi Model":

    st.header("📊 Informasi Model Random Forest")

    st.write(
        "Model dikembangkan menggunakan algoritma "
        "**Random Forest Regression** untuk memprediksi "
        "hasil panen rambutan."
    )

    st.subheader("Evaluasi Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R²",
            "0.7124"
        )

    with col2:
        st.metric(
            "MAE",
            "38.20 kg"
        )

    with col3:
        st.metric(
            "RMSE",
            "47.45 kg"
        )

    with col4:
        st.metric(
            "MSE",
            "2251.06"
        )

    st.info(
        "Nilai R² sebesar 0,7124 menunjukkan bahwa model "
        "mampu menjelaskan sekitar 71,24% variasi hasil panen "
        "pada data pengujian sintetis."
    )

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("Feature Importance")

    importance_data = pd.DataFrame({
        "Fitur": [
            "Jumlah Pohon",
            "Intensitas Serangan Hama",
            "Umur Pohon",
            "Frekuensi Penyiraman",
            "Frekuensi Pemupukan",
            "Curah Hujan"
        ],
        "Persentase": [
            42.20,
            20.25,
            13.84,
            8.58,
            8.19,
            6.93
        ]
    })

    st.bar_chart(
        importance_data,
        x="Fitur",
        y="Persentase"
    )

    st.dataframe(
        importance_data,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "**Jumlah Pohon** memiliki nilai feature importance "
        "tertinggi sebesar **42,20%**, diikuti oleh "
        "**Intensitas Serangan Hama sebesar 20,25%** dan "
        "**Umur Pohon sebesar 13,84%**."
    )

    st.warning(
        "Feature importance menunjukkan kontribusi relatif "
        "suatu fitur terhadap prediksi model dan tidak dapat "
        "langsung diinterpretasikan sebagai hubungan sebab-akibat."
    )

    st.divider()

    st.subheader("Konfigurasi Model")

    konfigurasi = pd.DataFrame({
        "Parameter": [
            "Algoritma",
            "n_estimators",
            "Train Data",
            "Test Data",
            "Jumlah Observasi",
            "Jumlah Fitur"
        ],
        "Nilai": [
            "Random Forest Regression",
            "100",
            "80%",
            "20%",
            "100 rumah",
            "6 fitur"
        ]
    })

    st.dataframe(
        konfigurasi,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# HALAMAN TENTANG SISTEM
# =========================================================
elif menu == "ℹ️ Tentang Sistem":

    st.header("ℹ️ Tentang Sistem")

    st.subheader("Prediksi Hasil Panen Rambutan")

    st.write(
        """
        Sistem ini dikembangkan sebagai prototipe penelitian
        untuk menerapkan algoritma **Random Forest Regression**
        dalam memprediksi hasil panen rambutan pada perkebunan
        skala rumahan.
        """
    )

    st.subheader("Variabel Penelitian")

    st.markdown(
        """
        Sistem menggunakan enam variabel prediktor:

        1. **Umur Pohon** – rata-rata umur pohon rambutan.
        2. **Jumlah Pohon** – jumlah pohon rambutan pada rumah/kebun.
        3. **Curah Hujan** – jumlah curah hujan selama periode pengamatan.
        4. **Frekuensi Pemupukan** – jumlah kegiatan pemupukan per periode.
        5. **Frekuensi Penyiraman** – jumlah kegiatan penyiraman per periode.
        6. **Intensitas Serangan Hama** – persentase intensitas serangan hama.

        Variabel target adalah **Hasil Panen Rambutan (kg/periode)**.
        """
    )

    st.subheader("Alur Sistem")

    st.code(
        """
Data Kebun
    ↓
Preprocessing Data
    ↓
Random Forest Regression
    ↓
Model Prediksi
    ↓
Input Data Kebun Baru
    ↓
Prediksi Hasil Panen (kg)
        """
    )

    st.subheader("Dataset")

    st.write(
        """
        Model prototipe dikembangkan menggunakan **100 observasi
        data sintetis**, dengan satu observasi merepresentasikan
        satu rumah/kebun rambutan. Setiap rumah memiliki sekitar
        **5–10 pohon rambutan**, dengan umur rata-rata pohon
        berada pada rentang **4–15 tahun**.
        """
    )

    st.warning(
        "Data yang digunakan pada tahap prototipe merupakan "
        "data sintetis/simulasi dan bukan hasil survei lapangan. "
        "Validasi menggunakan data observasi aktual diperlukan "
        "sebelum sistem digunakan untuk pengambilan keputusan nyata."
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Sistem Prediksi Hasil Panen Rambutan | "
    "Random Forest Regression"
)
