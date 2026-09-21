import streamlit as st
import pandas as pd
import joblib

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Panen Rambutan",
    page_icon="🌳",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_rambutan(1).pkl")
    fitur = joblib.load("fitur_rambutan(1).pkl")
    return model, fitur

model, fitur = load_model()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🌳 Prediksi Rambutan")

menu = st.sidebar.radio(
    "Menu",
    [
        "🌳 Prediksi Panen",
        "📊 Informasi Model",
        "ℹ️ Tentang Sistem"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Random Forest Regression • Dataset 100 hasil observasi "
    "dengan konteks Manokwari, Papua Barat"
)

# ============================================================
# MENU PREDIKSI
# ============================================================
if menu == "🌳 Prediksi Panen":

    st.title("🌳 Prediksi Hasil Panen Rambutan")

    st.write(
        "Masukkan kondisi kebun rambutan untuk memperoleh "
        "estimasi total hasil panen dalam satu periode."
    )

    st.info(
        "Model menggunakan enam variabel: umur pohon, jumlah pohon, "
        "curah hujan, frekuensi pemupukan, frekuensi penyiraman, "
        "dan intensitas serangan hama."
    )

    col1, col2 = st.columns(2)

    with col1:

        umur = st.number_input(
            "Umur Rata-rata Pohon (tahun)",
            min_value=1,
            max_value=20,
            value=8,
            step=1
        )

        jumlah = st.number_input(
            "Jumlah Pohon",
            min_value=1,
            max_value=20,
            value=3,
            step=1
        )

        hujan = st.number_input(
            "Curah Hujan (mm/periode)",
            min_value=0.0,
            max_value=1000.0,
            value=220.0,
            step=10.0
        )

    with col2:

        pupuk = st.number_input(
            "Frekuensi Pemupukan (kali/periode)",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        siram = st.number_input(
            "Frekuensi Penyiraman (kali/periode)",
            min_value=0,
            max_value=30,
            value=4,
            step=1
        )

        hama = st.number_input(
            "Intensitas Serangan Hama (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )

    # --------------------------------------------------------
    # Peringatan rentang data pelatihan
    # --------------------------------------------------------
    di_luar_rentang = (
        jumlah > 15
        or hujan < 90
        or hujan > 360
        or pupuk > 4
        or siram > 8
        or hama < 2
        or hama > 45
    )

    if di_luar_rentang:
        st.warning(
            "⚠️ Sebagian input berada di luar rentang yang "
            "direpresentasikan dalam dataset pelatihan. Prediksi tetap "
            "dapat dilakukan, tetapi hasilnya perlu diinterpretasikan "
            "dengan lebih hati-hati."
        )

    # --------------------------------------------------------
    # Prediksi
    # --------------------------------------------------------
    if st.button(
        "🔍 Prediksi Hasil Panen",
        type="primary",
        use_container_width=True
    ):

        data_input = pd.DataFrame(
            [[
                umur,
                jumlah,
                hujan,
                pupuk,
                siram,
                hama
            ]],
            columns=[
                "Umur Pohon (tahun)",
                "Jumlah Pohon",
                "Curah Hujan (mm/periode)",
                "Frekuensi Pemupukan (kali/periode)",
                "Frekuensi Penyiraman (kali/periode)",
                "Intensitas Serangan Hama (%)"
            ]
        )

        # Memastikan urutan fitur sama dengan saat training
        data_input = data_input[fitur]

        prediksi = model.predict(data_input)[0]

        rata_per_pohon = prediksi / jumlah

        st.markdown("---")
        st.subheader("📈 Hasil Prediksi")

        hasil1, hasil2 = st.columns(2)

        with hasil1:
            st.metric(
                "Estimasi Total Hasil Panen",
                f"{prediksi:.2f} kg/periode"
            )

        with hasil2:
            st.metric(
                "Estimasi Rata-rata per Pohon",
                f"{rata_per_pohon:.2f} kg/pohon"
            )

        st.caption(
            "Hasil merupakan estimasi model Random Forest Regression "
            "dan bukan nilai panen yang pasti."
        )


# ============================================================
# MENU INFORMASI MODEL
# ============================================================
elif menu == "📊 Informasi Model":

    st.title("📊 Informasi Model")

    st.subheader("Random Forest Regression")

    st.write(
        "Model dikembangkan menggunakan 100 data hasil observasi "
        "dengan konteks perkebunan rambutan skala rumahan di "
        "Manokwari, Papua Barat."
    )

    # --------------------------------------------------------
    # Hold-out
    # --------------------------------------------------------
    st.markdown("### Evaluasi Hold-out 80:20")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", "29.27 kg")
    c2.metric("MSE", "1595.62")
    c3.metric("RMSE", "39.95 kg")
    c4.metric("R²", "0.6200")

    st.caption(
        "Evaluasi hold-out menggunakan 80 observasi sebagai data "
        "training dan 20 observasi sebagai data testing."
    )

    # --------------------------------------------------------
    # Cross Validation
    # --------------------------------------------------------
    st.markdown("### 5-Fold Cross Validation")

    cv1, cv2, cv3, cv4 = st.columns(4)

    cv1.metric("MAE", "26.50 ± 5.11 kg")
    cv2.metric("MSE", "1439.37 ± 582.61")
    cv3.metric("RMSE", "37.19 ± 7.48 kg")
    cv4.metric("R²", "0.7818 ± 0.0605")

    st.caption(
        "Nilai menunjukkan rata-rata ± standar deviasi dari "
        "lima fold pengujian."
    )

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------
    st.markdown("### Feature Importance")

    importance_data = pd.DataFrame({
        "Variabel": [
            "Jumlah Pohon",
            "Umur Pohon (tahun)",
            "Frekuensi Penyiraman (kali/periode)",
            "Intensitas Serangan Hama (%)",
            "Curah Hujan (mm/periode)",
            "Frekuensi Pemupukan (kali/periode)"
        ],
        "Persentase (%)": [
            73.6059,
            11.5674,
            5.7702,
            4.6665,
            3.3019,
            1.0882
        ]
    })

    st.bar_chart(
        importance_data.set_index("Variabel")
    )

    st.dataframe(
        importance_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Feature importance menunjukkan kontribusi relatif variabel "
        "dalam proses prediksi model dan tidak menunjukkan hubungan "
        "sebab-akibat."
    )

    # --------------------------------------------------------
    # Konfigurasi
    # --------------------------------------------------------
    st.markdown("### Konfigurasi Penelitian")

    konfigurasi = pd.DataFrame({
        "Parameter": [
            "Algoritma",
            "Jumlah Tree",
            "Jumlah Observasi",
            "Jumlah Fitur",
            "Hold-out",
            "Cross Validation",
            "Random State"
        ],
        "Nilai": [
            "Random Forest Regression",
            "100",
            "100",
            "6",
            "80% Training : 20% Testing",
            "5-Fold",
            "42"
        ]
    })

    st.dataframe(
        konfigurasi,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MENU TENTANG
# ============================================================
else:

    st.title("ℹ️ Tentang Sistem")

    st.write(
        "Sistem ini merupakan prototipe prediksi hasil panen rambutan "
        "menggunakan algoritma Random Forest Regression."
    )

    st.markdown("### Variabel Model")

    st.write(
        """
        1. Umur rata-rata pohon (tahun)
        2. Jumlah pohon
        3. Curah hujan (mm/periode)
        4. Frekuensi pemupukan (kali/periode)
        5. Frekuensi penyiraman (kali/periode)
        6. Intensitas serangan hama (%)
        """
    )

    st.markdown("### Target Prediksi")

    st.write(
        "Target model adalah **total hasil panen rambutan "
        "(kg/periode)** pada satu rumah atau kebun."
    )

    st.markdown("### Alur Sistem")

    st.write(
        """
        Input kondisi kebun → pemrosesan enam variabel → 
        Random Forest Regression → prediksi hasil panen.
        """
    )

    st.markdown("### Dataset")

    st.write(
        "Model dikembangkan menggunakan **100 observasi sintetis** "
        "dengan konteks Manokwari, Papua Barat. Dataset dirancang "
        "untuk kebutuhan pengembangan dan pengujian prototipe."
    )

    st.warning(
        "Dataset yang digunakan merupakan data sintetis/simulasi, "
        "bukan hasil pengukuran 100 kebun secara langsung. "
        "Validasi menggunakan data lapangan diperlukan sebelum sistem "
        "digunakan sebagai dasar keputusan pertanian nyata."
    )
