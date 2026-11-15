import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
import io
import plotly.express as px
from streamlit_option_menu import option_menu

# ==================================================
# KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="FruitScan - Deteksi Buah",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "best.pt"
RIWAYAT_PATH = Path(__file__).parent / "riwayat_deteksi.csv"

KOLOM_RIWAYAT = ["Waktu", "Nama File", "Nama Buah", "Kondisi", "Confidence (%)"]
WARNA_KONDISI = {"Segar": "#3D9B62", "Busuk": "#D65B50"}

KELAS_BUAH = {
    "good apple": ("Apel", "Segar"),
    "bad apple": ("Apel", "Busuk"),
    "good banana": ("Pisang", "Segar"),
    "bad banana": ("Pisang", "Busuk"),
    "good orange": ("Jeruk", "Segar"),
    "bad orange": ("Jeruk", "Busuk"),
    "good pomegranate": ("Delima", "Segar"),
    "bad pomegranate": ("Delima", "Busuk")
}

# ==================================================
# CSS / TAMPILAN
# ==================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F8F9F6;
    }

    [data-testid="stSidebar"] {
        background-color: #263D32;
    }

    [data-testid="stSidebar"] * {
        color: #F5F7F3 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 10px;
        border-radius: 7px;
    }

    h1, h2, h3 {
        color: #29332D;
    }

    .page-subtitle {
        color: #78827B;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8ECE6;
        border-radius: 10px;
        padding: 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #78827B;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #263D32;
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #29332D;
        margin-bottom: 12px;
    }

    .info-box {
        background: #FFFFFF;
        border: 1px solid #E8ECE6;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    div.stButton > button {
        background-color: #527D62;
        color: white;
        border: none;
        border-radius: 7px;
        padding: 9px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #3D654D;
        color: white;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #B8C8B9;
        border-radius: 10px;
        padding: 12px;
    }

    hr {
        border-color: #E5EAE3;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #263D32 0%, #3D654D 60%, #527D62 100%);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 22px;
        color: #FFFFFF;
    }
    .hero h2 { color: #FFFFFF; margin: 0 0 6px 0; font-size: 26px; }
    .hero p  { color: #D5E3D8; margin: 0; font-size: 14px; line-height: 1.6; }

    .metric-card { box-shadow: 0 1px 3px rgba(38, 61, 50, 0.06); }
    .metric-card.hijau { border-left: 4px solid #3D9B62; }
    .metric-card.merah { border-left: 4px solid #D65B50; }
    .metric-card.abu   { border-left: 4px solid #527D62; }

    .info-box h3 { margin-top: 0; }

    .tips-box {
        background: #EEF4EF;
        border: 1px solid #D5E3D8;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 13.5px;
        color: #3A4A40;
        margin-bottom: 14px;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #FFFFFF;
        margin-right: 6px;
    }
    .badge.segar { background: #3D9B62; }
    .badge.busuk { background: #D65B50; }

    .chips { margin-top: 10px; }
    .chip {
        display: inline-block;
        background: #EEF4EF;
        border: 1px solid #D5E3D8;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 0 8px 8px 0;
        font-size: 13px;
        font-weight: 600;
    }

    .buah-card {
        background: #FFFFFF;
        border: 1px solid #E8ECE6;
        border-radius: 12px;
        padding: 20px 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(38, 61, 50, 0.06);
    }
    .buah-img svg { width: 84px; height: 84px; }
    .buah-img { margin-bottom: 10px; }
    .buah-dot {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        margin: 0 auto 12px auto;
    }
    .buah-nama {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #29332D !important;
    }

    .langkah {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .langkah .no {
        background: #527D62;
        color: #FFFFFF;
        min-width: 26px;
        height: 26px;
        border-radius: 50%;
        text-align: center;
        line-height: 26px;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Kunci warna teks supaya tetap terbaca di mode gelap maupun terang */
    .stApp, [data-testid="stMain"] {
        color: #29332D;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stHeading"],
    [data-testid="stHeading"] * {
        color: #29332D !important;
    }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricValue"] {
        color: #29332D;
    }
    .page-subtitle { color: #78827B !important; }

    .info-box, .info-box *,
    .tips-box, .tips-box * {
        color: #3A4A40 !important;
    }
    .info-box h3 { color: #263D32 !important; }

    /* Menu sidebar (option_menu) */
    [data-testid="stSidebar"] iframe {
        background: #263D32 !important;
        border: none !important;
        border-radius: 0 !important;
        color-scheme: normal;
    }

    .metric-label { color: #78827B !important; }
    .metric-value { color: #263D32 !important; }
    .section-title { color: #29332D !important; }

    .hero h2 { color: #FFFFFF !important; }
    .hero p  { color: #D5E3D8 !important; }
    .badge   { color: #FFFFFF !important; }
    .langkah .no { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# MODEL YOLO
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return YOLO(str(MODEL_PATH))


# ==================================================
# SESSION STATE
# ==================================================

def muat_riwayat():
    """Baca riwayat yang tersimpan permanen dari file CSV."""
    if RIWAYAT_PATH.exists():
        try:
            df = pd.read_csv(RIWAYAT_PATH)
            if all(k in df.columns for k in KOLOM_RIWAYAT):
                return df[KOLOM_RIWAYAT].to_dict("records")
        except Exception:
            pass
    return []


def simpan_riwayat(riwayat):
    """Simpan riwayat ke file CSV agar tidak hilang saat aplikasi ditutup."""
    try:
        pd.DataFrame(riwayat, columns=KOLOM_RIWAYAT).to_csv(
            RIWAYAT_PATH, index=False, encoding="utf-8-sig"
        )
    except OSError:
        pass


def hapus_file_riwayat():
    try:
        RIWAYAT_PATH.unlink(missing_ok=True)
    except OSError:
        pass


if "riwayat" not in st.session_state:
    st.session_state.riwayat = muat_riwayat()

if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None

if "nama_file" not in st.session_state:
    st.session_state.nama_file = None


# ==================================================
# FUNGSI GAMBAR
# ==================================================

def buka_gambar(file):
    gambar = Image.open(file)
    gambar = ImageOps.exif_transpose(gambar)

    if getattr(gambar, "is_animated", False):
        gambar.seek(0)

    return gambar.convert("RGB")


def gambar_font():
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


# ==================================================
# FUNGSI DETEKSI
# ==================================================

def deteksi_buah(gambar, confidence):
    model = load_model()

    if model is None:
        raise FileNotFoundError(
            "File best.pt tidak ditemukan. Letakkan best.pt satu folder dengan app_deteksi_buah.py."
        )

    hasil = model.predict(
        source=gambar,
        conf=confidence,
        iou=0.4,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = hasil[0]
    gambar_hasil = gambar.copy()
    draw = ImageDraw.Draw(gambar_hasil)
    font = gambar_font()

    data_deteksi = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            nilai_conf = float(box.conf[0].item())

            nama_kelas = result.names[class_id]

            if nama_kelas not in KELAS_BUAH:
                continue

            nama_buah, kondisi = KELAS_BUAH[nama_kelas]

            koordinat = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, koordinat)

            warna = "#3D9B62" if kondisi == "Segar" else "#D65B50"

            teks = f"{nama_buah} - {kondisi} ({nilai_conf * 100:.1f}%)"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=warna,
                width=4
            )

            bbox_teks = draw.textbbox((x1, max(0, y1 - 25)), teks, font=font)

            draw.rectangle(
                bbox_teks,
                fill=warna
            )

            draw.text(
                (x1, max(0, y1 - 25)),
                teks,
                fill="white",
                font=font
            )

            data_deteksi.append({
                "Nama Buah": nama_buah,
                "Kondisi": kondisi,
                "Confidence": round(nilai_conf * 100, 2)
            })

    return gambar_hasil, data_deteksi


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#FFFFFF;margin-bottom:0;'>🍃 FruitScan</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#C6D5C9;font-size:12px;'>Sistem Deteksi Kesegaran Buah</p>",
        unsafe_allow_html=True
    )

    st.divider()

    halaman = option_menu(
        menu_title="MENU UTAMA",
        options=[
            "Dashboard",
            "Deteksi Gambar",
            "Riwayat Deteksi",
            "Tentang Aplikasi"
        ],
        icons=["speedometer2", "camera", "clock-history", "info-circle"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#263D32",
                          "border-radius": "0", "border": "none"},
            "menu-title": {"font-size": "12px", "color": "#C6D5C9",
                           "letter-spacing": "1px"},
            "icon": {"color": "#C6D5C9", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "border-radius": "7px",
                         "color": "#F5F7F3", "margin": "2px 0",
                         "--hover-color": "#3D654D"},
            "nav-link-selected": {"background-color": "#527D62",
                                  "font-weight": "600"},
        },
    )

    st.divider()

    st.caption("FruitScan • YOLOv8n")


# ==================================================
# DASHBOARD
# ==================================================

if halaman == "Dashboard":

    st.title("Dashboard")
    st.markdown(
        '<p class="page-subtitle">Ringkasan aktivitas deteksi kesegaran buah.</p>',
        unsafe_allow_html=True
    )

    riwayat = st.session_state.riwayat

    total = len(riwayat)
    segar = sum(1 for x in riwayat if x["Kondisi"] == "Segar")
    busuk = sum(1 for x in riwayat if x["Kondisi"] == "Busuk")

    st.markdown("""
    <div class="hero">
        <h2>🍃 Selamat datang di FruitScan</h2>
        <p>Unggah foto buah dan ketahui kondisinya dalam hitungan detik.
        Mendukung apel, pisang, jeruk, dan delima, segar maupun busuk.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card abu">
            <div class="metric-label">Total Deteksi</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card hijau">
            <div class="metric-label">Buah Segar</div>
            <div class="metric-value">{segar}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card merah">
            <div class="metric-label">Buah Busuk</div>
            <div class="metric-value">{busuk}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if riwayat:
        df_dash = pd.DataFrame(riwayat)

        st.markdown('<div class="section-title">Statistik Deteksi</div>',
                    unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        with g1:
            hitung_kondisi = (
                df_dash["Kondisi"].value_counts().reset_index()
            )
            hitung_kondisi.columns = ["Kondisi", "Jumlah"]

            fig_kondisi = px.pie(
                hitung_kondisi,
                names="Kondisi",
                values="Jumlah",
                hole=0.55,
                color="Kondisi",
                color_discrete_map=WARNA_KONDISI,
                title="Perbandingan Segar dan Busuk"
            )
            fig_kondisi.update_layout(
                margin=dict(t=50, b=10, l=10, r=10),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_kondisi, use_container_width=True)

        with g2:
            per_buah = (
                df_dash.groupby(["Nama Buah", "Kondisi"])
                .size()
                .reset_index(name="Jumlah")
            )

            fig_buah = px.bar(
                per_buah,
                x="Nama Buah",
                y="Jumlah",
                color="Kondisi",
                barmode="group",
                color_discrete_map=WARNA_KONDISI,
                title="Jumlah Deteksi per Jenis Buah"
            )
            fig_buah.update_layout(
                margin=dict(t=50, b=10, l=10, r=10),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_buah, use_container_width=True)

        rata_conf = df_dash["Confidence (%)"].mean()
        st.caption(
            f"Rata-rata confidence seluruh deteksi: {rata_conf:.1f}%"
        )
    else:
        st.info(
            "Belum ada data deteksi. Buka menu **Deteksi Gambar** "
            "untuk mulai mendeteksi buah."
        )

    st.markdown('<div class="section-title">Informasi Aplikasi</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>FruitScan</b><br><br>
        Aplikasi berbasis web untuk mendeteksi jenis dan kondisi kesegaran buah
        menggunakan teknologi computer vision dengan model YOLOv8n.<br><br>
        <b>Jenis buah yang didukung:</b><br>
        Apel, Pisang, Jeruk, dan Delima.
    </div>
    """, unsafe_allow_html=True)

    if riwayat:
        st.markdown('<div class="section-title">Deteksi Terakhir</div>',
                    unsafe_allow_html=True)

        df = pd.DataFrame(riwayat)
        st.dataframe(
            df.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )


# ==================================================
# DETEKSI GAMBAR
# ==================================================

elif halaman == "Deteksi Gambar":

    st.title("Deteksi Gambar")
    st.markdown(
        '<p class="page-subtitle">Unggah gambar buah untuk mengetahui kondisi kesegarannya.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="tips-box">
        <b>Tips agar hasil lebih akurat</b><br>
        • Gunakan foto yang terang dan tidak buram.<br>
        • Hindari buah yang saling bertumpuk.<br>
        • Hanya apel, pisang, jeruk, dan delima yang dapat dikenali.
        Buah lain bisa saja salah terdeteksi.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        file_gambar = st.file_uploader(
            "Pilih gambar buah",
            type=[
                "jpg", "jpeg", "jfif", "png",
                "webp", "bmp", "gif", "tif", "tiff"
            ],
            help="Format yang didukung: JPG, JPEG, JFIF, PNG, WEBP, BMP, GIF, TIF, TIFF."
        )

    with col2:
        confidence = st.slider(
            "Batas Confidence",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05
        )

    if file_gambar is not None:

        try:
            gambar = buka_gambar(file_gambar)

            st.markdown("#### Pratinjau Gambar")

            _, preview_col, _ = st.columns([1, 2, 1])

            with preview_col:
                st.image(
                    gambar,
                    caption=file_gambar.name,
                    width=450
                )
            if st.button("Mulai Deteksi", use_container_width=True):

                with st.spinner("Sedang menganalisis gambar..."):
                    gambar_hasil, data = deteksi_buah(
                        gambar,
                        confidence
                    )

                st.session_state.hasil_deteksi = {
                    "gambar": gambar_hasil,
                    "data": data
                }

                st.session_state.nama_file = file_gambar.name

                waktu = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                for item in data:
                    st.session_state.riwayat.append({
                        "Waktu": waktu,
                        "Nama File": file_gambar.name,
                        "Nama Buah": item["Nama Buah"],
                        "Kondisi": item["Kondisi"],
                        "Confidence (%)": item["Confidence"]
                    })

                simpan_riwayat(st.session_state.riwayat)

            hasil = st.session_state.hasil_deteksi

            if hasil is not None and st.session_state.nama_file == file_gambar.name:

                st.divider()
                st.markdown("### Hasil Deteksi")

                col_asli, col_hasil = st.columns(2)

                with col_asli:
                    st.markdown("**Gambar Asli**")
                    st.image(gambar, use_container_width=True)

                with col_hasil:
                    st.markdown("**Hasil Analisis**")
                    st.image(hasil["gambar"], use_container_width=True)

                if hasil["data"]:

                    st.success(
                        f"Berhasil mendeteksi {len(hasil['data'])} objek buah."
                    )

                    df_hasil = pd.DataFrame(hasil["data"])

                    st.markdown("#### Detail Hasil")

                    st.dataframe(
                        df_hasil,
                        use_container_width=True,
                        hide_index=True
                    )

                    jumlah_segar = sum(
                        1 for x in hasil["data"]
                        if x["Kondisi"] == "Segar"
                    )

                    jumlah_busuk = sum(
                        1 for x in hasil["data"]
                        if x["Kondisi"] == "Busuk"
                    )

                    c1, c2 = st.columns(2)

                    c1.metric("Buah Segar", jumlah_segar)
                    c2.metric("Buah Busuk", jumlah_busuk)

                    if jumlah_busuk > 0:
                        st.warning(
                            f"Terdeteksi {jumlah_busuk} buah busuk. "
                            "Sebaiknya dipisahkan dari buah yang masih segar."
                        )
                    else:
                        st.success("Semua buah yang terdeteksi masih segar.")

                    buffer_png = io.BytesIO()
                    hasil["gambar"].save(buffer_png, format="PNG")

                    st.download_button(
                        label="Download Gambar Hasil",
                        data=buffer_png.getvalue(),
                        file_name=f"hasil_{Path(file_gambar.name).stem}.png",
                        mime="image/png"
                    )

                else:
                    st.warning(
                        "Tidak ditemukan buah yang sesuai dengan kelas model."
                    )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# ==================================================
# RIWAYAT DETEKSI
# ==================================================

elif halaman == "Riwayat Deteksi":

    st.title("Riwayat Deteksi")
    st.markdown(
        '<p class="page-subtitle">Daftar hasil deteksi yang tersimpan secara permanen.</p>',
        unsafe_allow_html=True
    )

    riwayat = st.session_state.riwayat

    if riwayat:

        df = pd.DataFrame(riwayat)

        f1, f2 = st.columns(2)

        with f1:
            filter_buah = st.multiselect(
                "Filter jenis buah",
                options=sorted(df["Nama Buah"].unique())
            )

        with f2:
            filter_kondisi = st.multiselect(
                "Filter kondisi",
                options=["Segar", "Busuk"]
            )

        df_tampil = df
        if filter_buah:
            df_tampil = df_tampil[df_tampil["Nama Buah"].isin(filter_buah)]
        if filter_kondisi:
            df_tampil = df_tampil[df_tampil["Kondisi"].isin(filter_kondisi)]

        st.caption(f"Menampilkan {len(df_tampil)} dari {len(df)} data.")

        st.dataframe(
            df_tampil.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download Riwayat CSV",
            data=csv,
            file_name="riwayat_deteksi_buah.csv",
            mime="text/csv"
        )

        if st.button("Hapus Riwayat"):
            st.session_state.riwayat = []
            hapus_file_riwayat()
            st.session_state.hasil_deteksi = None
            st.session_state.nama_file = None
            st.rerun()

    else:
        st.info("Belum ada riwayat deteksi.")


# ==================================================
# TENTANG APLIKASI
# ==================================================

elif halaman == "Tentang Aplikasi":

    st.title("Tentang Aplikasi")
    st.markdown(
        '<p class="page-subtitle">Informasi mengenai sistem FruitScan.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <h3>🍃 FruitScan</h3>
        FruitScan merupakan aplikasi deteksi kesegaran buah berbasis web
        yang menggunakan metode computer vision dan model YOLOv8n.
        Sistem ini dirancang untuk mengenali buah dan mengklasifikasikan
        kondisinya menjadi segar atau busuk.
        <hr>
        <b>Fitur utama</b><br>
        <div class="chips">
            <span class="chip">📷 Deteksi dari gambar</span>
            <span class="chip">🟩 Kotak deteksi pada gambar</span>
            <span class="chip">🎚️ Atur batas confidence</span>
            <span class="chip">🕘 Riwayat tersimpan</span>
            <span class="chip">📥 Download hasil &amp; CSV</span>
        </div>
        <hr>
        <b>Kelas deteksi</b><br>
        • Apel Segar dan Busuk<br>
        • Pisang Segar dan Busuk<br>
        • Jeruk Segar dan Busuk<br>
        • Delima Segar dan Busuk
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cara Menggunakan</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="langkah"><div class="no">1</div>
            <div>Buka menu <b>Deteksi Gambar</b>.</div></div>
        <div class="langkah"><div class="no">2</div>
            <div>Unggah foto buah (JPG, PNG, WEBP, dan format lainnya).</div></div>
        <div class="langkah"><div class="no">3</div>
            <div>Atur <b>Batas Confidence</b> sesuai kebutuhan.</div></div>
        <div class="langkah"><div class="no">4</div>
            <div>Klik <b>Mulai Deteksi</b> dan lihat hasilnya.</div></div>
        <div class="langkah"><div class="no">5</div>
            <div>Lihat atau unduh hasilnya di menu <b>Riwayat Deteksi</b>.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Jenis Buah yang Dideteksi</div>',
                unsafe_allow_html=True)

    daftar_buah = {
        "Apel": '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M50 30 C40 22 16 26 16 54 C16 76 34 92 44 92 C48 92 49 90 50 90 C51 90 52 92 56 92 C66 92 84 76 84 54 C84 26 60 22 50 30 Z" fill="#D65B50"/><path d="M50 30 C60 22 84 26 84 54 C84 66 78 78 70 85 C78 72 78 44 50 30 Z" fill="#B8453B" opacity="0.55"/><ellipse cx="32" cy="48" rx="6" ry="12" fill="#FFFFFF" opacity="0.28" transform="rotate(15 32 48)"/><path d="M50 30 C50 22 52 16 57 11" stroke="#6B4A2B" stroke-width="4" fill="none" stroke-linecap="round"/><path d="M57 22 C62 9 77 9 82 14 C78 25 65 29 57 22 Z" fill="#3D9B62"/></svg>',
        "Pisang": '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M22 22 C20 56 44 86 82 80 C86 79 87 74 83 73 C58 70 44 50 40 26 C39 20 23 17 22 22 Z" fill="#F2C94C"/><path d="M40 26 C44 50 58 70 83 73 C86 74 85 78 82 80 C60 80 44 64 36 44 C34 36 36 28 40 26 Z" fill="#D9A92B" opacity="0.6"/><path d="M20 30 C18 14 38 10 41 30 Z" fill="#7A5A2E"/><circle cx="82.5" cy="77" r="4" fill="#5C4322"/></svg>',
        "Jeruk": '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="56" r="34" fill="#F2994A"/><path d="M50 22 A34 34 0 0 1 50 90 A26 34 0 0 0 50 22 Z" fill="#D97F2F" opacity="0.5"/><ellipse cx="36" cy="44" rx="11" ry="6" fill="#FFFFFF" opacity="0.3" transform="rotate(-35 36 44)"/><path d="M50 22 C50 18 51 15 53 12" stroke="#6B4A2B" stroke-width="3.5" fill="none" stroke-linecap="round"/><path d="M53 20 C58 8 72 7 78 12 C74 23 62 27 53 20 Z" fill="#3D9B62"/></svg>',
        "Delima": '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="57" r="33" fill="#9B2C3B"/><path d="M50 24 A33 33 0 0 1 50 90 A24 33 0 0 0 50 24 Z" fill="#7A1F2D" opacity="0.55"/><ellipse cx="36" cy="46" rx="10" ry="6" fill="#FFFFFF" opacity="0.28" transform="rotate(-35 36 46)"/><path d="M36 28 L38 14 L45 22 L50 11 L55 22 L62 14 L64 28 Z" fill="#7A1F2D"/><path d="M42 28 L50 22 L58 28 Z" fill="#B23A4B"/></svg>',
    }

    kolom_buah = st.columns(4)

    for kolom, (nama, gambar_svg) in zip(kolom_buah, daftar_buah.items()):
        with kolom:
            st.markdown(f"""
            <div class="buah-card">
                <div class="buah-img">{gambar_svg}</div>
                <div class="buah-nama">{nama}</div>
                <span class="badge segar">Segar</span>
                <span class="badge busuk">Busuk</span>
            </div>
            """, unsafe_allow_html=True)
