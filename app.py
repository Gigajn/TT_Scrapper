import streamlit as st
try:
    import pandas as pd
except Exception:
    class DummyDataFrame(list):
        def __init__(self, data=None):
            super().__init__(data or [])
        @property
        def empty(self): return len(self) == 0
    class DummyPD:
        DataFrame = DummyDataFrame
        def concat(self, *args, **kwargs): return DummyDataFrame()
    pd = DummyPD()
from urllib.parse import urlparse
from scraper import scrape_tiktok_comments, CaptchaDetectedError

st.set_page_config(
    page_title="TikTok Comment Scraper + ML",
    page_icon="🎵",
    layout="wide",
)

# ─── Global styles ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0d0f14; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #1a0533 0%, #0d1a3a 50%, #051a2e 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(105,57,226,0.35) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 40%;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(0,205,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(105,57,226,0.25);
        border: 1px solid rgba(105,57,226,0.5);
        border-radius: 20px; padding: 4px 12px;
        font-size: 12px; font-weight: 600; color: #c4a8ff;
        letter-spacing: 0.5px; margin-bottom: 14px;
    }
    .hero-title {
        font-size: 36px; font-weight: 800; margin: 0 0 8px 0;
        background: linear-gradient(90deg, #ffffff 0%, #c4a8ff 60%, #69c1ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.2;
    }
    .hero-sub { font-size: 15px; color: #8892a4; margin: 0 0 20px 0; }

    .step-grid { display: flex; gap: 12px; flex-wrap: wrap; }
    .step-item {
        display: flex; align-items: flex-start; gap: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px; padding: 10px 14px; flex: 1; min-width: 160px;
    }
    .step-num {
        background: linear-gradient(135deg, #6939e2, #3b82f6);
        color: #fff; width: 22px; height: 22px; border-radius: 50%;
        font-size: 11px; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; margin-top: 1px;
    }
    .step-text { font-size: 13px; color: #b0bac8; line-height: 1.4; }

    /* ── Cards ── */
    .input-card {
        background: #131720; border: 1px solid #1e2535;
        border-radius: 16px; padding: 24px 28px; margin-bottom: 20px;
    }
    .card-label {
        font-size: 11px; font-weight: 600; color: #6939e2;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 16px;
    }
    .log-card {
        background: #0a0c10; border: 1px solid #1e2535;
        border-radius: 14px; padding: 18px 20px; margin-bottom: 20px;
    }
    .log-title {
        font-size: 11px; font-weight: 600; color: #6939e2;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px;
    }

    /* ── URL progress item ── */
    .url-progress {
        background: rgba(105,57,226,0.08);
        border: 1px solid rgba(105,57,226,0.25);
        border-radius: 10px; padding: 10px 16px; margin-bottom: 8px;
        font-size: 13px; color: #c4a8ff;
    }
    .url-done   { border-color: rgba(16,185,129,0.4); color: #6ee7b7; background: rgba(16,185,129,0.06); }
    .url-error  { border-color: rgba(239,68,68,0.4);  color: #fca5a5; background: rgba(239,68,68,0.06); }
    .url-active { border-color: rgba(59,130,246,0.5); color: #93c5fd; background: rgba(59,130,246,0.08); animation: pulse 1.5s infinite; }

    @keyframes pulse {
        0%, 100% { border-color: rgba(59,130,246,0.5); }
        50%       { border-color: rgba(59,130,246,1);   }
    }

    /* ── Streamlit overrides ── */
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #8892a4 !important; font-size: 13px !important; font-weight: 500 !important;
    }
    .stTextInput input, .stNumberInput input {
        background: #0d0f14 !important; border: 1px solid #1e2535 !important;
        border-radius: 10px !important; color: #e8ecf2 !important; font-size: 14px !important;
    }
    .stTextArea textarea {
        background: #0d0f14 !important; border: 1px solid #1e2535 !important;
        border-radius: 10px !important; color: #e8ecf2 !important;
        font-size: 13px !important; font-family: 'Inter', monospace !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #6939e2 !important; box-shadow: 0 0 0 3px rgba(105,57,226,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6939e2 0%, #3b82f6 100%) !important;
        border: none !important; border-radius: 10px !important;
        color: #fff !important; font-weight: 700 !important;
        font-size: 15px !important; padding: 12px 0 !important;
        letter-spacing: 0.3px; transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88 !important; transform: translateY(-1px); }

    /* ── Stat cards ── */
    .stat-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
    .stat-card {
        background: linear-gradient(135deg, #131720 0%, #0f1420 100%);
        border: 1px solid #1e2535; border-radius: 14px;
        padding: 20px 24px; flex: 1; min-width: 140px; text-align: center;
    }
    .stat-value {
        font-size: 32px; font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #c4a8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.1; margin-bottom: 4px;
    }
    .stat-label { font-size: 12px; color: #6b7588; font-weight: 500; }

    /* ── ML section ── */
    .ml-card {
        background: linear-gradient(135deg, #0f1a2e 0%, #130d2a 100%);
        border: 1px solid rgba(105,57,226,0.3);
        border-radius: 16px; padding: 24px 28px; margin-top: 24px;
    }
    .ml-title {
        font-size: 18px; font-weight: 700; color: #c4a8ff; margin-bottom: 16px;
    }

    /* ── Label badge ── */
    .badge-judi    { background: rgba(239,68,68,0.15);   border: 1px solid rgba(239,68,68,0.4);   color: #fca5a5; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
    .badge-promo   { background: rgba(245,158,11,0.15);  border: 1px solid rgba(245,158,11,0.4);  color: #fcd34d; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }
    .badge-normal  { background: rgba(16,185,129,0.15);  border: 1px solid rgba(16,185,129,0.4);  color: #6ee7b7; border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600; }

    /* ── Results ── */
    .results-label {
        font-size: 11px; font-weight: 600; color: #6939e2;
        letter-spacing: 1px; text-transform: uppercase; margin: 20px 0 12px 0;
    }
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
    .stDownloadButton > button {
        background: rgba(105,57,226,0.12) !important;
        border: 1px solid rgba(105,57,226,0.4) !important;
        border-radius: 10px !important; color: #c4a8ff !important;
        font-weight: 600 !important; font-size: 14px !important;
        margin-top: 12px !important; transition: background 0.2s;
    }
    .stDownloadButton > button:hover { background: rgba(105,57,226,0.25) !important; }

    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🎵 &nbsp;Playwright · Naive Bayes · Multi-URL</div>
        <p class="hero-title">TikTok Comment Scraper + ML Classifier</p>
        <p class="hero-sub">Scrape hingga 50 link TikTok sekaligus — data langsung dianalisis Machine Learning untuk deteksi promosi judi & spam.</p>
        <div class="step-grid">
            <div class="step-item"><div class="step-num">1</div><div class="step-text">Tempel URL video (1 link per baris)</div></div>
            <div class="step-item"><div class="step-num">2</div><div class="step-text">Tentukan jumlah komentar per link</div></div>
            <div class="step-item"><div class="step-num">3</div><div class="step-text">Klik <b>Scrape & Analisis</b></div></div>
            <div class="step-item"><div class="step-num">4</div><div class="step-text">Selesaikan CAPTCHA jika muncul</div></div>
            <div class="step-item"><div class="step-num">5</div><div class="step-text">Lihat hasil ML &amp; download CSV</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Session State ────────────────────────────────────────────────────────────
if "accumulated_df" not in st.session_state:
    st.session_state.accumulated_df = pd.DataFrame()
if "ml_result" not in st.session_state:
    st.session_state.ml_result = None


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_valid_tiktok_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.netloc or "").lower()
    return host == "tiktok.com" or host.endswith(".tiktok.com")


def parse_urls(raw: str) -> list[str]:
    """Extract and deduplicate valid TikTok URLs from multi-line input."""
    urls = []
    seen = set()
    for line in raw.strip().splitlines():
        line = line.strip()
        if line and is_valid_tiktok_url(line) and line not in seen:
            urls.append(line)
            seen.add(line)
    return urls


# ─── Input Card ───────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="card-label">⚙ Konfigurasi Scraping Batch</div>', unsafe_allow_html=True)

urls_raw = st.text_area(
    "URL Video TikTok (1 link per baris, maksimal 50 link)",
    placeholder=(
        "https://www.tiktok.com/@username/video/1234567890\n"
        "https://www.tiktok.com/@username2/video/0987654321\n"
        "..."
    ),
    height=180,
)

col_count, col_ml, col_reset = st.columns([2, 2, 1])
with col_count:
    max_comments = st.number_input(
        "Komentar per link",
        min_value=10,
        max_value=1000,
        value=50,
        step=10,
    )
with col_ml:
    run_ml = st.checkbox("🤖 Jalankan ML Classifier otomatis setelah scraping", value=True)
with col_reset:
    if st.button("🗑 Reset Data", use_container_width=True):
        st.session_state.accumulated_df = pd.DataFrame()
        st.session_state.ml_result = None
        st.rerun()

scrape_btn = st.button("🚀  Scrape & Analisis", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# ─── Akumulasi info ───────────────────────────────────────────────────────────
if not st.session_state.accumulated_df.empty:
    acc_df = st.session_state.accumulated_df
    st.info(
        f"📦 Data terakumulasi dari sesi sebelumnya: **{len(acc_df)} komentar** dari "
        f"**{acc_df['source_url'].nunique() if 'source_url' in acc_df.columns else '?'} link**. "
        "Klik **Scrape** lagi untuk menambah atau **Reset Data** untuk mulai ulang."
    )

# ─── Scraping Flow ────────────────────────────────────────────────────────────
if scrape_btn:
    valid_urls = parse_urls(urls_raw)

    if not valid_urls:
        st.error("⚠️  Tidak ada URL TikTok valid yang ditemukan. Periksa format URL.")
    elif len(valid_urls) > 50:
        st.error(f"⚠️  Maksimal 50 link per sesi. Kamu memasukkan {len(valid_urls)} link.")
    else:
        st.markdown(f"### 🔗 Memproses **{len(valid_urls)} link**")

        # Live log area
        st.markdown('<div class="log-card"><div class="log-title">📡 Log Proses</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # Per-URL status container
        url_status_container = st.container()

        all_comments: list[dict] = []
        url_results: list[dict] = []  # {url, count, status}

        logs: list[str] = []

        def push_log(message: str) -> None:
            logs.append(message)
            tail = logs[-30:]
            log_placeholder.code("\n".join(tail), language="text")

        # ── Per-URL scraping loop ──────────────────────────────────────────────
        for idx, url in enumerate(valid_urls):
            short_url = url if len(url) <= 60 else url[:57] + "…"
            push_log(f"\n[{idx+1}/{len(valid_urls)}] Mulai: {short_url}")

            result_entry = {"url": short_url, "count": 0, "status": "running"}
            url_results.append(result_entry)

            # Render status panel
            with url_status_container:
                for i, r in enumerate(url_results):
                    css = "url-done" if r["status"] == "done" else \
                          "url-error" if r["status"] == "error" else "url-active"
                    icon = "✅" if r["status"] == "done" else \
                           "❌" if r["status"] == "error" else "⏳"
                    st.markdown(
                        f'<div class="url-progress {css}">'
                        f'{icon} <b>[{i+1}/{len(valid_urls)}]</b> {r["url"]} '
                        f'— {r["count"]} komentar</div>',
                        unsafe_allow_html=True,
                    )

            try:
                with st.spinner(f"Scraping link {idx+1}/{len(valid_urls)}…"):
                    comments = scrape_tiktok_comments(url, max_comments, progress_callback=push_log)

                for c in comments:
                    c["source_url"] = url

                all_comments.extend(comments)
                result_entry["count"] = len(comments)
                result_entry["status"] = "done"
                push_log(f"[{idx+1}/{len(valid_urls)}] Selesai ✓ — {len(comments)} komentar")

            except CaptchaDetectedError as exc:
                result_entry["status"] = "error"
                push_log(f"[{idx+1}/{len(valid_urls)}] CAPTCHA gagal diselesaikan: {exc}")
                st.warning(f"🔒 Link {idx+1}: CAPTCHA terdeteksi. Lanjut ke link berikutnya.", icon="🔒")

            except Exception as exc:
                result_entry["status"] = "error"
                push_log(f"[{idx+1}/{len(valid_urls)}] Error: {exc}")
                st.warning(f"⚠️ Link {idx+1}: {exc}", icon="⚠️")

        push_log(f"\n=== Total baru: {len(all_comments)} komentar dari {len(valid_urls)} link ===")

        # ── Merge dengan data akumulasi ───────────────────────────────────────
        if all_comments:
            new_df = pd.DataFrame(all_comments)
            if not st.session_state.accumulated_df.empty:
                combined = pd.concat(
                    [st.session_state.accumulated_df, new_df], ignore_index=True
                )
                # Dedup berdasarkan account + comment
                combined = combined.drop_duplicates(
                    subset=["account", "comment"]
                ).reset_index(drop=True)
            else:
                combined = new_df

            st.session_state.accumulated_df = combined
            push_log(f"Total akumulasi: {len(combined)} komentar unik")

        # ── ML Pipeline ───────────────────────────────────────────────────────
        if run_ml and not st.session_state.accumulated_df.empty:
            push_log("\n[ML] Memulai pipeline Machine Learning…")
            try:
                from ml_pipeline import run_ml_pipeline
                ml_result = run_ml_pipeline(
                    st.session_state.accumulated_df,
                    log_callback=push_log,
                )
                st.session_state.ml_result = ml_result
                # Simpan df hasil ML kembali ke accumulated (agar kolom label ada)
                st.session_state.accumulated_df = ml_result["df_result"]
            except Exception as exc:
                push_log(f"[ML] Error: {exc}")
                st.error(f"❌ ML pipeline gagal: {exc}")


# ─── Tampilkan Hasil ──────────────────────────────────────────────────────────
df_display = st.session_state.accumulated_df
ml = st.session_state.ml_result

if not df_display.empty:
    total = len(df_display)
    unique_acc = df_display["account"].nunique() if "account" in df_display.columns else 0
    unique_links = df_display["source_url"].nunique() if "source_url" in df_display.columns else 0

    st.success(f"✅ Total data: **{total:,} komentar** dari **{unique_acc:,} akun** di **{unique_links} link**")

    # ── Stat Cards ────────────────────────────────────────────────────────────
    label_counts = ml["label_counts"] if ml else {}
    judi_count  = label_counts.get("Promosi Judi Online", 0)
    promo_count = label_counts.get("Promosi Non Judi Online", 0)
    norm_count  = label_counts.get("Komentar Normal", 0)
    acc_pct     = f"{ml['accuracy'] * 100:.1f}%" if ml else "—"

    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{total:,}</div>
                <div class="stat-label">Total Komentar</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{unique_links}</div>
                <div class="stat-label">Link Diproses</div>
            </div>
            <div class="stat-card" style="border-color: rgba(239,68,68,0.4);">
                <div class="stat-value" style="background: linear-gradient(135deg,#ef4444,#f97316); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;">{judi_count:,}</div>
                <div class="stat-label">Promosi Judi Online</div>
            </div>
            <div class="stat-card" style="border-color: rgba(245,158,11,0.4);">
                <div class="stat-value" style="background: linear-gradient(135deg,#f59e0b,#fbbf24); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;">{promo_count:,}</div>
                <div class="stat-label">Promosi Non-Judi</div>
            </div>
            <div class="stat-card" style="border-color: rgba(16,185,129,0.4);">
                <div class="stat-value" style="background: linear-gradient(135deg,#10b981,#34d399); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;">{norm_count:,}</div>
                <div class="stat-label">Komentar Normal</div>
            </div>
            <div class="stat-card" style="border-color: rgba(105,57,226,0.4);">
                <div class="stat-value">{acc_pct}</div>
                <div class="stat-label">Akurasi ML</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── ML Visualisasi & Akurasi ──────────────────────────────────────────────
    if ml:
        st.markdown('<div class="ml-card">', unsafe_allow_html=True)
        st.markdown('<div class="ml-title">🤖 Hasil Machine Learning — Naive Bayes Classifier</div>', unsafe_allow_html=True)

        # ── Kasus: semua komentar 1 kelas saja ───────────────────────────────
        if ml.get("single_class_warning"):
            satu_kelas = ml["single_class_warning"]
            st.markdown(
                f"""
                <div style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.4);
                            border-radius:12px;padding:18px 22px;margin-bottom:12px;">
                    <div style="font-size:15px;font-weight:700;color:#fcd34d;margin-bottom:8px;">
                        ⚠️  Model tidak bisa dilatih — Semua komentar berlabel sama
                    </div>
                    <div style="font-size:13px;color:#b0bac8;line-height:1.7;">
                        Dari <b>{len(ml['df_result'])} komentar</b>, seluruhnya terlabeli
                        <b style="color:#fbbf24;">"{satu_kelas}"</b>.<br>
                        Model Naive Bayes butuh minimal <b>2 kelas</b> berbeda untuk dilatih.<br><br>
                        <b style="color:#e8ecf2;">Solusi:</b>
                        <ul style="margin:6px 0 0 16px;color:#8892a4;">
                            <li>Tambahkan URL video lain yang kemungkinan ada komentar judi/promosi</li>
                            <li>Naikkan jumlah komentar per video</li>
                            <li>Data yang sudah ada tetap terakumulasi — klik Scrape lagi untuk menambah</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            if not ml["sufficient_data"]:
                st.warning("⚠️ Data masih sedikit (<20 komentar unik). Tambahkan lebih banyak data untuk hasil ML yang lebih akurat.")

            # ── Info split data ───────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
                    <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
                                border-radius:10px;padding:10px 18px;font-size:13px;color:#93c5fd;">
                        🏋️ <b>Data Pelatihan (80%)</b> &nbsp;—&nbsp; {ml['n_train']} sampel
                    </div>
                    <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
                                border-radius:10px;padding:10px 18px;font-size:13px;color:#6ee7b7;">
                        🧪 <b>Data Uji (20%)</b> &nbsp;—&nbsp; {ml['n_test']} sampel
                    </div>
                    <div style="background:rgba(105,57,226,0.15);border:1px solid rgba(105,57,226,0.5);
                                border-radius:10px;padding:10px 18px;color:#c4a8ff;
                                font-size:15px;font-weight:700;">
                        🎯 Akurasi Keseluruhan: {ml['accuracy']*100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Tabel Akurasi Per Kelas ───────────────────────────────────────
            st.markdown("**📊 Akurasi Per Kelas (Precision · Recall · F1-Score)**")
            df_metrics = pd.DataFrame(ml["per_class_metrics"])

            def _color_val(val):
                """Warnai cell berdasarkan nilai 0-1."""
                if isinstance(val, float):
                    if val >= 0.8:
                        color = "rgba(16,185,129,0.25)"
                    elif val >= 0.5:
                        color = "rgba(245,158,11,0.25)"
                    else:
                        color = "rgba(239,68,68,0.2)"
                    return f"background-color: {color}"
                return ""

            st.dataframe(
                df_metrics.style.map(_color_val, subset=["Precision", "Recall", "F1-Score"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── Visualisasi ───────────────────────────────────────────────────
            col_cm, col_dist = st.columns(2)
            with col_cm:
                st.pyplot(ml["fig_cm"])
            with col_dist:
                st.pyplot(ml["fig_dist"])

            with st.expander("📋 Classification Report Lengkap"):
                st.code(ml["report"], language="text")

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Download Data Pelatihan & Model ───────────────────────────────
            st.markdown('<div class="results-label">⬇️ Download Data & Model</div>', unsafe_allow_html=True)

            model_size_kb = len(ml["model_bytes"]) / 1024
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg,rgba(105,57,226,0.2),rgba(59,130,246,0.15));
                            border: 1px solid rgba(105,57,226,0.5); border-radius: 12px;
                            padding: 14px 20px; margin-bottom: 14px; display:flex;
                            align-items:center; gap:14px;">
                    <div style="font-size:28px;">🧠</div>
                    <div>
                        <div style="font-size:14px;font-weight:700;color:#c4a8ff;">
                            Model Terlatih — Naive Bayes Classifier
                        </div>
                        <div style="font-size:12px;color:#8892a4;margin-top:3px;">
                            File: <code style="color:#93c5fd;">{ml['model_filename']}</code>
                            &nbsp;·&nbsp; Ukuran: <b>{model_size_kb:.1f} KB</b>
                            &nbsp;·&nbsp; Akurasi: <b>{ml['accuracy']*100:.2f}%</b>
                            &nbsp;·&nbsp; Kelas: {', '.join(ml['model'].classes_.tolist())}
                        </div>
                        <div style="font-size:11px;color:#6b7588;margin-top:4px;">
                            Berisi: MultinomialNB + TF-IDF Vectorizer. Load ulang dengan <code>pickle.load()</code>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                f"🧠  Download Model (.pkl)  —  {model_size_kb:.1f} KB",
                ml["model_bytes"],
                ml["model_filename"],
                "application/octet-stream",
                use_container_width=True,
                key="dl_model_pkl",
                help=(
                    "File .pkl berisi model Naive Bayes + TF-IDF Vectorizer yang sudah dilatih. "
                    "Untuk memakainya: import pickle; bundle = pickle.load(open('file.pkl','rb')); "
                    "pred = bundle['model'].predict(bundle['tfidf'].transform(['teks komentar']))"
                ),
            )

            st.markdown("<br>", unsafe_allow_html=True)
            dl_train, dl_test, dl_report = st.columns(3)

            with dl_train:
                csv_train = ml["df_train"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📗 Data Pelatihan ({ml['n_train']} baris)",
                    csv_train,
                    "data_pelatihan_train.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Dataset yang digunakan untuk melatih model (80%). Berisi teks asli, teks bersih, dan label.",
                )

            with dl_test:
                csv_test = ml["df_test_eval"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📘 Data Evaluasi ({ml['n_test']} baris)",
                    csv_test,
                    "data_evaluasi_test.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Dataset uji (20%) dengan kolom label_aktual, label_prediksi_uji, dan kolom 'benar' (True/False).",
                )

            with dl_report:
                import datetime
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_txt = (
                    f"=== LAPORAN MODEL NAIVE BAYES ===\n"
                    f"Tanggal   : {now_str}\n"
                    f"Total Data: {len(df_display)} komentar\n"
                    f"Train     : {ml['n_train']} | Test: {ml['n_test']}\n"
                    f"Akurasi   : {ml['accuracy']*100:.2f}%\n\n"
                    f"--- Per Kelas ---\n"
                    f"{df_metrics.to_string(index=False)}\n\n"
                    f"--- Classification Report ---\n"
                    f"{ml['report']}\n"
                )
                st.download_button(
                    "📄 Model Report (TXT)",
                    report_txt.encode("utf-8"),
                    "model_report.txt",
                    "text/plain",
                    use_container_width=True,
                    help="Ringkasan lengkap hasil model dalam format teks.",
                )

    # ── Tabel Hasil ───────────────────────────────────────────────────────────
    st.markdown('<div class="results-label">📋 Tabel Komentar Terklasifikasi</div>', unsafe_allow_html=True)

    # Filter per kategori
    if "label_prediksi" in df_display.columns:
        filter_options = ["Semua"] + sorted(df_display["label_prediksi"].unique().tolist())
        filter_choice = st.selectbox("Filter kategori:", filter_options)
        df_show = df_display if filter_choice == "Semua" else df_display[df_display["label_prediksi"] == filter_choice]
    else:
        df_show = df_display

    # Kolom yang ditampilkan
    cols_to_show = [c for c in ["account", "comment", "source_url", "label_prediksi", "label_awal", "clean_text", "split"] if c in df_show.columns]
    col_cfg = {
        "account":        st.column_config.TextColumn("👤 Akun",          width="medium"),
        "comment":        st.column_config.TextColumn("💬 Komentar",       width="large"),
        "source_url":     st.column_config.TextColumn("🔗 Sumber URL",     width="medium"),
        "label_prediksi": st.column_config.TextColumn("🏷 Label ML",       width="medium"),
        "label_awal":     st.column_config.TextColumn("🏷 Label Awal",     width="medium"),
        "clean_text":     st.column_config.TextColumn("🧹 Teks Bersih",    width="large"),
        "split":          st.column_config.TextColumn("🔀 Split",          width="small"),
    }

    st.dataframe(
        df_show[cols_to_show],
        use_container_width=True,
        hide_index=True,
        column_config={k: v for k, v in col_cfg.items() if k in cols_to_show},
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="results-label">⬇️ Download Data Komentar</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_all = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📦 Download Semua Data Komentar (CSV)",
            csv_all,
            "tiktok_comments_all.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl2:
        if "label_prediksi" in df_display.columns:
            df_judi = df_display[df_display["label_prediksi"] == "Promosi Judi Online"]
            csv_judi = df_judi.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"🚨 Download Komentar Judi Saja ({len(df_judi)} baris)",
                csv_judi,
                "tiktok_judi_online.csv",
                "text/csv",
                use_container_width=True,
            )