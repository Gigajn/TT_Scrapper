# -*- coding: utf-8 -*-
"""
ml_pipeline.py
==============
Modul Machine Learning untuk klasifikasi komentar TikTok.
Menerima pandas DataFrame langsung dari memori (tidak perlu file CSV).

Kategori Output:
  - Promosi Judi Online
  - Promosi Non Judi Online
  - Komentar Normal
"""

import re
import pickle
import io
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (aman untuk Streamlit)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ─── Lazy-load Sastrawi (opsional) ────────────────────────────────────────────
_stemmer = None
_stopword_remover = None

def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
        except ImportError:
            _stemmer = False   # Tandai tidak tersedia
    return _stemmer if _stemmer else None


def _get_stopword_remover():
    global _stopword_remover
    if _stopword_remover is None:
        try:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            _stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        except ImportError:
            _stopword_remover = False
    return _stopword_remover if _stopword_remover else None


# ─── Kamus Normalisasi ────────────────────────────────────────────────────────
NORM_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "yg": "yang", "dm": "pesan",
    "bgt": "banget", "gacorr": "gacor", "jp": "jackpot",
    "sy": "saya", "kk": "kakak", "broo": "bro",
}

# ─── Kamus Kata Kunci ─────────────────────────────────────────────────────────
JUDI_UMUM = [
    'slot', 'judi', 'kasino', 'togel', 'bet', 'depo', 'deposit',
    'wd', 'withdraw', 'maxwin', 'scatter', 'scater', 'gacor',
    'gcr', 'jp', 'cuan', 'pola', 'kakek',
]
JUDI_KERAS = [
    '303', '138', '888', '777', '88', '69',
    'rtp', 'zeus', 'mahjong', 'olympus', 'pragmatic',
    'main di', 'daftar di', 'link di bio', 'link dibio',
    'klik link', 'tele', 'telegram', 'admin slot',
]
PROMO_NON_JUDI = [
    'jual', 'order', 'murah', 'promo', 'diskon', 'open po',
    'follow', 'subscribe', 'keranjang kuning', 'cek profil',
]
PENYANGKAL = [
    'jangan main', 'rugi', 'awas', 'bahaya', 'haram',
    'dampak buruk', 'berhenti',
]


# ─── Pre-labeling ─────────────────────────────────────────────────────────────

def pre_labeling(text: str) -> str:
    text = str(text).lower()
    score = 0

    for word in JUDI_UMUM:
        if word in text:
            score += 3
    for word in JUDI_KERAS:
        if word in text:
            score += 6
    for word in PROMO_NON_JUDI:
        if word in text:
            score += 2
    for word in PENYANGKAL:
        if word in text:
            score -= 4

    if score >= 6:
        return "Promosi Judi Online"
    elif 2 <= score <= 5:
        if any(j in text for j in JUDI_UMUM) and not any(p in text for p in PROMO_NON_JUDI):
            return "Promosi Judi Online"
        return "Promosi Non Judi Online"
    else:
        return "Komentar Normal"


# ─── Text Preprocessing ───────────────────────────────────────────────────────

def text_preprocessing(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalisasi kata
    words = text.split()
    words = [NORM_DICT.get(w, w) for w in words]
    text = " ".join(words)

    # Sastrawi (jika tersedia)
    sw = _get_stopword_remover()
    if sw:
        text = sw.remove(text)

    st = _get_stemmer()
    if st:
        text = st.stem(text)

    return text


# ─── Pipeline Utama ───────────────────────────────────────────────────────────

def run_ml_pipeline(df: pd.DataFrame, log_callback=None) -> dict:
    """
    Jalankan pipeline ML lengkap pada DataFrame komentar.

    Parameters
    ----------
    df : pd.DataFrame
        Harus memiliki kolom 'comment'. Kolom 'account' bersifat opsional.
    log_callback : callable | None
        Fungsi (str) -> None untuk live logging ke UI.

    Returns
    -------
    dict dengan keys:
        df_result        : DataFrame lengkap dengan kolom label_awal, clean_text, label_prediksi, split
        df_train         : DataFrame khusus data pelatihan (80%)
        df_test_eval     : DataFrame data uji dengan kolom label_aktual & label_prediksi_uji
        accuracy         : float (0–1) — akurasi pada data uji
        report           : str  classification_report
        per_class_metrics: list[dict] — precision/recall/f1 per kelas
        fig_cm           : matplotlib Figure  (Confusion Matrix)
        fig_dist         : matplotlib Figure  (Distribusi Kelas)
        model            : MultinomialNB yang sudah dilatih
        tfidf            : TfidfVectorizer yang sudah difit
        label_counts     : dict {label: count} distribusi prediksi seluruh data
        sufficient_data  : bool (apakah data cukup untuk train/test split)
        n_train          : int — jumlah sampel pelatihan
        n_test           : int — jumlah sampel uji
    """

    def _log(msg: str):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    # ── 1. Validasi kolom ─────────────────────────────────────────────────────
    if "comment" not in df.columns:
        raise ValueError("DataFrame harus memiliki kolom 'comment'.")

    df = df.copy()
    _log(f"[ML] Data diterima: {len(df)} baris komentar")

    # ── 2. Pre-labeling ───────────────────────────────────────────────────────
    _log("[ML] Menjalankan pre-labeling…")
    df["label_awal"] = df["comment"].apply(pre_labeling)

    label_dist = df["label_awal"].value_counts().to_dict()
    _log(f"[ML] Distribusi label: {label_dist}")

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    _log("[ML] Preprocessing teks… (stemming & stopword removal)")
    df["clean_text"] = df["comment"].apply(text_preprocessing)

    # Bersihkan baris kosong / duplikat setelah preprocessing
    df = df.replace("", np.nan)
    df = df.dropna(subset=["clean_text"])
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    _log(f"[ML] Setelah cleaning: {len(df)} baris unik")

    # Validasi jumlah data
    if len(df) < 4:
        raise ValueError(
            "Data terlalu sedikit untuk melatih model (minimum 4 komentar unik). "
            "Tambahkan lebih banyak komentar."
        )

    # Validasi jumlah kelas — harus >= 2 agar model bisa dilatih
    n_classes = df["label_awal"].nunique()
    if n_classes < 2:
        satu_kelas = df["label_awal"].iloc[0]
        _log(
            f"[ML] ⚠️  Semua {len(df)} komentar masuk kelas '{satu_kelas}'. "
            "Model tidak bisa dilatih dengan 1 kelas saja. "
            "Coba scrape video lain yang mengandung komentar judi/promosi."
        )
        # Kembalikan hasil parsial (label saja, tanpa model)
        df["label_prediksi"] = satu_kelas
        df["clean_text"]     = df["comment"].apply(text_preprocessing)
        df["split"]          = "none"
        return {
            "df_result":         df,
            "df_train":          pd.DataFrame(),
            "df_test_eval":      pd.DataFrame(),
            "accuracy":          None,
            "report":            f"Model tidak bisa dilatih: semua data berlabel '{satu_kelas}'.",
            "per_class_metrics": [],
            "fig_cm":            None,
            "fig_dist":          None,
            "model":             None,
            "tfidf":             None,
            "label_counts":      {satu_kelas: len(df)},
            "sufficient_data":   False,
            "n_train":           0,
            "n_test":            0,
            "model_bytes":       None,
            "model_filename":    None,
            "single_class_warning": satu_kelas,
        }

    # ── 4. TF-IDF Vectorization ───────────────────────────────────────────────
    _log("[ML] Membangun matriks TF-IDF…")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
    )
    X = tfidf.fit_transform(df["clean_text"])
    y = df["label_awal"]
    _log(f"[ML] Matriks TF-IDF: {X.shape[0]} × {X.shape[1]}")

    # ── 5. Train / Test Split ─────────────────────────────────────────────────
    sufficient_data = len(df) >= 20
    indices = np.arange(len(df))
    try:
        idx_train, idx_test, X_train, X_test, y_train, y_test = train_test_split(
            indices, X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        _log("[ML] Peringatan: data terlalu sedikit untuk stratify, pakai random split")
        idx_train, idx_test, X_train, X_test, y_train, y_test = train_test_split(
            indices, X, y, test_size=0.2, random_state=42
        )

    _log(f"[ML] Data latih: {X_train.shape[0]} | Data uji: {X_test.shape[0]}")

    # ── 6. Oversampling (jika imbalanced-learn tersedia & data balanced perlu) ──
    try:
        from imblearn.over_sampling import RandomOverSampler
        if len(np.unique(y_train)) >= 2:   # Hanya jika ada ≥ 2 kelas di train set
            _log("[ML] Melakukan RandomOverSampler pada data latih…")
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            _log(f"[ML] Setelah oversampling: {X_train.shape[0]} sampel latih")
        else:
            _log("[ML] Skip oversampling: hanya 1 kelas di train set")
    except ImportError:
        _log("[ML] imbalanced-learn tidak tersedia, skip oversampling")

    # ── 7. Training Model ─────────────────────────────────────────────────────
    _log("[ML] Melatih model Naive Bayes…")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ── 8. Evaluasi ───────────────────────────────────────────────────────────
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, zero_division=0)
    _log(f"[ML] Akurasi: {acc * 100:.2f}%")

    # Per-class metrics (precision, recall, f1)
    classes = model.classes_.tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred_test, labels=classes, zero_division=0
    )
    per_class_metrics = [
        {
            "Kelas": cls,
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1-Score": round(float(f), 4),
            "Support (Uji)": int(s),
        }
        for cls, p, r, f, s in zip(classes, prec, rec, f1, sup)
    ]

    # ── 9. Prediksi SELURUH dataset & tandai split ────────────────────────────
    _log("[ML] Memprediksi seluruh dataset…")
    X_all = tfidf.transform(df["clean_text"])
    df["label_prediksi"] = model.predict(X_all)

    # Tandai setiap baris apakah masuk train atau test
    df["split"] = "train"
    df.loc[idx_test, "split"] = "test"

    label_counts = df["label_prediksi"].value_counts().to_dict()
    _log(f"[ML] Hasil prediksi: {label_counts}")

    # ── DataFrame train & test untuk download ────────────────────────────────
    train_cols = [c for c in ["account", "comment", "clean_text", "label_awal", "source_url"] if c in df.columns]
    df_train = df.loc[idx_train, train_cols].copy().reset_index(drop=True)
    df_train["split"] = "train"

    # Test set: sertakan prediksi model vs label aktual
    test_cols = [c for c in ["account", "comment", "clean_text", "source_url"] if c in df.columns]
    df_test_eval = df.loc[idx_test, test_cols].copy().reset_index(drop=True)
    df_test_eval["label_aktual"]        = y_test.values
    df_test_eval["label_prediksi_uji"]  = y_pred_test
    df_test_eval["benar"]               = (df_test_eval["label_aktual"] == df_test_eval["label_prediksi_uji"])

    # ── 10. Visualisasi ───────────────────────────────────────────────────────
    _log("[ML] Membuat visualisasi…")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax_cm,
    )
    ax_cm.set_title("Confusion Matrix — Klasifikasi Komentar TikTok", fontsize=14, pad=12)
    ax_cm.set_xlabel("Label Prediksi Model")
    ax_cm.set_ylabel("Label Aktual (Ground Truth)")
    plt.tight_layout()

    # Distribusi Kelas Final
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    counts = df["label_prediksi"].value_counts()
    colors = ["#6939e2", "#3b82f6", "#10b981"]
    ax_dist.bar(counts.index, counts.values, color=colors[: len(counts)])
    ax_dist.set_title("Distribusi Final Kelas Komentar", fontsize=14, pad=12)
    ax_dist.set_xlabel("Kategori Kelas")
    ax_dist.set_ylabel("Jumlah Komentar")
    for bar, val in zip(ax_dist.patches, counts.values):
        ax_dist.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    plt.tight_layout()

    # ── 11. Serialize model ke bytes (untuk download) ────────────────────────
    _log("[ML] Menyimpan model ke bytes...")
    model_bundle = {
        "model":          model,
        "tfidf":          tfidf,
        "classes":        model.classes_.tolist(),
        "accuracy":       acc,
        "n_train":        len(idx_train),
        "n_test":         len(idx_test),
        "trained_at":     datetime.datetime.now().isoformat(),
        "norm_dict":      NORM_DICT,
        "label_counts":   label_counts,
    }
    buf = io.BytesIO()
    pickle.dump(model_bundle, buf)
    model_bytes = buf.getvalue()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"nb_tiktok_model_{ts}.pkl"

    _log(f"[ML] Pipeline selesai ✓  |  Ukuran model: {len(model_bytes)/1024:.1f} KB")

    return {
        "df_result":         df,
        "df_train":          df_train,
        "df_test_eval":      df_test_eval,
        "accuracy":          acc,
        "report":            report,
        "per_class_metrics": per_class_metrics,
        "fig_cm":            fig_cm,
        "fig_dist":          fig_dist,
        "model":             model,
        "tfidf":             tfidf,
        "label_counts":      label_counts,
        "sufficient_data":   sufficient_data,
        "n_train":           len(idx_train),
        "n_test":            len(idx_test),
        "model_bytes":       model_bytes,
        "model_filename":    model_filename,
    }


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = pd.DataFrame({
        "account": ["user1", "user2", "user3", "user4", "user5", "user6"],
        "comment": [
            "Bang bagi link gacor depo 10k auto wd maxwin",
            "Baju koko murah cek keranjang kuning diskon",
            "Wah videonya sangat mengedukasi, terima kasih",
            "Awas jangan main slot nanti rugi besar",
            "Cek bio link daftar slot gampang jp zeus olympus",
            "Open PO sepatu murah meriah mampir kak",
        ],
    })
    result = run_ml_pipeline(sample, log_callback=print)
    print("\n=== LAPORAN ===")
    print(result["report"])
    print(f"Akurasi: {result['accuracy'] * 100:.2f}%")
    print("\nLabel Counts:", result["label_counts"])
    plt.show()
