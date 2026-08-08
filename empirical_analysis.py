"""
empirical_analysis.py
=====================
Analisis data empiris (Subbab 3.4.2): prediksi pengeluaran rumah tangga
(EXPEND) Susenas Kota Bogor 2023 dengan selang prediksi jackknife &
jackknife+ untuk random forest, gradient boosting, dan SVR.

Seluruh model menggunakan transformasi logaritma natural pada peubah respons;
prediksi dan selang dikembalikan ke skala rupiah (lihat conformal_common.py).

Jalankan:  python empirical_analysis.py
Keluaran:  hasil_empiris.csv  (+ ringkasan di layar)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from susenas_preprocessing import load_metadata, clean_raw, SusenasEncoder, RESPONSE
from conformal_common import run_all_combinations, ALPHA

# ---------------------- Konfigurasi ----------------------
DATA_PATH = "bogor.csv"
META_PATH = "metadata.xlsx"
TEST_SIZE = 0.20
SEED = 42
MAX_TRAIN = None   # selalu gunakan seluruh data latih
# ---------------------------------------------------------


def eksplorasi(df):
    """Statistik deskriptif ringkas peubah respons & numerik (Subbab 4.1.1)."""
    y = df[RESPONSE]
    print("\n=== Eksplorasi data ===")
    print(f"n = {len(df)}")
    print(f"{RESPONSE}: min={y.min():,.0f}  median={y.median():,.0f}  "
          f"mean={y.mean():,.0f}  max={y.max():,.0f}  skew={y.skew():.2f}")
    print(f"  skew log({RESPONSE}) = {np.log(y).skew():.2f}")


def main(algorithms=None):
    df = pd.read_csv(DATA_PATH)
    meta = load_metadata(META_PATH)

    eksplorasi(df)                       # 4.1.1

    df = clean_raw(df, meta)             # 4.1.2 validasi + 4.1.4 hapus baris
    y = df[RESPONSE].values.astype(float)

    encoder = SusenasEncoder(meta)       # 4.1.5 encoding (+ 4.1.3 skip pattern via kategori)
    X = encoder.fit_transform(df)
    print(f"[encoding] matriks fitur: {X.shape[0]} x {X.shape[1]} "
          f"(numerik pada indeks {encoder.numeric_idx_}; kolom dibuang: {encoder.drop_cols_})")

    # Pembagian 80:20 (langkah 2 Subbab 3.4.2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    print(f"[split] latih={len(y_tr)}  uji={len(y_te)}")

    # Konstruksi selang prediksi & evaluasi (langkah 3-5).
    # RF, GB, dan SVR dilatih pada log(y); keluaran kembali pada skala asli.
    print(f"\n=== Konstruksi selang prediksi dengan transformasi log respons (alpha={ALPHA}, target cakupan {100*(1-ALPHA):.0f}%) ===")
    rows, detail = run_all_combinations(
        encoder.numeric_idx_, X_tr, y_tr, X_te, y_te, seed=SEED, alpha=ALPHA,
        algorithms=algorithms, collect_detail=True,
    )
    res = pd.DataFrame(rows)[["algoritme", "metode", "PCR", "AIW", "WS", "CT", "MU"]]
    res = res.sort_values(["algoritme", "metode"]).reset_index(drop=True)

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n=== Hasil data empiris ===")
    print(res.to_string(index=False))
    suffix = "" if algorithms is None else "_" + "_".join(algorithms)
    out = f"hasil_empiris{suffix}.csv"
    res.to_csv(out, index=False)

    # Tabel detail per amatan uji: y_true, y_pred, lower, upper, method, model
    detail_df = pd.DataFrame(detail)[
        ["model", "method", "y_true", "y_pred", "lower", "upper"]
    ]
    out_detail = f"hasil_empiris_detail{suffix}.csv"
    detail_df.to_csv(out_detail, index=False)
    print(f"\n=== Contoh tabel detail (5 baris pertama, total {len(detail_df):,}) ===")
    print(detail_df.head().to_string(index=False))
    print(f"\nDisimpan: {out}")
    print(f"Disimpan: {out_detail}")


if __name__ == "__main__":
    main()
