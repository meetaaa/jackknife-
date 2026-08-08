# Prediksi Pengeluaran Rumah Tangga (EXPEND) Susenas Kota Bogor 2023

Repositori ini berisi kode penelitian untuk memprediksi pengeluaran rumah tangga (EXPEND) berdasarkan data Susenas Kor Kota Bogor 2023 menggunakan tiga algoritme machine learning: **Random Forest**, **Gradient Boosting**, dan **Support Vector Regression (SVR)**. Seluruh model dilatih pada peubah respons yang telah ditransformasi dengan **logaritma natural**, kemudian prediksi dikembalikan secara otomatis ke skala asli (rupiah) menggunakan fungsi eksponensial melalui `TransformedTargetRegressor`.

Repositori terdiri atas dua modul analisis yang saling melengkapi:

| Modul | Fokus | Skrip utama |
|---|---|---|
| [`prediksi-titik/`](prediksi-titik/) | Prediksi titik + validasi silang 5 lipatan + analisis sisaan | `empirical_ml_only.py`, `simulation_ml_only.py` |
| [`selang-prediksi/`](selang-prediksi/) | Selang prediksi konformal Jackknife & Jackknife+ | `run_all.py` |

---

## 1. `prediksi-titik-ml/` — Prediksi Titik dengan CV 5 Lipatan dan Sisaan

Analisis prediksi titik EXPEND pada data empiris dan simulasi, tanpa selang prediksi.

**Fitur utama:**
- Ketiga model dilatih pada log(EXPEND); MAE, RMSE, MAPE, R², residual, dan absolute error dihitung pada skala rupiah.
- Validasi silang 5 lipatan pada data latih (`KFold(shuffle=True, random_state=42)`); RMSE (VS) dan R² (VS) pada tabel publikasi merupakan rata-rata dari lima lipatan, dengan metrik per lipatan disimpan terpisah agar dapat diaudit.
- Prediksi out-of-fold, sisaan validasi silang, serta prediksi dan sisaan data uji disimpan dalam CSV dan divisualisasikan (diagram pencar aktual vs dugaan dan diagram sisaan vs dugaan).
- Pada simulasi, random forest pembangkit respons juga dilatih pada log(EXPEND) dan respons simulasi dibatasi minimal 1 agar tetap positif.

**Cara menjalankan:**

```bash
cd prediksi-titik-ml

# Evaluasi data empiris
python empirical_ml_only.py
python empirical_ml_only.py --cv-folds 5
python empirical_ml_only.py --algorithms gradient_boosting random_forest svr
python empirical_ml_only.py --no-plots

# Evaluasi data simulasi (Gaussian copula semiparametrik)
python simulation_ml_only.py
```

**Keluaran utama:** `hasil_empiris_ml.csv`, `tabel_evaluasi_model_empiris.csv`, `hasil_empiris_ml_detail.csv`, `hasil_empiris_ml_cv_lipatan.csv`, `hasil_empiris_ml_cv_detail.csv`, `diagram_pencar_aktual_dugaan_data_uji.png`, `diagram_sisaan_dugaan_data_uji.png`, serta `hasil_simulasi_ml*.csv` untuk simulasi.

---

## 2. `selang-prediksi-konformal/` — Selang Prediksi Jackknife & Jackknife+

Analisis selang prediksi konformal untuk data empiris (Subbab 3.4.2) dan data simulasi (Subbab 3.4.1).

**Fitur utama:**
- Selang prediksi Jackknife dan Jackknife+ pada taraf nominal α = 0,10 (target cakupan 90%), dibangun pada skala respons asli (rupiah).
- Data simulasi dibangkitkan melalui random forest yang dilatih pada data empiris, dengan peubah penjelas dari Gaussian copula semiparametrik berdimensi 67.
- Sepuluh skenario rasio p/n (0,2 s.d. 2,0) dengan 100 pengulangan per skenario; evaluasi menggunakan PCR, AIW, WS, CT, dan MU.
- `RandomForestRegressor` dengan `n_estimators=50` dan `GradientBoostingRegressor` dengan `n_estimators=50`, `learning_rate=0.05`.

**Cara menjalankan:**

```bash
cd selang-prediksi-konformal
python run_all.py            # empiris + simulasi (konfigurasi penuh)
python empirical_analysis.py # hanya data empiris
python simulation_analysis.py # hanya simulasi
```

**Keluaran utama:** `hasil_empiris.csv` dan `hasil_simulasi.csv`.

> **Catatan:** prosedur leave-one-out untuk 10 skenario × 100 pengulangan sangat berat secara komputasi meskipun jumlah pohon/estimator telah dikurangi menjadi 50.

---

## Struktur Repositori

```
.
├── README.md
├── prediksi-titik-ml/
│   ├── empirical_ml_only.py        # Evaluasi empiris: test set + CV 5 lipatan + plot sisaan
│   ├── simulation_ml_only.py       # Evaluasi simulasi (tanpa selang prediksi)
│   ├── ml_common.py                # Model factory, run_models, cross_validate_model
│   ├── susenas_preprocessing.py    # Praproses & encoding data Susenas
│   └── README_transformasi_log.txt
└── selang-prediksi-konformal/
    ├── run_all.py                  # Menjalankan seluruh analisis empiris + simulasi
    ├── empirical_analysis.py       # Selang Jackknife/Jackknife+ pada data empiris
    ├── simulation_analysis.py      # Selang Jackknife/Jackknife+ pada data simulasi
    ├── conformal_common.py         # Implementasi jackknife, jackknife+, metrik selang
    ├── susenas_preprocessing.py    # Praproses & encoding data Susenas
    └── README.txt
```

## Data yang Diperlukan

Kedua modul membutuhkan dua berkas berikut di folder kerja masing-masing (tidak disertakan dalam repositori karena bersifat mikrodata):

- `bogor.csv` — mikrodata Susenas Kor Kota Bogor 2023
- `metadata.xlsx` — metadata peubah (domain nilai, tipe peubah)

Praproses (`susenas_preprocessing.py`) mencakup validasi domain, penanganan skip pattern, penghapusan baris non-respons (kode 8/9), encoding biner 0/1 dan one-hot, serta standardisasi peubah numerik yang hanya diterapkan pada pipeline SVR.

## Konfigurasi Model

| Model | Parameter |
|---|---|
| Random Forest | `n_estimators=50`, `n_jobs=-1`, `random_state=42` |
| Gradient Boosting | `n_estimators=50`, `learning_rate=0.05`, `random_state=42` |
| SVR | kernel RBF, `C=100`, `epsilon=0.15`, `gamma=0.008`, standardisasi peubah numerik |

Transformasi log respons diterapkan pada ketiga model melalui `TransformedTargetRegressor` (`func=log`, `inverse_func=exp`), sehingga `predict()` langsung menghasilkan nilai pada skala rupiah dan seluruh sisaan serta metrik dapat ditafsirkan langsung.

## Dependensi

```bash
pip install numpy pandas scikit-learn scipy matplotlib openpyxl
```

## Lisensi & Sitasi

Kode ini merupakan bagian dari penelitian akademik. Silakan sesuaikan bagian ini dengan lisensi dan format sitasi yang diinginkan.
