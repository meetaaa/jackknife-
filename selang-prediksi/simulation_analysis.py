"""
simulation_analysis.py
=====================
Analisis data simulasi (Subbab 3.4.1). Data dibangkitkan lewat random forest
yang dilatih pada data empiris (Pers. 3.1-3.2) dengan peubah penjelas dari
gaussian copula semiparametrik berdimensi 67 (struktur dependensi & marginal
diestimasi dari data empiris). Untuk tiap skenario p/n (Tabel 3.2) dan tiap
algoritme, dibangun selang prediksi jackknife & jackknife+, lalu dievaluasi
dengan PCR, AIW, WS, CT, MU dan dirata-ratakan atas pengulangan.

Seluruh model, termasuk random forest pembangkit respons, menggunakan
transformasi logaritma natural pada peubah respons. Prediksi dikembalikan ke
skala asli melalui fungsi eksponensial (conformal_common.py).

Jalankan:  python simulation_analysis.py
Keluaran:  hasil_simulasi.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from susenas_preprocessing import load_metadata, clean_raw, SusenasEncoder, RESPONSE
from conformal_common import run_all_combinations, make_rf, ALPHA

# ---------------------- Konfigurasi ----------------------
DATA_PATH = "bogor.csv"
META_PATH = "metadata.xlsx"
P = 67                      # jumlah peubah penjelas (tetap)
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]  # p/n (Tabel 3.2)
N_REPS = 100               # pengulangan tiap skenario
SIGMA_FRAC = 0.10          # sigma galat = 0.10 * s_Y
BASE_SEED = 2024

# ---------------------------------------------------------


def scenario_sizes(ratio, p=P):
    n = int(np.round(p / ratio))       # ukuran data latih
    m = int(np.round(0.25 * n))        # data uji ~ 20% (80:20)
    return max(n, 5), max(m, 2)


# ======================================================================
# Gaussian copula semiparametrik (Subbab 3.1.2)
# ======================================================================
def nearest_pd_corr(R, eps=1e-8):
    """Proyeksi ke matriks korelasi definit positif terdekat."""
    R = np.nan_to_num((R + R.T) / 2.0, nan=0.0)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.clip(np.diag(R), eps, None))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


class SemiparametricCopula:
    """Estimasi struktur dependensi (korelasi normal-score) & marginal empiris,
    lalu bangkitkan peubah penjelas baru yang meniru data empiris."""

    def __init__(self, num_vars, cat_vars):
        self.num_vars = list(num_vars)
        self.cat_vars = list(cat_vars)
        self.vars = self.num_vars + self.cat_vars

    def fit(self, df):
        df = df[self.vars].reset_index(drop=True)
        n = len(df)
        # pisahkan peubah konstan (varians nol) -> ditetapkan saat generate
        self.const_vals = {v: df[v].iloc[0] for v in self.vars if df[v].nunique() <= 1}
        self.active = [v for v in self.vars if v not in self.const_vals]

        self.num_quantiles = {v: np.sort(df[v].values.astype(float))
                              for v in self.num_vars if v in self.active}
        self.cat_levels, self.cat_cum = {}, {}
        scores = np.empty((n, len(self.active)))
        for j, v in enumerate(self.active):
            if v in self.num_vars:                       # van der Waerden
                ranks = pd.Series(df[v]).rank(method="average").values
                u = ranks / (n + 1.0)
                scores[:, j] = norm.ppf(u)
            else:                                        # kategorik: skor titik tengah
                vc = df[v].value_counts(normalize=True).sort_index()
                levels = vc.index.tolist()
                props = vc.values
                cum = np.cumsum(props)
                lower = np.concatenate([[0.0], cum[:-1]])
                mid = {lev: (lower[k] + cum[k]) / 2.0 for k, lev in enumerate(levels)}
                self.cat_levels[v] = levels
                self.cat_cum[v] = cum
                scores[:, j] = df[v].map(lambda x: norm.ppf(mid[x])).values

        R = np.corrcoef(scores, rowvar=False)
        self.R = nearest_pd_corr(R)
        self.L = np.linalg.cholesky(self.R)
        return self

    def sample(self, n, rng):
        p = len(self.active)
        Z = rng.standard_normal((n, p)) @ self.L.T
        U = norm.cdf(Z)
        out = {}
        for j, v in enumerate(self.active):
            u = np.clip(U[:, j], 1e-6, 1 - 1e-6)
            if v in self.num_vars:                       # inverse ECDF
                out[v] = np.quantile(self.num_quantiles[v], u, method="linear")
            else:                                        # peta u -> kategori via proporsi
                cum = self.cat_cum[v]
                idx = np.searchsorted(cum, u, side="right")
                idx = np.clip(idx, 0, len(cum) - 1)
                levels = np.array(self.cat_levels[v])
                out[v] = levels[idx]
        for v, val in self.const_vals.items():           # peubah konstan
            out[v] = np.full(n, val)
        return pd.DataFrame(out)[self.vars]


# ======================================================================
def main(algorithms=None):
    df = pd.read_csv(DATA_PATH)
    meta = load_metadata(META_PATH)
    df = clean_raw(df, meta, verbose=True)
    y_emp = df[RESPONSE].values.astype(float)
    s_Y = y_emp.std()
    sigma = SIGMA_FRAC * s_Y

    # encoder + fungsi pembangkit respons f_RF (dilatih pada data empiris; Pers. 3.1)
    encoder = SusenasEncoder(meta).fit(df)
    X_emp = encoder.transform(df)
    # Random forest pembangkit respons juga dilatih pada log(EXPEND), lalu
    # prediksinya dikembalikan ke skala rupiah sebelum galat aditif ditambahkan.
    f_rf = make_rf(seed=0)
    f_rf.fit(X_emp, y_emp)
    print(
        f"[generator] f_RF dilatih pada log({RESPONSE}) dan dikembalikan ke "
        f"skala asli; s_Y={s_Y:,.0f}; sigma={sigma:,.0f}"
    )

    # copula: struktur & marginal dari data empiris (peubah asli 67)
    copula = SemiparametricCopula(meta["num"], meta["cat"]).fit(df)
    print(f"[copula] peubah aktif={len(copula.active)}, konstan={list(copula.const_vals)}")

    def gen_Y(raw, rng):
        Xe = encoder.transform(raw)
        y = f_rf.predict(Xe) + rng.normal(0.0, sigma, size=len(raw))
        # Pengaman agar respons tetap positif dan dapat ditransformasi log.
        y = np.clip(y, 1.0, None)
        return y, Xe

    all_rows = []
    all_detail = []
    for ratio in RATIOS:
        n, m = scenario_sizes(ratio)
        print(f"\n[skenario] p/n={ratio}: n={n}, m={m}, ulangan={N_REPS}")
        for rep in range(N_REPS):
            rng = np.random.default_rng(BASE_SEED + int(ratio * 100) + rep)
            raw_tr = copula.sample(n, rng)
            raw_te = copula.sample(m, rng)
            y_tr, X_tr = gen_Y(raw_tr, rng)          # Pers. 3.2
            y_te, X_te = gen_Y(raw_te, rng)
            rows, detail = run_all_combinations(
                encoder.numeric_idx_, X_tr, y_tr, X_te, y_te, seed=rep, alpha=ALPHA,
                algorithms=algorithms, collect_detail=True,
            )
            for r in rows:
                r.update({"ratio_pn": ratio, "n": n, "m": m, "rep": rep})
            all_rows.extend(rows)
            for d in detail:
                d.update({"ratio_pn": ratio, "n": n, "rep": rep})
            all_detail.extend(detail)

    # ---- Tabel hasil per-ulangan (kolom sesuai permintaan) ----
    res = pd.DataFrame(all_rows).rename(columns={
        "ratio_pn": "ratio", "n": "n_train", "m": "n_test", "algoritme": "algoritma",
    })
    COLS = ["rep", "ratio", "n_train", "n_test", "algoritma", "metode",
            "PCR", "AIW", "WS", "CT", "MU"]
    perrep = (res[COLS]
              .sort_values(["ratio", "rep", "algoritma", "metode"])
              .reset_index(drop=True))
    suffix = "" if algorithms is None else "_" + "_".join(algorithms)
    out = f"hasil_simulasi{suffix}.csv"
    perrep.to_csv(out, index=False)

    # ---- Ringkasan rata-rata atas pengulangan (langkah 6 Subbab 3.4.1) ----
    ringkasan = (perrep.groupby(["ratio", "n_train", "n_test", "algoritma", "metode"])
                 [["PCR", "AIW", "WS", "CT", "MU"]].mean().reset_index())
    out_ring = f"hasil_simulasi_ringkasan{suffix}.csv"
    ringkasan.to_csv(out_ring, index=False)

    # ---- Tabel detail per amatan uji: y_true, y_pred, lower, upper, method, model ----
    detail_df = pd.DataFrame(all_detail)[
        ["ratio_pn", "n", "rep", "model", "method", "y_true", "y_pred", "lower", "upper"]
    ]
    out_detail = f"hasil_simulasi_detail{suffix}.csv"
    detail_df.to_csv(out_detail, index=False)

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(f"\n=== Hasil simulasi per-ulangan (5 baris pertama, total {len(perrep):,}) ===")
    print(perrep.head().to_string(index=False))
    print("\n=== Ringkasan (rata-rata pengulangan) ===")
    print(ringkasan.to_string(index=False))
    print(f"\nDisimpan: {out}")
    print(f"Disimpan: {out_ring}")
    print(f"Disimpan: {out_detail}")


if __name__ == "__main__":
    main()
