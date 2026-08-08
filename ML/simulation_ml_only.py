"""
simulation_ml_only.py
=====================
Analisis simulasi prediksi titik menggunakan Random Forest, Gradient Boosting,
dan SVR dengan transformasi logaritma natural pada peubah respons. Data simulasi tetap dibangkitkan dengan Gaussian copula
semiparametrik seperti pada kode penelitian, tetapi TIDAK dibangun selang
prediksi Jackknife maupun Jackknife+.

Berkas masukan yang harus tersedia pada folder kerja:
- bogor.csv
- metadata.xlsx
- susenas_preprocessing.py
- ml_common.py

Jalankan:
    python simulation_ml_only.py

Keluaran:
- hasil_simulasi_ml.csv
- hasil_simulasi_ml_ringkasan.csv
- hasil_simulasi_ml_detail.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ml_common import RANDOM_STATE, make_random_forest, run_models
from susenas_preprocessing import RESPONSE, SusenasEncoder, clean_raw, load_metadata

DATA_PATH = "bogor.csv"
META_PATH = "metadata.xlsx"
P = 67
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
N_REPS = 100
SIGMA_FRAC = 0.10
BASE_SEED = 2024


def scenario_sizes(ratio: float, p: int = P) -> tuple[int, int]:
    """Menghitung ukuran data latih dan uji untuk setiap rasio p/n."""
    n_train = int(np.round(p / ratio))
    n_test = int(np.round(0.25 * n_train))
    return max(n_train, 5), max(n_test, 2)


def nearest_pd_corr(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Memproyeksikan matriks ke matriks korelasi definit positif."""
    R = np.nan_to_num((R + R.T) / 2.0, nan=0.0)
    np.fill_diagonal(R, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = np.clip(eigenvalues, eps, None)
    R = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.clip(np.diag(R), eps, None))
    R = R / np.outer(diagonal, diagonal)
    np.fill_diagonal(R, 1.0)
    return R


class SemiparametricCopula:
    """Gaussian copula dengan marginal empiris untuk membangkitkan fitur."""

    def __init__(self, num_vars, cat_vars):
        self.num_vars = list(num_vars)
        self.cat_vars = list(cat_vars)
        self.vars = self.num_vars + self.cat_vars

    def fit(self, df: pd.DataFrame):
        df = df[self.vars].reset_index(drop=True)
        n = len(df)

        self.const_vals = {
            variable: df[variable].iloc[0]
            for variable in self.vars
            if df[variable].nunique() <= 1
        }
        self.active = [v for v in self.vars if v not in self.const_vals]

        self.num_quantiles = {
            variable: np.sort(df[variable].to_numpy(dtype=float))
            for variable in self.num_vars
            if variable in self.active
        }
        self.cat_levels = {}
        self.cat_cum = {}
        scores = np.empty((n, len(self.active)))

        for column_index, variable in enumerate(self.active):
            if variable in self.num_vars:
                ranks = pd.Series(df[variable]).rank(method="average").to_numpy()
                uniform_scores = ranks / (n + 1.0)
                scores[:, column_index] = norm.ppf(uniform_scores)
            else:
                frequencies = (
                    df[variable]
                    .value_counts(normalize=True)
                    .sort_index()
                )
                levels = frequencies.index.tolist()
                proportions = frequencies.to_numpy()
                cumulative = np.cumsum(proportions)
                lower = np.concatenate([[0.0], cumulative[:-1]])
                midpoint = {
                    level: (lower[k] + cumulative[k]) / 2.0
                    for k, level in enumerate(levels)
                }

                self.cat_levels[variable] = levels
                self.cat_cum[variable] = cumulative
                scores[:, column_index] = (
                    df[variable]
                    .map(lambda value: norm.ppf(midpoint[value]))
                    .to_numpy()
                )

        correlation = np.corrcoef(scores, rowvar=False)
        self.R = nearest_pd_corr(correlation)
        self.L = np.linalg.cholesky(self.R)
        return self

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        p = len(self.active)
        latent_normal = rng.standard_normal((n, p)) @ self.L.T
        uniforms = norm.cdf(latent_normal)
        output = {}

        for column_index, variable in enumerate(self.active):
            u = np.clip(uniforms[:, column_index], 1e-6, 1.0 - 1e-6)

            if variable in self.num_vars:
                output[variable] = np.quantile(
                    self.num_quantiles[variable],
                    u,
                    method="linear",
                )
            else:
                cumulative = self.cat_cum[variable]
                category_index = np.searchsorted(cumulative, u, side="right")
                category_index = np.clip(category_index, 0, len(cumulative) - 1)
                levels = np.asarray(self.cat_levels[variable])
                output[variable] = levels[category_index]

        for variable, value in self.const_vals.items():
            output[variable] = np.full(n, value)

        return pd.DataFrame(output)[self.vars]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menjalankan model ML prediksi titik pada data simulasi."
    )
    parser.add_argument("--data", default=DATA_PATH, help="Lokasi bogor.csv")
    parser.add_argument("--metadata", default=META_PATH, help="Lokasi metadata.xlsx")
    parser.add_argument("--n-reps", type=int, default=N_REPS, help="Jumlah ulangan")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=RATIOS,
        help="Daftar rasio p/n.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["random_forest", "gradient_boosting", "svr"],
        default=None,
        help="Model yang dijalankan; default seluruh model.",
    )
    return parser.parse_args()


def main(
    data_path: str = DATA_PATH,
    meta_path: str = META_PATH,
    ratios: list[float] | None = None,
    n_reps: int = N_REPS,
    algorithms: list[str] | None = None,
) -> None:
    ratios = RATIOS if ratios is None else ratios

    if n_reps < 1:
        raise ValueError("n_reps minimal 1.")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data tidak ditemukan: {data_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")

    df = pd.read_csv(data_path)
    meta = load_metadata(meta_path)
    df = clean_raw(df, meta, verbose=True)

    y_empirical = df[RESPONSE].to_numpy(dtype=float)
    response_sd = float(y_empirical.std())
    sigma = SIGMA_FRAC * response_sd

    encoder = SusenasEncoder(meta).fit(df)
    X_empirical = encoder.transform(df)

    # Pembangkit respons memakai RF dengan parameter yang sama dan dilatih
    # pada log(EXPEND). Prediksinya otomatis dikembalikan ke skala rupiah.
    response_generator = make_random_forest()
    response_generator.fit(X_empirical, y_empirical)
    print(
        f"[generator] RF dilatih pada log(EXPEND) data empiris; "
        f"random_state={RANDOM_STATE}; s_Y={response_sd:,.0f}; sigma={sigma:,.0f}"
    )

    copula = SemiparametricCopula(meta["num"], meta["cat"]).fit(df)
    print(
        f"[copula] peubah aktif={len(copula.active)}; "
        f"peubah konstan={list(copula.const_vals)}"
    )

    def generate_response(raw_data: pd.DataFrame, rng: np.random.Generator):
        encoded_features = encoder.transform(raw_data)
        mean_response = response_generator.predict(encoded_features)
        response = mean_response + rng.normal(0.0, sigma, size=len(raw_data))
        # EXPEND harus positif agar transformasi log terdefinisi.
        response = np.clip(response, 1.0, None)
        return response, encoded_features

    all_rows = []
    all_details = []

    for ratio in ratios:
        n_train, n_test = scenario_sizes(ratio)
        print(
            f"\n[skenario] p/n={ratio}; n_train={n_train}; "
            f"n_test={n_test}; ulangan={n_reps}"
        )

        for rep in range(n_reps):
            rng = np.random.default_rng(
                BASE_SEED + int(round(ratio * 1000)) + rep
            )

            raw_train = copula.sample(n_train, rng)
            raw_test = copula.sample(n_test, rng)
            y_train, X_train = generate_response(raw_train, rng)
            y_test, X_test = generate_response(raw_test, rng)

            rows, details, _ = run_models(
                numeric_idx=encoder.numeric_idx_,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                algorithms=algorithms,
                collect_detail=True,
            )

            for row in rows:
                row.update({
                    "rep": rep,
                    "ratio": ratio,
                    "n_train": n_train,
                    "n_test": n_test,
                })
            all_rows.extend(rows)

            for detail in details:
                detail.update({
                    "rep": rep,
                    "ratio": ratio,
                    "n_train": n_train,
                    "n_test": n_test,
                })
            all_details.extend(details)

    per_rep = (
        pd.DataFrame(all_rows)[
            [
                "rep", "ratio", "n_train", "n_test", "algoritma",
                "MAE", "RMSE", "MAPE", "R2", "CT", "MU",
            ]
        ]
        .sort_values(["ratio", "rep", "algoritma"])
        .reset_index(drop=True)
    )

    summary = (
        per_rep
        .groupby(["ratio", "n_train", "n_test", "algoritma"], as_index=False)[
            ["MAE", "RMSE", "MAPE", "R2", "CT", "MU"]
        ]
        .mean()
        .sort_values(["ratio", "algoritma"])
        .reset_index(drop=True)
    )

    detail_df = (
        pd.DataFrame(all_details)[
            [
                "rep", "ratio", "n_train", "n_test", "observation_id",
                "model", "y_true", "y_pred", "residual", "absolute_error",
            ]
        ]
        .sort_values(["ratio", "rep", "model", "observation_id"])
        .reset_index(drop=True)
    )

    suffix = "" if algorithms is None else "_" + "_".join(algorithms)
    output_per_rep = f"hasil_simulasi_ml{suffix}.csv"
    output_summary = f"hasil_simulasi_ml_ringkasan{suffix}.csv"
    output_detail = f"hasil_simulasi_ml_detail{suffix}.csv"

    per_rep.to_csv(output_per_rep, index=False)
    summary.to_csv(output_summary, index=False)
    detail_df.to_csv(output_detail, index=False)

    pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
    print("\n=== Hasil per ulangan: lima baris pertama ===")
    print(per_rep.head().to_string(index=False))
    print("\n=== Ringkasan rata-rata ===")
    print(summary.to_string(index=False))
    print(f"\nDisimpan: {output_per_rep}")
    print(f"Disimpan: {output_summary}")
    print(f"Disimpan: {output_detail}")


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data,
        meta_path=args.metadata,
        ratios=args.ratios,
        n_reps=args.n_reps,
        algorithms=args.algorithms,
    )
