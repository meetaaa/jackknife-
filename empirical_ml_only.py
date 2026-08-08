"""
empirical_ml_only.py
====================
Analisis prediksi titik EXPEND pada data empiris Susenas menggunakan:
- Gradient Boosting
- Random Forest
- Support Vector Regression

Seluruh model dilatih pada log(EXPEND), lalu prediksi dikembalikan ke skala
rupiah. Evaluasi mencakup data uji dan validasi silang 5 lipatan pada data
latih. Script juga menghasilkan diagram nilai aktual versus nilai dugaan dan
diagram sisaan versus nilai dugaan pada data uji.

Berkas masukan pada folder kerja:
- bogor.csv
- metadata.xlsx
- susenas_preprocessing.py
- ml_common.py

Jalankan:
    python empirical_ml_only.py

Keluaran utama:
- hasil_empiris_ml.csv                         (metrik lengkap)
- tabel_evaluasi_model_empiris.csv             (format tabel publikasi)
- hasil_empiris_ml_detail.csv                  (prediksi dan sisaan data uji)
- hasil_empiris_ml_cv_lipatan.csv              (metrik setiap lipatan)
- hasil_empiris_ml_cv_detail.csv               (prediksi OOF dan sisaan VS)
- diagram_pencar_aktual_dugaan_data_uji.png
- diagram_sisaan_dugaan_data_uji.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_common import (
    RANDOM_STATE,
    cross_validate_model,
    get_model_factories,
    run_models,
)
from susenas_preprocessing import RESPONSE, SusenasEncoder, clean_raw, load_metadata

DATA_PATH = "bogor.csv"
META_PATH = "metadata.xlsx"
TEST_SIZE = 0.20
CV_FOLDS = 5
MILLION = 1_000_000.0

MODEL_LABELS = {
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
    "svr": "Support Vector Regression",
}
MODEL_ORDER = ["gradient_boosting", "random_forest", "svr"]


def eksplorasi(df: pd.DataFrame) -> None:
    """Menampilkan statistik deskriptif ringkas peubah respons."""
    y = df[RESPONSE].astype(float)
    print("\n=== Eksplorasi data empiris ===")
    print(f"n = {len(df):,}")
    print(
        f"{RESPONSE}: min={y.min():,.0f}  median={y.median():,.0f}  "
        f"mean={y.mean():,.0f}  max={y.max():,.0f}  skew={y.skew():.2f}"
    )
    print(f"skew log({RESPONSE}) = {np.log(np.clip(y, 1.0, None)).skew():.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluasi model ML data empiris dengan validasi silang."
    )
    parser.add_argument("--data", default=DATA_PATH, help="Lokasi bogor.csv")
    parser.add_argument("--metadata", default=META_PATH, help="Lokasi metadata.xlsx")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["random_forest", "gradient_boosting", "svr"],
        default=None,
        help="Model yang dijalankan; default seluruh model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help="Jumlah lipatan validasi silang; default 5.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Tidak membuat gambar diagnostik.",
    )
    return parser.parse_args()


def _selected_order(models: list[str]) -> list[str]:
    return [model for model in MODEL_ORDER if model in models]


def plot_actual_vs_predicted(detail: pd.DataFrame, output_path: str) -> None:
    """Membuat diagram pencar aktual versus dugaan pada data uji."""
    models = _selected_order(detail["model"].drop_duplicates().tolist())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.5), squeeze=False)

    actual_max = float(detail["y_true"].max() / MILLION)
    pred_max = float(detail["y_pred"].max() / MILLION)
    upper = max(actual_max, pred_max)
    upper = max(1.0, np.ceil(upper / 10.0) * 10.0)

    for ax, model in zip(axes[0], models):
        subset = detail.loc[detail["model"] == model]
        actual = subset["y_true"].to_numpy() / MILLION
        predicted = subset["y_pred"].to_numpy() / MILLION

        ax.scatter(actual, predicted, s=18, alpha=0.65)
        ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.0)
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Nilai aktual (juta rupiah)")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Nilai dugaan (juta rupiah)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_vs_predicted(detail: pd.DataFrame, output_path: str) -> None:
    """Membuat diagram sisaan versus nilai dugaan pada data uji."""
    models = _selected_order(detail["model"].drop_duplicates().tolist())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.5), squeeze=False)

    residual_abs_max = float(np.abs(detail["residual"]).max() / MILLION)
    residual_limit = max(1.0, np.ceil(residual_abs_max / 10.0) * 10.0)

    for ax, model in zip(axes[0], models):
        subset = detail.loc[detail["model"] == model]
        predicted = subset["y_pred"].to_numpy() / MILLION
        residual = subset["residual"].to_numpy() / MILLION

        ax.scatter(predicted, residual, s=18, alpha=0.65)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylim(-residual_limit, residual_limit)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Nilai dugaan (juta rupiah)")
        ax.grid(alpha=0.25)

    axes[0, 0].set_ylabel("Sisaan (juta rupiah)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(
    data_path: str = DATA_PATH,
    meta_path: str = META_PATH,
    algorithms: list[str] | None = None,
    cv_folds: int = CV_FOLDS,
    make_plots: bool = True,
) -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data tidak ditemukan: {data_path}")
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
    if cv_folds < 2:
        raise ValueError("cv_folds minimal 2.")

    df = pd.read_csv(data_path)
    meta = load_metadata(meta_path)
    eksplorasi(df)

    df = clean_raw(df, meta, verbose=True)
    y = df[RESPONSE].to_numpy(dtype=float)

    encoder = SusenasEncoder(meta)
    X = encoder.fit_transform(df)
    observation_ids = np.arange(len(y), dtype=int)
    print(
        f"[encoding] matriks fitur: {X.shape[0]} x {X.shape[1]}; "
        f"indeks numerik={encoder.numeric_idx_}; kolom dibuang={encoder.drop_cols_}"
    )

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        observation_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"[split] latih={len(y_train):,}; uji={len(y_test):,}")
    print(
        "[transformasi] model dilatih pada log(EXPEND); prediksi, metrik, dan "
        "sisaan dilaporkan pada skala rupiah"
    )
    print(
        f"[validasi silang] {cv_folds} lipatan pada data latih; shuffle=True; "
        f"random_state={RANDOM_STATE}"
    )

    rows, details, _ = run_models(
        numeric_idx=encoder.numeric_idx_,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        algorithms=algorithms,
        collect_detail=True,
    )

    # Ganti indeks lokal data uji dengan indeks amatan setelah pembersihan data.
    for item in details:
        item["observation_id"] = int(id_test[int(item["observation_id"])])
        item["split"] = "uji"

    cv_summary_rows: list[dict[str, float | str]] = []
    cv_fold_rows: list[dict[str, float | int | str]] = []
    cv_detail_rows: list[dict[str, float | int | str]] = []
    factories = get_model_factories(encoder.numeric_idx_, algorithms=algorithms)

    for model_name, model_factory in factories.items():
        summary, fold_rows, cv_details = cross_validate_model(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            n_splits=cv_folds,
            random_state=RANDOM_STATE,
            observation_ids=id_train,
        )
        cv_summary_rows.append({"algoritma": model_name, **summary})

        for fold_row in fold_rows:
            cv_fold_rows.append({"model": model_name, **fold_row})
        for detail_row in cv_details:
            cv_detail_rows.append({"model": model_name, "split": "validasi_silang", **detail_row})

    result = pd.DataFrame(rows).merge(
        pd.DataFrame(cv_summary_rows),
        on="algoritma",
        how="left",
        validate="one_to_one",
    )
    result["model"] = result["algoritma"].map(MODEL_LABELS)
    ordered_models = _selected_order(result["algoritma"].tolist())
    order_map = {name: index for index, name in enumerate(ordered_models)}
    result["_order"] = result["algoritma"].map(order_map)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result = result[
        [
            "algoritma", "model",
            "MAE", "RMSE", "MAPE", "R2",
            "MAE_VS", "RMSE_VS", "MAPE_VS", "R2_VS",
            "RMSE_VS_SD", "R2_VS_SD", "RMSE_VS_OOF", "R2_VS_OOF",
            "CT", "MU",
        ]
    ]

    detail = (
        pd.DataFrame(details)[
            [
                "observation_id", "split", "model", "y_true", "y_pred",
                "residual", "absolute_error",
            ]
        ]
        .sort_values(["model", "observation_id"])
        .reset_index(drop=True)
    )
    cv_fold = (
        pd.DataFrame(cv_fold_rows)[
            ["model", "fold", "MAE", "RMSE", "MAPE", "R2"]
        ]
        .sort_values(["model", "fold"])
        .reset_index(drop=True)
    )
    cv_detail = (
        pd.DataFrame(cv_detail_rows)[
            [
                "observation_id", "split", "model", "fold", "y_true", "y_pred",
                "residual", "absolute_error",
            ]
        ]
        .sort_values(["model", "fold", "observation_id"])
        .reset_index(drop=True)
    )

    publication_table = result[["model", "RMSE", "RMSE_VS", "R2", "R2_VS"]].copy()
    publication_table["RMSE"] = publication_table["RMSE"] / MILLION
    publication_table["RMSE (VS)"] = publication_table.pop("RMSE_VS") / MILLION
    publication_table["R²"] = publication_table.pop("R2")
    publication_table["R² (VS)"] = publication_table.pop("R2_VS")
    publication_table = publication_table[["model", "RMSE", "RMSE (VS)", "R²", "R² (VS)"]]

    suffix = "" if algorithms is None else "_" + "_".join(algorithms)
    output_summary = f"hasil_empiris_ml{suffix}.csv"
    output_publication = f"tabel_evaluasi_model_empiris{suffix}.csv"
    output_detail = f"hasil_empiris_ml_detail{suffix}.csv"
    output_cv_fold = f"hasil_empiris_ml_cv_lipatan{suffix}.csv"
    output_cv_detail = f"hasil_empiris_ml_cv_detail{suffix}.csv"
    output_scatter = f"diagram_pencar_aktual_dugaan_data_uji{suffix}.png"
    output_residual = f"diagram_sisaan_dugaan_data_uji{suffix}.png"

    result.to_csv(output_summary, index=False)
    publication_table.to_csv(output_publication, index=False)
    detail.to_csv(output_detail, index=False)
    cv_fold.to_csv(output_cv_fold, index=False)
    cv_detail.to_csv(output_cv_detail, index=False)

    if make_plots:
        plot_actual_vs_predicted(detail, output_scatter)
        plot_residual_vs_predicted(detail, output_residual)

    pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
    print("\n=== Evaluasi model (RMSE dalam juta rupiah) ===")
    print(publication_table.to_string(index=False))
    print("\nCatatan: VS = validasi silang pada data latih.")
    print("RMSE (VS) dan R² (VS) adalah rata-rata dari metrik lima lipatan.")
    print("\n=== Metrik setiap lipatan ===")
    print(cv_fold.to_string(index=False))

    outputs = [
        output_summary,
        output_publication,
        output_detail,
        output_cv_fold,
        output_cv_detail,
    ]
    if make_plots:
        outputs.extend([output_scatter, output_residual])
    print("\nDisimpan:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data,
        meta_path=args.metadata,
        algorithms=args.algorithms,
        cv_folds=args.cv_folds,
        make_plots=not args.no_plots,
    )
