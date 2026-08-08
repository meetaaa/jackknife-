"""
ml_common.py
============
Modul bersama untuk menjalankan tiga model machine learning sebagai prediksi
titik dengan transformasi logaritma natural pada peubah respons.

Model:
1. Random Forest
   - n_estimators=50
   - n_jobs=-1
   - random_state=42
2. Gradient Boosting
   - n_estimators=50
   - learning_rate=0.05
   - random_state=42
3. Support Vector Regression
   - kernel='rbf'
   - C=100
   - epsilon=0.15
   - gamma=0.008
   - standardisasi hanya peubah numerik

Seluruh model dilatih pada log(y). Prediksi, metrik, dan sisaan dikembalikan
ke skala asli. Modul ini juga menyediakan evaluasi validasi silang K-lipatan
(default digunakan 5 lipatan pada empirical_ml_only.py).
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 50,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

GB_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
}

SVR_PARAMS = {
    "kernel": "rbf",
    "C": 100.0,
    "epsilon": 0.15,
    "gamma": 0.008,
}


def _safe_log(y: np.ndarray) -> np.ndarray:
    """Logaritma natural yang aman untuk respons bernilai nol atau negatif."""
    y = np.asarray(y, dtype=float)
    return np.log(np.clip(y, 1.0, None))


def _with_log_target(regressor: object) -> TransformedTargetRegressor:
    """Membungkus estimator dengan transformasi log(y) dan invers exp."""
    return TransformedTargetRegressor(
        regressor=regressor,
        func=_safe_log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def make_random_forest() -> TransformedTargetRegressor:
    """Membuat Random Forest dengan transformasi log pada respons."""
    return _with_log_target(RandomForestRegressor(**RF_PARAMS))


def make_gradient_boosting() -> TransformedTargetRegressor:
    """Membuat Gradient Boosting dengan transformasi log pada respons."""
    return _with_log_target(GradientBoostingRegressor(**GB_PARAMS))


def make_svr(numeric_idx: Iterable[int]) -> TransformedTargetRegressor:
    """
    Membuat SVR RBF. Peubah numerik distandardisasi, sedangkan peubah hasil
    penyandian kategorik diteruskan tanpa standardisasi.
    """
    numeric_idx = list(numeric_idx)
    preprocessor = ColumnTransformer(
        transformers=[("numeric_scaler", StandardScaler(), numeric_idx)],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("svr", SVR(**SVR_PARAMS)),
        ]
    )
    return _with_log_target(pipeline)


def get_model_factories(
    numeric_idx: Iterable[int],
    algorithms: Iterable[str] | None = None,
) -> dict[str, Callable[[], object]]:
    """Mengembalikan pembuat model yang dipilih."""
    factories: dict[str, Callable[[], object]] = {
        "gradient_boosting": make_gradient_boosting,
        "random_forest": make_random_forest,
        "svr": lambda: make_svr(numeric_idx),
    }

    if algorithms is None:
        return factories

    requested = list(algorithms)
    unknown = sorted(set(requested) - set(factories))
    if unknown:
        raise ValueError(
            f"Algoritme tidak dikenal: {unknown}. Pilihan: {sorted(factories)}"
        )
    return {name: factories[name] for name in requested}


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%) dengan penyebut aman untuk nilai aktual nol."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.clip(np.abs(y_true), 1.0, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Menghitung metrik pada skala asli peubah respons."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fit_predict_evaluate(
    model_factory: Callable[[], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[object, dict[str, float], np.ndarray]:
    """Melatih model, memprediksi data uji, serta mengukur waktu dan memori."""
    tracemalloc.start()
    start = time.perf_counter()

    try:
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        computation_time = time.perf_counter() - start
        peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()

    metrics = regression_metrics(y_test, y_pred)
    metrics.update({"CT": float(computation_time), "MU": float(peak_memory_mb)})
    return model, metrics, y_pred


def cross_validate_model(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    observation_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float | int]], list[dict[str, float | int]]]:
    """
    Menjalankan validasi silang K-lipatan pada data latih.

    Data diacak secara reproduktif, lalu setiap amatan diprediksi tepat sekali
    ketika menjadi bagian validasi. Nilai RMSE_VS dan R2_VS merupakan rata-rata
    metrik dari seluruh lipatan. Prediksi out-of-fold (OOF) juga dikembalikan
    untuk pemeriksaan sisaan validasi silang.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)

    if n_splits < 2:
        raise ValueError("n_splits minimal 2.")
    if len(y) < 2 * n_splits:
        raise ValueError(
            f"Jumlah data latih ({len(y)}) terlalu kecil untuk {n_splits} lipatan "
            "karena setiap lipatan perlu minimal dua amatan untuk menghitung R²."
        )

    if observation_ids is None:
        observation_ids_array = np.arange(len(y), dtype=int)
    else:
        observation_ids_array = np.asarray(observation_ids)
        if len(observation_ids_array) != len(y):
            raise ValueError("Panjang observation_ids harus sama dengan panjang y.")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    detail_rows: list[dict[str, float | int]] = []

    for fold, (fit_idx, validation_idx) in enumerate(splitter.split(X), start=1):
        model = model_factory()
        model.fit(X[fit_idx], y[fit_idx])
        fold_pred = np.asarray(model.predict(X[validation_idx]), dtype=float)
        oof_pred[validation_idx] = fold_pred

        fold_metrics = regression_metrics(y[validation_idx], fold_pred)
        fold_rows.append({"fold": fold, **fold_metrics})

        for local_idx, predicted in zip(validation_idx, fold_pred):
            actual = float(y[local_idx])
            residual = actual - float(predicted)
            detail_rows.append(
                {
                    "observation_id": int(observation_ids_array[local_idx]),
                    "fold": fold,
                    "y_true": actual,
                    "y_pred": float(predicted),
                    "residual": residual,
                    "absolute_error": abs(residual),
                }
            )

    if np.isnan(oof_pred).any():
        raise RuntimeError("Prediksi out-of-fold tidak lengkap.")

    rmse_folds = np.array([float(row["RMSE"]) for row in fold_rows])
    r2_folds = np.array([float(row["R2"]) for row in fold_rows])
    mae_folds = np.array([float(row["MAE"]) for row in fold_rows])
    mape_folds = np.array([float(row["MAPE"]) for row in fold_rows])
    pooled = regression_metrics(y, oof_pred)

    summary = {
        "MAE_VS": float(mae_folds.mean()),
        "RMSE_VS": float(rmse_folds.mean()),
        "MAPE_VS": float(mape_folds.mean()),
        "R2_VS": float(r2_folds.mean()),
        "RMSE_VS_SD": float(rmse_folds.std(ddof=1)),
        "R2_VS_SD": float(r2_folds.std(ddof=1)),
        "RMSE_VS_OOF": float(pooled["RMSE"]),
        "R2_VS_OOF": float(pooled["R2"]),
    }
    return summary, fold_rows, detail_rows


def run_models(
    numeric_idx: Iterable[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithms: Iterable[str] | None = None,
    collect_detail: bool = False,
):
    """Menjalankan seluruh model yang dipilih pada data latih dan data uji."""
    factories = get_model_factories(numeric_idx, algorithms=algorithms)
    rows: list[dict[str, float | str]] = []
    details: list[dict[str, float | str | int]] = []
    fitted_models: dict[str, object] = {}

    for algorithm, model_factory in factories.items():
        model, metrics, y_pred = fit_predict_evaluate(
            model_factory,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        fitted_models[algorithm] = model
        rows.append({"algoritma": algorithm, **metrics})

        if collect_detail:
            for observation_id, (actual, predicted) in enumerate(zip(y_test, y_pred)):
                residual = float(actual - predicted)
                details.append(
                    {
                        "observation_id": observation_id,
                        "model": algorithm,
                        "y_true": float(actual),
                        "y_pred": float(predicted),
                        "residual": residual,
                        "absolute_error": abs(residual),
                    }
                )

    if collect_detail:
        return rows, details, fitted_models
    return rows, fitted_models
