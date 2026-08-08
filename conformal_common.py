"""
conformal_common.py
===================
Modul bersama untuk analisis selang prediksi konformal (jackknife & jackknife+)
dengan tiga algoritme: random forest, gradient boosting, dan SVR.

Dipakai oleh empirical_analysis.py dan simulation_analysis.py.

Seluruh algoritme (random forest, gradient boosting, dan SVR) dilatih pada
peubah respons yang telah ditransformasi dengan logaritma natural. Prediksi
secara otomatis dikembalikan ke skala rupiah menggunakan fungsi eksponensial.

Parameter SVR sama untuk analisis empiris dan simulasi, yaitu:
    kernel RBF, C = 100, epsilon = 0.15, gamma = 0.008,
    dengan peubah numerik distandardisasi.
"""
import time
import tracemalloc
from math import ceil, floor

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ======================================================================
# Konfigurasi model
# - Transformasi log respons diterapkan pada RF, GB, dan SVR.
# - Prediksi dikembalikan ke skala asli melalui exp.
# - Parameter SVR tetap sama untuk data empiris dan simulasi.
# ======================================================================
SVR_PARAMS = dict(kernel="rbf", C=100.0, epsilon=0.15, gamma=0.008)

ALPHA = 0.10  # taraf nominal; target cakupan 1 - alpha = 90%


# ----------------------------------------------------------------------
# Pembuat model (model factory). Tiap pemanggilan menghasilkan estimator
# baru yang belum dilatih, sehingga dapat dilatih ulang pada prosedur LOO.
# ----------------------------------------------------------------------
def _safe_log(y):
    """Log natural yang aman untuk respons positif.

    Nilai di bawah 1 dipotong menjadi 1 agar log terdefinisi. Data EXPEND
    empiris bernilai positif; pemotongan ini terutama menjadi pengaman untuk
    data simulasi apabila penambahan galat menghasilkan nilai nonpositif.
    """
    return np.log(np.clip(np.asarray(y, dtype=float), 1.0, None))


def _log_target(regressor):
    """Bungkus estimator agar dilatih pada log(y) dan memprediksi pada skala asli."""
    return TransformedTargetRegressor(
        regressor=regressor,
        func=_safe_log,
        inverse_func=np.exp,
        check_inverse=False,
    )


def make_rf(seed=0):
    """Random forest dengan transformasi log pada peubah respons."""
    rf = RandomForestRegressor(
        n_estimators=50,
        random_state=seed,
        n_jobs=-1,
    )
    return _log_target(rf)


def make_gb(seed=0):
    """Gradient boosting dengan transformasi log pada peubah respons."""
    gb = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.05,
        random_state=seed,
    )
    return _log_target(gb)


def make_svr(numeric_idx, seed=0):
    """SVR RBF dengan standardisasi peubah numerik dan transformasi log respons.

    numeric_idx adalah indeks kolom numerik pada matriks X yang distandardisasi.
    Parameter seed dipertahankan agar antarmuka pembuat model konsisten.
    """
    del seed  # SVR bersifat deterministik untuk konfigurasi ini
    pre = ColumnTransformer(
        [("scale", StandardScaler(), list(numeric_idx))],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("svr", SVR(**SVR_PARAMS))])
    return _log_target(pipe)


def get_model_factories(numeric_idx, seed=0):
    """Kembalikan dict {nama_algoritme: fungsi_pembuat_model}."""
    return {
        "random_forest": lambda: make_rf(seed),
        "gradient_boosting": lambda: make_gb(seed),
        "svr": lambda: make_svr(numeric_idx, seed),
    }


# ----------------------------------------------------------------------
# Prosedur LOO (leave-one-out): dipakai baik jackknife maupun jackknife+.
# ----------------------------------------------------------------------
def _loo_pass(make_model, X_tr, y_tr, X_te, need_test_pred):
    """Satu putaran leave-one-out.
    Mengembalikan sisaan LOO |Y_i - f_{-i}(X_i)| dan (opsional) matriks
    prediksi tiap model LOO terhadap seluruh data uji (n x m)."""
    n = len(y_tr)
    loo_res = np.empty(n)
    loo_pred_te = np.empty((n, len(X_te))) if need_test_pred else None
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        mdl = make_model()
        mdl.fit(X_tr[mask], y_tr[mask])
        loo_res[i] = abs(y_tr[i] - mdl.predict(X_tr[i : i + 1])[0])
        if need_test_pred:
            loo_pred_te[i] = mdl.predict(X_te)
    return loo_res, loo_pred_te


def jackknife_interval(make_model, X_tr, y_tr, X_te, alpha=ALPHA):
    """Selang prediksi jackknife (Subbab 3.2.1)."""
    n = len(y_tr)
    loo_res, _ = _loo_pass(make_model, X_tr, y_tr, X_te, need_test_pred=False)
    k = ceil((1 - alpha) * (n + 1))
    k = min(k, n)  # bila k > n, gunakan sisaan terbesar
    d = np.sort(loo_res)[k - 1]
    full = make_model()
    full.fit(X_tr, y_tr)
    mu = full.predict(X_te)
    return mu - d, mu + d, mu  # lower, upper, y_pred (prediksi model data-penuh)


def jackknife_plus_interval(make_model, X_tr, y_tr, X_te, alpha=ALPHA):
    """Selang prediksi jackknife+ (Subbab 3.2.2; Barber et al. 2021)."""
    n = len(y_tr)
    loo_res, loo_pred_te = _loo_pass(make_model, X_tr, y_tr, X_te, need_test_pred=True)
    lo_mat = loo_pred_te - loo_res[:, None]  # f_{-i}(x) - R_i
    hi_mat = loo_pred_te + loo_res[:, None]  # f_{-i}(x) + R_i
    k_lo = max(1, floor(alpha * (n + 1)))
    k_hi = min(n, ceil((1 - alpha) * (n + 1)))
    lower = np.sort(lo_mat, axis=0)[k_lo - 1, :]
    upper = np.sort(hi_mat, axis=0)[k_hi - 1, :]
    y_pred = np.median(loo_pred_te, axis=0)  # prediksi titik = median prediksi LOO
    return lower, upper, y_pred


# ----------------------------------------------------------------------
# Metrik evaluasi (Subbab 3.3): PCR, AIW, WS, CT, MU
# ----------------------------------------------------------------------
def pcr(lower, upper, y):
    return float(np.mean((y >= lower) & (y <= upper)))


def aiw(lower, upper):
    return float(np.mean(upper - lower))


def winkler_score(lower, upper, y, alpha=ALPHA):
    width = upper - lower
    below = y < lower
    above = y > upper
    penalty = np.zeros_like(width, dtype=float)
    penalty[below] = (2.0 / alpha) * (lower[below] - y[below])
    penalty[above] = (2.0 / alpha) * (y[above] - upper[above])
    return float(np.mean(width + penalty))


def evaluate_interval(construct_fn, make_model, X_tr, y_tr, X_te, y_te, alpha=ALPHA):
    """Bangun selang prediksi sambil mengukur CT (detik) dan MU (MB),
    lalu hitung PCR, AIW, WS. construct_fn = jackknife_interval /
    jackknife_plus_interval."""
    tracemalloc.start()
    t0 = time.perf_counter()
    lower, upper, y_pred = construct_fn(make_model, X_tr, y_tr, X_te, alpha=alpha)
    ct = time.perf_counter() - t0
    mu_peak = tracemalloc.get_traced_memory()[1] / 1e6  # MB
    tracemalloc.stop()
    metrics = {
        "PCR": pcr(lower, upper, y_te),
        "AIW": aiw(lower, upper),
        "WS": winkler_score(lower, upper, y_te, alpha=alpha),
        "CT": ct,
        "MU": mu_peak,
    }
    return metrics, lower, upper, y_pred


def run_all_combinations(numeric_idx, X_tr, y_tr, X_te, y_te, seed=0, alpha=ALPHA,
                         algorithms=None, collect_detail=False):
    """Jalankan algoritme terpilih x 2 metode; kembalikan list baris metrik.
    algorithms: list nama algoritme (default: ketiganya).
    collect_detail: bila True, kembalikan (rows, detail) dengan detail berisi
    y_true, y_pred, lower, upper, method, model per amatan uji."""
    factories = get_model_factories(numeric_idx, seed=seed)
    if algorithms is not None:
        factories = {k: v for k, v in factories.items() if k in algorithms}
    methods = {
        "jackknife": jackknife_interval,
        "jackknife_plus": jackknife_plus_interval,
    }
    rows = []
    detail = []
    for algo, make_model in factories.items():
        for method_name, construct_fn in methods.items():
            m, lower, upper, y_pred = evaluate_interval(
                construct_fn, make_model, X_tr, y_tr, X_te, y_te, alpha=alpha
            )
            m.update({"algoritme": algo, "metode": method_name})
            rows.append(m)
            if collect_detail:
                for j in range(len(y_te)):
                    detail.append({
                        "model": algo,
                        "method": method_name,
                        "y_true": float(y_te[j]),
                        "y_pred": float(y_pred[j]),
                        "lower": float(lower[j]),
                        "upper": float(upper[j]),
                    })
    if collect_detail:
        return rows, detail
    return rows
