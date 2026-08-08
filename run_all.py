"""
run_all.py
==========
Menjalankan seluruh analisis data empiris dan simulasi menggunakan:

1. Random Forest dengan 50 pohon.
2. Gradient Boosting dengan 50 estimator/iterasi boosting.
3. Support Vector Regression dengan parameter pada conformal_common.py.
4. Transformasi logaritma natural pada peubah respons untuk seluruh model.
5. Selang prediksi Jackknife dan Jackknife+ pada skala respons asli.

Kode ini selalu menjalankan konfigurasi penuh:
- seluruh data latih empiris;
- 10 skenario rasio p/n;
- 100 pengulangan untuk setiap skenario simulasi.

Jalankan:
    python run_all.py
"""

import sys
import time

import pandas as pd

import empirical_analysis as emp
import simulation_analysis as sim


RUN_EMPIRICAL = True
RUN_SIMULATION = True


def _banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def main():
    t_start = time.perf_counter()

    # Konfigurasi penuh ditetapkan langsung, tanpa mode cepat.
    emp.MAX_TRAIN = None
    sim.RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    sim.N_REPS = 100

    print(">>> MODE PENUH: RF=50 pohon, GB=50 estimator, seluruh target menggunakan log")

    if RUN_EMPIRICAL:
        _banner("[1/2] ANALISIS DATA EMPIRIS")
        t0 = time.perf_counter()
        try:
            emp.main()
            elapsed = time.perf_counter() - t0
            print(f"[waktu] analisis empiris selesai dalam {elapsed:,.1f} detik")
        except Exception as exc:
            print(f"[GAGAL] analisis empiris: {exc}", file=sys.stderr)

    if RUN_SIMULATION:
        _banner("[2/2] ANALISIS DATA SIMULASI")
        t0 = time.perf_counter()
        try:
            sim.main()
            elapsed = time.perf_counter() - t0
            print(f"[waktu] analisis simulasi selesai dalam {elapsed:,.1f} detik")
        except Exception as exc:
            print(f"[GAGAL] analisis simulasi: {exc}", file=sys.stderr)

    _banner("RINGKASAN GABUNGAN")
    try:
        if RUN_EMPIRICAL:
            print("\n--- Hasil empiris (hasil_empiris.csv) ---")
            print(pd.read_csv("hasil_empiris.csv").to_string(index=False))

        if RUN_SIMULATION:
            print("\n--- Ringkasan simulasi rata-rata ---")
            sim_df = pd.read_csv("hasil_simulasi_ringkasan.csv")
            print(sim_df.to_string(index=False))
            print("\nTabel per-ulangan lengkap: hasil_simulasi.csv")
    except FileNotFoundError as exc:
        print(f"[info] berkas ringkasan belum lengkap: {exc}")

    total = time.perf_counter() - t_start
    _banner(
        f"SELESAI — total waktu {total:,.1f} detik "
        f"({total / 60:,.1f} menit)"
    )


if __name__ == "__main__":
    main()
