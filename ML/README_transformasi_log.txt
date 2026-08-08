TRANSFORMASI LOG, VALIDASI SILANG, DAN OUTPUT SISAAN
===================================================

Perubahan utama:
1. Random Forest, Gradient Boosting, dan SVR dilatih pada log(EXPEND).
2. Prediksi otomatis dikembalikan ke skala asli/rupiah menggunakan exp.
3. MAE, RMSE, MAPE, R2, residual, dan absolute_error dihitung pada skala rupiah.
4. Evaluasi data empiris sekarang dilengkapi validasi silang 5 lipatan pada
   DATA LATIH menggunakan KFold(shuffle=True, random_state=42).
5. RMSE (VS) dan R2 (VS) pada tabel publikasi adalah rata-rata metrik dari
   lima lipatan. Metrik per lipatan disimpan terpisah agar dapat diaudit.
6. Prediksi out-of-fold dan sisaan validasi silang disimpan dalam CSV.
7. Prediksi dan sisaan data uji disimpan dalam CSV serta divisualisasikan:
   - diagram_pencar_aktual_dugaan_data_uji.png
   - diagram_sisaan_dugaan_data_uji.png
8. Nilai RMSE pada tabel publikasi dinyatakan dalam juta rupiah; file metrik
   lengkap tetap menyimpan nilai pada satuan rupiah.
9. Pada simulasi, Random Forest pembangkit respons tetap dilatih pada
   log(EXPEND), dan respons simulasi dibatasi minimal 1.
10. Standardisasi peubah penjelas numerik tetap hanya diterapkan pada SVR.

Menjalankan evaluasi empiris:
    python empirical_ml_only.py

Opsi:
    python empirical_ml_only.py --cv-folds 5
    python empirical_ml_only.py --algorithms gradient_boosting random_forest svr
    python empirical_ml_only.py --no-plots
