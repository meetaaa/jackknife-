KODE REVISI RF50 + GRADIENT BOOSTING 50
=========================================

Perubahan utama:
1. Peubah respons ditransformasi dengan logaritma natural pada seluruh model:
   random forest, gradient boosting, dan support vector regression.
2. Prediksi model dikembalikan secara otomatis ke skala asli menggunakan exp.
3. Random forest pembangkit respons simulasi juga dilatih pada log(EXPEND),
   kemudian prediksinya dikembalikan ke skala asli sebelum galat ditambahkan.
4. Respons simulasi dipotong pada nilai minimum 1 agar tetap positif.
5. RandomForestRegressor menggunakan n_estimators=50.
6. GradientBoostingRegressor menggunakan n_estimators=50 dan learning_rate=0.05.
7. Nama algoritme keluaran menjadi "gradient_boosting".
8. Mode QUICK/QUICK_TEST dihilangkan.
9. run_all.py selalu menjalankan konfigurasi penuh:
   - seluruh data latih empiris;
   - rasio p/n 0.2 sampai 2.0;
   - 100 pengulangan per skenario.

File data yang tetap diperlukan pada folder kerja:
- bogor.csv
- metadata.xlsx

Cara menjalankan:
    python run_all.py

Catatan:
Prosedur leave-one-out untuk 10 skenario x 100 pengulangan tetap sangat berat
secara komputasi meskipun jumlah pohon/estimator telah dikurangi menjadi 50.


CATATAN TRANSFORMASI RESPONS
----------------------------
Transformasi dilakukan di dalam TransformedTargetRegressor. Oleh karena itu:
- proses pelatihan menggunakan log(y);
- predict() langsung menghasilkan nilai pada skala rupiah;
- sisaan leave-one-out, batas selang, PCR, AIW, dan WS tetap dihitung pada
  skala rupiah, sehingga hasil dapat ditafsirkan langsung.
