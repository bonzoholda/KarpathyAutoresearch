# Tethgard Autoresearch — Quant Trading Engine

Ini adalah sistem otomatisasi riset strategi trading kuantitatif untuk mesin "Tethgard". AI agent akan bereksperimen menciptakan, menguji, dan menyempurnakan strategi trading secara otonom menggunakan historical data crypto.

## Setup

Sebelum memulai loop eksperimen, lakukan langkah berikut bersama user:

1. **Sepakati Run Tag**: Usulkan tag berdasarkan tanggal hari ini (contoh: `quant-aug30`). Buat branch baru dari `master`: `git checkout -b autoresearch/<tag>`.
2. **Pahami File dalam Scope**:
   - `strategy_candidate.py` — **HANYA FILE INI YANG BOLEH DIEDIT AI**. Berisi logika generasi sinyal trading (`generate_signals`).
   - `evaluator.py` — Evaluator dan harness test. Memuat `strategy_candidate.py`, menguji sinyal pada data OHLCV market, dan menghitung score evaluasi (Sharpe Ratio & Penalti Drawdown). **Read-only**.
   - `main.py` — Engine orchestrator & cron scanner untuk mencari strategi pemenang via Optuna/VectorBT dan mempush ke Go Executor. **Read-only**.
   - `strategy_engine.py` — Backtesting engine menggunakan VectorBT dan Optuna. **Read-only**.
3. **Inisialisasi results.tsv**: Buat file `results.tsv` (jika belum ada) dengan header:
   ```tsv
   commit	sharpe_score	status	description

 * Jalankan Baseline: Jalankan python evaluator.py untuk mendapatkan baseline score awal sebelum melakukan modifikasi apa pun.
Experimentation
Eksperimen dilakukan secara otonom. Setiap ide strategi ditulis langsung ke dalam strategy_candidate.py, lalu dievaluasi dengan menjalankan:
python evaluator.py

Aturan Eksperimen:
 * Yang BOLEH Diubah: Modifikasi logika indikator, kalkulasi teknikal (RSI, EMA, MACD, Bollinger Bands, Volume, ATR, dll.), serta rule BUY (1), SELL (-1), atau HOLD (0) pada file strategy_candidate.py.
 * Yang TIDAK BOLEH Diubah: File evaluator.py, main.py, dan strategy_engine.py.
 * Target Utama: Mendapatkan Sharpe Ratio tertinggi dengan risiko drawdown yang terkontrol.
 * Batasan Memori & Kecepatan: Gunakan operasi tervektorisasi via pandas atau numpy. Hindari loop python murni (for-loop pada DataFrame) yang memperlambat evaluasi.
Output & Score Target
Output dari evaluator.py akan mencetak baris akhir seperti ini:
EVALUATION_SCORE: 1.8452

 * Score -999.0 menandakan strategi crash, menghasilkan sinyal pasif (tidak ada trade), atau mengalami Max Drawdown > 15%.
 * Score positif menagaskan strategi menghasilkan return yang disesuaikan dengan risiko (risk-adjusted return).
Logging Results
Catat setiap eksperimen ke file results.tsv (menggunakan separator TAB \t, BUKAN koma):
commit	sharpe_score	status	description

Kolom:
 * commit: Git commit hash singkat (7 karakter).
 * sharpe_score: Nilai EVALUATION_SCORE hasil run (misal: 1.8452). Gunakan -999.0 untuk crash/penalti.
 * status: keep (jika score membaik), discard (jika score turun/tetap), atau crash.
 * description: Penjelasan singkat strategi/indikator yang dicoba.
Contoh results.tsv:
commit	sharpe_score	status	description
a1b2c3d	0.4500	keep	baseline SMA crossover
b2c3d4e	1.2100	keep	add RSI oversold filter
c3d4e5f	-999.0	crash	div by zero on ATR calculation
d4e5f6g	0.8000	discard	switch to Stochastic Oscillator

The Experiment Loop
Jalankan eksperimen secara berulang tanpa henti (LOOP FOREVER):
 * Analisis kondisi strategi saat ini di strategy_candidate.py.
 * Tulis ide/hipotesis indikator kuantitatif baru di strategy_candidate.py.
 * Commit perubahan ke git: git commit -am "exp: <penjelasan singkat>".
 * Jalankan evaluasi: python evaluator.py > run.log 2>&1.
 * Baca score: grep "EVALUATION_SCORE:" run.log.
 * Jika terjadi error, periksa stack trace: tail -n 30 run.log. Jika mudah diperbaiki (misal typo/import error), perbaiki dan rerun. Jika logika rusak, tandai crash.
 * Catat baris baru ke results.tsv.
 * Pengambilan Keputusan:
   * KEEP: Jika sharpe_score lebih TINGGI dari pencapaian terbaik sebelumnya, pertahankan commit dan lanjutkan iterasi dari state baru ini.
   * DISCARD / CRASH: Jika sharpe_score lebih rendah atau bernilai -999.0, lakukan git reset ke commit sukses terakhir: git reset --hard HEAD~1.
 * ULANGI DARI STEP 1. JANGAN PERNAH BERHENTI ATAU MEMINTA KONFIRMASI MANUSIA.

---

### **Perubahan Utama yang Diterapkan pada File di Atas:**
1. **Target Metrik:** Mengubah target evaluasi dari `val_bpb` (pretraining LLM) menjadi `sharpe_score` (Trading Engine)[span_1](start_span)[span_1](end_span).
2. **File Scope:** Mengisolasi `strategy_candidate.py` sebagai satu-satunya file yang diizinkan untuk dimodifikasi oleh AI Agent[span_2](start_span)[span_2](end_span).
3. **Execution Command:** Mengubah *execution harness* dari `uv run train.py` menjadi `python evaluator.py`[span_3](start_span)[span_3](end_span).
4. **Kriteria Keep/Discard:** AI akan menyimpan (*keep*) strategi jika menghasilkan Sharpe Ratio lebih tinggi tanpa melanggar batasan *Max Drawdown*[span_4](start_span)[span_4](end_span).

