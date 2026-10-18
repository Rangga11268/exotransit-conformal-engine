# Product Requirements Document (PRD): ExoTransit-Conformal Engine

> **Project Name**: ExoTransit-Conformal  
> **Repository**: `exotransit-conformal-engine`  
> **Author**: Darell Rangga Putra Rachman  
> **Role Target**: Machine Learning Engineer / AI Engineer / Scientific Data Scientist  
> **Version**: 1.0.0 (Production Release)  
> **Status**: Production Ready & Fully Verified (18/18 Automated Tests Passing)

---

## 1. Objectives & Scientific Value

Membangun mesin inferensi machine learning dan fotometri astrofisika tingkat produksi yang ringan, deterministik, dan dapat dijalankan secara native di arsitektur CPU standar (Python 3.12) tanpa ketergantungan Docker atau kluster GPU mahal.


Sistem ini dirancang untuk mendeteksi kandidat eksoplanet dari kurva cahaya fotometrik teleskop antariksa (NASA Kepler & TESS) dengan garansi matematis kepastian error (**Distribution-Free Split Conformal Prediction**).

---

## 2. Key User Personas & Use Cases

1. **Exoplanet Survey Research Scientist**:
   Membutuhkan *prediction set* dengan jaminan $95\%$ validitas statistik agar tidak menyia-nyiakan proposal observasi teleskop antariksa lanjutan (JWST / Keck / VLT) pada sinyal palsu bintang (*eclipsing binary / stellar flares*).
2. **Astronomical Alert Broker Developer**:
   Membutuhkan endpoint REST API berkecepatan tinggi ($P99 < 20\text{ms}$) yang mampu menerima deret waktu fluks fotometrik, melakukan de-trending, phase-folding, dan mengembalikan klasifikasi terkalibrasi beserta interval ketidakpastian.

---

## 3. Detailed Technical Requirements

### R1. Photometry & Signal Processing Layer (`app/photometry/`)
* **Vectorized Box Least Squares (BLS)**:
  - Memindai deret waktu fotometri mentah $(t_i, F_i)$ untuk mencari periode $P \in [0.5, 30.0]\text{ hari}$, waktu transit pusat $t_0$, dan durasi transit $q$.
  - Mengimplementasikan penghitungan *Signal Residue* ($\mathrm{SR}$) dengan NumPy broadcasting murni (kecepatan $>75.000\text{ titik/detik}$).
* **Robust Detrending & Normalization**:
  - Filter adaptif (Savitzky-Golay / Running Median) untuk mengeliminasi variabilitas jangka panjang bintang (*stellar rotational modulation*) tanpa merusak kedalaman transit (*transit depth distortion*).
* **Dual-View Phase Folding**:
  - **Global View**: Memetakan $2.001$ titik binned pada rentang fase $[-0.5, +0.5]$ untuk memverifikasi ada/tidaknya gerhana sekunder.
  - **Local View**: Memetakan $201$ titik binned pada rentang transit terfokus $[-2 \times q, +2 \times q]$ untuk membedakan kelengkungan *limb darkening* U-shape vs V-shape.
* **Analytic Transit Simulator**:
  - Generator kurva cahaya sintetis menggunakan aproksimasi analitik Mandel-Agol dengan variasi rasio $R_p/R_\star$, limb darkening quadratic $(u_1, u_2)$, dan injeksi white Gaussian noise serta flare Poisson acak untuk pengujian ketahanan (*robustness testing*).

### R2. Dual-Branch Morphological Classification Architecture (`app/models/`)
* **Multi-Scale Dilated 1D Morphological Extractor**:
  - Menggunakan dua cabang representasi paralel (Global Branch & Local Branch) yang memanfaatkan *dilated convolution filters* (dilation rate 1, 2, 4, 8) dan scientific pooling untuk mengekstraksi gradien, kelengkungan limb darkening (U-shape vs V-shape), serta rasio kedalaman gerhana sekunder.
  - Vektor fitur gabungan berdimensi 43 ($\mathbf{z} \in \mathbb{R}^{43}$) yang mencakup morfologi global (16 fitur konvolusi + 6 indikator astrofisika), morfologi lokal (12 fitur konvolusi + 4 shape tests), dan 5 skalar BLS (periode, durasi, kedalaman, SDE, duty cycle).
  - Head klasifikasi berbasis LightGBM Gradient Boosted Decision Trees yang sangat cepat di CPU dengan probabilitas output $\hat{f}(X) \in [0, 1]^2$ (Kelas 0: False Positive / Stellar Noise, Kelas 1: Verified Exoplanet Transit).
* **Lightweight CPU Footprint**:
  - Dioptimalkan untuk inferensi CPU tanpa runtime PyTorch/CUDA yang berat, dengan model size $< 200\text{ KB}$, memori footprint $< 25\text{ MB}$, dan latensi forward pass $< 5\text{ ms}$.

### R3. Split Conformal Prediction Engine (`app/conformal/`)
* **Non-Conformity Scoring Function**:
  - Skor $s_i = 1 - \hat{f}(X_i)_{Y_i}$ dihitung pada $n = 1.000$ sampel dataset kalibrasi held-out.
* **Empirical Quantile Calibration**:
  - Menghitung nilai ambang $\hat{q} = \text{Quantile}\left(\{s_i\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$ secara dinamis berdasarkan parameter toleransi error $\alpha$ yang diminta klien (default $\alpha = 0.05 \implies 95\%$ coverage).
* **Prediction Set Construction**:
  - Mengembalikan himpunan $\mathcal{C}(X) = \{y \in \{0, 1\} : \hat{f}(X)_y \ge 1 - \hat{q}\}$.
  - Mengkategorikan status keputusan:
    * `SINGLETON_PLANET`: $\mathcal{C}(X) = \{1\}$ (Temuan pasti, aman untuk follow-up).
    * `SINGLETON_FALSE_POSITIVE`: $\mathcal{C}(X) = \{0\}$ (Artefak/bintang biner).
    * `AMBIGUOUS_REVIEW_REQUIRED`: $\mathcal{C}(X) = \{0, 1\}$ (Ketidakpastian tinggi / transit noise rendah, wajib ditinjau manusia).
* **Empirical Efficiency Tracking**:
  - Melacak metrik *Average Set Size* dan *Singleton Fraction* terhadap variasi $S/N$.

### R4. Async Serving Layer (`app/main.py`)
* **FastAPI Async REST API**:
  - `POST /v1/detect`: Menerima deret waktu fluks fotometrik, mengembalikan periode transit, kedalaman transit, estimasi $R_p/R_\star$, estimasi softmax, dan himpunan konformal $\mathcal{C}(X)$.
  - `POST /v1/calibrate`: Endpoint administratif untuk memperbarui skor kuantil $\hat{q}$ saat dataset kalibrasi baru tersedia.
  - `GET /v1/health`: Status latensi, alokasi memori, dan kalibrasi aktif.

### R5. Verification, Ablation & Benchmark Suite (`benchmarks/` & `tests/`)
* **Statistical Coverage Assertions**:
  - Validasi bahwa pada 5.000 kurva cahaya uji independen, rasio cakupan empiris berada dalam rentang $[1 - \alpha - 0.01, 1 - \alpha + 0.01]$.
* **Pareto Frontier Profiling**:
  - Eksperimen ablasi: Hubungan antara transit SNR ($2\sigma$ hingga $15\sigma$) terhadap persentase ambiguitas himpunan prediksi $\{0, 1\}$.
* **CPU Latency Benchmark**:
  - Pengujian $1.000$ request kontinu untuk mencatat distribusi P50, P90, dan P99 latency.

---

## 4. Acceptance Criteria & Quality Gates

| ID | Kategori | Kriteria Penerimaan | Metode Verifikasi |
| :--- | :--- | :--- | :--- |
| **AC-1** | Portabilitas | 100% berjalan native di lingkungan CPU modern (Python 3.12) tanpa ketergantungan Docker atau CUDA GPU | `python run_tests.py` / `pytest` |
| **AC-2** | Garansi Statistik | Empirical coverage $P(Y \in \mathcal{C}(X)) \ge 94.2\%$ pada target nominal $95.0\%$ ($\alpha = 0.05$) | `tests/test_conformal.py` |
| **AC-3** | BLS Performance | Ekstraksi periode dari $10.000$ titik fotometri selesai dalam $< 250\text{ ms}$ | `tests/test_photometry.py` |
| **AC-4** | End-to-End Latency | P99 latency pipeline ML (detrending + phase fold + classifier + conformal set) $< 20\text{ ms}$ | `benchmarks/benchmark_latency.py` |
| **AC-5** | Test Coverage | Seluruh unit test lulus $100\%$ tanpa *flaky behavior* | `pytest tests/ -v` |
| **AC-6** | Demo Interaktif | File `demo.py` dapat dieksekusi sekali jalan di terminal menampilkan kurva cahaya, BLS detection, dan prediction set dalam $< 15\text{ detik}$ | `python demo.py` |

---

## 5. Technology Stack

* **Language**: Python 3.12 (Cross-Platform 64-bit)
* **Core Numerics**: NumPy 1.26+, SciPy 1.13+ (SIMD accelerated)
* **Classifier Architecture**: Dual-Branch Multi-Scale Extractor + LightGBM CPU Booster
* **API Framework**: FastAPI 0.111+, Uvicorn 0.30+, Pydantic v2
* **Testing & Profiling**: PyTest, PyTest-Asyncio

