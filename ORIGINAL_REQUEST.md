# Original User Request & Implementation Prompt for Execution Agent

## 2026-10-18T12:30:00Z

Build **ExoTransit-Conformal**, a production-grade, ultra-lightweight, CPU-optimized exoplanet transit detection engine combining vectorized Box Least Squares (BLS) astrophysical photometry with multi-scale 1D Dilated Residual Deep Learning and distribution-free Split Conformal Prediction with finite-sample error guarantees ($1 - \alpha$).

Working directory: `D:\Ngoding\AI_ML_Projects\exotransit-conformal-engine`  
Target Environment: Windows 11 / Python 3.12 (Zero-Docker Native Vectorized Execution)  
Integrity mode: demo & production

## Architectural Specifications

### R1. Vectorized Photometry & Analytic Signal Processing (`app/photometry/`)
- Implement vectorized Box Least Squares (BLS) periodogram search in pure NumPy SIMD (`bls.py`), finding transit period $P \in [0.5, 30.0]\text{ days}$, epoch $t_0$, and duration $q$ across $10,000+$ photometric points in $<200\text{ms}$.
- Implement adaptive running-median / Savitzky-Golay detrending (`detrending.py`) preserving transit depth.
- Implement dual-view phase-folding (`phase_fold.py`): Global View ($2,001$ bins) and Local Transit View ($201$ bins).
- Implement Mandel-Agol analytic transit light curve simulator (`simulator.py`) with quadratic limb darkening and stochastic noise injection.

### R2. Dual-Branch 1D Dilated ResNet Architecture (`app/models/`)
- Build PyTorch 1D Dilated Residual Neural Network (`resnet1d.py`) taking Global View $(B, 1, 2001)$ and Local View $(B, 1, 201)$, processing through dilated convolutions (dilation rates $1, 2, 4, 8$) with skip connections, GELU activations, and feature fusion into a $64$-dim latent vector.
- Classification head outputting calibrated 2-class softmax probabilities ($0$: False Positive / Stellar Noise, $1$: Confirmed Exoplanet Transit).
- Pre-bake calibrated lightweight model weights or an ultra-fast deterministic pre-trained checkpoint initialization so the pipeline runs instantly without needing hours of GPU training.

### R3. Split Conformal Prediction Layer (`app/conformal/`)
- Implement non-conformity score calculation ($s_i = 1 - \hat{f}(X_i)_{Y_i}$) on calibration samples.
- Compute finite-sample corrected empirical quantile:
  $$\hat{q} = \text{Quantile}\left(\{s_i\}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
- Output rigorous prediction sets $\mathcal{C}(X) \subseteq \{0, 1\}$ with user-specified coverage $1 - \alpha$ (default $95\%$).
- Categorize outputs into `SINGLETON_PLANET` $\{1\}$, `SINGLETON_FALSE_POSITIVE` $\{0\}$, and `AMBIGUOUS_REVIEW_REQUIRED` $\{0, 1\}$.

### R4. FastAPI Production Serving Layer (`app/main.py`)
- High-throughput async REST endpoints:
  - `POST /v1/detect`: Ingests time-series flux array, executes BLS period extraction, dual-view binning, ResNet forward pass, and conformal prediction set generation. Returns JSON with period, transit depth, radius ratio $R_p/R_\star$, softmax scores, and conformal decision.
  - `POST /v1/calibrate`: Allows updating the calibration quantile $\hat{q}$ given new calibration residuals.
  - `GET /v1/health`: System health and latency telemetry.

### R5. Ablation, Benchmarks & Unit Test Suite (`benchmarks/` & `tests/`)
- Build `benchmarks/benchmark_conformal.py`: Profiles coverage error $|P(Y \in \mathcal{C}(X)) - (1 - \alpha)|$ and prediction set size across SNR variations ($2\sigma$ to $15\sigma$).
- Build `benchmarks/benchmark_latency.py`: Measures P50, P90, and P99 latency on CPU ($<20\text{ms}$ target).
- Build automated test suite (`tests/`): 100% test pass rate across photometry, BLS, model tensors, conformal guarantees, and FastAPI endpoints.
- Build single-command interactive terminal demo (`demo.py`): Generates simulated noisy NASA Kepler/TESS light curves, runs BLS detection, performs conformal classification, and displays ASCII transit curves and calibration metrics in $<15\text{ seconds}$.

## Acceptance Criteria
- [ ] 100% native Windows execution using Python 3.12 without requiring Docker or CUDA.
- [ ] Empirical coverage on test sets satisfies $P(Y \in \mathcal{C}(X)) \ge 94.2\%$ at nominal $\alpha = 0.05$.
- [ ] P99 end-to-end inference latency on CPU is strictly $< 20\text{ms}$.
- [ ] Full unit test suite passes with `pytest tests/ -v`.
- [ ] `demo.py` runs out of the box with zero errors.
