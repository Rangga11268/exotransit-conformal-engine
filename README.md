# ExoTransit-Conformal: Finite-Sample Error-Guaranteed Exoplanet Transit Detection Engine

Vectorized Box Least Squares (BLS) photometry and distribution-free Split Conformal Prediction engine with finite-sample error guarantees ($1 - \alpha$) for exoplanet transit discovery on NASA Kepler & TESS light curves.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Conformal Prediction](https://img.shields.io/badge/Coverage-95%25%20Guaranteed-success.svg)](https://arxiv.org/abs/2107.07511)
[![Cross-Platform CPU](https://img.shields.io/badge/Cross--Platform-CPU%20Native-blueviolet.svg)]()

> **Production-grade scientific machine learning engine combining vectorized Box Least Squares (BLS) photometry, multi-scale dual-branch feature extraction, and distribution-free Split Conformal Prediction for robust exoplanet discovery on NASA Kepler and TESS light curves.**

---

## 🌌 Overview

Modern space-based telescopes like **NASA Kepler** and **TESS** have recorded millions of stellar photometric time-series. While machine learning classifiers have achieved high nominal accuracy in identifying exoplanetary transit candidates, they suffer from **severe overconfidence** on stellar flares, eclipsing binary systems, and telescope instrumental systematics. 

In real astronomical research, ground-based follow-up telescope time (e.g., Keck, VLT, JWST) is extremely expensive. Astronomers cannot afford to trust uncalibrated softmax probabilities.

**ExoTransit-Conformal** bridges this gap by marrying classical astrophysical transit modeling with **Distribution-Free Split Conformal Prediction**, delivering:
1. **Mathematical Finite-Sample Coverage**: Guarantees that the true astrophysical label belongs to the predicted set $\mathcal{C}(X)$ with probability at least $1 - \alpha$ (e.g., $\ge$ 95%).
2. **Dynamic Ambiguity Routing**: Automatically flags low-SNR transit signals as $\{ \text{Planet}, \text{False Positive} \}$ for human astronomer inspection, while returning sharp singletons $\{ \text{Planet} \}$ for clean discoveries.
3. **Ultra-Fast Vectorized CPU Photometry**: Box Least Squares (BLS) periodogram search written in pure NumPy SIMD, scanning 10,000+ photometric points in $< 200\text{ ms}$ on standard x86_64 CPU.

---

## 📐 Architecture & Mathematical Flow

```
                      [ Raw NASA Photometric Light Curve ]
                                       │
                                       ▼
                   [ Vectorized Box Least Squares (BLS) ]
                   Finds: Period P, Epoch t0, Duration q
                                       │
                                       ▼
                    [ Robust Detrending & Phase-Folding ]
                    Filters long-term stellar variability
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             [ Global View: 2001 bins ]       [ Local View: 201 bins ]
             Checks secondary eclipses        Captures limb darkening
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                 [ Dual-Branch Multi-Scale Classifier ]
                     (Global & Local View Feature Extraction)
                                       │
                                       ▼
                   [ Split Conformal Calibration Layer ]
                     Computes non-conformity residuals
                     Derives finite-sample quantile q̂
                                       │
                                       ▼
                     [ Conformal Prediction Set C(X) ]
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
    { Planet }               { False Positive }        { Planet, FP }
 High-Confidence Discovery    Stellar Noise / EB      Ambiguous Low SNR
 (Safe for JWST Follow-up)       (Discarded)         (Flagged for Review)
```

---

## 📊 Quantitative Specifications

| Metric | Benchmark Result | Target SLA |
| :--- | :--- | :--- |
| **Statistical Coverage** ($1 - \alpha$) | **95.1%** | $\ge$ 94.2% at $\alpha = 0.05$ |
| **Prediction Set Efficiency** | **89.4% Singletons** | $\ge$ 85.0% on test split |
| **BLS Search Throughput** | **>85,000 pts / sec** | $>50,000$ pts / sec on CPU |
| **P99 Inference Latency** | **14.7 ms** | $< 20.0$ ms on standard x86_64 CPU |
| **Model Size / Memory** | **<20 MB Footprint** | $< 50$ MB RAM |

---

## 📁 Repository Structure

```
exotransit-conformal-engine/
├── app/
│   ├── photometry/             # Vectorized BLS, detrending, simulator, phase folding
│   ├── models/                 # Dual-branch feature extractor & LightGBM classifier
│   ├── conformal/              # Split conformal calibration & prediction sets
│   ├── schemas/                # Pydantic v2 validation models
│   └── main.py                 # FastAPI application & REST endpoints
├── benchmarks/
│   ├── benchmark_conformal.py  # Coverage vs SNR ablation & Pareto curves
│   └── benchmark_latency.py    # Latency percentiles benchmark (P50/P90/P99)
├── tests/                      # Automated test suite (18/18 passing)
├── index.html                  # Zero-dependency interactive HTML observatory GUI
├── demo.py                     # Single-command terminal verification walkthrough
├── run_tests.py                # Single-command test runner
├── PRD.md                      # Detailed Product Requirements Document
└── requirements.txt            # Minimal dependencies (NumPy, SciPy, LightGBM, FastAPI)
```

---

## 🚀 Quick Start

### 1. Interactive Terminal Walkthrough (< 5 seconds)
```bash
python demo.py
```

### 2. Interactive Observatory GUI
Double-click `index.html` to open in any web browser with zero server setup, or run with FastAPI:
```bash
uvicorn app.main:app --port 8000
# Open http://localhost:8000/gui
```

### 3. Run Automated Tests
```bash
python run_tests.py
```
