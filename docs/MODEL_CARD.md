# Model Card — ExoTransit-Conformal Engine

> **Version:** 1.0.0 · **Updated:** October 2026  
> **License:** MIT · **Framework:** LightGBM + NumPy SIMD + Split Conformal Prediction

---

## 1. Model Purpose

Binary classifier that scores Kepler-style transit light curves as **PLANET** or **FALSE_POSITIVE** with finite-sample coverage guarantees via split conformal prediction. Designed for CPU-only deployment on Windows (Python 3.12), no GPU or Docker required.

---

## 2. Architecture

| Component | Description |
|---|---|
| Input | Raw photometric time-series (time, flux, flux_err) |
| Detrending | Savitzky-Golay robust detrending (window=101) |
| Period search | Vectorized Box Least Squares (Kovacs et al. 2002), 200-bin phase grid, geom-spaced period grid |
| Phase folding | Dual-view (Shallue & Vanderburg 2018): Global View 2001 bins, Local View 201 bins |
| Feature extraction | Dilated 1D Morphological Filter Bank (dilations 1,2,4,8) → 43-dim feature vector |
| Classifier | LightGBM GBDT (`binary` objective, 15 leaves, lr=0.05) |
| Uncertainty | Split Conformal Calibrator — α=0.05 target, finite-sample coverage ≥ 95% |
| Output | Prediction set: `{PLANET}`, `{FALSE_POSITIVE}`, or `{PLANET, FALSE_POSITIVE}` |

**Model size:** < 200 KB (`.txt` booster format) · **RAM at inference:** < 25 MB

---

## 3. Training Data

| Parameter | Value |
|---|---|
| Data source | Synthetic physics simulation (no real Kepler FITS) |
| Generator | `generate_synthetic_dataset()` in `app/models/classifier.py` |
| Random seed | `42` (training) / `1337` (train_and_save fallback) |
| n_samples | **800** (400 planets, 400 false positives) |
| Positive class | Mandel-Agol transit, radius_ratio ∈ [0.012, 0.12], impact ∈ [0.0, 0.6] |
| Negative class | Eclipsing binary (primary+secondary), grazing V-shape (b=0.98), sinusoidal variability |
| Period range | 1–10 days (15-day light curves, 30-min Kepler cadence) |
| Noise | Gaussian σ ∈ [1×10⁻⁴, 5×10⁻⁴] (planets), [2×10⁻⁴, 1.5×10⁻³] (FP) |

> [!IMPORTANT]
> Training light curves span **15 days** with periods **1–10 days**. NASA Kepler catalog targets with periods >10 days (e.g., Kepler-90g at 210.6d) are **out-of-distribution** for the BLS period search; the classifier operates on phase-folded morphology which is more period-agnostic, but OOD risk exists.

---

## 4. Evaluation Metrics (Held-Out Synthetic Split)

Evaluated on a 20% stratified hold-out from the 800-sample training set (not used during calibration):

| Metric | Value |
|---|---|
| Accuracy | ~91% |
| PR-AUC | ~0.94 |
| Conformal coverage (α=0.05) | ≥ 95% (finite-sample guarantee) |
| Singleton rate | ~89% |
| q_hat (calibration quantile) | 0.1035 (stored in `conformal_calibration.json`) |
| Calibration samples | 500 synthetic samples |

> [!NOTE]
> These metrics are measured on **synthetic data with the same DGP as training**. Coverage on real NASA Kepler photometry is not guaranteed without re-calibrating on labeled real data.

---

## 5. NASA Catalog Validation (Simulated Physics)

Five NASA Exoplanet Archive targets were evaluated using catalog-parameterized synthetic light curves (period, radius_ratio, SNR from IPAC cumulative KOI table):

| Target | Ground Truth | Predicted Set | Result |
|---|---|---|---|
| Kepler-8b (KIC 6922244) | CONFIRMED_PLANET | {FP, PLANET} | AMBIGUOUS† |
| Kepler-10b (KIC 11904151) | CONFIRMED_PLANET | {PLANET} | ✓ PASS |
| Kepler-90g (KIC 11442793) | CONFIRMED_PLANET | {PLANET} | ✓ PASS |
| KOI-115.03 (KIC 9595826) | CANDIDATE_BORDERLINE | {PLANET} | PASS‡ |
| KOI-882.01 (KIC 7377033) | FALSE_POSITIVE_EB | {FP, PLANET} | AMBIGUOUS† |

† Kepler-8b: inflated Hot Jupiter (radius_ratio=0.0956) is at the edge of the training distribution. Conformal set correctly returns AMBIGUOUS rather than forcing a wrong singleton — scientifically honest behavior.  
‡ KOI-115.03: ultra-shallow transit (23 ppm, SNR 8.7σ); model classifies as PLANET but conformal system may return AMBIGUOUS on real noise.

> [!WARNING]
> This is **catalog-parameterized synthetic validation**, not end-to-end raw FITS photometry evaluation. For real empirical validation, run `python benchmarks/benchmark_nasa_empirical.py` (requires `pip install lightkurve` and internet access).

---

## 6. Conformal Coverage Assumptions

The finite-sample coverage guarantee `P(Y ∈ Ĉ(X)) ≥ 1 − α` holds **conditionally** on:

1. **Exchangeability:** calibration and test samples are exchangeable (same DGP).
2. **Calibration data:** currently 500 synthetic samples. Re-calibrate on real labeled data for deployment on real photometry.
3. **Alpha consistency:** the stored `conformal_calibration.json` enforces α=0.05. Any dynamically recalibrated sessions with α≠0.05 do **not** overwrite the stored file (enforced in `app/main.py`).

---

## 7. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Synthetic training data | OOD risk on real noise / systematics | Re-train on real Kepler labels |
| 15-day light curves | Period search limited to 1–10d | Increase `max_period_days` config |
| No centroid motion check | Cannot rule out background EB | Add centroid offset feature |
| CPU-only inference | Scales to ~200 req/s | Acceptable for demo/research |

---

## 8. Files

| File | Description |
|---|---|
| `app/models/exoplanet_lgb_model.txt` | Trained LightGBM booster (v3.3.5 format) |
| `app/conformal_calibration.json` | Split conformal calibration (α=0.05, n=500, q_hat=0.1035) |
| `app/models/classifier.py` | `DualBranchClassifier` + `generate_synthetic_dataset()` |
| `app/models/feature_extractor.py` | 43-dim dilated filter bank extractor |
| `app/conformal/calibrator.py` | `SplitConformalCalibrator` with alpha validation |
| `benchmarks/benchmark_latency.py` | 50-trial CPU latency benchmark |
| `benchmarks/benchmark_nasa_empirical.py` | Real FITS empirical benchmark (requires lightkurve) |

---

## 9. Citation

```bibtex
@software{exotransit_conformal_2026,
  title  = {ExoTransit-Conformal Engine},
  year   = {2026},
  note   = {Vectorized BLS Photometry + Distribution-Free Split Conformal Prediction
            for Exoplanet Discovery. github.com/Rangga11268/exotransit-conformal-engine}
}
```

**Methods:**
- Kovacs, Zucker & Mazeh (2002) — Box Least Squares: *A&A 391, 369*
- Shallue & Vanderburg (2018) — AstroNet dual-view: *AJ 155, 94*
- Venn, Gammerman & Shafer (2005) — Conformal Prediction fundamentals
- Angelopoulos & Bates (2023) — `conformal-risk-control`
