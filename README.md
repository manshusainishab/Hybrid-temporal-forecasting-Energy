# Hybrid Temporal Forecasting for Energy

A three-phase study on forecasting hourly electricity demand from the **PJME** (PJM East
region) load dataset. Each phase introduces a more expressive model class and a tighter
evaluation protocol; the project culminates in a **hybrid statistical + deep learning**
forecaster (Phase 3) that is benchmarked against both prior phases on an identical test split.

## Repository Structure

```text
.
├── dataset/
│   └── PJME_hourly.csv                                  # 145k rows of hourly load (MW)
├── report/
│   ├── Hybrid_temporal_forecaster_phase1.pdf            # Phase 1 write-up (ARIMA)
│   └── Hybrid_temporal_forcaster_phase2_report.pdf      # Phase 2 write-up (CNN-BiLSTM-Attn)
├── Hybrid_Temporal_Forcasting_(2) (1).ipynb             # Phase 1 notebook — ARIMA baseline
├── Phase2_Hybrid_Temporal_Forecasting (1) (1).ipynb     # Phase 2 notebook — Deep Learning
├── Phase3_Hybrid_Temporal_Forecasting.ipynb             # Phase 3 notebook — Hybrid (this phase)
└── README.md
```

## Dataset

The dataset is **PJME_hourly.csv** — estimated energy consumption in megawatts for the
PJM East region, recorded hourly from **2002-01-01** to **2018-08-03** (≈ 145,000 rows).

| Field | Description |
| --- | --- |
| `Datetime` | Hourly timestamp (used as index after parsing) |
| `PJME_MW`  | Electricity load in megawatts |

**Preprocessing** (identical across all phases):
- Datetime parsed and set as index, sorted, duplicates removed (keep first).
- Strict hourly frequency enforced via `asfreq('h')`.
- Missing values (30 NaNs) filled with the median of the load.

## Phases at a Glance

| Phase | Approach | Test MAE (MW) | Test RMSE (MW) | Notebook |
| :---: | --- | :---: | :---: | --- |
| 1 | ARIMA(3,1,3) — pure statistical baseline | 4986.03 | 6498.28 | [Phase 1](Hybrid_Temporal_Forcasting_%282%29%20%281%29.ipynb) |
| 2 | CNN-BiLSTM with additive attention — direct 24-h forecast | 1488.81 | 2045.43 | [Phase 2](Phase2_Hybrid_Temporal_Forecasting%20%281%29%20%281%29.ipynb) |
| **3A** | **Residual hybrid** (ARIMA + NN on residuals, Zhang 2003) | _run notebook_ | _run notebook_ | [Phase 3](Phase3_Hybrid_Temporal_Forecasting.ipynb) |
| **3B** | **Weighted ensemble** (linear stacking of Phase-1 + Phase-2) | _run notebook_ | _run notebook_ | [Phase 3](Phase3_Hybrid_Temporal_Forecasting.ipynb) |

> Phase 3 numbers are reported once you run the notebook end-to-end on Colab GPU.

## Phase 1 — ARIMA Baseline

- **Stationarity tests:** ADF (stationary) vs KPSS (non-stationary) — interpreted as
  *seasonally non-stationary* (a strong signal that pure ARIMA is insufficient).
- **Order selection:** `pmdarima.auto_arima` on the last 20,000 training samples → ARIMA(3, 1, 3).
- **Train/Test split:** 80 / 20 chronological.
- **Result:** MAE = 4986 MW, RMSE = 6498 MW. The single-block forecast quickly reverts to the
  series mean, which the writeup attributes to ARIMA's inability to model the dual seasonality
  (24-h daily + 168-h weekly).

## Phase 2 — Deep Learning (CNN-BiLSTM with Additive Attention)

- **Architecture:** dual-kernel Conv1D block (k=3, 5) → 2-layer Bidirectional LSTM → Bahdanau
  additive attention → MLP decoder predicting all 24 steps directly (MIMO).
- **Input window:** 168 h (full week, captures both seasonalities).
- **Output window:** 24 h direct multi-step (avoids recursive error accumulation).
- **Split:** 80 / 10 / 10 (train / val / test) chronological.
- **Scaling:** RobustScaler fit on train only (median/IQR — robust to demand spikes).
- **Training:** Adam + Huber loss, CosineAnnealingWarmRestarts, gradient clipping, early
  stopping (patience = 12). Converges in ~17 epochs.
- **Result:** MAE = 1489 MW, RMSE = 2045 MW, MAPE = 4.69 %. ~67 % MAE reduction over Phase 1.

## Phase 3 — Hybrid Forecaster (this phase)

Phase 2 already substantially outperforms Phase 1, but a pure deep model is opaque and prone to
over-confidence on regime shifts. Phase 3 tests two principled hybrid strategies on the same
80 / 10 / 10 split as Phase 2 — so all metrics are directly comparable.

### 3A — Residual Decomposition (Zhang, 2003)

Decompose the series as `y(t) = L(t) + N(t) + ε(t)` where ARIMA estimates `L̂(t)` and a CNN-BiLSTM-Attention model is trained to predict the residuals `r(t) = y(t) − L̂(t)`. Final prediction:

```
ŷ_hybrid(t) = ARIMA(t) + NN(residuals)
```

- ARIMA(3,1,3) fitted on train only.
- Out-of-sample ARIMA forecasts produced via 24-hour rolling blocks with
  `statsmodels`' `append(refit=False)` — production-style, parameter-frozen refitting.
- The DL component is identical in architecture to Phase 2; only the *target signal* changes.

### 3B — Weighted Ensemble (Linear Stacking)

Treat both phases as black-box forecasters and learn convex combination weights on the **validation set** via constrained linear regression:

```
ŷ_stack(t) = w_a · ŷ_ARIMA(t) + w_d · ŷ_DL(t) + b      (w_a, w_d ≥ 0)
```

The learned weights are themselves diagnostic — `w_a ≈ 0` indicates ARIMA carries no signal
the deep learner has not already absorbed.

### What you get from running the notebook

- ARIMA-only metrics on the **same 10 % test partition** as Phase 2 (apples-to-apples).
- A standalone Phase-2 reproduction (used as the DL component in Hybrid-B).
- A residual model and the reconstructed Hybrid-A forecast.
- A Hybrid-B stacker fit on validation and evaluated on test.
- A four-way comparison table, bar chart, per-horizon MAE plot, multi-day overlay plot,
  and per-model residual histograms.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels pmdarima torch
```

For Phase 2 and Phase 3, a CUDA-capable GPU is strongly recommended (the notebooks default
to Colab's `/content/PJME_hourly.csv` path — adjust to `./dataset/PJME_hourly.csv` if running
locally).

### Running Phase 3

1. Open [Phase3_Hybrid_Temporal_Forecasting.ipynb](Phase3_Hybrid_Temporal_Forecasting.ipynb)
   in Google Colab (GPU runtime) or a local Jupyter environment with CUDA.
2. Execute cells sequentially. The full pipeline:
   - Loads PJME and reproduces the 80/10/10 split.
   - Fits ARIMA(3,1,3) and produces rolling 24-h forecasts on val + test.
   - Trains a standalone CNN-BiLSTM-Attention model (≈ 15–20 epochs).
   - Trains a residual CNN-BiLSTM-Attention model on ARIMA residuals.
   - Fits the linear stacker on validation predictions.
   - Computes and visualizes the four-way comparison.

End-to-end runtime is dominated by the two DL trainings (~10–20 min on a Colab T4).

## Future Work

- **Exogenous regressors** — temperature, calendar (holiday/weekend), day-type embeddings.
  Phase-2 diagnostics traced the bulk of remaining error to evening peak hours and
  weather-driven regime shifts, both of which require external signals.
- **SARIMA(p,d,q)(P,D,Q,24)** in place of plain ARIMA — explicit daily seasonality should
  leave a smaller, more noise-like residual for the DL component.
- **Probabilistic forecasts** — switch from point estimates to quantile or distributional
  outputs (Temporal Fusion Transformer, DeepAR) to quantify uncertainty.
- **Online retraining** — periodically refit on the most recent window to track distribution
  drift (heatwaves, post-COVID demand patterns, EV adoption).

## References

1. Zhang, G. P. (2003). *Time series forecasting using a hybrid ARIMA and neural network model.* Neurocomputing, 50, 159–175.
2. Wolpert, D. H. (1992). *Stacked generalization.* Neural Networks, 5(2), 241–259.
3. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
4. Bahdanau, D., Cho, K. & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
5. Lai, G., Chang, W.-C., Yang, Y. & Liu, H. (2018). *Modeling Long- and Short-term Temporal Patterns with Deep Neural Networks (LSTNet).* SIGIR.
6. Lim, B., Arık, S. Ö., Loeff, N. & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting.
7. Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR.
8. Ben Taieb, S., Bontempi, G., Atiya, A. F. & Sorjamaa, A. (2012). *A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition.* Expert Systems with Applications, 39(8), 7067–7083.
