# Overcoming Simulator Misspecification in Protein Structure Inference Using Foundation Model Embeddings

Final project for the Simulation-Based Inference course, TU Dortmund University (2026).

---

## Summary

This project builds an amortized Bayesian inference pipeline that estimates hidden Markov transition parameters governing protein secondary structure. A 2-state HMM simulator generates synthetic training data in the latent space of Meta's ESM-2 protein language model, and a BayesFlow Flow Matching network learns the posterior p(theta | embedding). The central finding is that progressively matching the simulator's noise structure to real ESM-2 statistics — mean anchoring, per-dimension variance, cross-dimension covariance, temporal correlation, and residual noise — monotonically improves both point accuracy and posterior calibration on real protein data, without ever training on real labels.

**Final results (2000 real PDB sequences):**

| Metric | Forward-Backward Baseline | Neural (Flow Matching) |
|---|---|---|
| Mean Absolute Error | 0.343 | 0.080 |
| Pearson Correlation | 0.714 | 0.904 |
| 90% CI Coverage (temp-scaled) | N/A | 90.4% |

The neural method outperforms the classical baseline by 77% on MAE while providing calibrated uncertainty intervals that the baseline cannot produce.

---

## Research Question

Standard simulation-based inference assumes the simulator accurately reflects reality. A simple 2-state HMM for protein secondary structure violates this assumption — it lacks biophysical constraints, evolutionary context, and long-range cooperative folding dynamics. Training a summary network on naive HMM output produces posteriors that collapse when applied to real proteins (MAE ~0.41).

**Hypothesis:** Projecting both simulated and real sequences into the frozen ESM-2 embedding space, and anchoring the simulator's statistical properties to the empirical distribution of those embeddings, can bridge the sim-to-real gap without requiring a biophysically realistic simulator.

---

## Architecture

The pipeline has four decoupled components, designed to fit within a 16 GB VRAM budget:

**Simulator** — A vectorized 2-state HMM generating synthetic sequences of length 512 with 480-dimensional emissions. Emissions are drawn from empirically-anchored class-conditional distributions: real ESM-2 mean profiles, rank-64 covariance factors (via SVD), residual per-dimension noise, and AR(1) temporal correlation along the sequence.

**Feature extractor (offline)** — Meta's ESM-2 (`esm2_t12_35M_UR50D`, 35M parameters). Converts real amino acid sequences into [512, 480] embeddings. Runs once; results cached to disk. Not part of the training loop.

**Summary network** — Masked self-attention pooling (with zero-padding detection) followed by a 2-layer MLP, compressing [512, 480] inputs to a 64-dimensional summary vector.

**Inference network** — BayesFlow Flow Matching density estimator mapping the 64-D summary to posterior samples of theta = [p_stay_coil, p_stay_helix].

---

## How to Run

### Prerequisites

Python 3.10+, PyTorch with CUDA, and the packages listed in `requirements.txt`. A GPU with at least 16 GB VRAM is recommended.

```bash
pip install -r requirements.txt
```

The raw dataset (`2018-06-06-ss.cleaned.csv`) must be placed in `data/empirical_raw/`. This is the Kaggle protein secondary structure dataset (Q3/Q8 labels).

### Full pipeline

```bash
python main.py
```

This runs three phases in order:

1. **Preprocessing** (`preprocess_data.py`) — Reads the raw CSV, filters to sequences <= 512 residues, samples 2000 for evaluation, writes FASTA sequences and binarized labels (H=1, else 0).

2. **Embedding extraction** (`compute_embeddings.py`) — Loads frozen ESM-2, streams FASTA sequences through it in batches, extracts and caches [2000, 512, 480] embeddings to disk. This is the slowest phase (~5 min on GPU) but only runs once.

3. **Training and evaluation** (`pipeline_entry.py`) — Builds the empirically-anchored simulator, trains the Flow Matching network with simulated-data validation and early stopping, runs posterior inference on real embeddings, computes a fair Forward-Backward baseline on the same inputs, fits post-hoc temperature scaling, and generates all plots and the final report.

### Running individual phases

```bash
python preprocess_data.py       # Phase 1 only
python compute_embeddings.py    # Phase 2 only
python pipeline_entry.py        # Phase 3 only (requires phases 1-2 outputs)
```

---

## File Structure

```
.
├── main.py                  # Pipeline orchestrator (runs phases 1-2-3)
├── preprocess_data.py       # Phase 1: CSV -> FASTA + binary labels
├── compute_embeddings.py    # Phase 2: FASTA -> frozen ESM-2 embeddings
├── pipeline_entry.py        # Phase 3: simulator + training + evaluation + report
├── hmm_baseline.py          # Fair Forward-Backward baseline (GaussianHMM on embeddings)
├── requirements.txt
├── README.md
│
└── data/
    ├── empirical_raw/                       # Raw Kaggle dataset (not tracked)
    │   ├── 2018-06-06-ss.cleaned.csv
    │   └── 2018-06-06-pdb-intersect-pisces.csv
    ├── empirical_processed/                 # All pipeline outputs
    │   ├── test_sequences.fasta             # Phase 1: protein sequences
    │   ├── test_labels.npy                  # Phase 1: ground-truth labels [2000, 512]
    │   ├── test_esm_embeddings.pt           # Phase 2: ESM-2 features [2000, 512, 480]
    │   ├── final_report.txt                 # Phase 3: full results with diagnostics
    │   ├── training_log.csv                 # Phase 3: per-checkpoint metrics
    │   ├── training_dashboard.png           # Phase 3: 6-panel training health display
    │   ├── calibration_plot.png             # Phase 3: raw vs temp-scaled calibration
    │   ├── helix_scatter_comparison.png     # Phase 3: neural vs baseline scatter
    │   └── flow_matching_loss.png           # Phase 3: training loss curve
    └── pdb_cases/                           # Individual PDB structures for case studies
        ├── 1A7F.fasta / 1A7F.pdb
        └── 1CRN.fasta / 1CRN.pdb
```

---

## Key Technical Decisions

**Why ESM-2 instead of one-hot encoding?** One-hot encoded HMM emissions live in a completely different space than real protein features. ESM-2 embeddings encode physicochemical context, evolutionary conservation, and local structural signals that a 2-state HMM cannot generate. By operating in ESM-2 space, the simulator only needs to match the *statistical* properties of real embeddings (means, covariances), not the underlying biology.

**Why a fair baseline matters.** The original Forward-Backward baseline fed ground-truth labels as observations into `predict_proba()`, achieving MAE 0.033 — but this is reading the answer key, not doing inference. The rebuilt baseline runs a GaussianHMM on PCA-reduced ESM-2 embeddings (the same inputs the neural network sees), achieving MAE 0.343. This 10x difference in baseline difficulty completely changes the interpretation of the neural method's performance.

**Why temperature scaling.** After all simulator improvements, the raw posterior is still ~5x too narrow on real data. The remaining gap comes from the 2-state HMM conflating beta-sheet and coil into one state (systematic bias of +0.05) and residual distributional mismatch. A single scalar T=4.95 stretches the posterior samples to achieve 90.4% coverage. This is reported transparently as a post-hoc correction.

---

## Simulator Evolution (Ablation)

Each row adds one change to the simulator. All other settings held constant.

| Stage | MAE | Correlation | 90% Coverage |
|---|---|---|---|
| Naive scalar emissions (+1/-1) | 0.41 | — | — |
| ESM-2 mean anchoring (480-D profiles) | 0.17 | 0.71 | — |
| + per-dimension variance | 0.14 | 0.77 | 27% |
| + rank-64 cross-dimension covariance | 0.072 | 0.91 | 50% |
| + AR(1) temporal correlation (rho=0.7) | 0.070 | 0.90 | 57% |
| + residual diagonal noise | 0.069 | 0.89 | 58% |
| + post-hoc temperature scaling (T=4.95) | 0.080 | 0.90 | 90.4% |

---

## Outputs

After a successful run, `data/empirical_processed/final_report.txt` contains the complete analysis with six sections: head-to-head comparison, temperature scaling details, raw theta diagnostics, error decomposition (bias vs. variance), calibration curves (raw and recalibrated), and training summary. The training dashboard and calibration plot provide the visual evidence.

---

## Known Limitations and Future Work

- **2-state HMM** conflates beta-sheet (E) and coil (C) into one state, causing +0.05 systematic bias. A 3-state simulator (H/E/C) would eliminate this and reduce the need for temperature scaling.
- **Temperature T=4.95 is large**, indicating substantial residual sim-to-real mismatch. The raw posterior is informative but not calibrated without post-hoc correction.
- **Rank-64 covariance** captures only 58-73% of real per-class variance. Higher rank or full covariance (if tractable) would narrow the gap.
- **Single attention head** in the summary network. Multi-head attention or a small transformer encoder could capture richer sequence-level patterns.
- **2000 test sequences** from one Kaggle dataset. Validation on the full PDB (~200k chains) would test generalization to rare folds.

---

## References

- Radev, S.T., et al. (2023). BayesFlow: Amortized Bayesian Workflows with Neural Networks.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2).
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks.
- Kuleshov, V., et al. (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression.
