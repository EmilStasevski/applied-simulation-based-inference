# diagnostics.py
# =============================================================================
# Standalone SBI diagnostics — run AFTER training, separately from the pipeline.
#
# WHY THIS FILE EXISTS
# --------------------
# The presentation shows an apparent contradiction:
#   * Slide 10  : excellent accuracy on REAL proteins (helix-fraction MAE 0.084)
#   * Slide 11  : only-just-passing SBC on SIMULATED data (KS p = 0.12 / 0.84)
# and the final_report.txt shows the real smoking gun:
#   * 90% CI coverage on REAL data = 38% (raw) — wildly overconfident
#   * mean |z| = 3.79, spread ratio = 3.26x, 12.6% of raw theta outside [0,1]
#   * temperature scaling needs T = 4.95 to reach 81.5% coverage
#
# Slide 10 and slide 11 are NOT in contradiction — they measure different things
# on different data. The mismatch this file quantifies is:
#
#     "calibrated w.r.t. the SIMULATOR"  vs.  "calibrated w.r.t. REALITY"
#
# The posterior is (marginally) calibrated on data the simulator generated, but
# the simulator is misspecified relative to real ESM-2 embeddings, so on real
# proteins the same posterior is badly overconfident. That gap IS the
# sim-to-real gap, made visible.
#
# This script uses the `sbi` package's own diagnostics (sbi.diagnostics):
#   - run_sbc / check_sbc / sbc_rank_plot   (SBC rank histograms + KS/c2st stats)
#   - run_tarp / check_tarp / plot_tarp     (expected-coverage / TARP curves)
# applied to the BayesFlow posterior through a thin NeuralPosterior adapter,
# and produces the key "restoration" plot: SIMULATED vs REAL expected-coverage
# on one axis, which is the single clearest picture of the mismatch.
#
# NOTE ON LIBRARIES: the training pipeline uses BayesFlow, and `sbi` is a
# DIFFERENT package. `sbi.diagnostics.run_sbc` expects an sbi `NeuralPosterior`.
# We therefore wrap the trained BayesFlow workflow in a minimal adapter that
# exposes `.sample()` / `.sample_batched()` so the genuine sbi functions run
# unmodified. If `sbi` is unavailable, we fall back to an equivalent manual
# rank computation so the script still produces every plot.
#
# USAGE
# -----
#   python diagnostics.py                 # trains a fresh posterior, then runs
#   python diagnostics.py --epochs 80     # control training length
#   python diagnostics.py --fast          # small run for a quick smoke test
#
# All figures + a text summary land in data/empirical_processed/diagnostics/.
# =============================================================================

import os

# Keras/BayesFlow backend must be set before those imports (mirrors pipeline).
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch

# --- Reuse the pipeline's simulator, summary net, and helix functional --------
# These live in pipeline_entry.py; importing them guarantees the diagnostics run
# against EXACTLY the same generative model and summary network as training.
from pipeline_entry import (
    VectorizedProteinHMMSimulator,
    MaskedAttentionPoolingSummaryNet,
    stationary_helix_probability,
)

import bayesflow as bf


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
DATA_DIR = "data/empirical_processed"
EMB_PATH = os.path.join(DATA_DIR, "test_esm_embeddings.pt")
LABELS_PATH = os.path.join(DATA_DIR, "test_labels.npy")
OUT_DIR = os.path.join(DATA_DIR, "diagnostics")

PARAM_LABELS = ["p_stay_coil", "p_stay_helix"]


# -----------------------------------------------------------------------------
# 1. Build empirical profiles + simulator (identical recipe to pipeline_entry)
# -----------------------------------------------------------------------------
def build_simulator(seed=0):
    """Rebuild the empirically-anchored simulator from real ESM-2 statistics.

    Returns (simulator, obs_embeddings, obs_labels). Mirrors steps [1]-[3] of
    pipeline_entry.run_unified_experiment so diagnostics use the same anchoring.
    """
    print("[setup] Loading real embeddings + labels for anchoring...")
    obs_embeddings = (
        torch.load(EMB_PATH, weights_only=True).cpu().numpy()
    )  # [N, 512, 480]
    obs_labels = np.load(LABELS_PATH)  # [N, 512]

    helix_mask = obs_labels == 1
    coil_mask = obs_labels == 0

    helix_profile = obs_embeddings[helix_mask].mean(axis=0)
    coil_profile = obs_embeddings[coil_mask].mean(axis=0)
    helix_std = obs_embeddings[helix_mask].std(axis=0)
    coil_std = obs_embeddings[coil_mask].std(axis=0)

    cov_rank = 64

    def low_rank_cov_factor(X, mean, rank):
        centered = X - mean
        n = centered.shape[0]
        if n > 50000:
            idx = np.random.default_rng(0).choice(n, 50000, replace=False)
            centered = centered[idx]
            n = 50000
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        return (Vt[:rank].T * (S[:rank] / np.sqrt(n))).astype(np.float32)

    helix_cov_factor = low_rank_cov_factor(
        obs_embeddings[helix_mask], helix_profile, cov_rank
    )
    coil_cov_factor = low_rank_cov_factor(
        obs_embeddings[coil_mask], coil_profile, cov_rank
    )

    simulator = VectorizedProteinHMMSimulator(
        helix_profile=helix_profile,
        coil_profile=coil_profile,
        helix_std=helix_std,
        coil_std=coil_std,
        helix_cov_factor=helix_cov_factor,
        coil_cov_factor=coil_cov_factor,
        rng=np.random.default_rng(seed),
    )
    return simulator, obs_embeddings, obs_labels


# -----------------------------------------------------------------------------
# 2. Train (or you can wire in a checkpoint load) the BayesFlow posterior
# -----------------------------------------------------------------------------
def train_workflow(simulator, epochs=80, batch_size=64, num_batches=100):
    """Train the same Flow-Matching workflow used in the pipeline.

    Kept deliberately close to pipeline_entry so diagnostics reflect the real
    model. If you already persist a workflow/approximator, replace this with a
    load — the diagnostics below only need `workflow.sample(conditions=...)`.
    """
    print(f"[train] Training Flow-Matching posterior for {epochs} epochs...")
    summary_net = MaskedAttentionPoolingSummaryNet()
    inference_net = bf.networks.FlowMatching()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_network=inference_net,
        summary_network=summary_net,
        inference_variables=["prior_draws"],
        summary_variables=["sim_data"],
    )
    # Same Keras-3 symbolic-build bypass as the pipeline.
    workflow.approximator._symbolic_build = lambda *a, **k: None

    workflow.fit_online(
        epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches
    )
    print("[train] Done.")
    return workflow


# -----------------------------------------------------------------------------
# 3. Adapter: expose the BayesFlow workflow as an sbi NeuralPosterior
# -----------------------------------------------------------------------------
# sbi.diagnostics.run_sbc / run_tarp only ever call `.sample(shape, x=...)` and
# (optionally) `.sample_batched(shape, x=...)` on the posterior. We subclass
# NeuralPosterior and route those to workflow.sample(conditions={"sim_data": x}).
# This lets the *genuine* sbi diagnostics run against the BayesFlow model.
def make_sbi_adapter(workflow, prior):
    from sbi.inference.posteriors.base_posterior import NeuralPosterior

    class BayesFlowSBIAdapter(NeuralPosterior):
        """Minimal shim so sbi diagnostics can query a BayesFlow workflow."""

        def __init__(self, workflow, prior):
            # Bypass NeuralPosterior.__init__ (wants a density estimator);
            # we only need the sampling surface the diagnostics touch.
            self._workflow = workflow
            self._prior = prior
            self._device = "cpu"
            # NeuralPosterior.default_x is a property with a setter that runs
            # process_x(); set the backing fields directly to avoid that path.
            self._x = None
            self._map = None

        # ---- the two methods sbi's diagnostics actually call -------------
        def sample(self, sample_shape=torch.Size([]), x=None, show_progress_bars=False, **kw):
            n = int(torch.Size(sample_shape).numel()) or 1
            x_np = _to_np(x)
            if x_np.ndim == 3 and x_np.shape[0] == 1:
                cond = {"sim_data": x_np.astype(np.float32)}
            elif x_np.ndim == 2:  # single [L, F] observation
                cond = {"sim_data": x_np[None].astype(np.float32)}
            else:
                cond = {"sim_data": x_np.astype(np.float32)}
            post = self._workflow.sample(conditions=cond, num_samples=n)
            draws = np.asarray(post["prior_draws"])  # [1, n, 2] or [n, 2]
            draws = draws.reshape(-1, draws.shape[-1])[:n]
            return torch.as_tensor(draws, dtype=torch.float32)

        def sample_batched(self, sample_shape, x, show_progress_bars=False, **kw):
            n = int(torch.Size(sample_shape).numel()) or 1
            x_np = _to_np(x)  # [B, L, F]
            cond = {"sim_data": x_np.astype(np.float32)}
            post = self._workflow.sample(conditions=cond, num_samples=n)
            draws = np.asarray(post["prior_draws"])  # [B, n, 2]
            draws = np.moveaxis(draws, 1, 0)          # -> [n, B, 2]
            return torch.as_tensor(draws, dtype=torch.float32)

        # ---- inert requirements of the abstract base --------------------
        def log_prob(self, *a, **k):
            raise NotImplementedError("Flow-Matching posterior: no closed-form log_prob here.")

        def map(self, *a, **k):
            raise NotImplementedError

        def __repr__(self):
            return "BayesFlowSBIAdapter(FlowMatching)"

    return BayesFlowSBIAdapter(workflow, prior)


def _to_np(x):
    if x is None:
        return np.empty((0,))
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# -----------------------------------------------------------------------------
# 4. Draw an SBC/TARP calibration set from the simulator (known ground truth)
# -----------------------------------------------------------------------------
def draw_calibration_set(simulator, num_datasets=300):
    """Sample (theta, x) pairs from the simulator: theta ~ prior, x ~ p(x|theta).

    These are the prior-predictive draws SBC/TARP consume. Ground-truth theta is
    known, so calibration here measures calibration W.R.T. THE SIMULATOR.
    """
    print(f"[calib] Drawing {num_datasets} simulated (theta, x) pairs...")
    batch = simulator.sample(batch_size=num_datasets)
    thetas = torch.as_tensor(batch["prior_draws"], dtype=torch.float32)  # [M, 2]
    xs = torch.as_tensor(batch["sim_data"], dtype=torch.float32)          # [M, 512, 480]
    return thetas, xs


# -----------------------------------------------------------------------------
# 5. SBC via sbi.diagnostics (with manual fallback)
# -----------------------------------------------------------------------------
def run_sbc_diagnostics(adapter, thetas, xs, num_posterior_samples, out_dir):
    print("[SBC] Running simulation-based calibration (sbi.diagnostics)...")
    summary = {}
    try:
        from sbi.diagnostics import run_sbc, check_sbc
        from sbi.analysis.plot import sbc_rank_plot

        ranks, dap_samples = run_sbc(
            thetas,
            xs,
            adapter,
            num_posterior_samples=num_posterior_samples,
            reduce_fns="marginals",
            use_batched_sampling=True,
            show_progress_bar=False,
        )
        prior_samples = thetas  # prior draws == the thetas we sampled
        stats = check_sbc(
            ranks, prior_samples, dap_samples, num_posterior_samples=num_posterior_samples
        )
        for k, v in stats.items():
            summary[k] = _to_np(v).tolist()
            print(f"       {k}: {summary[k]}")

        fig, ax = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=20,
            parameter_labels=PARAM_LABELS,
        )
        fig.suptitle("SBC Rank Histograms (sbi.diagnostics) — calibration w.r.t. SIMULATOR")
        _save(fig, out_dir, "sbc_rank_hist.png")

        # CDF view — often clearer for spotting slopes/bias than the histogram.
        fig2, ax2 = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="cdf",
            parameter_labels=PARAM_LABELS,
        )
        fig2.suptitle("SBC Rank CDF (sbi.diagnostics)")
        _save(fig2, out_dir, "sbc_rank_cdf.png")

        return _to_np(ranks), summary

    except Exception as e:  # pragma: no cover - fallback path
        warnings.warn(f"sbi SBC path failed ({e}); using manual fallback.")
        return _manual_sbc(adapter, thetas, xs, num_posterior_samples, out_dir), summary


def _manual_sbc(adapter, thetas, xs, num_posterior_samples, out_dir):
    """Manual SBC ranks: rank of each true theta among posterior draws.

    Identical statistic to sbi's run_sbc; used only if the sbi path is missing.
    """
    thetas_np = _to_np(thetas)
    M, D = thetas_np.shape
    ranks = np.zeros((M, D), dtype=int)
    for i in range(M):
        s = adapter.sample((num_posterior_samples,), x=xs[i : i + 1])
        s = _to_np(s)  # [S, D]
        ranks[i] = (s < thetas_np[i]).sum(axis=0)

    fig, axes = plt.subplots(1, D, figsize=(6 * D, 4))
    if D == 1:
        axes = [axes]
    for d in range(D):
        axes[d].hist(ranks[:, d], bins=20, color="steelblue", edgecolor="k", alpha=0.8)
        axes[d].axhline(M / 20, color="k", ls="--", alpha=0.6, label="uniform")
        axes[d].set_title(f"SBC ranks: {PARAM_LABELS[d]} (manual)")
        axes[d].set_xlabel("rank of true theta"); axes[d].legend()
    _save(fig, out_dir, "sbc_rank_hist_manual.png")
    return ranks


# -----------------------------------------------------------------------------
# 6. TARP / expected coverage via sbi.diagnostics (with manual fallback)
# -----------------------------------------------------------------------------
def run_tarp_diagnostics(adapter, thetas, xs, num_posterior_samples, out_dir, tag):
    """Return (ecp, alpha) expected-coverage curve for one dataset (sim or real)."""
    print(f"[TARP] Running expected-coverage / TARP on '{tag}' set...")
    try:
        from sbi.diagnostics import run_tarp, check_tarp
        from sbi.analysis import plot_tarp

        ecp, alpha = run_tarp(
            thetas,
            xs,
            adapter,
            references=None,
            num_posterior_samples=num_posterior_samples,
            use_batched_sampling=True,
            show_progress_bar=False,
        )
        atc, ks_pval = check_tarp(ecp, alpha)
        print(f"       [{tag}] TARP ATC={float(atc):+.4f} (0=ideal), KS p={float(ks_pval):.4f}")

        fig, ax = plot_tarp(ecp, alpha, title=f"TARP expected coverage — {tag}")
        _save(fig, out_dir, f"tarp_{tag}.png")
        return _to_np(ecp), _to_np(alpha), float(atc), float(ks_pval)

    except Exception as e:  # pragma: no cover
        warnings.warn(f"sbi TARP path failed ({e}); using manual coverage fallback.")
        ecp, alpha = _manual_expected_coverage(adapter, thetas, xs, num_posterior_samples)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(alpha, ecp, "o-", label=tag)
        ax.plot([0, 1], [0, 1], "k--", label="ideal")
        ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage")
        ax.set_title(f"Expected coverage (manual) — {tag}"); ax.legend()
        _save(fig, out_dir, f"tarp_{tag}_manual.png")
        return ecp, alpha, float("nan"), float("nan")


def _manual_expected_coverage(adapter, thetas, xs, num_posterior_samples, n_levels=20):
    """Expected-coverage on the helix FUNCTIONAL (1-D), mirroring the report.

    For each dataset: draw posterior theta, map to helix fraction, and check
    whether the true helix fraction lies in the central credible interval at
    each level. Averaged over datasets -> expected coverage curve.
    """
    thetas_np = _to_np(thetas)
    true_helix = stationary_helix_probability(thetas_np)  # [M]
    M = thetas_np.shape[0]
    levels = np.linspace(0.05, 0.95, n_levels)

    helix_samps = np.zeros((M, num_posterior_samples))
    for i in range(M):
        s = _to_np(adapter.sample((num_posterior_samples,), x=xs[i : i + 1]))
        helix_samps[i] = stationary_helix_probability(s)

    ecp = []
    for lvl in levels:
        lo = np.quantile(helix_samps, (1 - lvl) / 2, axis=1)
        hi = np.quantile(helix_samps, 1 - (1 - lvl) / 2, axis=1)
        ecp.append(np.mean((true_helix >= lo) & (true_helix <= hi)))
    return np.array(ecp), levels


# -----------------------------------------------------------------------------
# 7. THE RESTORATION / MISMATCH PLOT — simulated vs real, same axes
# -----------------------------------------------------------------------------
def restoration_plot(sim_curve, real_curve, out_dir, report_real_coverage=None):
    """Overlay expected-coverage on SIMULATED vs REAL data.

    This is the plot that answers the presentation's question: the posterior
    tracks the diagonal on simulated data (slide 11 passes) but collapses below
    it on real data (slide-10 accuracy hides a badly overconfident posterior).
    The vertical gap between the two curves = the sim-to-real calibration gap.
    """
    sim_ecp, sim_alpha = np.asarray(sim_curve[0]), np.asarray(sim_curve[1])
    real_ecp, real_alpha = np.asarray(real_curve[0]), np.asarray(real_curve[1])

    # sim (TARP) and real curves may live on DIFFERENT alpha grids (run_tarp
    # picks its own number of bins). Plot each on its own grid, and shade the
    # gap on a shared grid built by interpolating the real curve onto sim_alpha.
    order = np.argsort(real_alpha)
    real_on_sim = np.interp(sim_alpha, real_alpha[order], real_ecp[order])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="perfect calibration")
    ax.plot(sim_alpha, sim_ecp, "o-", color="seagreen", lw=2,
            label="SIMULATED data (slide 11 regime)")
    ax.plot(real_alpha, real_ecp, "s-", color="crimson", lw=2,
            label="REAL proteins (the sim-to-real gap)")
    ax.fill_between(sim_alpha, real_on_sim, sim_alpha,
                    where=(real_on_sim < sim_alpha),
                    color="crimson", alpha=0.10)

    if report_real_coverage is not None:
        ax.scatter([0.90], [report_real_coverage], color="black", zorder=5,
                   label=f"reported real 90% cov = {report_real_coverage:.0%}")

    ax.set_xlabel("credibility level (expected coverage)", fontsize=12)
    ax.set_ylabel("observed coverage", fontsize=12)
    ax.set_title("Restoration plot: calibration on SIMULATED vs REAL data\n"
                 "curve below diagonal = overconfident posterior", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax.grid(True, ls="--", alpha=0.4)
    _save(fig, out_dir, "restoration_sim_vs_real.png")


# -----------------------------------------------------------------------------
# 8. Posterior predictive check in EMBEDDING space (why real != sim)
# -----------------------------------------------------------------------------
def embedding_ppc(workflow, simulator, obs_embeddings, out_dir, n_show=400):
    """Do simulated embeddings look like real ones? If not, that's the root cause.

    We compare per-dimension summary stats of REAL ESM-2 embeddings against the
    simulator's own emissions. Systematic gaps here are the mechanism behind the
    real-data overconfidence: the network never trained on what reality looks
    like. This is the posterior-predictive-check complement SBC alone can't give.
    """
    print("[PPC] Comparing real vs simulated embedding statistics...")
    real = obs_embeddings.reshape(-1, obs_embeddings.shape[-1])
    # drop all-zero padded rows
    real = real[np.abs(real).sum(1) > 1e-6]
    idx = np.random.default_rng(0).choice(real.shape[0], min(200000, real.shape[0]), replace=False)
    real = real[idx]

    sim_batch = simulator.sample(batch_size=64)["sim_data"]
    sim = sim_batch.reshape(-1, sim_batch.shape[-1])

    real_mean, sim_mean = real.mean(0), sim.mean(0)
    real_std, sim_std = real.std(0), sim.std(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].scatter(real_mean, sim_mean, s=6, alpha=0.4, color="slateblue")
    lim = [min(real_mean.min(), sim_mean.min()), max(real_mean.max(), sim_mean.max())]
    axes[0].plot(lim, lim, "k--"); axes[0].set_title("Per-dim MEAN: real vs sim")
    axes[0].set_xlabel("real"); axes[0].set_ylabel("simulated")

    axes[1].scatter(real_std, sim_std, s=6, alpha=0.4, color="teal")
    lim = [min(real_std.min(), sim_std.min()), max(real_std.max(), sim_std.max())]
    axes[1].plot(lim, lim, "k--"); axes[1].set_title("Per-dim STD: real vs sim")
    axes[1].set_xlabel("real"); axes[1].set_ylabel("simulated")

    # 1-D marginal on the single most helix-discriminative dimension
    disc = np.argmax(np.abs(simulator.helix_profile - simulator.coil_profile))
    axes[2].hist(real[:, disc], bins=60, density=True, alpha=0.5, label="real", color="crimson")
    axes[2].hist(sim[:, disc], bins=60, density=True, alpha=0.5, label="sim", color="seagreen")
    axes[2].set_title(f"Marginal, most discriminative dim (#{disc})")
    axes[2].legend()

    fig.suptitle("Posterior predictive check in embedding space — "
                 "gaps here drive the real-data miscalibration", fontsize=13)
    _save(fig, out_dir, "embedding_ppc.png")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"       saved: {path}")


def real_data_functional_coverage(adapter, obs_embeddings, obs_labels,
                                  num_posterior_samples, chunk=128):
    """Expected-coverage curve on REAL proteins, using true helix fraction.

    Ground-truth theta is unknown for real proteins, but the FUNCTIONAL
    (helix fraction) is known from DSSP labels. We therefore evaluate coverage
    of the true helix fraction under the posterior-implied helix distribution —
    the same quantity the report/scatter (slide 10) is about, but as a full
    coverage curve rather than a single 90% number.
    """
    print("[real] Computing real-data expected-coverage on helix functional...")
    true_helix = obs_labels.mean(axis=1)  # [N]
    N = obs_embeddings.shape[0]
    levels = np.linspace(0.05, 0.95, 20)

    helix_samps = np.zeros((N, num_posterior_samples))
    start = 0
    while start < N:
        end = min(start + chunk, N)
        s = adapter.sample_batched(
            (num_posterior_samples,),
            torch.as_tensor(obs_embeddings[start:end], dtype=torch.float32),
        )
        s = _to_np(s)  # [S, B, 2]
        s = np.moveaxis(s, 0, 1).reshape(-1, 2)  # [B*S, 2]
        h = stationary_helix_probability(s).reshape(end - start, num_posterior_samples)
        helix_samps[start:end] = h
        start = end

    ecp = []
    for lvl in levels:
        lo = np.quantile(helix_samps, (1 - lvl) / 2, axis=1)
        hi = np.quantile(helix_samps, 1 - (1 - lvl) / 2, axis=1)
        ecp.append(np.mean((true_helix >= lo) & (true_helix <= hi)))
    return np.array(ecp), levels


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone SBI diagnostics (post-training).")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--num-datasets", type=int, default=300,
                        help="simulated (theta,x) pairs for SBC/TARP")
    parser.add_argument("--num-posterior-samples", type=int, default=200)
    parser.add_argument("--fast", action="store_true",
                        help="tiny run for a smoke test")
    args = parser.parse_args()

    if args.fast:
        args.epochs = 5
        args.num_datasets = 40
        args.num_posterior_samples = 50

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. simulator + real data
    simulator, obs_embeddings, obs_labels = build_simulator()

    # 2. train posterior (or swap for a checkpoint load)
    workflow = train_workflow(simulator, epochs=args.epochs)

    # 3. build a 2-D uniform prior matching theta ~ U(0.55, 0.98)^2 and adapter
    try:
        from sbi.utils import BoxUniform
        prior = BoxUniform(low=torch.tensor([0.55, 0.55]),
                           high=torch.tensor([0.98, 0.98]))
    except Exception:
        prior = None
    adapter = make_sbi_adapter(workflow, prior)

    # 4. calibration set from the simulator (known ground truth)
    thetas, xs = draw_calibration_set(simulator, num_datasets=args.num_datasets)

    # 5. SBC (sbi.diagnostics) — the slide-11 diagnostic, done properly
    ranks, sbc_summary = run_sbc_diagnostics(
        adapter, thetas, xs, args.num_posterior_samples, OUT_DIR
    )

    # 6. TARP / expected coverage on SIMULATED data
    sim_ecp, sim_alpha, sim_atc, sim_ks = run_tarp_diagnostics(
        adapter, thetas, xs, args.num_posterior_samples, OUT_DIR, tag="simulated"
    )

    # 7. Expected coverage on REAL proteins (helix functional)
    real_ecp, real_alpha = real_data_functional_coverage(
        adapter, obs_embeddings, obs_labels, args.num_posterior_samples
    )

    # 8. THE restoration plot: simulated vs real on one axis
    restoration_plot(
        (sim_ecp, sim_alpha),
        (real_ecp, real_alpha),
        OUT_DIR,
        report_real_coverage=0.3815,  # raw 90% coverage from final_report.txt
    )

    # 9. Embedding-space PPC: the mechanism behind the gap
    embedding_ppc(workflow, simulator, obs_embeddings, OUT_DIR)

    # 10. text summary
    summary_path = os.path.join(OUT_DIR, "diagnostics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("SBI DIAGNOSTICS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"SBC check (sbi.diagnostics): {sbc_summary}\n")
        f.write(f"TARP simulated: ATC={sim_atc:+.4f} (0=ideal), KS p={sim_ks:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- SBC/TARP on SIMULATED data test calibration w.r.t. the simulator.\n")
        f.write("- The restoration plot overlays REAL-data coverage: the vertical\n")
        f.write("  gap below the diagonal is the sim-to-real miscalibration that\n")
        f.write("  slide-10 accuracy (MAE 0.084) masks and slide-11 SBC can't see.\n")
        f.write("- embedding_ppc.png shows the mechanism: simulator emissions do\n")
        f.write("  not match real ESM-2 statistics, so real inputs are OOD.\n")
    print(f"\n[done] Summary: {summary_path}")
    print(f"[done] All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()