# hmm_baseline.py
# =============================================================================
# Fair Forward-Backward Baseline
#
# The original baseline fed ground-truth labels as "observations" into
# predict_proba(), which is not inference — it's reading the answer key.
#
# This version runs a proper Gaussian HMM on the same ESM-2 embeddings
# that the neural network sees, making the comparison fair.
# =============================================================================

import numpy as np
from hmmlearn import hmm
from sklearn.decomposition import PCA


def run_forward_backward_baseline(
    embeddings_path: str,
    labels_path: str,
    num_sequences: int = 2000,
    pca_dim: int = 16,
    max_fit_sequences: int = 500,
    n_iter: int = 100,
):
    """
    Fit a 2-state diagonal-covariance Gaussian HMM on PCA-reduced ESM-2
    embeddings, then run Forward-Backward to infer per-residue helix
    posterior probabilities.

    Parameters
    ----------
    embeddings_path : str
        Path to the ESM-2 embedding tensor (.pt file), shape [N, L, 480].
    labels_path : str
        Path to the ground-truth binary labels (.npy), used ONLY to align
        HMM state indices (helix vs coil) after unsupervised fitting.
    num_sequences : int
        Number of sequences to evaluate.
    pca_dim : int
        Dimensionality after PCA reduction (480-D is too high for hmmlearn).
    max_fit_sequences : int
        Number of sequences used to fit the HMM (Baum-Welch is slow).
    n_iter : int
        Maximum EM iterations for Baum-Welch.

    Returns
    -------
    predictions : np.ndarray, shape [num_sequences, max_len]
        Per-residue posterior probability of the helix state.
    """
    import torch

    print("🐢 Running FAIR Forward-Backward baseline on ESM-2 embeddings...")
    print(f"   (PCA to {pca_dim}-D, fitting on {max_fit_sequences} sequences, "
          f"evaluating {num_sequences})")

    # ------------------------------------------------------------------
    # 1. Load the SAME inputs the neural network sees
    # ------------------------------------------------------------------
    embeddings = torch.load(embeddings_path, weights_only=True).cpu().numpy()
    embeddings = embeddings[:num_sequences]  # [N, 512, 480]
    labels = np.load(labels_path)[:num_sequences]  # [N, 512]  — only for index alignment

    N, L, F = embeddings.shape
    print(f"   Loaded embeddings: {embeddings.shape}, labels: {labels.shape}")

    # ------------------------------------------------------------------
    # 2. Reduce dimensionality so hmmlearn can handle it
    # ------------------------------------------------------------------
    print("   Fitting PCA...")
    flat_embeddings = embeddings.reshape(-1, F)  # [N*L, 480]
    pca = PCA(n_components=pca_dim, random_state=42)
    flat_reduced = pca.fit_transform(flat_embeddings)  # [N*L, pca_dim]
    reduced = flat_reduced.reshape(N, L, pca_dim)
    explained = pca.explained_variance_ratio_.sum()
    print(f"   PCA variance explained: {explained:.2%}")

    # ------------------------------------------------------------------
    # 3. Fit the Gaussian HMM via Baum-Welch (EM) on a subset
    # ------------------------------------------------------------------
    n_fit = min(max_fit_sequences, N)
    fit_data = reduced[:n_fit].reshape(-1, pca_dim)  # [n_fit*L, pca_dim]
    lengths = [L] * n_fit

    print(f"   Fitting GaussianHMM (n_components=2, covariance='diag')...")
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    model.fit(fit_data, lengths)
    print(f"   HMM converged: {model.monitor_.converged}")

    # ------------------------------------------------------------------
    # 4. Align state indices: hmmlearn assigns states arbitrarily.
    #    We check which HMM state correlates with label=1 (helix).
    # ------------------------------------------------------------------
    sample_preds = model.predict(reduced[0])
    corr_same = np.corrcoef(sample_preds, labels[0])[0, 1]
    swap_states = corr_same < 0  # If negative correlation, states are flipped
    helix_idx = 0 if swap_states else 1
    print(f"   State alignment: helix = HMM state {helix_idx} "
          f"(swap={swap_states}, corr={corr_same:.3f})")

    # ------------------------------------------------------------------
    # 5. Run Forward-Backward on every sequence
    # ------------------------------------------------------------------
    print("   Running posterior inference on all sequences...")
    predictions = np.zeros((N, L), dtype=np.float64)

    for i in range(N):
        try:
            posteriors = model.predict_proba(reduced[i])  # [L, 2]
            predictions[i] = posteriors[:, helix_idx]
        except Exception as e:
            print(f"   ⚠️ Sequence {i} failed: {e}")
            predictions[i] = 0.0

    print(f"✅ Fair Forward-Backward completed. Output shape: {predictions.shape}")
    return predictions


# Legacy wrapper: drop-in replacement for old call signature in pipeline_entry.py
def run_forward_backward_baseline_legacy(test_labels_path, num_sequences=2000):
    """
    OLD interface kept for backwards compatibility.
    Now delegates to the fair version using embeddings.
    Expects embeddings to be at the standard path.
    """
    import os
    data_dir = os.path.dirname(test_labels_path)
    embeddings_path = os.path.join(data_dir, "test_esm_embeddings.pt")
    return run_forward_backward_baseline(
        embeddings_path=embeddings_path,
        labels_path=test_labels_path,
        num_sequences=num_sequences,
    )


if __name__ == "__main__":
    preds = run_forward_backward_baseline(
        embeddings_path="data/empirical_processed/test_esm_embeddings.pt",
        labels_path="data/empirical_processed/test_labels.npy",
    )
    helix_fractions = preds.mean(axis=1)
    print(f"   Mean helix fraction across sequences: {helix_fractions.mean():.4f}")