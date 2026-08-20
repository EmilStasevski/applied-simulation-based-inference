import os
import torch
from transformers import AutoTokenizer, EsmModel
from Bio import SeqIO
import time


def main():
    print("🚀 Initializing ESM-2 Embedding Extraction Pipeline...")
    start_time = time.time()

    # --- Configuration ---
    MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
    HIDDEN_DIM = 480
    MAX_LEN = 512
    BATCH_SIZE = 64  # Increased for RTX 5060 Ti throughput

    FASTA_PATH = "data/empirical_processed/test_sequences.fasta"
    TENSOR_OUT_PATH = "data/empirical_processed/test_esm_embeddings.pt"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(TENSOR_OUT_PATH), exist_ok=True)

    # --- Hardware & Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {device}")
    if device.type == "cuda":
        print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")

    print(f"📦 Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # .eval() disables dropout for deterministic inference
    model = EsmModel.from_pretrained(MODEL_NAME).to(device).eval()

    # --- Data Loading (Biopython Fix) ---
    print(f"\n📖 Stream reading parsed FASTA sequences from: {FASTA_PATH}")
    try:
        # Biopython flawlessly handles multi-line sequences and broken headers
        sequences = [str(record.seq) for record in SeqIO.parse(FASTA_PATH, "fasta")]
        print(f"   Successfully loaded {len(sequences)} valid protein chains.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {FASTA_PATH}. Please check your data directory.")
        return

    if not sequences:
        print("⚠️ Warning: FASTA file was empty. Exiting.")
        return

    # --- Embedding Extraction Loop ---
    embedded_tensors = []
    print(f"\n⚡ Streaming sequences through Frozen ESM-2 layers...")

    # torch.no_grad() prevents tracking history, drastically reducing VRAM usage
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]

            # Tokenize with max_length boundaries +2 to safely allocate start/stop tokens (<cls>, <eos>)
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                max_length=MAX_LEN + 2,
                truncation=True,
                padding=True
            ).to(device)

            outputs = model(**inputs)
            # Extract features matrix shape: [Batch, Tokens, Hidden_Dimension]
            hidden_states = outputs.last_hidden_state

            # Unpack the active sequences batch from the batch matrix
            for b_idx in range(len(batch_seqs)):
                # Slice away the special bounding tokens (<cls>, <eos>)
                seq_features = hidden_states[b_idx, 1:-1, :]

                # Pad tensor to the exact uniform configuration matrix expected by the Summary Network
                padded_features = torch.zeros(MAX_LEN, HIDDEN_DIM)
                curr_len = min(seq_features.shape[0], MAX_LEN)

                # Send arrays back to CPU to prevent VRAM memory leaks
                padded_features[:curr_len, :] = seq_features[:curr_len, :].cpu()
                embedded_tensors.append(padded_features)

            print(f"   Processed sequences: {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)}")

    # --- Finalization & Caching ---
    final_tensor = torch.stack(embedded_tensors)
    torch.save(final_tensor, TENSOR_OUT_PATH)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Phase 2 complete! Finished in {elapsed_time:.2f} seconds.")
    print(f"💾 Features cached to: {TENSOR_OUT_PATH}")
    print(f"📊 Extracted Validation Tensor Geometry Shape: {final_tensor.shape} -> [Batch, Sequence, Embedding_Dim]")


if __name__ == "__main__":
    main()