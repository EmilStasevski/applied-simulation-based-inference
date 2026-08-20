# preprocess_data.py
import pandas as pd
import numpy as np
import os

# Create the processed asset destination folder dynamically
os.makedirs("data/empirical_processed", exist_ok=True)

raw_csv_path = "data/empirical_raw/2018-06-06-ss.cleaned.csv"

print(f"📖 Reading raw Kaggle Q3/Q8 CSV dataset from: {raw_csv_path}...")
# Reading only columns necessary for character & label mapping to preserve RAM
df = pd.read_csv(raw_csv_path, usecols=["seq", "sst3"])

# Enforce uniform max-length constraints matching your 16GB VRAM profiling target
MAX_LEN = 512
df = df[df["seq"].str.len() <= MAX_LEN].dropna().reset_index(drop=True)

# Sample a clean, fast validation tracking array
df = df.sample(n=2000, random_state=42).reset_index(drop=True)

fasta_out_path = "data/empirical_processed/test_sequences.fasta"
print(f"📝 Writing parsed text streams to: {fasta_out_path}...")
with open(fasta_out_path, "w") as f:
    for idx, row in df.iterrows():
        f.write(f">seq_{idx}\n{row['seq']}\n")

labels_out_path = "data/empirical_processed/test_labels.npy"
print(f"💾 Binarizing and structural-padding secondary states into: {labels_out_path}...")
# Mapping categorical sequence classes: H (Alpha-Helix) = 1, all others (Sheet, Coil) = 0
label_map = {'H': 1, 'E': 0, 'C': 0, '_': 0}

processed_labels = []
for sst in df["sst3"]:
    encoded = [label_map.get(char, 0) for char in sst]
    # Continuous right-padding to lock shapes onto the identical uniform dimension
    padded = encoded + [0] * (MAX_LEN - len(encoded))
    processed_labels.append(padded)

# Save as zero-overhead binary NumPy array
np.save(labels_out_path, np.array(processed_labels))
print("✅ Phase 1 data parsing complete! Ground truths isolated perfectly.")