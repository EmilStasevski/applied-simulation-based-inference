
import subprocess, sys, os

scripts = [
    ("Phase 1: Preprocessing",        "preprocess_data.py"),
    ("Phase 2: ESM-2 Embeddings",     "compute_embeddings.py"),
    ("Phase 3: Training + Evaluation", "pipeline_entry.py"),
]

for name, script in scripts:
    print(f"\n{'='*60}\n  {name} — {script}\n{'='*60}")
    if not os.path.isfile(script):
        print(f"  ❌ {script} not found, skipping.")
        continue
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"  ❌ {name} failed (exit code {result.returncode})")
        sys.exit(1)

print(f"\n✅ Pipeline complete. Results in data/empirical_processed/")