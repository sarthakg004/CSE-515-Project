"""Run every baseline needed for the midterm, skipping anything already recorded.

Safe to re-run: results already in results/baselines.jsonl are not repeated, so an
interrupted sweep can simply be started again.

    /home/sarthak/miniconda3/envs/op05/bin/python code/run_baselines.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "..", "results", "baselines.jsonl")
PY = sys.executable

MODELS = ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
          "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"]

# ViT on the NTU-Fi datasets runs out of memory at batch 64 on an 8 GB card. Its input
# is 3 x 114 x 500, which becomes about 760 patch tokens, and attention is quadratic in
# that, so one attention tensor alone needs roughly 2.8 GB. Inference fits; training does
# not, because the autograd graph has to be kept as well. The evaluation batch matters
# more than the training one here, since it defaults to 256: at 256 the attention tensor
# alone is 2.75 GB. Both are lowered for these two runs.
BATCH = {("NTU-Fi_HAR", "ViT"): 16, ("NTU-Fi-HumanID", "ViT"): 16}
EVAL_BATCH = {("NTU-Fi_HAR", "ViT"): 32, ("NTU-Fi-HumanID", "ViT"): 32}

# (dataset, protocol). Widar 'original' is the in-domain number our project questions,
# so it belongs in the midterm table as the reference point.
JOBS = [
    ("UT_HAR_data", "original"),
    ("NTU-Fi_HAR", "original"),
    ("NTU-Fi-HumanID", "original"),
    ("Widar", "original"),
]


def done():
    if not os.path.exists(JSONL):
        return set()
    out = set()
    for line in open(JSONL):
        r = json.loads(line)
        if r.get("tag") == "baseline":
            out.add((r["dataset"], r["protocol"], r["model"]))
    return out


def main():
    have = done()
    todo = [(d, p, m) for d, p in JOBS for m in MODELS if (d, p, m) not in have]
    print(f"{len(have)} runs already recorded, {len(todo)} to go\n")

    t0 = time.time()
    for i, (dataset, protocol, model) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {dataset} {protocol} {model}", flush=True)
        cmd = [PY, os.path.join(HERE, "train.py"), "--dataset", dataset,
               "--model", model, "--protocol", protocol, "--max-epochs", "200",
               "--batch", str(BATCH.get((dataset, model), 64)),
               "--eval-batch", str(EVAL_BATCH.get((dataset, model), 256))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr.strip().splitlines()[-1][:120]}", flush=True)
        else:
            print("    " + r.stdout.strip().splitlines()[-1].strip(), flush=True)
    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
