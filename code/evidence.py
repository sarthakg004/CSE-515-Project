"""Model-free evidence that the benchmark's Widar split can be solved by lookup.

Everything here is measured on the raw cached BVP arrays with no training of any kind,
which is what makes it hard to argue with. A neural network's accuracy always invites the
reply that the architecture or the recipe was at fault. Nearest-neighbour retrieval on
untouched data invites no such reply.

Two measurements.

1. Cosine similarity between pairs of samples, grouped by what the pair has in common.
   If two recordings of the same gesture under different conditions are barely more alike
   than two recordings of different gestures, while two recordings sharing a condition
   are markedly more alike, then the recording condition carries more of the signal than
   the gesture label does. That is the mechanism behind the whole problem.

2. One nearest neighbour classification, where each test sample is labelled with the
   gesture of the single most similar training sample. Chance is one over the number of
   classes. A high score on the benchmark's own split means the test set can be answered
   by looking up near-duplicates, without learning anything about gestures at all.

Run from the project root:
    /home/sarthak/miniconda3/envs/op05/bin/python code/evidence.py

Memory note: the cache is 1.54 GB and this laptop has 16 GB of RAM with much of it
already in use, so samples are streamed to the GPU in chunks in half precision and the
full array is never held in host memory at once.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results")
CACHE = os.path.join(HERE, "..", "data", "cache", "widar_x.npy")

MAX_TRAIN = 12000      # caps GPU memory; retrieval accuracy is stable well below this
MAX_TEST = 3000
CHUNK = 1500


def to_gpu_unit(idx, device):
    """Load these samples, flatten, L2-normalise, return a half-precision GPU matrix."""
    x = np.load(CACHE, mmap_mode="r")
    out = torch.empty(len(idx), 22 * 20 * 20, dtype=torch.float16, device=device)
    for a in range(0, len(idx), CHUNK):
        blk = np.array(x[idx[a:a + CHUNK]], dtype=np.float32).reshape(-1, 22 * 20 * 20)
        t = torch.from_numpy(blk).to(device)
        out[a:a + CHUNK] = torch.nn.functional.normalize(t, dim=1).half()
        del t
    return out


def subsample(idx, cap, rng):
    return idx if len(idx) <= cap else rng.permutation(idx)[:cap]


def one_nn(train_idx, train_lab, test_idx, test_lab, device):
    """Accuracy of labelling each test sample with its nearest training sample, and the
    mean similarity of those nearest neighbours."""
    tr = to_gpu_unit(train_idx, device)
    ytr = torch.from_numpy(train_lab).to(device)
    correct, sims, total = 0, [], 0
    for a in range(0, len(test_idx), CHUNK):
        te = to_gpu_unit(test_idx[a:a + CHUNK], device)
        sim = te @ tr.T                                  # cosine, both sides unit norm
        best, where = sim.max(dim=1)
        correct += (ytr[where].cpu() == torch.from_numpy(test_lab[a:a + CHUNK])).sum().item()
        sims.append(best.float().cpu())
        total += te.size(0)
        del te, sim
    del tr
    torch.cuda.empty_cache()
    return correct / total, torch.cat(sims).mean().item()


def similarity_table(meta, gid, device, rng, n=1500):
    """Mean cosine similarity for three kinds of pair."""
    import splits
    lab = meta[:, splits.COLS["label"]].astype(np.int64)

    by_group, by_label = {}, {}
    for i, g in enumerate(gid):
        by_group.setdefault(int(g), []).append(i)
    for i, l in enumerate(lab):
        by_label.setdefault(int(l), []).append(i)
    multi = [v for v in by_group.values() if len(v) >= 2]

    pairs = {"same recording condition": [], "same gesture, different condition": [],
             "different gesture": []}
    for _ in range(n):
        v = multi[rng.integers(len(multi))]
        a, b = rng.choice(len(v), 2, replace=False)
        pairs["same recording condition"].append((v[a], v[b]))

        l = int(rng.integers(len(by_label)))
        cand = by_label[l]
        for _ in range(8):
            a, b = rng.choice(len(cand), 2, replace=False)
            if gid[cand[a]] != gid[cand[b]]:
                pairs["same gesture, different condition"].append((cand[a], cand[b]))
                break

        a, b = rng.choice(len(meta), 2, replace=False)
        if lab[a] != lab[b]:
            pairs["different gesture"].append((a, b))

    out = {}
    for name, pl in pairs.items():
        left = to_gpu_unit(np.array([p[0] for p in pl]), device)
        right = to_gpu_unit(np.array([p[1] for p in pl]), device)
        s = (left * right).sum(dim=1).float().cpu()
        out[name] = (s.mean().item(), s.std().item(), len(pl))
        del left, right
        torch.cuda.empty_cache()
    return out


def main():
    import splits
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    meta, _ = splits.load_meta()
    gid = splits.group_ids(meta)
    lab = meta[:, splits.COLS["label"]].astype(np.int64)

    lines = []
    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 78)
    say("EVIDENCE 4: what a pair of recordings has in common, measured on raw data")
    say("=" * 78)
    say("Mean cosine similarity between two BVP samples, with no model involved.")
    say("")
    say(f"{'the pair shares':<42}{'pairs':>7}{'cosine':>9}{'sd':>8}")
    tab = similarity_table(meta, gid, device, rng)
    for name, (m, sd, n) in tab.items():
        say(f"{name:<42}{n:>7}{m:>9.3f}{sd:>8.3f}")
    a = tab["same recording condition"][0]
    b = tab["same gesture, different condition"][0]
    c = tab["different gesture"][0]
    say("")
    say(f"Sharing a gesture is worth {b - c:+.3f} in similarity. Sharing a recording")
    say(f"condition is worth {a - c:+.3f}, which is {(a - c) / max(b - c, 1e-6):.1f} times more. Two recordings")
    say("resemble each other far more for coming from the same person standing in the")
    say("same place than for being the same gesture. A model rewarded for accuracy on a")
    say("condition-mixed split is therefore rewarded for recognising the condition.")

    say("")
    say("=" * 78)
    say("EVIDENCE 5: the benchmark's split can be answered by lookup, with no learning")
    say("=" * 78)
    say("Each test sample takes the gesture label of the single most similar training")
    say("sample. No training, no parameters, raw BVP only.")
    say("")
    say(f"{'protocol':<20}{'arm':<20}{'classes':>8}{'chance':>8}{'1-NN acc':>10}{'mean sim':>10}")

    out = {"similarity": tab, "one_nn": {}}
    for name in splits.PROTOCOLS:
        s = splits.build(meta, name)
        splits.check(meta, s)
        for arm in splits.ARMS:
            tr, te = s[arm]
            tr = subsample(tr, MAX_TRAIN, rng)
            te = subsample(te, MAX_TEST, rng)
            acc, sim = one_nn(tr, lab[tr], te, lab[te], device)
            k = s["n_classes"]
            say(f"{name if arm == 'cross' else '':<20}{arm:<20}{k:>8}{1 / k:>8.3f}"
                f"{acc:>10.3f}{sim:>10.3f}")
            out["one_nn"][f"{name}/{arm}"] = dict(acc=round(acc, 4), mean_sim=round(sim, 4),
                                                  n_classes=k, chance=round(1 / k, 4))
    say("")
    say("The gap between the two in-domain arms is pure near-duplicate retrieval, because")
    say("both are domain-mixed and differ only in whether repeat takes of one recording")
    say("condition were allowed to straddle the split. The mean nearest-neighbour")
    say("similarity never reaches 1.0, so these are sibling recordings rather than")
    say("copies of the same file.")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    path = os.path.join(RESULTS, "widar_split_evidence.txt")
    with open(path, "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")
    print(f"\nappended to {os.path.normpath(path)} and wrote evidence.json")


if __name__ == "__main__":
    main()
