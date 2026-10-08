"""Few-shot adaptation: how many labelled examples from the new condition do we need?

Sections 8.6 to 8.9 of the report establish that accuracy collapses when the model meets
a condition it was never trained on, and that augmentation recovers only a small part of
that. This asks the question a person deploying the system would actually ask instead.
Given a model already trained on the conditions available, how many labelled examples
from the new orientation, position or person does it take before the system becomes
usable again.

The experiment reuses the checkpoints saved by the cross-domain runs, so it costs
seconds per point rather than a full retraining.

Fairness matters here and is easy to get wrong. A pool of the largest k samples per
class is reserved out of the held-out test set first, and every value of k draws its
adaptation examples from inside that pool. Accuracy is always measured on the test set
with the whole pool removed, so the evaluation set is identical for k equal to zero and
for k equal to twenty, and the numbers are directly comparable to each other. They are
not directly comparable to the headline table, which evaluates on the full test set, and
the k equal to zero row is there so the curve has its own baseline.

    /home/sarthak/miniconda3/envs/op05/bin/python code/adapt.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "WiFi-CSI-Sensing-Benchmark"))
RESULTS = os.path.join(HERE, "..", "results")
CKPT = os.path.join(RESULTS, "ckpt")
CACHE = os.path.join(HERE, "..", "data", "cache", "widar_x.npy")

# A ladder from one labelled example per gesture up to 200, which for the held out
# orientation is roughly half of everything available in the new condition.
KS = [0, 1, 5, 20, 50, 100, 200, 300]
POOL_PER_CLASS = max(KS)


def freeze_to_head(model):
    """Freeze everything except the final linear layer, and return its parameters.

    This is the half of the comparison that isolates where the failure lives. Retraining
    only the last layer keeps the features the model learned from the source conditions
    and can move nothing but the decision boundaries. Retraining everything lets the
    features change too. If only the second one recovers accuracy, then the features are
    what failed to transfer, which is a different and much more awkward problem than a
    misplaced boundary.
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    for p in model.parameters():
        p.requires_grad_(False)
    for p in last.parameters():
        p.requires_grad_(True)
    return list(last.parameters())


def stratified_pool(labels, idx, per_class, rng):
    """Reserve per_class samples of every class, and return (pool, rest).

    The pool is ordered so that the first k of each class form a subset of the first
    k + 1, which is what lets every value of k share one evaluation set.
    """
    pool = []
    for c in np.unique(labels[idx]):
        have = idx[labels[idx] == c]
        take = rng.permutation(have)[:per_class]
        pool.append(take)
    pool_flat = np.concatenate(pool)
    rest = np.setdiff1d(idx, pool_flat)
    return pool, rest


def main():
    import train as T
    import splits

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LeNet")
    ap.add_argument("--protocols", nargs="*",
                    default=["cross-orientation", "cross-position", "cross-user"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="much lower than training. The head starts from the trained "
                         "weights, so a large step throws away what it already knew. "
                         "We swept this and 1e-4 was the best of the values tried")
    ap.add_argument("--modes", nargs="*", default=["head", "full"],
                    help="head retrains only the last layer, full retrains everything. "
                         "Running both is the point: the contrast between them says "
                         "whether the features or the decision boundary are at fault")
    ap.add_argument("--lr-full", type=float, default=3e-4,
                    help="learning rate when the whole network is being retrained")
    ap.add_argument("--seeds", type=int, default=3,
                    help="k is small, so which examples are drawn matters. Each point "
                         "is averaged over this many draws")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, _ = splits.load_meta()
    labels = meta[:, splits.COLS["label"]].astype(np.int64)
    out_rows = []

    for protocol in args.protocols:
        ck = os.path.join(CKPT, f"{args.model}_{protocol}_s0.pt")
        if not os.path.exists(ck):
            print(f"skipping {protocol}, no checkpoint at {os.path.basename(ck)}")
            continue
        s = splits.build(meta, protocol)
        splits.check(meta, s)
        _, test_idx = s["cross"]
        k_classes = s["n_classes"]
        state = torch.load(ck, map_location="cpu")

        print(f"\n{protocol}: {len(test_idx)} held-out samples, {k_classes} classes")
        for mode in args.modes:
            lr = args.lr_full if mode == "full" else args.lr
            print(f"  [{mode}] learning rate {lr}")
            for k in KS:
                # a large k eats the evaluation set, so stop before it is too small
                # to mean anything. cross-user is the binding case at 374 per class.
                if k and k * k_classes > len(test_idx) - 500:
                    print(f"    k={k:<4d} skipped, would leave under 500 to score on")
                    continue
                accs = []
                for seed in range(args.seeds):
                    rng = np.random.default_rng(1000 + seed)
                    pool, rest = stratified_pool(labels, test_idx, POOL_PER_CLASS, rng)
                    shots = (np.concatenate([p[:k] for p in pool]) if k
                             else np.array([], int))

                    model = T.build_model(args.model, k_classes)
                    model.load_state_dict(state)
                    model = model.to(device)

                    if k:
                        ds = T.WidarCache(shots, labels, CACHE, aug=True, seed=seed)
                        dl = torch.utils.data.DataLoader(
                            ds, batch_size=min(64, len(shots)), shuffle=True,
                            num_workers=0)
                        tune = (list(model.parameters()) if mode == "full"
                                else freeze_to_head(model))
                        opt = torch.optim.Adam(tune, lr=lr)
                        crit = nn.CrossEntropyLoss()
                        model.train()
                        for _ in range(args.epochs):
                            for x, y in dl:
                                opt.zero_grad(set_to_none=True)
                                crit(model(x.to(device)), y.to(device).long()).backward()
                                opt.step()

                    el = torch.utils.data.DataLoader(
                        T.WidarCache(rest, labels, CACHE), batch_size=256, num_workers=2)
                    acc, f1, _ = T.evaluate(model, el, device, k_classes)
                    accs.append(acc)
                    del model
                    torch.cuda.empty_cache()

                m, sd = float(np.mean(accs)), float(np.std(accs))
                print(f"    k={k:<4d} {k * k_classes:>5d} labelled   "
                      f"accuracy {m:.4f} +/- {sd:.4f}", flush=True)
                out_rows.append(dict(tag="endsem-adapt", dataset="Widar",
                                     protocol=protocol, model=args.model, k=k,
                                     n_shots=int(k * k_classes), n_eval=int(len(rest)),
                                     n_classes=k_classes, acc=round(m, 4),
                                     acc_sd=round(sd, 4), seeds=args.seeds,
                                     epochs=args.epochs, mode=mode, lr=lr))

    if out_rows:
        with open(os.path.join(RESULTS, "adaptation.jsonl"), "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(out_rows)} rows to results/adaptation.jsonl")


if __name__ == "__main__":
    main()
