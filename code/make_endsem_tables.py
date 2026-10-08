"""Turn the end-term runs into the tables that go into the end-term report.

Reads results/baselines.jsonl and results/evidence.json, writes
results/endsem_tables.md. Nothing here is typed by hand, so the report and the
experiments cannot drift apart. Safe to run while the sweep is still going: any table
whose runs have not finished yet is simply left out, and the header says how many runs
the file was built from.

    /home/sarthak/miniconda3/envs/op05/bin/python code/make_endsem_tables.py
"""
import json
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

ARMS = ["in_domain", "in_domain_grouped", "cross"]
ARM_NAME = {"in_domain": "Benchmark style",
            "in_domain_grouped": "Grouped in domain",
            "cross": "Cross domain"}
PROTO_NAME = {"cross-orientation": "Held out orientation",
              "cross-position": "Held out position",
              "cross-user": "Held out volunteers"}
MODELS = ["LeNet", "GRU", "ViT", "ResNet18"]
ACC = "val_sel_acc"          # the honest column: epoch chosen on validation


def read(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def split_protocol(p):
    base, _, arm = p.partition("/")
    name, _, held = base.partition(":")
    return name, held, (arm or "cross")


def main():
    runs = [r for r in read("baselines.jsonl") if str(r.get("tag", "")).startswith("endsem")]
    if not runs:
        print("no end-term runs recorded yet")
        return
    ev = {}
    p = os.path.join(RESULTS, "evidence.json")
    if os.path.exists(p):
        ev = json.load(open(p))

    out = ["# End term result tables", "",
           f"Generated from {len(runs)} recorded end-term runs in "
           "`results/baselines.jsonl`, plus `results/evidence.json`.",
           "Regenerate with `code/make_endsem_tables.py`.", "",
           "Accuracy is quoted at the epoch chosen by a validation set carved out of the",
           "training data at recording-condition level. It is never the best epoch on the",
           "test set, which would be peeking. Table 7 shows what that choice costs.", ""]

    # ---- Table 1: the three-arm decomposition, the headline of the whole project
    idx = {}
    for r in runs:
        if r["tag"] != "endsem" or r["protocol"] == "original":
            continue
        name, held, arm = split_protocol(r["protocol"])
        if not held:
            idx[(r["model"], name, arm)] = r

    rows = []
    for model in MODELS:
        for proto in PROTO_NAME:
            got = [idx.get((model, proto, a)) for a in ARMS]
            if not all(got):
                continue
            a, b, c = (g[ACC] for g in got)
            rows.append([model, PROTO_NAME[proto], f"{a:.3f}", f"{b:.3f}", f"{c:.3f}",
                         f"{b - a:+.3f}", f"{c - b:+.3f}", f"{c - a:+.3f}"])
    if rows:
        out += ["## Table 1. The accuracy drop, split into its two causes", "",
                "Every arm of a row uses exactly the same samples and the same test size,",
                "so the only thing that changes across a row is what the split holds back.",
                "The benchmark style arm is a random split over samples, which is what",
                "SenseFi ships. The grouped arm is also random and still contains every",
                "volunteer, position and orientation, but no repeat take of one recording",
                "condition is allowed to straddle it. The cross arm holds out a whole",
                "condition.", "",
                "| Model | Protocol | Benchmark style | Grouped in domain | Cross domain "
                "| Cost of removing repeat takes | Cost of the domain shift | Total |",
                "|---|---|---|---|---|---|---|---|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        out.append("")

    # ---- Table 2: does it matter which value was held out?
    loo = defaultdict(dict)
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["model"] != "LeNet" or arm != "cross":
            continue
        if r["tag"] == "endsem-loo" and held:
            loo[name][int(held)] = r[ACC]
        elif r["tag"] == "endsem" and not held:
            loo[name][5] = r[ACC]           # the default protocols hold out value 5
    blocks = []
    for name in ("cross-orientation", "cross-position"):
        d = loo.get(name, {})
        if len(d) >= 4:
            ks = sorted(d)
            blocks.append((name, ks, [d[k] for k in ks]))
    if blocks:
        out += ["## Table 2. Leave one condition out, every value in turn", "",
                "LeNet, cross arm only. If the drop were a property of the dataset as a",
                "whole the row would be flat. It is not.", ""]
        for name, ks, vs in blocks:
            out += [f"| {PROTO_NAME[name].replace('Held out ', '').capitalize()} held out | "
                    + " | ".join(str(k) for k in ks) + " | Spread |",
                    "|---" * (len(ks) + 2) + "|",
                    "| Accuracy | " + " | ".join(f"{v:.3f}" for v in vs)
                    + f" | {max(vs) - min(vs):.3f} |", ""]

    # ---- Table 3: recovery
    base = {p: idx[("LeNet", p, "cross")][ACC] for p in PROTO_NAME
            if ("LeNet", p, "cross") in idx}
    def rec_label(r):
        lab = ("augment " + str(int(r.get("aug_deg", 20))) + " deg"
               if r.get("augment") else "")
        if r.get("mixup"):
            lab = (lab + " + mixup").strip(" +") if lab else "mixup"
        return lab

    # Only LeNet on the cross arm belongs in the recovery table. The in-domain runs and
    # the ResNet18 run share a protocol base name with these, so without filtering on
    # both model and arm they silently overwrite the numbers they are meant to check.
    rec, side = defaultdict(dict), []
    for r in runs:
        if r["tag"] != "endsem-rec":
            continue
        name, _, arm = split_protocol(r["protocol"])
        if r["model"] == "LeNet" and arm == "cross":
            rec[rec_label(r)][name] = r[ACC]
        else:
            side.append((r, name, arm))
    if rec and base:
        cols = [p for p in PROTO_NAME if p in base]
        out += ["## Table 3. How much of the loss each recovery method wins back", "",
                "LeNet on the cross arm. The first row is the same model with no recovery",
                "applied, and every later row is the change against it. A negative number",
                "means the method made things worse, which is reported as readily as a",
                "positive one.", "",
                "| Method | " + " | ".join(PROTO_NAME[c] for c in cols) + " |",
                "|---" * (len(cols) + 1) + "|",
                "| No recovery | " + " | ".join(f"{base[c]:.3f}" for c in cols) + " |"]
        for label in sorted(rec):
            cells = []
            for c in cols:
                v = rec[label].get(c)
                cells.append(f"{v:.3f} ({v - base[c]:+.3f})" if v is not None else "-")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")

    # ---- the checks those extra runs were for: does recovery cost in-domain accuracy,
    #      and does it work on another architecture
    if side:
        srows = []
        for r, name, arm in sorted(side, key=lambda t: (t[0]["model"], t[1])):
            key = (r["model"], name, arm)
            ref = idx.get(key)
            base_txt = f"{ref[ACC]:.3f}" if ref else "-"
            delta = f"{r[ACC] - ref[ACC]:+.3f}" if ref else "-"
            srows.append([r["model"], PROTO_NAME.get(name, name),
                          ARM_NAME.get(arm, arm), rec_label(r),
                          base_txt, f"{r[ACC]:.3f}", delta])
        out += ["## Table 4. Does the recovery cost anything, and does it transfer", "",
                "A method that raises cross domain accuracy is only worth having if it "
                "does not",
                "quietly cost in domain accuracy to get there. The first rows apply the "
                "same",
                "augmentation to the grouped in domain arm. The last row repeats it on a "
                "different",
                "architecture.", "",
                "| Model | Protocol | Arm | Method | Without | With | Change |",
                "|---|---|---|---|---|---|---|"]
        out += ["| " + " | ".join(r) + " |" for r in srows]
        out.append("")

    # ---- Table 5: model-free retrieval, no training at all
    if ev.get("one_nn"):
        out += ["## Table 5. The same split answered by lookup, with no model", "",
                "Each test sample is given the gesture of the single most similar training",
                "sample, measured by cosine similarity on the raw data. No training and no",
                "parameters are involved, so nothing here can be blamed on an architecture",
                "or a learning rate.", "",
                "| Protocol | Arm | Classes | Chance | Nearest neighbour accuracy |",
                "|---|---|---|---|---|"]
        order = {a: i for i, a in enumerate(ARMS)}
        for k, v in sorted(ev["one_nn"].items(),
                           key=lambda kv: (list(PROTO_NAME).index(kv[0].split("/")[0])
                                           if kv[0].split("/")[0] in PROTO_NAME else 9,
                                           order.get(kv[0].split("/")[1], 9))):
            proto, arm = k.split("/")
            out.append(f"| {PROTO_NAME.get(proto, proto)} | {ARM_NAME[arm]} | "
                       f"{v['n_classes']} | {v['chance']:.3f} | {v['acc']:.3f} |")
        out.append("")
    if ev.get("similarity"):
        out += ["## Table 6. What makes two recordings look alike", "",
                "| The pair shares | Pairs | Mean cosine similarity |", "|---|---|---|"]
        for name, (m, sd, n) in ev["similarity"].items():
            out.append(f"| {name.capitalize()} | {n} | {m:.3f} |")
        out.append("")

    # ---- Table 6: what the choice of epoch is worth
    sel = [r for r in runs if r["tag"] == "endsem" and r["protocol"] != "original"]
    if sel:
        out += ["## Table 7. What the choice of reporting epoch is worth", "",
                "Three ways of reading the same runs. The final epoch is what SenseFi's",
                "own script reports, since it trains for a fixed number of epochs and",
                "tests once at the end. The validation selected column is what this report",
                "quotes. The oracle column is the best epoch measured on the test set,",
                "which is not a number anyone could obtain in practice and is shown only",
                "to size the temptation.", "",
                "| Reading | Mean accuracy over "
                f"{len(sel)} cross-domain and in-domain runs |", "|---|---|"]
        for label, field in [("Final epoch, as SenseFi reports", "final_acc"),
                             ("Validation selected, as this report quotes", "val_sel_acc"),
                             ("Best test epoch, an oracle", "best_acc")]:
            out.append(f"| {label} | {sum(r[field] for r in sel) / len(sel):.3f} |")
        out.append("")

    # ---- Table 7: the benchmark's double softmax
    fix = [r for r in runs if r["tag"] == "endsem-softmax"]
    if fix:
        out += ["## Table 8. The benchmark's CNN+GRU applies softmax twice", "",
                "`Widar_CNN_GRU` ends in `nn.Softmax`, and the training loop then applies",
                "`nn.CrossEntropyLoss`, which softmaxes again. The same mistake is present",
                "in the UT-HAR and NTU-Fi versions of that model. Removing the layer is a",
                "one line change and it is worth this much accuracy.", "",
                "| Version | Accuracy |", "|---|---|"]
        mid = [r for r in read("baselines.jsonl")
               if r.get("tag") == "baseline" and r["dataset"] == "Widar"
               and r["model"] == "CNN+GRU"]
        if mid:
            out.append(f"| As shipped, softmax applied twice | {mid[0]['best_acc']:.3f} |")
        out.append(f"| Trailing softmax removed | {fix[0]['best_acc']:.3f} |")
        out.append("")

    path = os.path.join(RESULTS, "endsem_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwritten to {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
