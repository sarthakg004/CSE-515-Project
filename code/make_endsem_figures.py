"""Draw the end-term figures from the recorded results.

Slide 10 of the course notes forbids reusing anyone else's figures, so every figure here
is drawn by us from our own measurements. Nothing is traced or copied from a paper.

Safe to run while the sweep is still going. A figure whose runs have not finished yet is
skipped with a message rather than drawn from incomplete data.

    /home/sarthak/miniconda3/envs/op05/bin/python code/make_endsem_figures.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RESULTS, "figures")

INK = "#15202b"
ACCENT = "#123a5e"
MID = "#5b8db8"
WARN = "#a4423a"
GREY = "#8b97a3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#b9c3cc",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#e3e9ee",
    "grid.linewidth": 0.8,
    "figure.dpi": 200,
})

ARMS = ["in_domain", "in_domain_grouped", "cross"]
ARM_LABEL = {"in_domain": "Benchmark style\n(random over samples)",
             "in_domain_grouped": "Grouped in domain\n(no repeat takes)",
             "cross": "Cross domain\n(condition held out)"}
PROTO = {"cross-orientation": "Orientation\nheld out",
         "cross-position": "Position\nheld out",
         "cross-user": "Volunteers\nheld out"}
ACC = "val_sel_acc"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def split_protocol(p):
    base, _, arm = p.partition("/")
    name, _, held = base.partition(":")
    return name, held, (arm or "cross")


def load():
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "baselines.jsonl"))
            if l.strip()]
    return [r for r in runs if str(r.get("tag", "")).startswith("endsem")]


def fig7_decomposition(runs):
    """The headline: the drop and the two separate things causing it."""
    idx = {}
    for r in runs:
        if r["tag"] != "endsem" or r["protocol"] == "original":
            continue
        name, held, arm = split_protocol(r["protocol"])
        if not held and r["model"] == "LeNet":
            idx[(name, arm)] = r[ACC]
    protos = [p for p in PROTO if all((p, a) in idx for a in ARMS)]
    if not protos:
        return print("  fig7 skipped, LeNet arms not finished")

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(protos))
    w = 0.26
    for i, arm in enumerate(ARMS):
        vals = [idx[(p, arm)] for p in protos]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABEL[arm],
                      color=[ACCENT, MID, WARN][i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x, [PROTO[p] for p in protos], fontsize=8.5)
    ax.set_ylabel("Test accuracy, epoch chosen on validation")
    ax.set_ylim(0, max(idx.values()) * 1.22)
    ax.legend(fontsize=7.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.21),
              frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "fig7_three_arm_decomposition.png")


def fig8_leave_one_out(runs):
    """Whether the size of the drop depends on which value was held out. It does."""
    loo = defaultdict(dict)
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["model"] != "LeNet" or arm != "cross":
            continue
        if r["tag"] == "endsem-loo" and held:
            loo[name][int(held)] = r[ACC]
        elif r["tag"] == "endsem" and not held:
            loo[name][5] = r[ACC]
    have = [p for p in ("cross-orientation", "cross-position") if len(loo.get(p, {})) >= 4]
    if not have:
        return print("  fig8 skipped, leave-one-out runs not finished")

    base = {}
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["tag"] == "endsem" and r["model"] == "LeNet" \
                and arm == "in_domain_grouped" and not held:
            base[name] = r[ACC]

    fig, axes = plt.subplots(1, len(have), figsize=(3.5 * len(have), 3.0), squeeze=False)
    for ax, name in zip(axes[0], have):
        ks = sorted(loo[name])
        vs = [loo[name][k] for k in ks]
        top = max(max(vs), base.get(name, 0))
        if name in base:
            ax.axhline(base[name], color=GREY, ls="--", lw=1.1, zorder=1)
            # sits above the line at the right edge, clear of the data labels
            ax.text(ks[-1] + 0.35, base[name] + 0.018, "nothing held out",
                    fontsize=6.8, color=GREY, ha="right")
        ax.plot(ks, vs, "-o", color=ACCENT, lw=1.8, ms=6, zorder=3)
        worst = int(np.argmin(vs))
        ax.plot([ks[worst]], [vs[worst]], "o", color=WARN, ms=8, zorder=4)
        for k, v in zip(ks, vs):
            # keep the value label off the dashed baseline where they would collide
            below = abs(v - base.get(name, -9)) < 0.05 * top
            ax.annotate(f"{v:.3f}", (k, v), fontsize=7,
                        xytext=(0, -13 if below else 8),
                        textcoords="offset points", ha="center")
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 0.45, ks[-1] + 0.45)
        ax.set_xlabel(f"Which {name.split('-')[1]} was held out")
        ax.set_ylim(0, top * 1.22)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Test accuracy")
    save(fig, "fig8_leave_one_out.png")


def fig9_model_free():
    """Why the gap exists, measured on raw data with no model involved."""
    p = os.path.join(RESULTS, "evidence.json")
    if not os.path.exists(p):
        return print("  fig9 skipped, run code/evidence.py first")
    ev = json.load(open(p))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    fig.subplots_adjust(wspace=0.45)

    names = ["different gesture", "same gesture, different condition",
             "same recording condition"]
    vals = [ev["similarity"][n][0] for n in names]
    labels = ["Nothing\n(different gestures)", "The gesture,\nnot the condition",
              "The condition,\nnot necessarily the gesture"]
    y = np.arange(3)
    ax1.barh(y, vals, color=[GREY, MID, WARN], height=0.55)
    for i, v in enumerate(vals):
        ax1.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
    ax1.set_yticks(y, labels, fontsize=7.6)
    ax1.set_xlim(0, max(vals) * 1.22)
    ax1.set_xlabel("Mean cosine similarity of a pair")
    ax1.set_title("What makes two recordings alike", fontsize=9, color=ACCENT, pad=7)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", visible=False)

    order = ["cross-orientation/in_domain", "cross-orientation/in_domain_grouped",
             "cross-orientation/cross"]
    got = [(k, ev["one_nn"][k]) for k in order if k in ev["one_nn"]]
    if got:
        vals = [v["acc"] for _, v in got]
        chance = got[0][1]["chance"]
        y = np.arange(len(got))
        ax2.barh(y, vals, color=[ACCENT, MID, WARN], height=0.55)
        ax2.axvline(chance, color=INK, ls=":", lw=1.2)
        ax2.text(chance + 0.008, -0.62, f"chance {chance:.3f}", fontsize=7, color=INK)
        for i, v in enumerate(vals):
            ax2.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
        ax2.set_yticks(y, ["Benchmark style", "Grouped in domain", "Cross domain"],
                       fontsize=7.8)
        ax2.set_xlim(0, max(vals) * 1.25)
        ax2.set_xlabel("Accuracy of one nearest neighbour")
        ax2.set_title("The split answered by lookup, no model",
                      fontsize=9, color=ACCENT, pad=7)
        ax2.set_axisbelow(True)
        ax2.grid(axis="y", visible=False)
    save(fig, "fig9_model_free_evidence.png")


def fig10_recovery(runs):
    """How much of the loss each recovery method wins back, including the failures."""
    base, rec = {}, defaultdict(dict)
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["model"] != "LeNet" or arm != "cross" or held:
            continue
        if r["tag"] == "endsem":
            base[name] = r[ACC]
        elif r["tag"] == "endsem-rec":
            label = f"augment {int(r.get('aug_deg', 20))}" if r.get("augment") else ""
            if r.get("mixup"):
                label = f"{label} + mixup" if label else "mixup"
            rec[label][name] = r[ACC]
    protos = [p for p in PROTO if p in base and any(p in v for v in rec.values())]
    if not protos or not rec:
        return print("  fig10 skipped, recovery runs not finished")

    labels = sorted(rec)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = np.arange(len(protos))
    w = 0.8 / len(labels)
    alld = [rec[l].get(p, base[p]) - base[p] for l in labels for p in protos]
    # offsets have to scale with the data, since these deltas can be a few thousandths
    span = max(max(alld, default=0) - min(alld, default=0), 1e-6)
    pad = 0.05 * span
    for i, lab in enumerate(labels):
        deltas = [rec[lab].get(p, base[p]) - base[p] for p in protos]
        off = (i - (len(labels) - 1) / 2) * w
        bars = ax.bar(x + off, deltas, w, label=lab,
                      color=plt.cm.Blues(0.40 + 0.45 * i / max(len(labels) - 1, 1)))
        for b, d in zip(bars, deltas):
            ax.text(b.get_x() + b.get_width() / 2, d + (pad if d >= 0 else -pad * 2.4),
                    f"{d:+.3f}", ha="center", fontsize=6.6,
                    color=WARN if d < 0 else INK)
    ax.set_ylim(min(min(alld, default=0) - 6 * pad, -pad),
                max(max(alld, default=0) + 4 * pad, pad))
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(x, [PROTO[p].replace("\n", " ") for p in protos], fontsize=8.5)
    ax.set_ylabel("Change in cross-domain accuracy")
    ax.legend(fontsize=7.4, ncol=len(labels), loc="upper center",
              bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    save(fig, "fig10_recovery.png")


def fig11_confusion(runs):
    """Which gestures survive a change of orientation and which collapse."""
    got = None
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["tag"] == "endsem" and r["model"] == "LeNet" and not held \
                and name == "cross-orientation" and r.get("confusion"):
            got = got or {}
            got[arm] = np.array(r["confusion"], dtype=float)
    if not got or "cross" not in got or "in_domain_grouped" not in got:
        return print("  fig11 skipped, confusion matrices not available yet")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, arm in zip(axes, ("in_domain_grouped", "cross")):
        cm = got[arm]
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("Grouped in domain" if arm == "in_domain_grouped"
                     else "Orientation held out", fontsize=9, color=ACCENT, pad=6)
        ax.set_xlabel("Predicted gesture")
        ax.grid(False)
        ax.set_xticks([0, 5, 10, 15, 21])
        ax.set_yticks([0, 5, 10, 15, 21])
    axes[0].set_ylabel("True gesture")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Share of that gesture")
    save(fig, "fig11_confusion.png")


def fig12_cost(runs):
    """Accuracy against the cost of running the model where it would actually run."""
    prof = {}
    p = os.path.join(RESULTS, "model_profile.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            if d["dataset"] == "Widar":
                prof[d["model"]] = d
    pts = []
    for r in runs:
        name, held, arm = split_protocol(r["protocol"])
        if r["tag"] == "endsem" and arm == "cross" and name == "cross-orientation" \
                and not held and r["model"] in prof:
            pts.append((r["model"], prof[r["model"]], r[ACC]))
    key = "cpu_ms_batch1"
    if len(pts) < 2 or not all(key in p_ for _, p_, _ in pts):
        return print("  fig12 skipped, needs model_profile.py rerun with CPU latency")

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = [p_[key] for _, p_, _ in pts]
    ys = [a for _, _, a in pts]
    ax.scatter(xs, ys, s=44, color=ACCENT, zorder=3)
    for (m, p_, a) in pts:
        ax.annotate(m, (p_[key], a), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency for one sample, milliseconds (log scale)")
    ax.set_ylabel("Cross-domain accuracy")
    ax.set_title("What it costs to be wrong on an access point",
                 fontsize=9.5, color=ACCENT, pad=8)
    save(fig, "fig12_cost_vs_cross_domain.png")


def fig13_adaptation():
    """How far a handful of labelled examples from the new condition gets you."""
    p = os.path.join(RESULTS, "adaptation.jsonl")
    if not os.path.exists(p):
        return print("  fig13 skipped, run code/adapt.py first")
    rows = [json.loads(l) for l in open(p) if l.strip()]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["protocol"]][r.get("mode", "head")].append(r)
    if not by:
        return print("  fig13 skipped, no adaptation rows")

    protos = sorted(by)
    fig, axes = plt.subplots(1, len(protos), figsize=(3.3 * len(protos), 3.2),
                             squeeze=False, sharey=True)
    style = {"head": (MID, "o", "last layer only"),
             "full": (WARN, "s", "whole network")}
    for ax, proto in zip(axes[0], protos):
        base = None
        for mode, rs in sorted(by[proto].items()):
            rs = sorted(rs, key=lambda r: r["k"])
            xs = [max(r["k"], 0.4) for r in rs]   # 0.4 stands in for zero on a log axis
            ys = [r["acc"] for r in rs]
            c, mk, lab = style.get(mode, (GREY, "^", mode))
            ax.errorbar(xs, ys, yerr=[r.get("acc_sd", 0) for r in rs], fmt="-" + mk,
                        color=c, lw=1.7, ms=5, capsize=3, label=lab, zorder=3)
            base = base or ys[0]
        if base:
            ax.axhline(base, color=INK, ls=":", lw=1.0, zorder=1)
            ax.text(0.45, base + 0.008, "no adaptation", fontsize=6.8, color=INK)
        ax.set_xscale("log")
        ax.set_xticks([0.4, 1, 5, 20, 100, 300])
        ax.set_xticklabels(["0", "1", "5", "20", "100", "300"])
        ax.set_xlabel("Labelled examples per gesture")
        ax.set_title(PROTO.get(proto, proto).replace("\n", " "),
                     fontsize=9, color=ACCENT, pad=6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Accuracy on the rest of the new condition")
    axes[0][-1].legend(fontsize=7.4, frameon=False, loc="lower right")
    save(fig, "fig13_adaptation.png")


def main():
    runs = load()
    print(f"drawing end-term figures from {len(runs)} recorded runs")
    fig7_decomposition(runs)
    fig8_leave_one_out(runs)
    fig9_model_free()
    fig10_recovery(runs)
    fig11_confusion(runs)
    fig12_cost(runs)
    fig13_adaptation()
    print(f"written to {os.path.normpath(FIGS)}")


if __name__ == "__main__":
    main()
