"""Draw every figure that goes into the report, from the recorded results.

Slide 10 of the course notes forbids reusing anyone else's figures, so every figure here
is drawn by us from our own measurements. Nothing is traced, copied or redrawn from a
paper.

    /home/sarthak/miniconda3/envs/op05/bin/python code/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RESULTS, "figures")

INK = "#15202b"
ACCENT = "#123a5e"
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

ORDER = ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
         "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"]
NICE = {"UT_HAR_data": "UT-HAR", "NTU-Fi_HAR": "NTU-Fi HAR",
        "NTU-Fi-HumanID": "NTU-Fi HumanID", "Widar": "Widar3.0"}


def load():
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "baselines.jsonl"))
            if l.strip()]
    runs = [r for r in runs if r.get("tag") == "baseline"]
    return {(r["dataset"], r["model"]): r for r in runs}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def fig1_split_evidence():
    """Why the shipped Widar split proves nothing, and what our splits change."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    fig.subplots_adjust(wspace=0.55)
    factors = ["Volunteers", "Standing\npositions", "Body\norientations"]
    totals = [17, 8, 5]

    panels = [("Benchmark split", [17, 8, 5]),
              ("Our cross-user split", [14, 8, 5])]
    for ax, (title, shared) in zip(axes, panels):
        y = np.arange(len(factors))
        ax.barh(y, totals, color="#e6ecf1", height=0.5)
        cols = [WARN if s_ == t else ACCENT for s_, t in zip(shared, totals)]
        ax.barh(y, shared, color=cols, height=0.5)
        for i, (t, s_) in enumerate(zip(totals, shared)):
            note = "all shared" if s_ == t else f"{t - s_} held out"
            ax.text(t + 0.6, i, f"{s_}/{t}  {note}", va="center", fontsize=7.2,
                    color=WARN if s_ == t else ACCENT)
        ax.set_yticks(y, factors, fontsize=8)
        ax.set_xlim(0, 34)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_title(title, fontsize=9, color=ACCENT, pad=7)
        ax.invert_yaxis()
    axes[0].set_xlabel("Values appearing in BOTH train and test", fontsize=8)
    axes[1].set_xlabel("Values appearing in BOTH train and test", fontsize=8)
    save(fig, "fig1_split_evidence.png")


def fig2_accuracy_by_dataset(acc):
    """Accuracy of all eleven models across the four datasets."""
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    datasets = list(NICE)
    x = np.arange(len(ORDER))
    w = 0.2
    colors = [ACCENT, "#2f6f9f", "#6ea3c4", GREY]
    for i, d in enumerate(datasets):
        vals = [acc[(d, m)]["best_acc"] if (d, m) in acc else 0 for m in ORDER]
        ax.bar(x + (i - 1.5) * w, vals, w, label=NICE[d], color=colors[i])
    ax.set_xticks(x, ORDER, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Best test accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#e3e9ee", lw=0.8)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              frameon=False)
    save(fig, "fig2_accuracy_by_dataset.png")


def fig3_cost_vs_accuracy(acc):
    """The central efficiency picture: accuracy against model size."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, d in zip(axes, ["NTU-Fi_HAR", "Widar"]):
        xs, ys, names = [], [], []
        for m in ORDER:
            r = acc.get((d, m))
            if r:
                xs.append(r["params"] / 1e6)
                ys.append(r["best_acc"])
                names.append(m)
        ax.scatter(xs, ys, s=34, color=ACCENT, zorder=3)
        best = names[int(np.argmax(ys))]
        for xv, yv, n in zip(xs, ys, names):
            ax.annotate(n, (xv, yv), fontsize=6.5, xytext=(3, 4),
                        textcoords="offset points",
                        color=WARN if n == best else INK,
                        fontweight="bold" if n == best else "normal")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters, millions (log scale)")
        ax.set_title(NICE[d], fontsize=9, color=ACCENT)
    axes[0].set_ylabel("Best test accuracy")
    save(fig, "fig3_cost_vs_accuracy.png")


def fig4_widar_size_trend(acc):
    """On Widar, accuracy peaks at a small model and falls for every larger one.
    Ordered by size so labels cannot collide, which they did when this was a scatter."""
    rows = sorted(((acc[("Widar", m)]["params"], acc[("Widar", m)]["best_acc"], m)
                   for m in ORDER if ("Widar", m) in acc))
    names = [f"{m}   {p / 1e6:.2f}M" for p, _, m in rows]
    vals = [a_ for _, a_, _ in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))
    best = int(np.argmax(vals))
    cols = [WARN if i == best else ACCENT for i in range(len(rows))]
    ax.barh(y, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8,
                color=WARN if i == best else INK,
                fontweight="bold" if i == best else "normal")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("Best test accuracy")
    ax.set_ylabel("Model, ordered smallest to largest", fontsize=8.5)
    ax.set_title("Widar3.0: the best model is a small one",
                 fontsize=9.5, color=ACCENT, pad=8)
    save(fig, "fig4_widar_size_trend.png")


def fig5_instability(acc):
    """Runs that ended below their own best epoch. Drawn as a dumbbell chart because the
    earlier scatter plot piled several labels on top of one another."""
    bad = sorted(((r["final_acc"] - r["best_acc"], d, m, r)
                  for (d, m), r in acc.items()
                  if r["best_acc"] - r["final_acc"] > 0.05))
    names = [f"{m} on {NICE[d]}" for _, d, m, _ in bad]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(bad))
    for i, (_, _, _, r) in enumerate(bad):
        ax.plot([r["final_acc"], r["best_acc"]], [i, i], color="#c3ced7", lw=2.2,
                zorder=2, solid_capstyle="round")
        ax.scatter([r["final_acc"]], [i], s=42, color=WARN, zorder=3)
        ax.scatter([r["best_acc"]], [i], s=42, color=ACCENT, zorder=3)
        ax.text(r["best_acc"] + 0.018, i, f"{r['final_acc'] - r['best_acc']:+.3f}",
                va="center", fontsize=7.5, color=WARN)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Test accuracy")
    ax.scatter([], [], s=42, color=ACCENT, label="best epoch")
    ax.scatter([], [], s=42, color=WARN, label="final epoch")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_title("Accuracy lost between the best epoch and the last one",
                 fontsize=9.5, color=ACCENT, pad=8)
    save(fig, "fig5_training_instability.png")


def fig6_pipeline_speedup():
    """The measured effect of caching the Widar data."""
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    labels = ["Benchmark loader\n(cold cache)", "Benchmark loader\n(warm cache)",
              "Our cached array"]
    minutes = [3.30, 1.52, 0.0012]
    y = np.arange(len(labels))
    bars = ax.barh(y, minutes, color=[WARN, "#c8836f", ACCENT], height=0.55)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Minutes of file reading per training epoch (log scale)")
    for b, v in zip(bars, minutes):
        ax.text(v * 1.25, b.get_y() + b.get_height() / 2,
                f"{v * 60:.2f} s" if v < 0.1 else f"{v:.2f} min",
                va="center", fontsize=8)
    ax.set_xlim(0.0005, 60)
    ax.set_title("File reading cost per training epoch", fontsize=9.5, color=ACCENT,
                 pad=8)
    save(fig, "fig6_pipeline_speedup.png")


def main():
    acc = load()
    print(f"drawing figures from {len(acc)} recorded runs")
    fig1_split_evidence()
    fig2_accuracy_by_dataset(acc)
    fig3_cost_vs_accuracy(acc)
    fig4_widar_size_trend(acc)
    fig5_instability(acc)
    fig6_pipeline_speedup()
    print(f"written to {os.path.normpath(FIGS)}")


if __name__ == "__main__":
    main()
