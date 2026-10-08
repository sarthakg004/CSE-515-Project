"""Measure parameter count and inference latency for all eleven benchmark models.

The SenseFi benchmark reports accuracy and nothing else, but accuracy alone cannot tell
you whether a model could ever run on a real access point. These two numbers are the
missing half of that picture, and they cost seconds to collect rather than GPU hours,
so we can have them before any training has been done.

Run from the project root with the op05 environment:
    /home/sarthak/miniconda3/envs/op05/bin/python code/model_profile.py
"""
import os
import sys
import time

import torch

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "WiFi-CSI-Sensing-Benchmark")
sys.path.insert(0, BENCH)

MODELS = ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
          "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"]

# The input each dataset hands to the model, and the batch size the benchmark uses.
DATASETS = {
    "UT_HAR_data": (1, 250, 90),
    "NTU-Fi_HAR": (3, 114, 500),
    "NTU-Fi-HumanID": (3, 114, 500),
    "Widar": (22, 20, 20),
}
BATCH = 64


def latency_ms(model, shape, device, batch=BATCH, iters=30):
    """Median milliseconds to classify one batch, after warm-up."""
    x = torch.randn(batch, *shape, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return times[len(times) // 2]


def cpu_latency_ms(model, shape, threads=4, iters=20):
    """Median milliseconds to classify ONE sample on the processor.

    This is the number that decides whether a model could run on an access point. The
    graphics processor figure above is measured at batch 64 on a desktop card, and a
    router has neither. It also has roughly four cores rather than the 28 on this
    laptop, so the thread count is pinned, otherwise the measurement flatters itself by
    using hardware the target does not have.
    """
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        m = model.cpu().eval()
        x = torch.randn(1, *shape)
        with torch.no_grad():
            for _ in range(3):
                m(x)
            times = []
            for _ in range(iters):
                t = time.perf_counter()
                m(x)
                times.append((time.perf_counter() - t) * 1000)
    finally:
        torch.set_num_threads(old)
    times.sort()
    return times[len(times) // 2]


def main():
    import util

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else ''})")
    print(f"batch size {BATCH}, latency is the median of 30 timed runs after warm-up\n")

    rows = []
    for dataset, shape in DATASETS.items():
        for name in MODELS:
            # load_data_n_model also builds the loaders, which we do not need here, but
            # it is the only place the benchmark maps a name to a correctly sized model.
            _, _, model, epochs = util.load_data_n_model(dataset, name, BENCH + "/Data/")
            params = sum(p.numel() for p in model.parameters())
            model = model.to(device).eval()
            try:
                ms = latency_ms(model, shape, device)
            except RuntimeError as e:  # out of memory on a big model, worth recording
                ms = float("nan")
                print(f"  {name} on {dataset}: {type(e).__name__}", file=sys.stderr)
            try:
                cpu_ms = cpu_latency_ms(model, shape)
            except RuntimeError as e:
                cpu_ms = float("nan")
                print(f"  {name} on {dataset} (cpu): {type(e).__name__}", file=sys.stderr)
            rows.append((dataset, name, params, ms, ms / BATCH, epochs, cpu_ms))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = os.path.join(os.path.dirname(BENCH), "..", "results", "model_profile.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lines = []
    for dataset in DATASETS:
        lines.append(f"\n{dataset}")
        lines.append(f"{'model':<12}{'parameters':>14}{'MB':>8}"
                     f"{'batch ms':>10}{'ms/sample':>11}{'cpu ms/1':>10}{'epochs':>8}")
        for d, name, params, ms, per, epochs, cpu_ms in rows:
            if d != dataset:
                continue
            lines.append(f"{name:<12}{params:>14,}{params * 4 / 1e6:>8.1f}"
                         f"{ms:>10.2f}{per:>11.3f}{cpu_ms:>10.2f}{epochs:>8}")
    text = "\n".join(lines)
    print(text)
    import json
    with open(out.replace(".txt", ".jsonl"), "w") as f:
        for d, name, params, ms, per, epochs, cpu_ms in rows:
            f.write(json.dumps(dict(dataset=d, model=name, params=params,
                                    batch_ms=round(ms, 3), ms_per_sample=round(per, 4),
                                    cpu_ms_batch1=round(cpu_ms, 3),
                                    shipped_epochs=epochs)) + "\n")
    with open(out, "w") as f:
        f.write("Parameter count and inference latency for the SenseFi models.\n")
        f.write(f"Measured on {torch.cuda.get_device_name(0)}, batch size {BATCH}, "
                f"float32, median of 30 runs.\n")
        f.write("MB is the weight size in float32. 'epochs' is what the benchmark "
                "ships as its training length.\n")
        f.write("cpu ms/1 is the time to classify a single sample on 4 processor "
                "threads with no GPU, which is the\nsituation an access point is "
                "actually in.\n")
        f.write(text + "\n")
    print(f"\nwritten to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
