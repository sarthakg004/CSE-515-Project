"""Convert the 43652 Widar3.0 CSV files into one cached array plus a metadata table.

Why this exists: the SenseFi loader calls np.genfromtxt once per sample, which costs
about 5.7 ms. That is 3.3 minutes of single-threaded parsing for every pass over the
training set, before any learning happens. Caching turns that into a memory-mapped read.

The metadata table is the other half of the point. Each Widar filename encodes the
volunteer, where they stood, and which way they faced, and we need those columns to
build honest train/test splits later (see splits.py).

Run from the project root:  python3 code/build_widar_cache.py
Needs only numpy. Writes about 1.5 GB into data/cache/.
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "data", "Data", "Widardata")
OUT = os.path.join(ROOT, "data", "cache")

# user<N>-<gesture>-<torsoLocation>-<faceOrientation>-<repetition>-<receiver>-<rest>
NAME = re.compile(r"^user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-([^-]+)")


def scan():
    """Return (paths, meta rows, label names). The folder name is the true label.

    The gesture field inside the filename is NOT reliable: Widar3.0 renumbered gestures
    between recording sessions, so folder 5-Draw-N(H) holds files whose gesture field
    reads 3, 4, 6 and 7. The benchmark uses the folder and so do we.
    """
    labels = sorted(os.listdir(os.path.join(SRC, "train")),
                    key=lambda d: int(d.split("-")[0]))
    label_id = {name: i for i, name in enumerate(labels)}

    paths, rows = [], []
    for split_id, split in enumerate(("train", "test")):
        for folder in sorted(os.listdir(os.path.join(SRC, split))):
            d = os.path.join(SRC, split, folder)
            for fname in sorted(os.listdir(d)):
                m = NAME.match(fname)
                if not m:
                    raise ValueError(f"unparseable filename: {fname}")
                paths.append(os.path.join(d, fname))
                rows.append((label_id[folder], int(m.group(1)), int(m.group(3)),
                             int(m.group(4)), int(m.group(5)), split_id))
    return paths, np.array(rows, dtype=np.int16), labels


def _load(path):
    # loadtxt is roughly 3x faster than the genfromtxt the benchmark uses.
    return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(22, 20, 20)


def main():
    os.makedirs(OUT, exist_ok=True)
    paths, meta, labels = scan()
    n = len(paths)
    print(f"found {n} samples across {len(labels)} gesture classes")

    # Measure the loader the benchmark actually uses, so the report has a before number.
    sample = paths[:60]
    t = time.time()
    for p in sample:
        np.genfromtxt(p, delimiter=",")
    per = (time.time() - t) / len(sample)
    print(f"baseline  np.genfromtxt : {per * 1000:6.2f} ms/sample "
          f"-> {per * 34926 / 60:5.2f} min per training epoch")

    print(f"converting with {os.cpu_count()} processes ...")
    t = time.time()
    x = np.lib.format.open_memmap(os.path.join(OUT, "widar_x.npy"), mode="w+",
                                  dtype=np.float32, shape=(n, 22, 20, 20))
    with Pool() as pool:
        for i, arr in enumerate(pool.imap(_load, paths, chunksize=64)):
            x[i] = arr
            if i % 5000 == 0:
                print(f"  {i:6d} / {n}", flush=True)
    x.flush()
    convert = time.time() - t
    print(f"converted in {convert / 60:.1f} min")

    np.savez(os.path.join(OUT, "widar_meta.npz"), meta=meta,
             columns=np.array(["label", "user", "position", "orientation",
                               "repetition", "orig_split"]),
             labels=np.array(labels))

    # And the after number: how fast a training epoch can now read its data.
    x = np.load(os.path.join(OUT, "widar_x.npy"), mmap_mode="r")
    idx = np.random.default_rng(0).permutation(n)[:5000]
    t = time.time()
    for i in idx:
        _ = np.asarray(x[i])
    per_cached = (time.time() - t) / len(idx)
    print(f"cached    memmap read   : {per_cached * 1000:6.3f} ms/sample "
          f"-> {per_cached * 34926:5.1f} s per training epoch")
    print(f"speedup on the data path : {per / per_cached:.0f}x")

    size = os.path.getsize(os.path.join(OUT, "widar_x.npy")) / 1e9
    print(f"\nwrote {OUT}/widar_x.npy ({size:.2f} GB) and widar_meta.npz")


if __name__ == "__main__":
    sys.exit(main())
