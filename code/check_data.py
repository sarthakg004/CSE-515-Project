"""Sanity check that all four CSI datasets are present and shaped as the SenseFi paper says.

Run from the project root:  python3 code/check_data.py
Needs only numpy. Exits non-zero if anything is missing or the wrong shape.
"""
import glob
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "Data")


def main():
    # UT-HAR. Note the files end in .csv but are actually numpy .npy dumps.
    x = np.load(os.path.join(ROOT, "UT_HAR/data/X_train.csv"))
    y = np.load(os.path.join(ROOT, "UT_HAR/label/y_train.csv"))
    assert x.shape == (3977, 250, 90), x.shape
    assert len(y) == 3977 and len(set(y.tolist())) == 7
    print(f"UT-HAR          X{x.shape} -> (n,1,250,90), 7 classes, {len(y)} train samples")

    for name, n_classes in [("NTU-Fi_HAR", 6), ("NTU-Fi-HumanID", 14)]:
        counts = {}
        for split in ("train_amp", "test_amp"):
            classes = sorted(os.listdir(os.path.join(ROOT, name, split)))
            assert len(classes) == n_classes, (name, split, len(classes))
            counts[split] = sum(
                len(os.listdir(os.path.join(ROOT, name, split, c))) for c in classes
            )
        print(f"{name:15s} {n_classes} classes, "
              f"{counts['train_amp']} train / {counts['test_amp']} test")

    for split, expected in [("train", 34926), ("test", 8726)]:
        classes = sorted(os.listdir(os.path.join(ROOT, "Widardata", split)))
        assert len(classes) == 22, len(classes)
        total = sum(
            len(os.listdir(os.path.join(ROOT, "Widardata", split, c))) for c in classes
        )
        assert total == expected, (split, total, expected)
        print(f"Widar3.0 {split:5s}  22 classes, {total} samples")

    sample = np.genfromtxt(glob.glob(os.path.join(ROOT, "Widardata/train/*/*"))[0],
                           delimiter=",")
    assert sample.reshape(22, 20, 20).shape == (22, 20, 20)
    print(f"Widar3.0 sample {sample.shape} -> (22,20,20)")

    print("\nAll four datasets present and correctly shaped.")


if __name__ == "__main__":
    main()
