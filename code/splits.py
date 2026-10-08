"""Train/test splits for Widar3.0, plus the evidence that the shipped split is unfair.

The SenseFi benchmark ships a random split. Because Widar3.0 records who performed each
gesture, where they stood and which way they faced, we can check whether that split holds
any of those conditions out of training. It does not: all 17 volunteers, all 8 standing
positions and all 5 orientations appear on both sides. The published accuracy therefore
measures how well a model recognises people it has already trained on.

Measuring that at the level of the individual recording condition makes it far worse than
a factor-by-factor count suggests. Group every sample by the tuple of gesture, volunteer,
standing position and body orientation and there are 4265 such conditions, holding 10.2
recordings each on average. 4259 of them straddle the shipped split, and every one of the
8726 test samples has at least one sibling in training recorded under the identical
condition. The benchmark is largely testing on repeat takes of recordings it has already
learned.

That gives us three arms rather than two, and the three-way decomposition is the point of
the whole project:

  * cross               one whole orientation, position or volunteer group held out
  * in_domain           random over samples, which is what the benchmark does
  * in_domain_grouped   random over condition groups, so no repeat take crosses the
                        split, but every volunteer, position and orientation still
                        appears on both sides

in_domain minus in_domain_grouped is the near-duplicate leakage the benchmark gives
itself. in_domain_grouped minus cross is the real domain gap. All three arms run over
exactly the same samples with the same test size, so nothing else differs between them.

Two things about Widar3.0 constrain how the cross arm can be built, and both were found
by the coverage check below rather than assumed:

  * The volunteers did not all perform the same gestures. Users 1 and 2 performed all 22
    and between them hold 48 percent of the data, while ten volunteers performed only 9.
    Gestures 13 to 22 were performed by two people. So a cross-user split cannot use all
    22 classes, and we restrict it to the 9 gestures that 12 volunteers share.
  * Standing positions 6, 7 and 8 cover only 6 gestures and about 600 samples each, so a
    cross-position split has to hold out one of positions 1 to 5.

Orientation is the clean factor: all five cover all 22 gestures with about 8730 samples
each, so the cross-orientation experiment can use the whole dataset.

One trap. The repetition field is NOT a unique instance identifier. 4370 files carry
"L0(1)", "L0(2)" or "L0(3)" suffixes and collapse onto a repetition number some other
file already owns, so 5873 (gesture, user, position, orientation, repetition) keys own
more than one file. They are separate recordings rather than copies, verified by
checksum, but it means repetition must never be used as a grouping key. The condition
group deliberately leaves it out.

Run from the project root:  python3 code/splits.py
Needs only numpy, and the cache built by build_widar_cache.py.
"""
import os

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cache")
COLS = {"label": 0, "user": 1, "position": 2, "orientation": 3,
        "repetition": 4, "orig_split": 5}

ARMS = ("cross", "in_domain", "in_domain_grouped")

# Volunteers who performed all of the first 9 gestures. Used by the cross-user protocol.
USERS_9 = [1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17]

PROTOCOLS = {
    "cross-orientation": {
        "factor": "orientation", "held": [5], "gestures": range(22), "users": None,
        "why": "all 5 orientations cover all 22 gestures evenly",
    },
    "cross-position": {
        "factor": "position", "held": [5], "gestures": range(22), "users": None,
        "why": "positions 6 to 8 are too sparse to hold out, so we hold out position 5",
    },
    "cross-user": {
        "factor": "user", "held": [15, 16, 17], "gestures": range(9), "users": USERS_9,
        "why": "restricted to the 9 gestures that 12 volunteers all performed",
    },
}


def load_meta():
    z = np.load(os.path.join(CACHE, "widar_meta.npz"), allow_pickle=True)
    return z["meta"], list(z["labels"])


def original_split(meta):
    """The split the benchmark ships, as index arrays."""
    orig = meta[:, COLS["orig_split"]]
    return np.where(orig == 0)[0], np.where(orig == 1)[0]


def group_ids(meta):
    """One id per recording condition: gesture, volunteer, position, orientation.

    Repetition is deliberately excluded, because it is not a unique instance id. Samples
    sharing a group are repeat takes of the same person performing the same gesture in
    the same place facing the same way, so they are near-duplicates of one another.
    """
    key = meta[:, [COLS["label"], COLS["user"], COLS["position"],
                   COLS["orientation"]]].astype(np.int64)
    # pack the four small fields into one integer, then rank them
    packed = ((key[:, 0] * 32 + key[:, 1]) * 16 + key[:, 2]) * 8 + key[:, 3]
    return np.unique(packed, return_inverse=True)[1]


def describe_overlap(meta, train_idx, test_idx):
    """For each factor, how many of its values appear on both sides of the split."""
    out = {}
    for factor in ("user", "position", "orientation"):
        col = meta[:, COLS[factor]]
        tr, te = set(col[train_idx].tolist()), set(col[test_idx].tolist())
        out[factor] = (len(tr), len(te), len(tr & te))
    return out


def group_straddle(gid, train_idx, test_idx):
    """(groups split across the two sides, total groups, test samples with a train sibling)."""
    tr, te = set(gid[train_idx].tolist()), set(gid[test_idx].tolist())
    both = tr & te
    return len(both), len(tr | te), int(np.isin(gid[test_idx], list(both)).sum())


def _subset(meta, spec):
    keep = np.isin(meta[:, COLS["label"]], list(spec["gestures"]))
    if spec["users"] is not None:
        keep &= np.isin(meta[:, COLS["user"]], spec["users"])
    return np.where(keep)[0]


def _grouped_split(gid, pool, n_test, seed):
    """Random split of the pool at condition-group level, hitting n_test as closely as
    the group sizes allow. Whole groups go to one side, so no repeat take crosses over."""
    rng = np.random.default_rng(seed + 991)
    groups = np.unique(gid[pool])
    sizes = {g: int((gid[pool] == g).sum()) for g in groups}
    test_groups, n = set(), 0
    for g in rng.permutation(groups):
        if n >= n_test:
            break
        if n + sizes[g] > n_test:      # skip a group that would overshoot
            continue
        test_groups.add(g)
        n += sizes[g]
    is_test = np.isin(gid[pool], list(test_groups))
    return pool[~is_test], pool[is_test]


def build(meta, name, held=None, seed=0):
    """Return all three arms of one protocol, over exactly the same pool of samples.

    held overrides which value of the factor is held out, so the leave-one-out sweep can
    ask for each orientation in turn rather than only the default.
    """
    spec = PROTOCOLS[name]
    held = spec["held"] if held is None else list(held)
    pool = _subset(meta, spec)
    gid = group_ids(meta)

    is_held = np.isin(meta[pool, COLS[spec["factor"]]], held)
    train_idx, test_idx = pool[~is_held], pool[is_held]
    n_test = len(test_idx)

    # Matched baseline: same samples, same test size, but split at random so the held-out
    # condition leaks in exactly the way the benchmark's split does.
    shuffled = np.random.default_rng(seed).permutation(pool)

    return {
        "name": name,
        "factor": spec["factor"],
        "held": held,
        "n_classes": len(list(spec["gestures"])),
        "cross": (train_idx, test_idx),
        "in_domain": (shuffled[n_test:], shuffled[:n_test]),
        "in_domain_grouped": _grouped_split(gid, pool, n_test, seed),
    }


def check(meta, split):
    """A split is only usable if it leaks nothing it claims to hold out, and still
    covers every gesture class on both sides."""
    gid = group_ids(meta)
    labels = meta[:, COLS["label"]]
    k = split["n_classes"]
    total = sum(len(a) for a in split["cross"])

    for arm in ARMS:
        tr, te = split[arm]
        assert not (set(tr.tolist()) & set(te.tolist())), f"{arm}: a sample is on both sides"
        assert len(tr) + len(te) == total, f"{arm}: uses a different amount of data"
        for part, idx in (("train", tr), ("test", te)):
            got = len(np.unique(labels[idx]))
            assert got == k, f"{arm} {part} has {got} gestures, expected {k}"

    # the cross arm must hold its factor out completely
    col = meta[:, COLS[split["factor"]]]
    tr, te = split["cross"]
    assert not (set(col[tr].tolist()) & set(col[te].tolist())), \
        f"{split['factor']} leaked across the cross split"

    # the grouped arm must hold whole condition groups out, and must still be domain-mixed
    tr, te = split["in_domain_grouped"]
    both, _, _ = group_straddle(gid, tr, te)
    assert both == 0, f"{both} condition groups straddle the grouped split"
    for factor, (a, b, shared) in describe_overlap(meta, tr, te).items():
        assert shared == min(a, b), \
            f"grouped arm dropped a {factor} value, it is meant to stay domain-mixed"


def main():
    meta, labels = load_meta()
    gid = group_ids(meta)
    print(f"Widar3.0: {len(meta)} samples, {len(labels)} gestures, "
          f"{len(np.unique(meta[:, COLS['user']]))} volunteers, "
          f"{gid.max() + 1} recording conditions\n")

    print("=" * 78)
    print("EVIDENCE 1: what the benchmark's own split does")
    print("=" * 78)
    tr, te = original_split(meta)
    print(f"train {len(tr)} samples, test {len(te)} samples\n")
    print(f"{'factor':<14}{'in train':>10}{'in test':>10}{'in BOTH':>10}   verdict")
    for factor, (a, b, both) in describe_overlap(meta, tr, te).items():
        verdict = "leaked, nothing held out" if both == a == b else "ok"
        print(f"{factor:<14}{a:>10}{b:>10}{both:>10}   {verdict}")
    print("\nEvery volunteer, position and orientation appears on both sides, so the test")
    print("set contains no condition the model has not already trained on.")

    print("\n" + "=" * 78)
    print("EVIDENCE 2: the leakage measured per recording condition, which is worse")
    print("=" * 78)
    both, tot, sib = group_straddle(gid, tr, te)
    print(f"A recording condition is one (gesture, volunteer, position, orientation).")
    print(f"conditions in the dataset                     {tot:>8}")
    print(f"recordings per condition, on average          {len(meta) / tot:>8.1f}")
    print(f"conditions appearing on BOTH sides            {both:>8}   "
          f"{100 * both / tot:.1f}%")
    print(f"test samples with a sibling in training       {sib:>8}   "
          f"{100 * sib / len(te):.1f}%")
    print("\nEvery test sample has a training sibling recorded under the identical")
    print("condition, so the benchmark is largely testing on repeat takes of recordings")
    print("it has already learned. This is why we add a grouped in-domain arm.")

    print("\n" + "=" * 78)
    print("EVIDENCE 3: the dataset is not balanced across volunteers")
    print("=" * 78)
    u, g = meta[:, COLS["user"]], meta[:, COLS["label"]]
    print(f"{'volunteer':<12}{'gestures':>10}{'samples':>10}{'share':>9}")
    for uu in range(1, 18):
        m = u == uu
        print(f"user {uu:<7d}{len(np.unique(g[m])):>10}{int(m.sum()):>10}"
              f"{100 * m.sum() / len(meta):>8.1f}%")
    print("\nUsers 1 and 2 hold 48 percent of the data and are the only two who performed")
    print("gestures 13 to 22. This is why the cross-user protocol uses 9 gestures.")

    print("\n" + "=" * 78)
    print("THE THREE ARMS WE WILL USE")
    print("=" * 78)
    print(f"{'protocol':<20}{'arm':<20}{'classes':>8}{'train':>8}{'test':>8}"
          f"{'conditions in both':>20}")
    for name in PROTOCOLS:
        s = build(meta, name)
        check(meta, s)
        for arm in ARMS:
            a, b = s[arm]
            both, tot, _ = group_straddle(gid, a, b)
            print(f"{name if arm == 'cross' else '':<20}{arm:<20}{s['n_classes']:>8}"
                  f"{len(a):>8}{len(b):>8}{both:>14} / {tot}")
        print(f"{'':20}held out: {s['factor']} {s['held']}. {PROTOCOLS[name]['why']}")
    print("\nAll three protocols pass the leak check on all three arms. The grouped arm")
    print("holds out zero conditions' worth of factors, so it is still domain-mixed, but")
    print("no repeat take crosses it. That is the control the comparison needs.")


if __name__ == "__main__":
    main()
