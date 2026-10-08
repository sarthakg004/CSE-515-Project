"""Train one benchmark model on one dataset or cross-domain arm, and record the result.

This reuses the SenseFi model definitions and its training recipe (Adam at 1e-3,
cross entropy, batch 64) so that our numbers stay comparable to the published ones.
The differences from the benchmark are deliberate and each one is a correction.

  * For Widar it reads our cached array instead of parsing 43652 CSV files on every
    epoch, and loading is spread over worker processes instead of one thread.
  * Test accuracy is counted as correct predictions divided by total samples. The
    benchmark averages the per-batch accuracy instead, which over-weights the final
    partial batch.
  * The Widar model is built with the class count the split actually has. The benchmark
    hardcodes 22, so the 9-class cross-user protocol would otherwise train a 22-way head.
  * Three accuracies are recorded per run rather than one, because how you pick the
    epoch changes the answer and the benchmark hides that choice:
        final_acc    accuracy after the last epoch. This is what SenseFi's run.py
                     reports, since it trains N epochs and calls test() once at the end.
        val_sel_acc  accuracy at the epoch with the best validation accuracy, where the
                     validation set is carved out of training at condition-group level
                     so no repeat take leaks into it. This is the honest number and it
                     is what the end-term tables quote.
        best_acc     the maximum over epochs measured on the test set. This is an
                     oracle, not a usable number, and it is recorded only so the cost
                     of test-set peeking can be reported.
  * Every run appends a JSON line to results/baselines.jsonl, so tables are generated
    from a file rather than copied out of terminal scrollback.

Examples:
    python code/train.py --dataset UT_HAR_data --model ResNet18
    python code/train.py --dataset Widar --protocol cross-user --model LeNet
    python code/train.py --dataset Widar --protocol cross-orientation/in_domain_grouped --model LeNet
    python code/train.py --dataset Widar --protocol cross-orientation:3 --model LeNet
    python code/train.py --dataset Widar --protocol cross-user --model LeNet --augment --mixup
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "WiFi-CSI-Sensing-Benchmark")
RESULTS = os.path.join(HERE, "..", "results")
CKPT = os.path.join(RESULTS, "ckpt")
sys.path.insert(0, BENCH)
sys.path.insert(0, HERE)

MODELS = ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
          "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"]
DATASETS = ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"]

WIDAR_MEAN, WIDAR_STD = 0.0025, 0.0119
AUG_DEG = 20.0   # see augment(): large rotations map Draw-N(H) onto Draw-N(V)


# --------------------------------------------------------------------- augmentation
def augment(t, rng, max_deg=AUG_DEG):
    """Augment one raw (22, 20, 20) BVP sample, before normalisation.

    A BVP is not an image. It is a 20 by 20 grid of velocities per time step, and each
    time step is a distribution that sums to about 1.01. Three consequences shape this:

      * Rotating the velocity plane corresponds to a change of body orientation. The
        rotation has to stay small. Widar's own gesture set contains Draw-N(H) and
        Draw-N(V), Draw-O(H) and Draw-O(V), Draw-Zigzag(H) and Draw-Zigzag(V), which are
        the same shape drawn horizontally and vertically and are separate classes. A
        rotation near 90 degrees therefore maps one class onto another and destroys the
        label. That is also the physically correct choice, because a BVP is already
        expressed in body coordinates, so what survives a change of orientation is only
        the residual error in estimating which way the torso faced, not the full angle.
        Zero velocity sits at grid index (9, 9), not the array centre of 9.5, so the
        rotation is centred there or every sample shifts by half a bin.
      * Scaling the plane is a change of movement speed, which is the most obvious way
        one volunteer differs from another, so it is the prior for the cross-user case.
      * Interpolation and erasing both destroy mass, so each time step is rescaled back
        to the mass it started with. Otherwise the fixed normalisation constants the
        benchmark uses stop being true of the augmented data.

    There is no clean physical prior for standing position, so we expect this to help
    least there. That prediction is part of the result.
    """
    s0 = t.sum(dim=(1, 2), keepdim=True)

    angle = float(rng.uniform(-max_deg, max_deg))
    scale = float(rng.uniform(0.85, 1.18))
    if angle or abs(scale - 1.0) > 1e-3:
        t = TF.affine(t, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0.0,
                      center=[9.0, 9.0])
    if rng.random() < 0.3:                                  # erase part of the grid
        i, j = rng.integers(0, 16, 2)
        h, w = rng.integers(3, 6, 2)
        t = t.clone()
        t[:, i:i + h, j:j + w] = 0.0
    if rng.random() < 0.5:
        t = (t + torch.randn_like(t) * (0.02 * float(t.mean()))).clamp_(min=0.0)

    t = t * (s0 / t.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8))

    if rng.random() < 0.5:                                  # shift in time
        t = torch.roll(t, int(rng.integers(-3, 4)), dims=0)
    return t


class WidarCache(torch.utils.data.Dataset):
    """Widar samples read from the memory-mapped cache built by build_widar_cache.py."""

    def __init__(self, idx, labels, path, aug=False, seed=0, aug_deg=AUG_DEG):
        self.idx, self.labels, self.path = idx, labels, path
        self.aug, self.seed, self.aug_deg = aug, seed, aug_deg
        self.x = None       # opened lazily so each worker process gets its own handle
        self.rng = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.x is None:
            self.x = np.load(self.path, mmap_mode="r")
            wi = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed + 7919 * (wi.id if wi else 0))
        j = self.idx[i]
        v = torch.from_numpy(np.array(self.x[j], dtype=np.float32))
        if self.aug:
            v = augment(v, self.rng, self.aug_deg)
        return (v - WIDAR_MEAN) / WIDAR_STD, int(self.labels[j])


# --------------------------------------------------------------------- data and model
def parse_protocol(p):
    """'cross-orientation:3/in_domain_grouped' -> ('cross-orientation', [3], arm)."""
    base, _, arm = p.partition("/")
    name, _, held = base.partition(":")
    return name, ([int(v) for v in held.split(",")] if held else None), (arm or "cross")


def widar_indices(protocol, seed, split_seed=0):
    """Return (train, val, test, n_classes). Validation is carved out of training at
    condition-group level, so a repeat take cannot sit on both sides of it.

    split_seed fixes which samples form the split and seed varies only initialisation
    and the validation carve. Keeping them apart matters for the seed-variance figure:
    if one seed moved both, the in-domain arms would get wider error bars than the cross
    arm purely because their split is random and the cross arm's is not."""
    import splits
    meta, _ = splits.load_meta()
    labels = meta[:, splits.COLS["label"]].astype(np.int64)

    if protocol == "original":
        tr, te = splits.original_split(meta)
        n_classes = 22
    else:
        name, held, arm = parse_protocol(protocol)
        s = splits.build(meta, name, held=held, seed=split_seed)
        splits.check(meta, s)
        tr, te = s[arm]
        n_classes = s["n_classes"]

    gid = splits.group_ids(meta)
    tr, va = splits._grouped_split(gid, tr, max(1, int(0.10 * len(tr))), seed)
    assert len(np.unique(labels[tr])) == n_classes, "validation split cost us a class"
    return tr, va, te, labels, n_classes


def build_model(name, n_classes, fix_softmax=False):
    """Build a Widar model with the class count this split actually has.

    fix_softmax removes the trailing nn.Softmax from CNN+GRU. The benchmark applies
    nn.CrossEntropyLoss on top of that softmax, so the loss sees a double softmax, which
    flattens the gradient. The vendored repository is left untouched on purpose, since
    the project's job is to evaluate it rather than to quietly repair it, so the fix is
    applied here and reported as a measured difference."""
    import widar_model
    m = getattr(widar_model, "Widar_" + name.replace("+", "_"))(num_classes=n_classes)
    if fix_softmax and isinstance(getattr(m, "classifier", None), nn.Sequential) \
            and isinstance(m.classifier[-1], nn.Softmax):
        m.classifier[-1] = nn.Identity()
    return m


def get_setup(args):
    """Return loaders, model, the benchmark's shipped epoch count, and sizes."""
    cache = os.path.join(HERE, "..", "data", "cache", "widar_x.npy")

    def loader(ds, shuffle, bs, workers):
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
            drop_last=shuffle, persistent_workers=workers > 0)

    if args.dataset == "Widar":
        import util
        # only to read the benchmark's shipped epoch count for this model
        shipped = {"MLP": 30, "LeNet": 100, "ResNet18": 100, "ResNet50": 100,
                   "ResNet101": 100, "RNN": 500, "GRU": 200, "LSTM": 200,
                   "BiLSTM": 200, "CNN+GRU": 200, "ViT": 200}[args.model]
        tr, va, te, labels, k = widar_indices(args.protocol, args.seed,
                                              args.split_seed)
        mk = lambda idx, shuf, aug, bs, w: loader(
            WidarCache(idx, labels, cache, aug=aug, seed=args.seed,
                       aug_deg=args.aug_deg), shuf, bs, w)
        return (mk(tr, True, args.augment, args.batch, args.workers),
                mk(va, False, False, args.eval_batch, 2),
                mk(te, False, False, args.eval_batch, 2),
                build_model(args.model, k, args.fix_softmax), shipped, len(tr), len(va), len(te), k)

    import util
    train_loader, test_loader, model, shipped = util.load_data_n_model(
        args.dataset, args.model, BENCH + "/Data/")
    n_tr, n_te = len(train_loader.dataset), len(test_loader.dataset)
    return (loader(train_loader.dataset, True, args.batch, args.workers),
            None,
            loader(test_loader.dataset, False, args.eval_batch, 2),
            model, shipped, n_tr, 0, n_te, None)


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, n_classes=None):
    """Return (accuracy, macro F1, confusion matrix). Confusion is truth by prediction."""
    model.eval()
    correct = total = 0
    cm = None
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).argmax(dim=1).cpu()
        y = y.long()
        if cm is None:
            k = n_classes or int(max(y.max().item(), pred.max().item()) + 1)
            cm = torch.zeros(k, k, dtype=torch.long)
        cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
        correct += (pred == y).sum().item()
        total += y.size(0)
    tp = cm.diag().double()
    f1 = (2 * tp / (cm.sum(0) + cm.sum(1)).double().clamp(min=1)).mean().item()
    return correct / total, f1, cm.tolist()


def mixup(x, y, k, alpha=0.4):
    """Convex combination of two samples and their labels. A blend of two BVPs is still
    BVP-shaped, so this is a legitimate augmentation here rather than a visual trick."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    oh = torch.zeros(y.size(0), k, device=x.device).scatter_(1, y[:, None], 1.0)
    return lam * x + (1 - lam) * x[perm], lam * oh + (1 - lam) * oh[perm]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--protocol", default="original",
                    help="Widar only: original, or <name>[:held]/[cross|in_domain|"
                         "in_domain_grouped]")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=100,
                    help="cap, because the benchmark ships values up to 3000")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4,
                    help="kept low on purpose: this laptop has 16 GB of RAM and each "
                         "worker is a process. Loading is not the bottleneck any more")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0,
                    help="initialisation and validation carve only")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="which samples form the split. Held fixed across seeds so the "
                         "seed-variance figure measures initialisation, not the split")
    ap.add_argument("--tag", default="endsem")
    ap.add_argument("--augment", action="store_true", help="physics-motivated BVP augmentation")
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--aug-deg", type=float, default=AUG_DEG,
                    help="max rotation in degrees. Keep small: Draw-N(H) and Draw-N(V) "
                         "are different classes, so a big rotation destroys the label")
    ap.add_argument("--fix-softmax", action="store_true",
                    help="drop CNN+GRU's trailing Softmax, which the benchmark feeds "
                         "straight into CrossEntropyLoss")
    ap.add_argument("--save", action="store_true", help="write the trained weights to results/ckpt/")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tl, vl, testl, model, shipped, n_tr, n_va, n_te, k = get_setup(args)
    epochs = args.epochs or min(shipped, args.max_epochs)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    flags = "".join(f" +{f}" for f in ("augment", "mixup") if getattr(args, f))
    print(f"{args.dataset} / {args.protocol} / {args.model}{flags}: {n_tr} train, "
          f"{n_va} val, {n_te} test, {k} classes, {params:,} params, {epochs} epochs "
          f"(benchmark ships {shipped})", flush=True)

    best_acc = best_val = 0.0
    val_sel_acc, val_sel_f1, sel_epoch, sel_cm = 0.0, 0.0, 0, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        loss_sum = seen = 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device).long()
            opt.zero_grad(set_to_none=True)
            if args.mixup and k:
                xm, ysoft = mixup(x, y, k)
                loss = -(torch.log_softmax(model(xm), 1) * ysoft).sum(1).mean()
            else:
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            seen += x.size(0)

        acc, f1, cm = evaluate(model, testl, device, k)
        best_acc = max(best_acc, acc)
        if vl is not None:
            vacc, _, _ = evaluate(model, vl, device, k)
            if vacc > best_val:
                best_val, val_sel_acc, val_sel_f1, sel_epoch, sel_cm = \
                    vacc, acc, f1, ep + 1, cm
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:4d}/{epochs}  loss {loss_sum / seen:.4f}  "
                  f"test {acc:.4f}  val-selected {val_sel_acc:.4f}  "
                  f"oracle {best_acc:.4f}", flush=True)
    mins = (time.time() - t0) / 60

    if vl is None:                       # non-Widar datasets have no validation arm
        val_sel_acc, val_sel_f1, sel_epoch, sel_cm, best_val = acc, f1, epochs, cm, 0.0

    row = dict(tag=args.tag, dataset=args.dataset, protocol=args.protocol,
               model=args.model, params=params, epochs=epochs,
               shipped_epochs=shipped, n_train=n_tr, n_val=n_va, n_test=n_te,
               n_classes=k, batch=args.batch, augment=args.augment, mixup=args.mixup, aug_deg=args.aug_deg,
               fix_softmax=args.fix_softmax, split_seed=args.split_seed,
               final_acc=round(acc, 4), best_acc=round(best_acc, 4),
               val_sel_acc=round(val_sel_acc, 4), val_acc=round(best_val, 4),
               val_sel_f1=round(val_sel_f1, 4), sel_epoch=sel_epoch,
               final_f1=round(f1, 4), confusion=sel_cm,
               train_min=round(mins, 2), seed=args.seed, note=args.note)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "baselines.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")

    if args.save:
        os.makedirs(CKPT, exist_ok=True)
        key = f"{args.model}_{args.protocol.replace('/', '_')}_s{args.seed}{flags.replace(' +', '_')}"
        torch.save(model.state_dict(), os.path.join(CKPT, key + ".pt"))
        print(f"  weights saved as {key}.pt")

    print(f"  done in {mins:.1f} min, final {acc:.4f}, val-selected {val_sel_acc:.4f} "
          f"(epoch {sel_epoch}), oracle {best_acc:.4f}")


if __name__ == "__main__":
    main()
