"""Run every end-term experiment, skipping anything already recorded.

Safe to re-run. A job is identified by everything that could change its result, so an
interrupted overnight sweep is restarted with the same command and picks up where it
stopped. Each job runs as a separate process, which also means its GPU and host memory
are released before the next one starts. That matters on this laptop, which has 16 GB of
RAM and an 8 GB card.

Jobs are ordered by how much the report needs them rather than by how long they take.
LeNet is the most accurate model on Widar3.0 and it is also one of the cheapest, so the
entire headline result is finished inside the first hour. The expensive models come last
and only confirm that the effect is not an artefact of one architecture. If the sweep is
cut short, what is lost is confirmation rather than the finding itself.

    /home/sarthak/miniconda3/envs/op05/bin/python code/run_endsem.py
    /home/sarthak/miniconda3/envs/op05/bin/python code/run_endsem.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(HERE, "..", "results", "baselines.jsonl")
PY = sys.executable

PROTOCOLS = ["cross-orientation", "cross-position", "cross-user"]
ARMS = ["cross", "in_domain", "in_domain_grouped"]

# roughly how long one 100-to-200 epoch Widar run takes, from the midterm timings
COST = {"LeNet": 5, "GRU": 5, "CNN+GRU": 13, "ViT": 13, "ResNet18": 21}


def job(model, protocol, tag, **kw):
    return dict(model=model, protocol=protocol, tag=tag, **kw)


def build_jobs():
    jobs = []

    # A. The headline, on the cheapest good model first. Three arms per protocol:
    #    cross is the honest number, in_domain is what the benchmark does, and
    #    in_domain_grouped sits between them with repeat takes removed.
    for model in ("LeNet", "GRU"):
        for p in PROTOCOLS:
            for arm in ARMS:
                jobs.append(job(model, p if arm == "cross" else f"{p}/{arm}",
                                "endsem", save=(arm == "cross" and model == "LeNet")))

    # B. The benchmark's CNN+GRU ends in a Softmax that its own loss softmaxes again.
    #    One run turns that grep into a measurement.
    jobs.append(job("CNN+GRU", "original", "endsem-softmax", fix_softmax=True))

    # C. Leave one condition out, every value in turn. Answers whether the size of the
    #    drop depends on which value was held out, which the single-value design cannot.
    #    The fifth orientation and fifth position are already covered by experiment A.
    for v in (1, 2, 3, 4):
        jobs.append(job("LeNet", f"cross-orientation:{v}", "endsem-loo"))
        jobs.append(job("LeNet", f"cross-position:{v}", "endsem-loo"))

    # D. Seed variance. Only initialisation moves; the split is pinned at split-seed 0,
    #    otherwise the in-domain arms would look noisier than the cross arm for a reason
    #    that has nothing to do with domains.
    for seed in (1, 2):
        for arm in ("cross", "in_domain_grouped"):
            jobs.append(job("LeNet", "cross-orientation" if arm == "cross"
                            else f"cross-orientation/{arm}", "endsem-seed", seed=seed))

    # E. Recovery. Two rotation limits because the physical argument says to keep the
    #    angle small, since Draw-N(H) and Draw-N(V) are separate classes, but a short
    #    calibration run was not conclusive, so both get measured.
    for p in PROTOCOLS:
        jobs.append(job("LeNet", p, "endsem-rec", augment=True, aug_deg=20))
        jobs.append(job("LeNet", p, "endsem-rec", augment=True, aug_deg=45))
        jobs.append(job("LeNet", p, "endsem-rec", mixup=True))
        jobs.append(job("LeNet", p, "endsem-rec", augment=True, aug_deg=20, mixup=True))

    # E2. A recovery method that raises cross-domain accuracy is only interesting if it
    #     does not quietly cost in-domain accuracy to get there, so the same augmentation
    #     runs against the grouped in-domain arm. ResNet18 gets one run to check the
    #     effect is not specific to LeNet.
    for p in PROTOCOLS:
        jobs.append(job("LeNet", f"{p}/in_domain_grouped", "endsem-rec",
                        augment=True, aug_deg=20))
    jobs.append(job("ResNet18", "cross-orientation", "endsem-rec",
                    augment=True, aug_deg=20))

    # F. The expensive models, last. These only confirm that the drop is not specific
    #    to one architecture.
    for model in ("ViT", "ResNet18"):
        for p in PROTOCOLS:
            for arm in ARMS:
                jobs.append(job(model, p if arm == "cross" else f"{p}/{arm}", "endsem"))
    return jobs


def key(j):
    """Everything that could change the result. Tag is included so that the same model
    and protocol can legitimately appear under endsem, endsem-rec and endsem-seed."""
    return (j["tag"], j["model"], j["protocol"], j.get("seed", 0),
            bool(j.get("augment")), float(j.get("aug_deg", 20)) if j.get("augment") else 0.0,
            bool(j.get("mixup")), bool(j.get("fix_softmax")))


def recorded():
    if not os.path.exists(JSONL):
        return set()
    out = set()
    for line in open(JSONL):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("tag", "").startswith("endsem"):
            continue
        out.add((r["tag"], r["model"], r["protocol"], r.get("seed", 0),
                 bool(r.get("augment")),
                 float(r.get("aug_deg", 20)) if r.get("augment") else 0.0,
                 bool(r.get("mixup")), bool(r.get("fix_softmax"))))
    return out


def cmd(j):
    c = [PY, os.path.join(HERE, "train.py"), "--dataset", "Widar",
         "--model", j["model"], "--protocol", j["protocol"], "--tag", j["tag"],
         "--max-epochs", "200", "--seed", str(j.get("seed", 0)), "--split-seed", "0"]
    if j.get("augment"):
        c += ["--augment", "--aug-deg", str(j.get("aug_deg", 20))]
    for flag in ("mixup", "fix_softmax", "save"):
        if j.get(flag):
            c.append("--" + flag.replace("_", "-"))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    have = recorded()
    jobs = build_jobs()
    todo = [j for j in jobs if key(j) not in have]
    est = sum(COST.get(j["model"], 10) for j in todo)
    print(f"{len(jobs)} jobs defined, {len(jobs) - len(todo)} already recorded, "
          f"{len(todo)} to run, roughly {est / 60:.1f} hours\n")

    if args.dry_run:
        for i, j in enumerate(todo, 1):
            print(f"[{i:3d}] {' '.join(cmd(j)[3:])}")
        return

    t0 = time.time()
    failed = []
    for i, j in enumerate(todo, 1):
        left = sum(COST.get(x["model"], 10) for x in todo[i - 1:])
        print(f"[{i}/{len(todo)}] {j['tag']} {j['model']} {j['protocol']}"
              f"{' aug' + str(j.get('aug_deg')) if j.get('augment') else ''}"
              f"{' mixup' if j.get('mixup') else ''}"
              f"{' seed' + str(j['seed']) if j.get('seed') else ''}"
              f"   (~{left / 60:.1f} h left)", flush=True)
        r = subprocess.run(cmd(j), capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr.strip().splitlines() or ["no stderr"])[-1][:160]
            print(f"    FAILED: {tail}", flush=True)
            failed.append((j, tail))
        else:
            print("    " + r.stdout.strip().splitlines()[-1].strip(), flush=True)

    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(todo) - len(failed)} succeeded, {len(failed)} failed")
    for j, why in failed:
        print(f"  {j['tag']} {j['model']} {j['protocol']}: {why}")


if __name__ == "__main__":
    main()
