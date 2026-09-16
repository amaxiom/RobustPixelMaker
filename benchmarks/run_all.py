"""One-command reproduction of the full RobustPixelMaker benchmark suite.

Runs every component benchmark in sequence with the configurations documented in
BENCHMARKS.md, writing all CSVs, summaries and figures into ``results/``. Components
run as subprocesses so one failure cannot poison another's state; a failure is
reported and the remaining components still run.

Tiers, because the full suite is hours of compute:

  --tier smoke     one seed, synthetic quick pass + one real dataset (~10 minutes)
  --tier standard  the configurations used for FINDINGS.md (the default; the last two
                   runs took 248 and 268 min on a machine that was not idle)
  --tier full      standard plus every registered MedMNIST dataset and both
                   frontier datasets (an overnight run)

Wall-clock durations depend heavily on background load; every component prints its
own timing, and relative comparisons within one run remain valid under contention.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def component(name, args):
    return {"name": name, "cmd": [PY, str(HERE / args[0]), *args[1:]]}


def plan(tier: str):
    if tier == "smoke":
        return [
            component("synthetic (quick)", ["run_synthetic_benchmark.py", "--quick"]),
            component("real: mnist-3v8 (quick)",
                      ["run_real_benchmark.py", "--datasets", "mnist-3v8", "--quick"]),
            component("nonlinear label rules (quick)",
                      ["run_nonlinear_control.py", "--seeds", "2", "--n", "300"]),
        ]
    standard = [
        component("synthetic sweep", ["run_synthetic_benchmark.py"]),
        component("nonlinear label rules", ["run_nonlinear_control.py", "--seeds", "5"]),
        component("gate ablation",
                  ["run_synthetic_benchmark.py", "--axes", "noise,region_side",
                   "--out-prefix", "gate_ablation"]),
        component("lam frontier: mnist-3v8",
                  ["run_lam_frontier.py", "--dataset", "mnist-3v8", "--n-max", "2000"]),
        component("real: digits", ["run_real_benchmark.py",
                                   "--datasets", "mnist-3v8,mnist-all",
                                   "--n-max", "4000", "--out-prefix", "real"]),
        component("real: medmnist core",
                  ["run_real_benchmark.py",
                   "--datasets", "medmnist-pneumonia,medmnist-blood,"
                                 "medmnist-organa,medmnist-breast",
                   "--n-max", "3000", "--out-prefix", "medmnist"]),
        component("representation ablation",
                  ["run_representation_ablation.py",
                   "--datasets", "synthetic,medmnist-breast"]),
        component("cross-dataset table", ["compare_real.py"]),
        component("selection maps: blood",
                  ["compare_selection_maps.py", "--dataset", "medmnist-blood"]),
        component("selection maps: pneumonia",
                  ["compare_selection_maps.py", "--dataset", "medmnist-pneumonia"]),
    ]
    if tier == "standard":
        return standard
    extra = [
        component("real: medmnist extended",
                  ["run_real_benchmark.py",
                   "--datasets", "medmnist-oct,medmnist-derma,medmnist-tissue,"
                                 "medmnist-path,medmnist-retina",
                   "--n-max", "3000", "--out-prefix", "medmnist_extended"]),
        component("lam frontier: medmnist-breast",
                  ["run_lam_frontier.py", "--dataset", "medmnist-breast",
                   "--n-max", "780", "--folds", "5"]),
    ]
    return standard + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["smoke", "standard", "full"], default="standard")
    args = ap.parse_args()

    comps = plan(args.tier)
    print(f"benchmark suite, tier '{args.tier}': {len(comps)} components", flush=True)
    failures = []
    for i, comp in enumerate(comps, 1):
        t0 = time.perf_counter()
        print(f"\n[{i}/{len(comps)}] {comp['name']}", flush=True)
        r = subprocess.run(comp["cmd"], cwd=str(HERE.parent))
        mins = (time.perf_counter() - t0) / 60
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"[{i}/{len(comps)}] {comp['name']}: {status} ({mins:.1f} min)", flush=True)
        if r.returncode != 0:
            failures.append(comp["name"])

    print("\n" + "=" * 70)
    if failures:
        print(f"suite finished with {len(failures)} failed component(s): {failures}")
        sys.exit(1)
    print("suite finished: all components ok. See benchmarks/results/ and FINDINGS.md")


if __name__ == "__main__":
    main()
