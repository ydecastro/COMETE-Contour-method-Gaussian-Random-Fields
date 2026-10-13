"""Idempotent launcher for the round-2 computations (author's rule 2026-09-23: at most 4 workers).

Runs, in order and only if their output does not exist yet:
  1. the simulation grid (results_full/<tag>.csv), one (model, kappa) configuration at a time;
  2. the window-size study (n = 500 and n = 2000);
  3. the Planck calibration on 100 simulated isotropic maps (results_planck/planck_sims_*.csv);
  4. the figure scripts (figures_r2.py, planck_figures.py).

Usage:  python3 resume_grid.py [--workers 4] [--dry-run]
Log:    results_resume.log (when launched through resume_all.sh)
The number of workers is capped at 4 (W = min(workers, 4) below); edit the cap to use more.
The runs of the paper were made with 14, 7 and 4 workers (see the README); the results do not
depend on the number of workers (seeds seed0 + i per FFT).
"""
import argparse, glob, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Python used for the Planck steps (healpy needed): the author's venv if it exists, else the current interpreter
VENV = Path.home() / ".venvs" / "anisotropy" / "bin" / "python"
if not VENV.exists():
    VENV = Path(sys.executable)

GRID = []
for model, extra, suffix in [("gaussian", [], ""), ("matern", ["--nu", "2.5"], "_nu2.5"), ("cauchy", ["--alpha", "1.5"], "_al1.5"), ("cauchy", ["--alpha", "0.75"], "_al0.75")]:
    for kappa in [0.0, 0.1, 0.25, 0.5, 0.9]:
        GRID.append(dict(model=model, extra=extra, suffix=suffix, kappa=kappa, n=1000, nrep=5000, N=[10, 5, 20, 25]))
for model, extra, suffix in [("gaussian", [], ""), ("cauchy", ["--alpha", "1.5"], "_al1.5")]:
    for kappa in [0.0, 0.25]:
        GRID.append(dict(model=model, extra=extra, suffix=suffix, kappa=kappa, n=500, nrep=5000, N=[5, 10]))
        GRID.append(dict(model=model, extra=extra, suffix=suffix, kappa=kappa, n=2000, nrep=2000, N=[20, 10, 40, 50]))


def tag(c):
    return f"{c['model']}_k{c['kappa']:.2f}_n{c['n']}_ul5_N{c['N'][0]}_s0{c['suffix']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    W = min(a.workers, 4)   # hard cap decided by the author on 2026-09-23
    os.chdir(HERE)
    todo = [c for c in GRID if not (HERE / "results_full" / f"{tag(c)}.csv").exists()]
    print(f"[resume] {len(GRID) - len(todo)} configurations saved, {len(todo)} to run, {W} workers", flush=True)
    for c in todo:
        cmd = [sys.executable, "-u", "comete_r2.py", "--model", c["model"], *c["extra"], "--kappa", str(c["kappa"]), "--level", "0", "1", "2",
               "--nrep", str(c["nrep"]), "--n", str(c["n"]), "--workers", str(W), "--N-cells", *map(str, c["N"]), "--out", "results_full"]
        print("[resume]", tag(c), "->", " ".join(cmd[2:]), flush=True)
        if not a.dry_run:
            t0 = time.time(); r = subprocess.run(cmd); print(f"[resume] exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
    sims = glob.glob(str(HERE / "results_planck" / "planck_sims_*.csv"))
    if not sims:
        cmd = [str(VENV), "-u", "planck_r2.py", "sims", "--nsim", "100", "--workers", str(W)]
        print("[resume] Planck simulations ->", " ".join(cmd[1:]), flush=True)
        if not a.dry_run:
            with open(HERE / "results_planck_sims.log", "w") as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT); print(f"[resume] exit {r.returncode}", flush=True)
    else:
        print("[resume] Planck simulations already saved:", sims, flush=True)
    if not a.dry_run:
        subprocess.run([sys.executable, "figures_r2.py"])
        subprocess.run([str(VENV), "planck_figures.py"])
    print("RESUME ALL DONE", flush=True)


if __name__ == "__main__":
    main()
