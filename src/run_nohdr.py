# Train + tune + evaluate the robust (no header length) model, with step checkpoints.
# Each step is skipped if its output already exists, so a crash or Ctrl+C never repeats finished work.
# Usage:  python -m src.run_nohdr            (resume from the first unfinished step)
#         python -m src.run_nohdr --force    (redo every step)
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
TAG = "xgb_v1_nofp_nohdr_clean_r1500"
LOG = REPORTS / f"run_{TAG}.log"

STEPS = [  # (name, command, files that prove the step finished, rough minutes)
    ("1/3 train XGBoost (GPU)",
     ["-m", "src.train_xgb", "--nofp", "--nohdr", "--clean", "--rounds", "1500"],
     [MODELS / f"{TAG}.json", MODELS / f"{TAG}_config.json"], "3-5"),
    ("2/3 base thresholds (CPU)",
     ["-m", "src.xgb_thresholds", "--tag", TAG],
     [MODELS / f"{TAG}_thresholds.json"], "1-2"),
    ("3/3 Variant B + incidents + bootstrap CI (CPU)",
     ["-m", "src.refine_final", "--tag", TAG],
     [MODELS / f"{TAG}_variantB_thresholds.json", REPORTS / f"bootstrap_ci_final_{TAG}.csv"], "3-5"),
]


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def main():
    force = "--force" in sys.argv
    REPORTS.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        log(f"\n===== run {datetime.now():%Y-%m-%d %H:%M} (force={force}) =====", fh)
        t_all = time.time()
        for name, cmd, outputs, eta in STEPS:
            if not force and all(p.exists() for p in outputs):
                log(f"[skip] {name}: already done ({outputs[0].name})", fh)
                continue
            log(f"[run ] {name}  (about {eta} min)", fh)
            t0 = time.time()
            proc = subprocess.Popen([sys.executable, "-u", *cmd], cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)
            proc.wait()
            if proc.returncode != 0:
                for p in outputs:   # a half-written output must not count as a checkpoint
                    if p.exists() and p.stat().st_mtime >= t0:
                        p.unlink()
                log(f"[FAIL] {name} (exit {proc.returncode}). Fix and rerun: python -m src.run_nohdr", fh)
                sys.exit(proc.returncode)
            log(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", fh)
        log(f"All steps finished in {(time.time() - t_all) / 60:.1f} min. Full log: {LOG}", fh)


if __name__ == "__main__":
    main()
