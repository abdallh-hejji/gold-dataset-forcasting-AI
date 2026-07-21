"""
main.py (project root)
Orchestrator - runs the full Gold Price Forecasting pipeline end-to-end,
in the same order the models were developed and compared (Phase 1-4).

Usage (run from the project root):
    python main.py                  # run everything
    python main.py --skip-lstm      # skip the two slow LSTM trainings
    python main.py --skip-prophet   # skip Prophet (slowest to install/run)
    python main.py --only baseline arima   # run only specific steps
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# (step_name, script_filename, is_slow)
PIPELINE_STEPS = [
    ("data_loader",      "data_loader.py",       False),
    ("eda",              "eda.py",               False),
    ("baseline",         "train_baseline.py",    False),
    ("arima",            "train_arima.py",       False),
    ("lstm_raw",         "train_lstm.py",        True),
    ("lstm_returns",     "train_lstm_returns.py", True),
    ("xgboost",          "train_xgboost.py",     False),
    ("prophet",          "train_prophet.py",     True),
    ("compare_models",   "compare_models.py",    False),
    ("evaluation",       "evaluation.py",        False),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full gold forecasting pipeline.")
    parser.add_argument("--skip-lstm", action="store_true",
                         help="Skip both LSTM training steps (slowest, several minutes each).")
    parser.add_argument("--skip-prophet", action="store_true",
                         help="Skip Prophet training.")
    parser.add_argument("--only", nargs="+", default=None,
                         help="Run only these step names (space-separated), e.g. --only baseline arima")
    return parser.parse_args()


def run_step(name, script_filename):
    script_path = SRC_DIR / script_filename
    print(f"\n{'=' * 70}")
    print(f"  STEP: {name}  ({script_filename})")
    print(f"{'=' * 70}\n")

    result = subprocess.run([sys.executable, str(script_path)], cwd=SRC_DIR)

    if result.returncode != 0:
        print(f"\n[FAILED] Step '{name}' exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

    print(f"\n[OK] Step '{name}' completed successfully.")


if __name__ == "__main__":
    args = parse_args()

    steps_to_run = PIPELINE_STEPS
    if args.only:
        steps_to_run = [s for s in PIPELINE_STEPS if s[0] in args.only]
    if args.skip_lstm:
        steps_to_run = [s for s in steps_to_run if s[0] not in ("lstm_raw", "lstm_returns")]
    if args.skip_prophet:
        steps_to_run = [s for s in steps_to_run if s[0] != "prophet"]

    print(f"Running {len(steps_to_run)} pipeline step(s): "
          f"{', '.join(s[0] for s in steps_to_run)}")

    for name, script_filename, is_slow in steps_to_run:
        run_step(name, script_filename)

    print(f"\n{'=' * 70}")
    print("  PIPELINE COMPLETE - all steps finished successfully.")
    print(f"{'=' * 70}")