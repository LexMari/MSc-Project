"""CLI entry point for running a batch experiment

Usage: python main_batch.py batches/some_batch.yaml

The per-row results already go to output_csv. This also saves the console's own
progress log to results/logs/, the same way main_headless.py does -
the CSV alone doesn't capture which combination ran in which order, or
the total run count.
"""
import os
import sys
from datetime import datetime

from sim_core.batch import run_batch
from main_headless import Tee

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_batch.py <batch_config.yaml>")
        sys.exit(1)

    batch_path = sys.argv[1]

    os.makedirs("results/logs", exist_ok=True)
    batch_stem = os.path.splitext(os.path.basename(batch_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/logs/batch_{batch_stem}_{timestamp}.txt"

    real_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        sys.stdout = Tee(real_stdout, log_file)
        try:
            rows = run_batch(batch_path)
            print(f"\n{len(rows)} rows written")
        finally:
            sys.stdout = real_stdout
    print(f"(output also saved to {log_path})")
