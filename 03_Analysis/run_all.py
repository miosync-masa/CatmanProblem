from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "00_clean.py",
    "01_manipulation.py",
    "02_judgment_profiles.py",
    "03_parallel.py",
    "04_correlations_sensitivity.py",
    "05_figures.py",
    "06_verify_outputs.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"[Cat-Man] running {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    print("[Cat-Man] all analysis outputs regenerated successfully", flush=True)


if __name__ == "__main__":
    main()
