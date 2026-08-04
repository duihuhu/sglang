#!/usr/bin/env python3
"""Regenerate node_scalability_energy_clustered.pdf from node_scalability_all.json."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHARTS = Path(__file__).resolve().parent.parent / "charts"


def main() -> None:
    subprocess.run(
        [sys.executable, str(CHARTS / "plot_node_scalability_energy.py")],
        check=True,
        cwd=str(CHARTS),
    )


if __name__ == "__main__":
    main()
