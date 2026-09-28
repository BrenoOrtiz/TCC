"""Refaz só a análise (estatística, gráficos, relatório) a partir de results/runs.csv.

Uso:
    uv run python analyze.py                 # results/
    uv run python analyze.py results_smoke   # outro diretório
"""

import logging
import sys
from pathlib import Path

from config import RESULTS_DIR
from src import analysis

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    analysis.run(Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR)
