#!/usr/bin/env python3
"""Calcule la précision (et son intervalle de confiance) d'un échantillon annoté.

  python scripts/qa_report.py echantillon_annote.csv --by pays
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catmap.qa import precision_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("annotated_csv"); ap.add_argument("--by"); ap.add_argument("--sep", default=";")
    a = ap.parse_args()
    df = pd.read_csv(a.annotated_csv, sep=a.sep, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(precision_report(df, a.by).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
