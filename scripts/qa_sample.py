#!/usr/bin/env python3
"""Produit un échantillon reproductible à contrôler à la main.

  python scripts/qa_sample.py sites.csv --n 500 --seed 42 --strata pays --out echantillon.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catmap.qa import make_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites_csv"); ap.add_argument("--out", default="echantillon_qa.csv")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strata", nargs="*", default=["pays"])
    ap.add_argument("--sep", default=";")
    a = ap.parse_args()
    sites = pd.read_csv(a.sites_csv, sep=a.sep, encoding="utf-8-sig")
    s = make_sample(sites, a.n, a.seed, a.strata)
    s.to_csv(a.out, sep=a.sep, index=False, encoding="utf-8-sig")
    print(f"{len(s)} sites écrits dans {a.out} (graine {a.seed}) : remplir 'precision' (1/0/2) puis lancer qa_report.py")


if __name__ == "__main__":
    main()
