"""Contrôle qualité des coordonnées : échantillon reproductible + précision avec intervalle.

Remplace ``testsCoordonnees.py`` (A11) : pas de Selenium ni de saisie clavier bloquante, pas
d'effet de bord à l'import, échantillon à graine fixe et stratifié, colonnes lues par nom,
sauvegarde atomique, métriques avec intervalle de confiance de Wilson.

Protocole : ``make_sample`` produit un CSV avec liens (OSM, Google Maps, Nominatim) et des
colonnes vides ``precision`` / ``commentaire`` à remplir à la main (1 = bonne, 0 = mauvaise,
2 = incertaine) ; ``precision_report`` calcule ensuite les indicateurs.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import pandas as pd

CODES = {1: "bonne", 0: "mauvaise", 2: "incertaine"}


def make_sample(sites: pd.DataFrame, n: int = 500, seed: int = 42,
                strata: Optional[Sequence[str]] = ("pays",)) -> pd.DataFrame:
    """Échantillon reproductible. Si ``strata`` est fourni, l'allocation est proportionnelle
    par strate avec au moins un site par strate (dans la limite de ``n``)."""
    if sites.empty:
        return sites.assign(precision=None, commentaire=None)
    n = min(n, len(sites))
    strata = [c for c in (strata or ()) if c in sites.columns]
    if strata:
        key = sites[strata].fillna("?").astype(str).agg("|".join, axis=1)
        counts = key.value_counts()
        alloc = (counts / counts.sum() * n).apply(math.floor).clip(lower=1)
        while alloc.sum() > n:                       # trop de strates minuscules : on retire aux plus grosses
            alloc[alloc.idxmax()] -= 1
        parts = [sites[key == k].sample(n=min(int(a), int(counts[k])), random_state=seed) for k, a in alloc.items()]
        sample = pd.concat(parts)
        extra = n - len(sample)
        if extra > 0:
            rest = sites.drop(sample.index)
            sample = pd.concat([sample, rest.sample(n=min(extra, len(rest)), random_state=seed)])
    else:
        sample = sites.sample(n=n, random_state=seed)
    sample = sample.sort_index().copy()
    lat, lon = sample["lat"].round(6).astype(str), sample["long"].round(6).astype(str)
    sample["lien_google_maps"] = "https://www.google.com/maps/search/?api=1&query=" + lat + "," + lon
    sample["lien_nominatim"] = "https://nominatim.openstreetmap.org/ui/reverse.html?lat=" + lat + "&lon=" + lon + "&zoom=18"
    sample["precision"] = None
    sample["commentaire"] = None
    return sample


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def precision_report(annotated: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Précision = bonnes / (bonnes + mauvaises). Les « incertaines » sont exclues et
    encadrent la valeur (borne basse = elles comptent comme mauvaises, haute = bonnes)."""
    df = annotated.copy()
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df = df[df["precision"].isin([0, 1, 2])]

    def one(g: pd.DataFrame) -> pd.Series:
        good, bad, unsure = int((g.precision == 1).sum()), int((g.precision == 0).sum()), int((g.precision == 2).sum())
        n = good + bad
        lo, hi = wilson_interval(good, n)
        return pd.Series({
            "n_controles": good + bad + unsure, "bonnes": good, "mauvaises": bad, "incertaines": unsure,
            "precision": good / n if n else float("nan"), "ic95_bas": lo, "ic95_haut": hi,
            "precision_min": good / (n + unsure) if (n + unsure) else float("nan"),
            "precision_max": (good + unsure) / (n + unsure) if (n + unsure) else float("nan"),
        })

    if by and by in df.columns:
        return df.groupby(by).apply(one, include_groups=False).reset_index()
    return one(df).to_frame().T
