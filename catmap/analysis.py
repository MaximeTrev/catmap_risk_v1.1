"""Indicateurs et agrégations pour la restitution."""
from __future__ import annotations

import pandas as pd


def completeness(df: pd.DataFrame, cols=("pays", "naf_section", "surface_m2")) -> dict[str, float]:
    """Part (%) de lignes renseignées pour chaque colonne ; 0 si aucun site."""
    if df.empty:
        return {c: 0.0 for c in cols}
    return {c: round(100.0 * float(df[c].notna().mean()), 1) if c in df else 0.0 for c in cols}


def top_with_other(series: pd.Series, top_n: int = 10, other_label: str = "Autres") -> pd.DataFrame:
    """Comptage par modalité, regroupant au-delà de ``top_n`` dans « Autres »."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return pd.DataFrame({"modalite": [], "n": []})
    if len(counts) > top_n:
        head = counts.iloc[:top_n]
        counts = pd.concat([head, pd.Series({other_label: int(counts.iloc[top_n:].sum())})])
    return pd.DataFrame({"modalite": counts.index.astype(str), "n": counts.values.astype(int)})


def summary_by_company(sites: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par entreprise demandée (y compris celles sans résultat)."""
    base = report[["nom_demande", "nom_recherche", "statut"]].copy()
    if sites.empty:
        base["n_sites"] = 0
        base["surface_totale_m2"] = 0.0
        base["n_pays"] = 0
        base["pays_principal"] = None
        base["secteur_principal"] = None
        return base

    def mode(s: pd.Series):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else None

    g = sites.groupby("nom_demande")
    agg = pd.DataFrame({
        "n_sites": g.size(),
        "surface_totale_m2": g["surface_m2"].sum(min_count=1),
        "n_pays": g["pays"].nunique(),
        "pays_principal": g["pays"].agg(mode),
        "secteur_principal": g["naf_section"].agg(mode),
    }).reset_index()
    out = base.merge(agg, on="nom_demande", how="left")
    out["n_sites"] = out["n_sites"].fillna(0).astype(int)
    return out
