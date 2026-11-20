"""Accès au catalogue Copernicus (CDS) : paramètres réels des jeux de données utiles au vent et aux tempêtes.

Étape de diagnostic : ``fetch_form`` lit la description publique d'un jeu de données (liste des champs de la requête et des valeurs
autorisées, sans clé d'API) pour que les requêtes ultérieures soient construites avec les vrais noms de paramètres, et non avec des
suppositions. NON testé en réel depuis l'environnement de développement (réseau bloqué).

Jeux de données :
* ``reanalysis-era5-single-levels-timeseries`` : ERA5 horaire en série temporelle par point, 1940 à aujourd'hui, mondial, 0,25° (CC-BY) ;
* ``sis-european-wind-storm-reanalysis`` : trajectoires (CSV), empreintes de rafales par tempête (NetCDF-4) et indicateurs annuels, Europe,
  1940 à aujourd'hui (CC-BY, DOI 10.24381/bf1f06a9).
"""
from __future__ import annotations

import json
from typing import Any, Optional

CATALOGUE_URL = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{dataset}/form.json"
DATASETS = {
    "timeseries": "reanalysis-era5-single-levels-timeseries",
    "storms": "sis-european-wind-storm-reanalysis",
}


class CopernicusError(Exception):
    pass


def fetch_form(dataset: str, session=None, timeout: int = 30) -> Any:
    """Description JSON du formulaire de requête (champs et valeurs autorisées). Ne demande aucune clé."""
    import requests
    try:
        r = (session or requests).get(CATALOGUE_URL.format(dataset=dataset), timeout=timeout, headers={"Accept": "application/json"})
    except Exception as e:                                       # noqa: BLE001
        raise CopernicusError(f"{type(e).__name__}: {str(e)[:160]}") from e
    if r.status_code != 200:
        raise CopernicusError(f"HTTP {r.status_code}")
    try:
        return r.json()
    except Exception as e:                                       # noqa: BLE001
        raise CopernicusError(f"réponse non JSON ({type(e).__name__})") from e


def _collect(node: Any, out: list, labels: dict) -> None:
    if isinstance(node, dict):
        vals = node.get("values")
        if isinstance(vals, list):
            out.extend(str(v) for v in vals if isinstance(v, (str, int, float)))
        lab = node.get("labels")
        if isinstance(lab, dict):
            labels.update({str(k): str(v) for k, v in lab.items()})
        for v in node.values():
            _collect(v, out, labels)
    elif isinstance(node, list):
        for v in node:
            _collect(v, out, labels)


def summarize_form(form: Any) -> list[dict]:
    """Une ligne par champ : nom de l'API, type, libellé, nombre de valeurs, exemples (et valeurs contenant « gust »)."""
    widgets = form if isinstance(form, list) else (form.get("widgets") or form.get("form") or []) if isinstance(form, dict) else []
    rows = []
    for w in widgets:
        if not isinstance(w, dict):
            continue
        vals: list = []
        labs: dict = {}
        _collect(w.get("details", w), vals, labs)
        vals = list(dict.fromkeys(vals))
        gust = [v for v in vals if "gust" in v.lower() or "gust" in labs.get(v, "").lower()]
        rows.append({"champ (API)": w.get("name"), "type": w.get("type"), "libellé": w.get("label"), "nb valeurs": len(vals),
                     "exemples": ", ".join(vals[:10]) + (" …" if len(vals) > 10 else ""),
                     "valeurs « gust »": ", ".join(gust[:8])})
    return rows


def pretty(form: Any) -> str:
    return json.dumps(form, ensure_ascii=False, indent=1)


def key_present(secrets: dict, env: Optional[dict] = None) -> bool:
    """Présence (sans l'afficher) d'une clé d'API Copernicus dans les secrets ou l'environnement."""
    env = env or {}
    for name in ("CDSAPI_KEY", "CDS_API_KEY", "cdsapi_key"):
        if str(env.get(name) or secrets.get(name) or "").strip():
            return True
    return False


def get_key(secrets: dict, env: Optional[dict] = None) -> str:
    """Valeur de la clé d'API Copernicus (secrets puis environnement). Ne jamais l'afficher ni la journaliser."""
    env = env or {}
    for name in ("CDSAPI_KEY", "CDS_API_KEY", "cdsapi_key"):
        v = str(secrets.get(name) or env.get(name) or "").strip()
        if v:
            return v
    return ""
