"""Risques naturels et technologiques par point GPS, via l'API Géorisques (BRGM, https://georisques.gouv.fr/api/v1/).

Ce qui est établi (documentation publique lue) : adresse de base ``https://georisques.gouv.fr/api/v1/``, accès ouvert sans
clé, limite annoncée de 1000 requêtes par minute et par adresse IP, point d'accès ``gaspar/risques``.
Ce qui n'a PAS pu être vérifié depuis l'environnement de développement (accès réseau bloqué) : les autres chemins et les noms
exacts des champs des réponses. Le module est donc tolérant : chaque jeu de données est interrogé séparément, un chemin ou un
champ inattendu donne un statut lisible dans le tableau (jamais une exception), et ``probe`` renvoie la réponse brute pour
diagnostic.

Colonnes ajoutées (préfixe ``gr_``) : voir ``new_columns``. Valeurs absentes = ``None`` ; ``gr_statut`` résume la réussite.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

BASE_URL = "https://www.georisques.gouv.fr/api/v1/"
# Quotas annoncés : 1 appel/s sur resultats_rapport_risque, 5 appels/s sur les autres routes /api/v1, 1000/min au total.
RATE_PER_MINUTE = {"rapport": 55, "default": 270}


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    paths: tuple[str, ...]            # chemins essayés dans l'ordre (le premier qui ne renvoie pas 404 est retenu)
    radius: bool = False              # accepte ``rayon`` (m) : on compte alors les objets autour du point
    kind: str = "count"               # count | scalar | report | labels
    fields: tuple[str, ...] = ()      # champs candidats pour ``scalar``
    default: bool = True
    by_insee: bool = False            # demandé par code INSEE de la commune (radon) ; rapport : INSEE + coordonnées


DATASETS: tuple[Dataset, ...] = (
    Dataset("rapport", "Rapport de risques (synthèse par risque ; lent et instable côté API)", ("resultats_rapport_risque",),
            kind="report", by_insee=True, default=False),
    Dataset("commune", "Risques recensés dans la commune (GASPAR)", ("gaspar/risques",), kind="labels"),
    Dataset("sismique", "Zonage sismique", ("zonage_sismique",), kind="scalar", fields=("zone_sismicite", "code_zone")),
    Dataset("radon", "Potentiel radon (commune)", ("radon",), kind="scalar", fields=("classe_potentiel",), by_insee=True),
    Dataset("argiles", "Retrait-gonflement des argiles", ("argiles",), kind="scalar",
            fields=("exposition", "codeExposition", "code_exposition")),
    Dataset("catnat", "Arrêtés de catastrophe naturelle (commune)", ("gaspar/catnat",), radius=True),
    Dataset("mvt", "Mouvements de terrain", ("mvt",), radius=True),
    Dataset("cavites", "Cavités souterraines", ("cavites",), radius=True),
    Dataset("sis", "Secteurs d'information sur les sols (SIS)", ("ssp/sis", "sis"), radius=True),
    Dataset("casias", "Anciens sites industriels (CASIAS)", ("ssp/casias",), radius=True, default=False),
    Dataset("icpe", "Installations classées (ICPE)", ("installations_classees",), radius=True),
    Dataset("tri", "Territoires à risque important d'inondation (TRI)", ("gaspar/tri",), radius=True, default=False),
    Dataset("azi", "Atlas des zones inondables (AZI)", ("gaspar/azi",), radius=True, default=False),
    Dataset("pprn", "Plans de prévention des risques naturels", ("gaspar/pprn",), radius=True, default=False),
)
BY_KEY = {d.key: d for d in DATASETS}
_LABEL_KEYS = ("libelle_risque_jo", "libelle_risque_long", "libelle", "nom", "nom_installation", "raisonSociale", "typeMouvement",
               "type", "lib_risque_jo", "nom_tri", "libelle_tri", "lib_azi", "nom_azi", "libelle_ppr", "nom_ppr")


def new_columns(selected: list[str]) -> list[str]:
    cols = ["gr_statut", "gr_erreurs", "gr_code_insee"]
    for k in selected:
        d = BY_KEY[k]
        if d.kind == "report":
            cols += ["gr_commune_rapport", "gr_adresse_rapport", "gr_rapport_pdf"]   # + colonnes dynamiques gr_nat_*, gr_tech_*
        elif d.kind == "scalar":
            cols.append(f"gr_{k}")
        elif d.kind == "labels":
            cols += [f"gr_{k}_nb", f"gr_{k}_liste"]
        else:
            cols += [f"gr_{k}_nb", f"gr_{k}_detail"]
    return cols


class _RateLimiter:
    def __init__(self, per_minute: int):
        self.gap = 60.0 / max(1, per_minute)
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


def make_limiters() -> dict:
    return {k: _RateLimiter(v) for k, v in RATE_PER_MINUTE.items()}


def _get(url: str, params: dict, session, limiter: Optional[_RateLimiter], timeout: float = 25.0, retries: int = 3):
    """(payload, erreur, http). Réessaie sur 429 / 5xx / erreurs réseau ; ne lève jamais."""
    import requests
    s = session or requests
    err, code = "", 0
    for i in range(retries):
        if limiter:
            limiter.wait()
        try:
            r = s.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
        except Exception as e:  # noqa: BLE001
            err, code = f"réseau ({type(e).__name__})", 0
            time.sleep(1.5 * (i + 1))
            continue
        code = r.status_code
        if code == 200:
            try:
                return r.json(), "", 200
            except ValueError:
                return None, "réponse non JSON", 200
        if code in (429, 502, 503, 504):
            err = f"HTTP {code}"
            time.sleep(2.0 * (i + 1))
            continue
        return None, f"HTTP {code}", code
    return None, err or "échec", code


def _items(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "results", "items", "features"):
            if isinstance(payload.get(k), list):
                return payload[k]
        if "data" in payload and isinstance(payload["data"], dict):
            return [payload["data"]]
    return []


def _label(item) -> str:
    if isinstance(item, dict):
        for k in _LABEL_KEYS:
            if item.get(k):
                return str(item[k])
    return ""


def _count(payload) -> int:
    """Nombre total d'objets : le champ ``results`` (l'API pagine par 10) ; à défaut, la longueur de la liste."""
    items = _items(payload)
    nested = 0
    for it in items:                                   # GASPAR catnat : une ligne par commune avec une liste d'arrêtés
        if isinstance(it, dict):
            sub = next((it[k] for k in ("liste_catnat", "listCatnat", "catnat", "arretes") if isinstance(it.get(k), list)), None)
            nested += len(sub) if sub is not None else 0
    if nested:
        return nested
    if isinstance(payload, dict) and isinstance(payload.get("results"), int):
        return payload["results"]
    return len(items)


def insee_from(payload) -> Optional[str]:
    for it in _items(payload) or ([payload] if isinstance(payload, dict) else []):
        if isinstance(it, dict):
            for k in ("code_insee", "codeInsee", "insee", "code_commune"):
                if it.get(k):
                    return str(it[k]).strip()
    return None


def _detail(payload, limit: int = 3) -> str:
    out: list[str] = []
    for it in _items(payload):
        subs = []
        if isinstance(it, dict):
            subs = next((it[k] for k in ("liste_catnat", "listCatnat", "catnat", "arretes") if isinstance(it.get(k), list)), [])
        for x in (subs or [it]):
            lab = _label(x)
            if lab and lab not in out:
                out.append(lab)
    return " | ".join(out[:limit]) + (" …" if len(out) > limit else "")


def _scalar(payload, fields: tuple[str, ...]) -> Optional[str]:
    items = _items(payload) or ([payload] if isinstance(payload, dict) else [])
    for it in items:
        if isinstance(it, dict):
            for f in fields:
                if it.get(f) not in (None, ""):
                    return str(it[f])
    return None


def _flatten_report(payload) -> dict:
    """Rapport de risques : une colonne Oui/Non par risque (+ statut). Structure tolérante : à défaut, feuilles scalaires."""
    out: dict = {}
    if not isinstance(payload, dict):
        return out
    root = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    adr = root.get("adresse")
    com = root.get("commune")
    out["gr_adresse_rapport"] = (adr.get("libelle") if isinstance(adr, dict) else adr) or None
    out["gr_commune_rapport"] = (com.get("libelle") if isinstance(com, dict) else com) or None
    out["gr_rapport_pdf"] = root.get("url") or root.get("urlPdf") or root.get("url_pdf") or None
    found = False
    for block, pref in (("risquesNaturels", "gr_nat_"), ("risquesTechnologiques", "gr_tech_")):
        sub = root.get(block)
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            if isinstance(v, dict) and "present" in v:
                found = True
                out[f"{pref}{k}"] = "Oui" if v.get("present") else "Non"
                st = v.get("libelleStatutAdresse") or v.get("libelleStatutCommune")
                if st:
                    out[f"{pref}{k}_statut"] = str(st)
    if not found:                                       # structure inattendue : on garde les feuilles scalaires lisibles
        def walk(o, path):
            if len(out) > 80:
                return
            if isinstance(o, dict):
                for kk, vv in o.items():
                    walk(vv, path + [str(kk)])
            elif isinstance(o, (str, int, float, bool)) and path:
                out["gr_rapport_" + re.sub(r"\W+", "_", "_".join(path))[:60]] = o
        walk(root, [])
    return out


def _latlon(lat: float, lon: float) -> str:
    return f"{lon:.6f},{lat:.6f}"          # l'API attend « longitude,latitude »


def query_dataset(ds: Dataset, lat: float, lon: float, radius_m: int, session=None, limiter=None,
                  path_hint: Optional[dict] = None, insee: Optional[str] = None):
    """Interroge un jeu de données pour un point. Retourne (valeurs, erreur, payload brut)."""
    params = {"latlon": _latlon(lat, lon)}
    if ds.by_insee:
        if not insee:
            return {}, "code INSEE de la commune introuvable", None
        params = {"code_insee": insee} if ds.kind != "report" else {"code_insee": insee, "latlon": params["latlon"]}
    if ds.radius:
        params["rayon"] = int(radius_m)
    paths = ds.paths
    if path_hint and path_hint.get(ds.key) in ds.paths:
        paths = (path_hint[ds.key],)
    payload, err, code = None, "", 0
    for p in paths:
        payload, err, code = _get(BASE_URL + p, params, session, limiter)
        if code != 404 and err != "réponse non JSON":
            if code == 200 and path_hint is not None:
                path_hint[ds.key] = p
            break
    if payload is None:
        return {}, ("chemin introuvable (HTTP 404)" if code == 404 else err), None
    k = ds.key
    if ds.kind == "report":
        return _flatten_report(payload), "", payload
    if ds.kind == "scalar":
        return {f"gr_{k}": _scalar(payload, ds.fields)}, "", payload
    if ds.kind == "labels":
        labs: list[str] = []
        for it in _items(payload):
            for d in (it.get("risques_detail") or []) if isinstance(it, dict) else []:
                lab = _label(d)
                if lab and lab not in labs:
                    labs.append(lab)
        return {f"gr_{k}_nb": len(labs), f"gr_{k}_liste": " | ".join(labs) or None}, "", payload
    return {f"gr_{k}_nb": _count(payload), f"gr_{k}_detail": _detail(payload) or None}, "", payload


def enrich_points(df: pd.DataFrame, lat_col: str, lon_col: str, selected: list[str], radius_m: int = 1000,
                  max_points: int = 300, session=None, workers: int = 6,
                  on_progress: Optional[Callable[[int, int], None]] = None) -> tuple[pd.DataFrame, dict]:
    """Ajoute les colonnes ``gr_*``. Points identiques (6 décimales) interrogés une seule fois. Ne lève pas pour une erreur d'API."""
    selected = [k for k in selected if k in BY_KEY]
    out = df.copy().reset_index(drop=True)
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    keys = [(round(a, 6), round(b, 6)) if v else None for a, b, v in zip(lat, lon, valid)]
    uniq = list(dict.fromkeys(k for k in keys if k))
    skipped = max(0, len(uniq) - max_points)
    uniq = uniq[:max_points]
    limiters, hints = make_limiters(), {}
    results: dict = {}
    raw_errors: dict[str, int] = {}
    lock = threading.Lock()
    done = [0]
    need_insee = any(BY_KEY[k].by_insee for k in selected)

    def work(pt):
        vals, errs, insee = {}, [], None
        if need_insee or "commune" in selected:      # la commune (GASPAR) donne aussi le code INSEE exigé par rapport et radon
            v, e, payload = query_dataset(BY_KEY["commune"], pt[0], pt[1], radius_m, session, limiters["default"], hints)
            insee = insee_from(payload)
            if "commune" in selected:
                vals.update(v)
                if e:
                    errs.append(f"commune: {e}")
            elif e:
                errs.append(f"commune (code INSEE): {e}")
        vals["gr_code_insee"] = insee
        for k in selected:
            if k == "commune":
                continue
            ds = BY_KEY[k]
            v, e, _ = query_dataset(ds, pt[0], pt[1], radius_m, session, limiters["rapport" if k == "rapport" else "default"],
                                    hints, insee)
            vals.update(v)
            if e:
                errs.append(f"{k}: {e}")
        with lock:
            for e in errs:
                raw_errors[e] = raw_errors.get(e, 0) + 1
        vals["gr_erreurs"] = " ; ".join(errs) or None
        vals["gr_statut"] = "ok" if not errs else ("échec" if len(errs) >= len(selected) else "partiel")
        with lock:
            results[pt] = vals

    # ``on_progress`` (barre Streamlit) doit être appelé depuis le thread principal, jamais depuis un thread de travail
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(work, pt) for pt in uniq]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - un point en échec ne doit pas faire tomber les autres
                with lock:
                    raw_errors[f"erreur interne: {type(e).__name__}"] = raw_errors.get(f"erreur interne: {type(e).__name__}", 0) + 1
            done[0] += 1
            if on_progress:
                on_progress(done[0], len(uniq))
    rows = [results.get(k, {"gr_statut": "non interrogé" if k else "coordonnées invalides"}) for k in keys]
    add = pd.DataFrame(rows, index=out.index)
    cols = [c for c in new_columns(selected) if c in add.columns] + [c for c in add.columns if c not in new_columns(selected)]
    add = add.reindex(columns=list(dict.fromkeys(cols)))
    summary = {"points": len(out), "interroges": len(uniq), "non_interroges": skipped, "invalides": int((~valid).sum()),
               "erreurs": raw_errors, "chemins": hints}
    return pd.concat([out, add], axis=1), summary


def probe(lat: float, lon: float, selected: list[str], radius_m: int = 1000, session=None) -> list[dict]:
    """Diagnostic : pour chaque jeu de données, chemin retenu, statut, nombre d'objets, extrait brut de la réponse."""
    import json
    rows = []
    _, _, cp = query_dataset(BY_KEY["commune"], lat, lon, radius_m, session, None, {})
    insee = insee_from(cp)
    for k in selected:
        ds = BY_KEY[k]
        hints: dict = {}
        vals, err, payload = query_dataset(ds, lat, lon, radius_m, session, None, hints, insee)
        rows.append({"jeu de données": ds.label, "chemin": hints.get(k, ds.paths[0]), "statut": err or "ok",
                     "objets": _count(payload) if payload is not None and ds.kind == "count" else None,
                     "valeurs lues": json.dumps(vals, ensure_ascii=False)[:200] if vals else "",
                     "réponse brute (extrait)": (json.dumps(payload, ensure_ascii=False)[:300] if payload is not None else "")})
    rows.insert(0, {"jeu de données": "(code INSEE de la commune, via GASPAR)", "chemin": "gaspar/risques", "statut": "ok" if insee else "introuvable",
                    "objets": None, "valeurs lues": insee or "", "réponse brute (extrait)": ""})
    return rows
