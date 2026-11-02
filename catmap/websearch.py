"""Recherche web de documents et extraction de fournisseurs via Linkup (https://www.linkup.so).

Clé à définir dans le secret ``LINKUP_API_KEY``. D'après la documentation de Linkup (lue, non testée ici) :
* point d'accès ``GET https://api.linkup.so/v1/search``, en-tête ``Authorization: Bearer <clé>`` ;
* ``outputType=searchResults`` : liste de pages (``results[].name/url/content``), 0,005 $ l'appel en profondeur « standard » ;
* ``outputType=structured`` + ``structuredOutputSchema`` : un objet JSON conforme au schéma, 0,055 $ l'appel en « deep ».
Le format exact des réponses n'a pas pu être vérifié : toute réponse inattendue produit une erreur lisible, jamais une
exception brute.
"""
from __future__ import annotations

import json
import re
from typing import Optional
from urllib.parse import urlparse

import requests

LINKUP_URL = "https://api.linkup.so/v1/search"
COST_SEARCH_USD = 0.005          # par requête searchResults, profondeur standard
COST_STRUCTURED_USD = 0.055      # par requête structured, profondeur deep

SUPPLIER_SCHEMA = {
    "type": "object",
    "properties": {
        "suppliers": {
            "type": "array",
            "description": "Entreprises explicitement présentées comme fournisseurs, fabricants ou sous-traitants du client.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nom de l'entreprise fournisseur"},
                    "country": {"type": "string", "description": "Pays du fournisseur ou du site, si connu"},
                    "relation": {"type": "string", "enum": ["fournisseur direct", "fabricant", "sous-traitant", "matières premières", "logistique", "autre"],
                                 "description": "Nature de la relation (une seule valeur de la liste)"},
                    "tier": {"type": "string", "description": "1 pour un fournisseur direct, 2 ou plus pour un fournisseur de fournisseur"},
                    "source_url": {"type": "string", "description": "Adresse exacte de la page ou du document consulté"},
                    "quote": {"type": "string", "description": "Citation exacte, mot pour mot, du document qui justifie la relation"},
                },
                "required": ["name", "source_url", "quote"],
            },
        }
    },
    "required": ["suppliers"],
}


class SearchError(RuntimeError):
    pass


def build_queries(client: str) -> list[str]:
    c = client.strip()
    return [f"{c} list of suppliers", f"{c} supplier list factories pdf", f"{c} liste des fournisseurs",
            f"{c} sustainability report supplier disclosure"]


def _call(params: dict, key: str, session=None, timeout: float = 90.0):
    """Appelle l'API Linkup. POST (JSON) d'abord ; si le serveur répond 404 ou 405, nouvel essai en GET (paramètres d'adresse).
    En cas de refus de la requête (400, 422), le message du service est repris pour faciliter le diagnostic."""
    s = session or requests
    headers = {"Authorization": f"Bearer {key}"}
    r = None
    try:
        r = s.post(LINKUP_URL, json=params, headers=dict(headers, **{"Content-Type": "application/json"}), timeout=timeout)
        if r.status_code in (404, 405):
            r = s.get(LINKUP_URL, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise SearchError(f"Service de recherche injoignable ({type(e).__name__}).") from e
    if r.status_code in (401, 403):
        raise SearchError("Clé de recherche refusée (vérifiez LINKUP_API_KEY).")
    if r.status_code == 402:
        raise SearchError("Crédits Linkup épuisés.")
    if r.status_code == 429:
        raise SearchError("Quota de recherche dépassé.")
    if r.status_code != 200:
        detail = ""
        try:
            detail = " — " + re.sub(r"\s+", " ", str(getattr(r, "text", "") or ""))[:200]
        except Exception:  # noqa: BLE001
            pass
        raise SearchError(f"Service de recherche : HTTP {r.status_code}{detail}")
    try:
        return r.json()
    except ValueError as e:
        raise SearchError("Réponse du service de recherche illisible.") from e


def search_documents(queries: list[str], key: str, *, session=None, max_results: int = 8) -> list[dict]:
    """Recherche de pages et de PDF. Retourne [{title, url, snippet}] sans doublons."""
    seen: dict[str, dict] = {}
    for q in queries:
        data = _call({"q": q, "depth": "standard", "outputType": "searchResults", "maxResults": max_results}, key, session)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise SearchError("Format de réponse inattendu du service de recherche.")
        for it in results:
            it = it or {}
            url = str(it.get("url") or "")
            if url.startswith("http") and url not in seen:
                seen[url] = {"title": str(it.get("name") or url)[:150], "url": url,
                             "snippet": str(it.get("content") or it.get("snippet") or "")[:300]}
    return list(seen.values())


FOCUS = {
    "production": "Production : usines, fabricants, matières premières",
    "all": "Tous les fournisseurs (y compris logistique, services, informatique)",
}


def extract_suppliers(client: str, key: str, *, session=None, depth: str = "deep", focus: str = "production") -> list[dict]:
    """Extraction structurée par l'IA de Linkup : [{name, country, relation, tier, source_url, quote}]."""
    scope = ("Only include companies that supply products, components, raw materials or manufacturing capacity "
             "(factories, manufacturers, subcontract producers, raw material suppliers). EXCLUDE software, IT, recruitment, "
             "marketing, consulting and other service providers. "
             if focus == "production" else
             "Include suppliers of products, components, raw materials, manufacturing, logistics and services. ")
    q = (f"List as many as possible (up to 100) of the suppliers that {client.strip()} publicly declares: official supplier "
         "lists (often published as PDF or Excel on the company's website), sustainability / ESG / annual reports, sourcing "
         "disclosures, factory lists. Search in several languages. " + scope +
         "Only include companies explicitly named as suppliers of that company in a source you consulted, with the exact URL "
         "and a verbatim quote from it. Do not include customers, competitors or subsidiaries. Do not invent anything.")
    data = _call({"q": q, "depth": depth, "outputType": "structured",
                  "structuredOutputSchema": json.dumps(SUPPLIER_SCHEMA)}, key, session, timeout=120.0)
    if isinstance(data, dict) and "suppliers" not in data:           # certaines enveloppes possibles : data / result / output
        for k in ("data", "result", "output", "structured"):
            if isinstance(data.get(k), dict) and "suppliers" in data[k]:
                data = data[k]
                break
    sup = data.get("suppliers") if isinstance(data, dict) else None
    if not isinstance(sup, list):
        raise SearchError("Format de réponse inattendu pour l'extraction des fournisseurs.")
    return [d for d in sup if isinstance(d, dict) and str(d.get("name") or "").strip()]


def rank(results: list[dict], client: str, limit: int = 8) -> list[dict]:
    """Privilégie les PDF, les adresses contenant « supplier/fournisseur » et le site de l'entreprise elle-même."""
    token = re.sub(r"[^a-z0-9]", "", client.lower())

    def score(r):
        u = r["url"].lower()
        host = urlparse(u).hostname or ""
        return ((3 if re.search(r"suppl|fournisseur|factory|factories|vendor|sourcing", u) else 0)
                + (2 if u.split("?")[0].endswith(".pdf") else 0)
                + (2 if token and token in re.sub(r"[^a-z0-9]", "", host) else 0))
    return sorted(results, key=score, reverse=True)[:limit]
