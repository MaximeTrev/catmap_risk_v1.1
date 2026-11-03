"""Supply chain : recensement des noms de fournisseurs d'une entreprise (le « client »), puis géolocalisation par la
recherche OpenStreetMap existante.

Sources de noms prises en charge (gratuites, sans API d'IA) :
* tableaux d'une liste de fournisseurs publiée (page web, PDF, Excel, CSV) ;
* « mode assisté » : on fabrique une consigne à coller dans Claude / ChatGPT, puis on relit leur réponse ;
* saisie manuelle.

Niveaux de confiance (relation client-fournisseur) :
    A = liste publiée par l'entreprise elle-même (document fourni) ;
    B = réservé (base ouverte de sites de production, ex. Open Supply Hub) ;
    C = réponse d'un assistant IA avec adresse de source et citation (à vérifier) ;
    D = réponse d'un assistant IA sans source exploitable ;
    n/d = saisie manuelle, non documentée.
"""
from __future__ import annotations

import csv
import io
import ipaddress
import re
import socket
import unicodedata
from typing import Optional
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests

from .names import clean_company_name

CANDIDATE_COLUMNS = ["fournisseur", "pays", "type_relation", "rang", "source", "source_url", "citation", "confiance"]
CONF_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "n/d": 4}
RELATION_TYPES = ["fournisseur direct", "fabricant", "sous-traitant", "matières premières", "logistique", "autre"]
MAX_BYTES = 20 * 1024 * 1024

NAME_HINTS = ["supplier name", "vendor name", "factory name", "facility name", "manufacturer name", "company name",
              "nom du fournisseur", "nom de l'usine", "raison sociale", "supplier", "vendor", "factory", "facility",
              "manufacturer", "fournisseur", "usine", "producer", "mill", "company", "entreprise", "société", "societe",
              "name", "nom", "site"]
COUNTRY_HINTS = ["country/region", "country", "pays", "land", "país", "pais", "paese", "nation"]


class SupplyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- téléchargement sécurisé
def _check_public_host(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise SupplyError("Adresse invalide : seules les adresses http(s) sont acceptées.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except OSError as e:
        raise SupplyError(f"Adresse introuvable ({p.hostname}).") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise SupplyError("Adresse refusée (réseau interne).")


def fetch_url(url: str, *, session=None, timeout: float = 40.0, max_redirects: int = 4) -> tuple[bytes, str]:
    """Télécharge un document public. Retourne (contenu, type MIME). Refuse les adresses internes (protection SSRF)
    et les fichiers de plus de 20 Mo."""
    s = session or requests
    cur = url.strip()
    for _ in range(max_redirects + 1):
        _check_public_host(cur)
        try:
            r = s.get(cur, headers={"User-Agent": "Mozilla/5.0 CatMap/3.3 (supply-chain reader)"}, timeout=timeout,
                      allow_redirects=False, stream=True)
        except requests.RequestException as e:
            raise SupplyError(f"Téléchargement impossible ({type(e).__name__}).") from e
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            cur = urljoin(cur, r.headers["Location"])
            continue
        if r.status_code != 200:
            raise SupplyError(f"Le site a répondu HTTP {r.status_code}.")
        data = bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise SupplyError("Document trop volumineux (plus de 20 Mo).")
        return bytes(data), (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    raise SupplyError("Trop de redirections.")


# --------------------------------------------------------------------------- lecture de tableaux
def _looks_like_header(cells: list) -> bool:
    cells = [str(c or "").strip() for c in cells]
    filled = [c for c in cells if c]
    if not filled:
        return False
    numeric = sum(bool(re.fullmatch(r"[\d\s.,%-]+", c)) for c in filled)
    return numeric <= len(filled) // 3 and all(len(c) <= 60 for c in filled)


def _df_from_rows(rows: list[list]) -> Optional[pd.DataFrame]:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows if r and any(c for c in r)]
    if len(rows) < 2:
        return None
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    if _looks_like_header(rows[0]):
        head = [h or f"col{i + 1}" for i, h in enumerate(rows[0])]
        body = rows[1:]
    else:
        head = [f"col{i + 1}" for i in range(n)]
        body = rows
    seen: dict = {}
    cols = []
    for h in head:                              # noms de colonnes uniques
        seen[h] = seen.get(h, 0) + 1
        cols.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    df = pd.DataFrame(body, columns=cols)
    if len(df) and (df.apply(lambda r: list(r) == head, axis=1)).any():      # en-têtes répétés à chaque page
        df = df[~df.apply(lambda r: list(r) == head, axis=1)]
    return df.reset_index(drop=True) if len(df) else None


def tables_from_pdf(data: bytes) -> list[tuple[str, pd.DataFrame]]:
    try:
        import pdfplumber
    except ImportError as e:
        raise SupplyError("La lecture de PDF demande la bibliothèque `pdfplumber` (à ajouter à requirements.txt).") from e
    groups: list[list[list]] = []
    last_n = None
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:300]:
                for tb in page.extract_tables() or []:
                    rows = [r for r in tb if r and any(c for c in r)]
                    if not rows:
                        continue
                    n = max(len(r) for r in rows)
                    if groups and last_n == n:
                        groups[-1].extend(rows)
                    else:
                        groups.append(list(rows))
                    last_n = n
    except Exception as e:  # noqa: BLE001
        raise SupplyError(f"PDF illisible : {type(e).__name__}.") from e
    out = []
    for i, g in enumerate(groups, 1):
        df = _df_from_rows(g)
        if df is not None:
            out.append((f"Tableau PDF {i} ({len(df)} lignes, {df.shape[1]} colonnes)", df))
    return out


def tables_from_html(text: str) -> list[tuple[str, pd.DataFrame]]:
    try:
        dfs = pd.read_html(io.StringIO(text))
    except (ValueError, ImportError):
        return []
    out = []
    for i, d in enumerate(dfs, 1):
        d = d.copy()
        d.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in d.columns]
        d = d.astype(str).replace({"nan": ""})
        if len(d) >= 2:
            out.append((f"Tableau web {i} ({len(d)} lignes, {d.shape[1]} colonnes)", d.reset_index(drop=True)))
    return out


def tables_from_bytes(data: bytes, name: str = "", mime: str = "") -> list[tuple[str, pd.DataFrame]]:
    """Détecte le format (PDF, Excel, CSV, HTML) et retourne la liste des tableaux trouvés."""
    low = (name or "").lower()
    head = data[:8]
    if head.startswith(b"%PDF") or low.endswith(".pdf") or "pdf" in mime:
        return tables_from_pdf(data)
    if head.startswith(b"PK") or low.endswith((".xlsx", ".xlsm")) or "spreadsheet" in mime:
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
        except Exception as e:  # noqa: BLE001
            raise SupplyError(f"Excel illisible : {e}") from e
        return [(f"Feuille « {k} » ({len(v)} lignes)", v.fillna("")) for k, v in sheets.items() if len(v)]
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = text or ""
    if low.endswith((".htm", ".html")) or "html" in mime or re.search(r"<\s*(table|html|body)", text[:5000], re.I):
        return tables_from_html(text)
    if not text.strip():
        raise SupplyError("Document vide.")
    try:
        delim = csv.Sniffer().sniff(text[:5000], delimiters=";,\t|").delimiter
    except csv.Error:
        delim = ";"
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise SupplyError(f"CSV illisible : {e}") from e
    return [(f"Fichier CSV ({len(df)} lignes)", df)] if len(df) else []


# --------------------------------------------------------------------------- choix des colonnes
def guess_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Devine (colonne du nom, colonne du pays) d'après les en-têtes ; à défaut la colonne texte la plus variée."""
    low = {c: str(c).lower().strip() for c in df.columns}

    def by_hints(hints, exclude=()):
        for h in hints:
            for c, l in low.items():
                if c not in exclude and h in l:
                    if h in ("name", "nom", "site") and any(x in l for x in ("country", "pays", "address", "adresse", "city", "ville")):
                        continue
                    return c
        return None

    name = by_hints(NAME_HINTS)
    country = by_hints(COUNTRY_HINTS, exclude=(name,))
    if name is None:
        best, score = None, 0.0
        for c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s[s != ""]
            if len(s) < 2:
                continue
            alpha = s.str.contains(r"[A-Za-zÀ-ÿ]", regex=True).mean()
            uniq = s.nunique() / len(s)
            if alpha > 0.7 and uniq * alpha > score:
                best, score = c, uniq * alpha
        name = best
    return name, country


# --------------------------------------------------------------------------- candidats
def candidates_from_table(df: pd.DataFrame, name_col: str, country_col: Optional[str] = None, *, source: str = "",
                          source_url: str = "", confiance: str = "A", type_relation: str = "") -> pd.DataFrame:
    out = pd.DataFrame({"fournisseur": df[name_col].astype(str).str.strip()})
    out["pays"] = df[country_col].astype(str).str.strip() if country_col else ""
    out["type_relation"] = type_relation
    out["rang"] = ""
    out["source"] = source
    out["source_url"] = source_url
    out["citation"] = ""
    out["confiance"] = confiance
    return out[CANDIDATE_COLUMNS]


def candidates_from_text(text: str, *, source: str = "Saisie manuelle", confiance: str = "n/d") -> pd.DataFrame:
    names = [ln.strip(" \t-•*;") for ln in str(text).splitlines()]
    rows = []
    for n in names:
        if not n:
            continue
        parts = [p.strip() for p in re.split(r"[;\t]", n)]
        rows.append({"fournisseur": parts[0], "pays": parts[1] if len(parts) > 1 else "", "type_relation": "", "rang": "",
                     "source": source, "source_url": "", "citation": "", "confiance": confiance})
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _norm_key(name: str) -> str:
    t = unicodedata.normalize("NFKD", clean_company_name(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def consolidate(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les lignes vides ou non plausibles, fusionne les doublons (même nom normalisé) en gardant la meilleure
    confiance, regroupe les pays et les sources."""
    if df is None or df.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    d = df.copy()
    for c in CANDIDATE_COLUMNS:
        if c not in d.columns:
            d[c] = ""
    d = d[CANDIDATE_COLUMNS].fillna("").astype(str)
    d["fournisseur"] = d["fournisseur"].str.strip()
    d = d[(d["fournisseur"].str.len() >= 2) & d["fournisseur"].str.contains(r"[A-Za-zÀ-ÿ]", regex=True)]
    d["_k"] = d["fournisseur"].map(_norm_key)
    d = d[d["_k"] != ""]
    d["_o"] = d["confiance"].map(lambda c: CONF_ORDER.get(c, 9))
    d = d.sort_values("_o", kind="stable")
    rows = []
    for _, g in d.groupby("_k", sort=False):
        r = g.iloc[0].copy()
        r["pays"] = "; ".join(dict.fromkeys(x for x in g["pays"] if x))
        r["source"] = "; ".join(dict.fromkeys(x for x in g["source"] if x))[:200]
        for c in ("source_url", "citation", "type_relation", "rang"):
            if not r[c]:
                r[c] = next((x for x in g[c] if x), "")
        rows.append(r)
    out = pd.DataFrame(rows)[CANDIDATE_COLUMNS] if rows else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- mode assisté
def build_prompt(client: str) -> str:
    return (
        f"Tu es analyste supply chain. Recherche sur le web les fournisseurs de « {client} » déclarés publiquement : liste "
        "officielle de fournisseurs, rapports de durabilité / RSE / ESG, rapport annuel, documents d'approvisionnement, "
        "publications institutionnelles. Cherche dans plusieurs langues (anglais et langue du siège de l'entreprise).\n\n"
        "RÈGLES\n"
        f"- N'inclus que des entreprises explicitement présentées comme fournisseurs, fabricants ou sous-traitants de « {client} » "
        "dans une source que tu as réellement consultée.\n"
        "- N'invente rien. Si tu n'en trouves pas, réponds uniquement : AUCUN FOURNISSEUR TROUVÉ.\n"
        "- Pour chaque ligne, donne l'adresse exacte de la source et une courte citation reprise mot pour mot du document.\n"
        "- Ne cite pas les clients, les concurrents, les filiales du groupe ni les simples partenaires commerciaux.\n\n"
        "FORMAT DE RÉPONSE : uniquement un tableau CSV séparé par des points-virgules, sans texte avant ni après, avec cet "
        "en-tête exact :\n"
        "fournisseur;pays;type_relation;rang;source_url;citation\n"
        f"- type_relation : {', '.join(RELATION_TYPES)}.\n"
        "- rang : 1 pour un fournisseur direct, 2 ou plus pour un fournisseur de fournisseur, vide si inconnu.\n"
        "- Entoure d'apostrophes doubles tout champ contenant un point-virgule."
    )


def parse_assisted_reply(text: str) -> pd.DataFrame:
    """Relit la réponse collée (CSV à points-virgules, tableau Markdown, JSON ou simple liste de noms)."""
    t = str(text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.M).strip()
    if not t or "AUCUN FOURNISSEUR" in t.upper():
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    rows: list[dict] = []
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if t.startswith("[") or t.startswith("{"):
        import json
        try:
            data = json.loads(t)
            data = data if isinstance(data, list) else data.get("fournisseurs", [])
            rows = [{str(k).lower(): v for k, v in d.items()} for d in data if isinstance(d, dict)]
        except (ValueError, AttributeError):
            rows = []
    if not rows:
        if sum(ln.lstrip().startswith("|") for ln in lines) >= 2:                      # tableau Markdown
            cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.lstrip().startswith("|")]
            cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c or "-") or not c for c in r)]
            header = [h.lower() for h in cells[0]]
            rows = [dict(zip(header, r)) for r in cells[1:]]
        else:
            hi = next((i for i, ln in enumerate(lines) if re.search(r"fournisseur|supplier", ln, re.I) and ";" in ln), None)
            if hi is not None:
                reader = csv.reader(io.StringIO("\n".join(lines[hi:])), delimiter=";", quotechar='"')
                recs = [r for r in reader if r]
                header = [h.strip().lower() for h in recs[0]]
                rows = [dict(zip(header, r)) for r in recs[1:]]
            else:                                                                       # simple liste de noms
                rows = [{"fournisseur": re.sub(r"^[\s\-•*\d.)]+", "", ln).strip()} for ln in lines]
    alias = {"supplier": "fournisseur", "name": "fournisseur", "nom": "fournisseur", "country": "pays", "relation": "type_relation",
             "type": "type_relation", "tier": "rang", "url": "source_url", "source": "source_url", "quote": "citation"}
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            d[alias.get(k, k)] = "" if v is None else str(v).strip()
        if not d.get("fournisseur"):
            continue
        url_ok = bool(re.match(r"https?://", d.get("source_url", "")))
        out.append({"fournisseur": d["fournisseur"], "pays": d.get("pays", ""), "type_relation": d.get("type_relation", ""),
                    "rang": d.get("rang", ""), "source": "Assistant IA (mode assisté)",
                    "source_url": d.get("source_url", "") if url_ok else "", "citation": d.get("citation", ""),
                    "confiance": "C" if (url_ok and d.get("citation")) else "D"})
    return pd.DataFrame(out, columns=CANDIDATE_COLUMNS)


# --------------------------------------------------------------------------- sortie
def build_output(client: str, sites: pd.DataFrame, cands: pd.DataFrame, include_client: bool = False) -> pd.DataFrame:
    """Fichier « suply » : colonne ``client`` en premier, puis ``role`` (client / fournisseur), puis les colonnes de la
    géolocalisation, puis la qualification de la relation (vide pour les lignes du client)."""
    extra = ["type_relation", "rang", "confiance", "source", "source_url", "citation"]
    if sites is None or sites.empty:
        return pd.DataFrame(columns=["client", "role"] + list(sites.columns if sites is not None else []) + extra)
    c = cands.copy()
    c["_k"] = c["fournisseur"].str.casefold()
    c = c.drop_duplicates("_k")
    s = sites.copy()
    s["_k"] = s["nom_demande"].astype(str).str.casefold()
    s = s.merge(c[["_k"] + extra], on="_k", how="left")
    is_client = (s["_k"] == str(client).strip().casefold()) & bool(include_client)
    s = s.drop(columns="_k")
    s.insert(0, "role", ["client" if x else "fournisseur" for x in is_client])
    s.insert(0, "client", client)
    return s


def not_found(report: pd.DataFrame, cands: pd.DataFrame, client: str) -> pd.DataFrame:
    """Fournisseurs sans site trouvé sur OpenStreetMap (avec la qualification) : rien ne doit disparaître."""
    if report is None or report.empty:
        return pd.DataFrame()
    bad = report[(report["n_sites"].fillna(0) == 0) | (report["statut"] == "erreur")]
    out = bad[["nom_demande", "statut", "message"]].copy()
    out["_k"] = out["nom_demande"].astype(str).str.casefold()
    c = cands.copy()
    c["_k"] = c["fournisseur"].str.casefold()
    out = out.merge(c.drop_duplicates("_k")[["_k", "pays", "type_relation", "confiance", "source_url"]], on="_k", how="left").drop(columns="_k")
    out.insert(0, "client", client)
    return out


# --------------------------------------------------------------------------- analyse automatique de documents trouvés
def useful_tables(tables: list[tuple[str, pd.DataFrame]], min_rows: int = 5) -> list[tuple[str, pd.DataFrame]]:
    """Tableaux qui ressemblent à une liste de fournisseurs : une colonne de noms et au moins ``min_rows`` lignes."""
    out = []
    for label, df in tables:
        if len(df) < min_rows:
            continue
        name, _ = guess_columns(df)
        if name is not None:
            out.append((label, df))
    return out


def analyse_urls(urls: list[str], *, workers: int = 4) -> list[dict]:
    """Télécharge et analyse plusieurs documents en parallèle. Ne lève jamais : chaque résultat porte son erreur."""
    from concurrent.futures import ThreadPoolExecutor

    def one(url):
        try:
            data, mime = fetch_url(url, timeout=30)
            name = urlparse(url).path.rsplit("/", 1)[-1]
            return {"url": url, "tables": useful_tables(tables_from_bytes(data, name, mime)), "error": ""}
        except SupplyError as e:
            return {"url": url, "tables": [], "error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"url": url, "tables": [], "error": f"{type(e).__name__}"}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(urls) or 1))) as ex:
        return list(ex.map(one, urls))


# --------------------------------------------------------------------------- IA : conversion et vérification des citations
def _squash(t: str) -> str:
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def page_text(data: bytes, name: str = "", mime: str = "") -> str:
    """Texte brut d'un document (PDF ou HTML) pour vérifier une citation."""
    low = (name or "").lower()
    if data[:4] == b"%PDF" or low.endswith(".pdf") or "pdf" in mime:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join((p.extract_text() or "") for p in pdf.pages[:200])
        except Exception:  # noqa: BLE001
            return ""
    text = data.decode("utf-8", "ignore")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    return re.sub(r"<[^>]+>", " ", text)


def candidates_from_ai(items: list[dict], *, verify: bool = True, source: str = "IA Linkup") -> pd.DataFrame:
    """Convertit les fournisseurs extraits par l'IA. Confiance C si la citation est retrouvée mot pour mot dans la page
    source, D sinon (page illisible, citation absente ou inventée)."""
    rows = []
    texts: dict[str, str] = {}
    urls = list(dict.fromkeys(str(i.get("source_url") or "") for i in items if str(i.get("source_url") or "").startswith("http")))
    if verify and urls:
        from concurrent.futures import ThreadPoolExecutor

        def grab(u):
            try:
                data, mime = fetch_url(u, timeout=30)
                return u, _squash(page_text(data, urlparse(u).path.rsplit("/", 1)[-1], mime))
            except Exception:  # noqa: BLE001
                return u, ""
        with ThreadPoolExecutor(max_workers=4) as ex:
            texts = dict(ex.map(grab, urls[:12]))
    for i in items:
        url = str(i.get("source_url") or "")
        quote = str(i.get("quote") or "")
        ok = bool(verify and quote and texts.get(url) and _squash(quote)[:90] in texts[url])
        rows.append({"fournisseur": str(i.get("name") or "").strip(), "pays": str(i.get("country") or "").strip(),
                     "type_relation": str(i.get("relation") or "").strip(), "rang": str(i.get("tier") or "").strip(),
                     "source": source + (" (citation vérifiée)" if ok else " (à vérifier)"),
                     "source_url": url if url.startswith("http") else "", "citation": quote, "confiance": "C" if ok else "D"})
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
