"""Lecture robuste des listes d'entreprises (CSV, XLSX, texte libre).

Corrige A6 (lecture CSV fragile) : détection d'encodage et de séparateur, en-tête
auto-détecté, colonne ``wikidata`` optionnelle, doublons retirés, erreurs explicites
(plus de ``(None, [])`` non géré).
"""
from __future__ import annotations

import csv
import io
from typing import Optional

import pandas as pd

from .models import NameQuery
from .names import normalize_name, valid_wikidata

_NAME_HEADERS = {"nom", "name", "entreprise", "societe", "company", "raison sociale", "nom de l entreprise",
                 "nom entreprise", "denomination", "marque", "brand"}
_QID_HEADERS = {"wikidata", "brand wikidata", "qid", "wikidata id", "identifiant wikidata"}


class InputError(ValueError):
    """Fichier d'entrée illisible ou vide."""


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise InputError("Encodage du fichier non reconnu")


def _sniff_delimiter(text: str) -> str:
    sample = next((l for l in text.splitlines() if l.strip()), "")
    counts = {d: sample.count(d) for d in (";", "|", "\t", ",")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ";"


def _looks_like_header(first_row: list[str]) -> bool:
    cells = {normalize_name(c) for c in first_row if c}
    return bool(cells & (_NAME_HEADERS | _QID_HEADERS))


def _rows_from_table(rows: list[list[str]], has_header: Optional[bool]) -> tuple[list[NameQuery], list[str]]:
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise InputError("Le fichier ne contient aucune ligne exploitable")
    header_present = _looks_like_header(rows[0]) if has_header is None else has_header
    warnings: list[str] = []
    name_col, qid_col = 0, None
    if header_present:
        norm = [normalize_name(c) for c in rows[0]]
        for i, h in enumerate(norm):
            if h in _NAME_HEADERS:
                name_col = i
                break
        for i, h in enumerate(norm):
            if h in _QID_HEADERS:
                qid_col = i
                break
        rows = rows[1:]
    elif len(rows[0]) > 1:
        # sans en-tête : une 2e colonne composée d'identifiants Wikidata est reconnue
        col1 = [r[1] for r in rows if len(r) > 1 and r[1]]
        if col1 and sum(1 for v in col1 if valid_wikidata(v)) / len(col1) >= 0.5:
            qid_col = 1
    queries, seen = [], set()
    for r in rows:
        name = r[name_col] if name_col < len(r) else ""
        if not name:
            continue
        qid_raw = r[qid_col] if qid_col is not None and qid_col < len(r) else ""
        qid = valid_wikidata(qid_raw)
        if qid_raw and not qid:
            warnings.append(f"Identifiant Wikidata invalide ignoré pour « {name} » : {qid_raw}")
        key = (normalize_name(name), qid)
        if key in seen:
            continue
        seen.add(key)
        queries.append(NameQuery.make(name, qid))
    if not queries:
        raise InputError("Aucun nom d'entreprise trouvé (vérifiez la colonne et l'en-tête)")
    return queries, warnings


def read_company_list(data: bytes, filename: str = "", has_header: Optional[bool] = None) -> tuple[list[NameQuery], list[str]]:
    """``has_header`` : True/False pour forcer, None pour détecter."""
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        except Exception as e:  # noqa: BLE001
            raise InputError(f"Lecture Excel impossible : {e}") from e
        return _rows_from_table(df.fillna("").values.tolist(), has_header)
    text = _decode(data)
    delim = _sniff_delimiter(text)
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    except csv.Error as e:
        raise InputError(f"Lecture CSV impossible : {e}") from e
    return _rows_from_table(rows, has_header)


def parse_names_text(text: str) -> tuple[list[NameQuery], list[str]]:
    """Une entreprise par ligne ; identifiant Wikidata optionnel après « ; » (ex. ``Total;Q154037``)."""
    rows = []
    for line in text.splitlines():
        if line.strip():
            parts = [p.strip() for p in line.split(";", 1)]
            rows.append(parts)
    return _rows_from_table(rows, has_header=False)
