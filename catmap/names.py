"""Traitement des noms d'entreprises.

Principe de robustesse : le **serveur** Overpass ne reçoit qu'une regex volontairement
simple (sous-chaîne tolérante, sans classes ni échappements dépendant du dialecte),
et la **précision** (« mot entier », score) est appliquée ensuite en Python, où elle est
testable. Cela supprime la fragilité d'échappement de la v1 (A5).
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Optional

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
_SPECIAL = str.maketrans({
    "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH", "ı": "i",
})


def _strip_latin_accents(text: str) -> str:
    """Retire les accents des lettres latines uniquement (le cyrillique, le grec, etc.
    ne sont pas altérés)."""
    out = []
    for ch in text:
        d = unicodedata.normalize("NFD", ch)
        if len(d) > 1 and d[0].isascii() and d[0].isalpha():
            out.append(d[0])
        else:
            out.append(ch)
    return "".join(out)


def normalize_name(text: str) -> str:
    """Forme canonique : sans accents latins, minuscules, ponctuation et séparateurs
    (espace, tiret, underscore, apostrophes droite/typographique, &, /, .) réduits à un
    seul espace. « L’Oréal-Paris » et « l'oreal paris » donnent le même résultat."""
    if not text:
        return ""
    s = text.translate(_SPECIAL)
    s = _strip_latin_accents(s)
    s = s.casefold()
    return re.sub(r"[\W_]+", " ", s).strip()


# --------------------------------------------------------------------------
# Nettoyage des formes juridiques (corrige A3)
# --------------------------------------------------------------------------
_LEGAL = {
    # France / Belgique / Suisse / Luxembourg
    "sa", "sas", "sasu", "sarl", "eurl", "ei", "snc", "scs", "sca", "scop", "gie", "sem",
    "se", "sci", "scm", "selarl", "nv", "bv", "sprl", "srl", "sl",
    # Anglo-saxon
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", "plc", "lp", "llp",
    # Allemagne / Nord / Sud
    "gmbh", "ag", "kg", "ohg", "ug", "ab", "oy", "oyj", "asa", "aps", "as", "spa",
}
# Mots courants par ailleurs : on ne les retire que s'ils sont écrits en MAJUSCULES ou avec points.
_AMBIGUOUS = {"as", "ab", "ag", "se", "sl", "oy", "lp", "kg", "ug", "sa", "sci", "scm", "sem"}


def _token_key(tok: str) -> str:
    return re.sub(r"[.\s,;]", "", tok.lower())


def clean_company_name(raw: str) -> str:
    """Retire les formes juridiques **finales** (mots entiers uniquement).

    - « Groupe Sanofi » et « Total Sante » restent intacts (la v1 produisait « Groupenofi »,
      « Totalnte »), « Carrefour SCA » est correctement nettoyé.
    - Plusieurs suffixes successifs sont gérés (« Foo GmbH & Co. KG »).
    - La casse et la ponctuation du reste du nom sont conservées.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    tokens = text.split()

    def drop_conjunctions(toks: list[str]) -> list[str]:
        while len(toks) > 1 and toks[-1].lower().strip(",;") in {"&", "and", "et", "+"}:
            toks = toks[:-1]
        return toks

    for _ in range(5):
        if len(tokens) < 2:
            break
        if len(tokens) >= 4 and [_token_key(t) for t in tokens[-3:]] == ["sa", "de", "cv"]:
            tokens = drop_conjunctions(tokens[:-3])
            continue
        last = tokens[-1]
        key = _token_key(last)
        stripped_last = last.strip(".,;")
        if key in _LEGAL:
            if key in _AMBIGUOUS and not (stripped_last.upper() == stripped_last or "." in stripped_last):
                break
            tokens = drop_conjunctions(tokens[:-1])
            continue
        if key == "co" and (last.endswith(".") or last.isupper()):
            tokens = drop_conjunctions(tokens[:-1])
            continue
        break
    cleaned = " ".join(tokens).strip().rstrip(",;")
    return cleaned or text


# --------------------------------------------------------------------------
# Regex envoyée à Overpass : sous-chaîne tolérante, dialecte-agnostique
# --------------------------------------------------------------------------
_ACCENTS: dict[str, set[str]] = defaultdict(set)
for _cp in list(range(0x00C0, 0x0180)) + list(range(0x1E00, 0x1F00)):
    _ch = chr(_cp)
    _d = unicodedata.normalize("NFD", _ch)
    if len(_d) > 1 and _d[0].isascii() and _d[0].isalpha():
        for _v in (_ch.lower(), _ch.upper()):
            if len(_v) == 1:
                _ACCENTS[_d[0].lower()].add(_v)
for _base, _extra in {"o": "øØ", "l": "łŁ", "d": "đĐð", "i": "ı"}.items():
    _ACCENTS[_base].update(_extra)


def _char_pattern(ch: str) -> str:
    """Alternation ``(a|A|à|…)`` : robuste même si le serveur ne gère pas les classes
    multi-octets. Les lettres sans variante restent nues (le drapeau ``,i`` suffit)."""
    if ch in _ACCENTS:
        alts = sorted({ch, ch.upper(), *_ACCENTS[ch]})
        return "(" + "|".join(alts) + ")"
    if ch.isascii():
        return ch
    up = ch.upper()
    return f"({ch}|{up})" if up != ch and len(up) == 1 else ch


def server_regex(search_name: str) -> str:
    """Regex de recherche pour l'opérateur ``~`` d'Overpass.

    Uniquement : lettres/chiffres, alternations ``(a|b)`` et ``.{0,2}`` entre les mots
    (couvre espace, tiret, underscore, apostrophe, point ou absence de séparateur).
    Aucun caractère à échapper : impossible d'injecter du QL ou de casser la requête.
    """
    norm = normalize_name(search_name)
    if not norm:
        raise ValueError("Nom d'entreprise vide après normalisation")
    words = [w for w in norm.split(" ") if w]
    return ".{0,2}".join("".join(_char_pattern(c) for c in w) for w in words)


# --------------------------------------------------------------------------
# Appariement strict et score (corrige H1/H2 et A16)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchInfo:
    field: str       # tag OSM qui a matché : brand:wikidata, brand, name, operator
    score: float     # 0-1, voir NameMatcher
    exact: bool


_FIELD_PENALTY = {"brand:wikidata": 0.0, "brand": 0.0, "name": 0.0, "operator": 0.2}
_FIELD_PRIORITY = ("brand:wikidata", "brand", "name", "operator")


class NameMatcher:
    """Vérifie qu'un tag OSM contient bien le nom demandé **en mots entiers**.

    Score :
      - 1.00  identifiant ``brand:wikidata`` identique (cas le plus fiable) ;
      - 0.90  nom identique (à la casse, aux accents et aux séparateurs près) ;
      - 0.75  le nom demandé est le début du nom OSM (« IKEA Lyon ») ;
      - 0.60  le nom demandé est contenu en mots entiers (« Magasin IKEA ») ;
      - le tag ``operator`` retranche 0.20 (souvent une société tierce).
    « xxIKEAxxx » ne correspond pas.
    """

    def __init__(self, query: str, wikidata: Optional[str] = None):
        self.q = normalize_name(query)
        self.q_compact = self.q.replace(" ", "")
        self.wikidata = wikidata.strip().upper() if wikidata else None
        if not self.q:
            raise ValueError("Nom vide")

    def score_text(self, candidate: str, field: str) -> Optional[tuple[float, bool]]:
        c = normalize_name(candidate)
        if not c:
            return None
        pen = _FIELD_PENALTY.get(field, 0.0)
        if c == self.q or c.replace(" ", "") == self.q_compact:
            return (0.90 - pen, True)
        if c.startswith(self.q + " "):
            return (0.75 - pen, False)
        if f" {self.q} " in f" {c} ":
            return (0.60 - pen, False)
        return None

    def match_tags(self, tags: Mapping[str, str], include_operator: bool = False) -> Optional[MatchInfo]:
        if self.wikidata and str(tags.get("brand:wikidata", "")).strip().upper() == self.wikidata:
            return MatchInfo("brand:wikidata", 1.0, True)
        best: Optional[MatchInfo] = None
        for field in _FIELD_PRIORITY[1:]:
            if field == "operator" and not include_operator:
                continue
            val = tags.get(field)
            if not val:
                continue
            for part in str(val).split(";"):
                res = self.score_text(part, field)
                if res and (best is None or res[0] > best.score):
                    best = MatchInfo(field, round(res[0], 2), res[1])
        return best


_QID = re.compile(r"^Q[1-9][0-9]{0,12}$")


def valid_wikidata(value: Optional[str]) -> Optional[str]:
    """Retourne l'identifiant Wikidata normalisé ou None s'il est invalide (sécurité :
    la valeur est insérée dans la requête)."""
    if not value:
        return None
    v = str(value).strip().upper()
    return v if _QID.match(v) else None

# --------------------------------------------------------------------------
# Recherche exacte (rapide) : variantes de graphie pour l'index d'Overpass
# --------------------------------------------------------------------------
# Overpass n'indexe que les égalités exactes ``["name"="IKEA"]``. Une regex insensible à la
# casse ``["name"~"ikea",i]`` oblige le serveur à parcourir tous les éléments nommés du
# périmètre (très lent sur la planète : timeouts et 504 sur les serveurs publics). On interroge
# donc quelques graphies courantes en égalité exacte ; la tolérance fine (casse, accents,
# séparateurs, mot entier) est ensuite appliquée en Python par ``NameMatcher``.
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def ql_escape(value: str) -> str:
    """Échappe une valeur littérale pour Overpass QL (entre guillemets) : une saisie utilisateur
    ne peut pas fermer la chaîne ni injecter d'instruction."""
    v = _CTRL.sub(" ", str(value))
    return v.replace("\\", "\\\\").replace('"', '\\"')


def exact_variants(name: str, max_variants: int = 24) -> list[str]:
    """Graphies à interroger en égalité exacte, par ordre de priorité.

    Saisie telle quelle, MAJUSCULES, minuscules, « Majuscule À Chaque Mot », « Majuscule initiale »,
    puis version sans accents, tirets à la place des espaces et apostrophe typographique.
    """
    base = " ".join(str(name).split())
    if not normalize_name(base):
        raise ValueError("Nom d'entreprise vide après normalisation")
    bases = [base]
    plain = _strip_latin_accents(base.translate(_SPECIAL))
    if plain != base:
        bases.append(plain)

    forms: list[str] = []

    def add(x: str) -> None:
        if x and x not in forms:
            forms.append(x)

    for b in bases:
        add(b)
        add(b.upper())
        add(b.lower())
        add(" ".join(w[:1].upper() + w[1:].lower() for w in b.split(" ")))
        add(b[:1].upper() + b[1:].lower())
    if " " in base:
        for f in list(forms):
            add(f.replace(" ", "-"))
    for f in list(forms):
        if "'" in f:
            add(f.replace("'", "’"))
        elif "’" in f:
            add(f.replace("’", "'"))
    return forms[:max_variants]
