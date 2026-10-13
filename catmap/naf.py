"""Classification sectorielle NAF rév. 2 à partir des tags OSM.

Corrige A1/A15 :
- nomenclature complète (88 divisions -> 21 sections) intégrée : plus de dictionnaire vide ;
- table ``data/naf_mapping.csv`` curatée, avec niveau de confiance et commentaire ;
- règle de priorité explicite et tracée (colonnes ``naf_source`` et ``naf_confidence``) ;
- valeurs OSM non reconnues listées (``unmapped``) pour maintenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIR

SECTIONS = {
    "A": "Agriculture, sylviculture et pêche",
    "B": "Industries extractives",
    "C": "Industrie manufacturière",
    "D": "Production et distribution d'électricité, de gaz, de vapeur et d'air conditionné",
    "E": "Production et distribution d'eau ; assainissement, gestion des déchets et dépollution",
    "F": "Construction",
    "G": "Commerce ; réparation d'automobiles et de motocycles",
    "H": "Transports et entreposage",
    "I": "Hébergement et restauration",
    "J": "Information et communication",
    "K": "Activités financières et d'assurance",
    "L": "Activités immobilières",
    "M": "Activités spécialisées, scientifiques et techniques",
    "N": "Activités de services administratifs et de soutien",
    "O": "Administration publique",
    "P": "Enseignement",
    "Q": "Santé humaine et action sociale",
    "R": "Arts, spectacles et activités récréatives",
    "S": "Autres activités de services",
    "T": "Activités des ménages en tant qu'employeurs",
    "U": "Activités extra-territoriales",
}
_RANGES = [
    (1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"), (41, 43, "F"), (45, 47, "G"),
    (49, 53, "H"), (55, 56, "I"), (58, 63, "J"), (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"),
    (84, 84, "O"), (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S"), (97, 98, "T"), (99, 99, "U"),
]
DIVISIONS = {
    "01": "Culture et production animale, chasse et services annexes", "02": "Sylviculture et exploitation forestière",
    "03": "Pêche et aquaculture", "05": "Extraction de houille et de lignite", "06": "Extraction d'hydrocarbures",
    "07": "Extraction de minerais métalliques", "08": "Autres industries extractives",
    "09": "Services de soutien aux industries extractives", "10": "Industries alimentaires",
    "11": "Fabrication de boissons", "12": "Fabrication de produits à base de tabac", "13": "Fabrication de textiles",
    "14": "Industrie de l'habillement", "15": "Industrie du cuir et de la chaussure",
    "16": "Travail du bois et fabrication d'articles en bois et en liège", "17": "Industrie du papier et du carton",
    "18": "Imprimerie et reproduction d'enregistrements", "19": "Cokéfaction et raffinage",
    "20": "Industrie chimique", "21": "Industrie pharmaceutique",
    "22": "Fabrication de produits en caoutchouc et en plastique",
    "23": "Fabrication d'autres produits minéraux non métalliques", "24": "Métallurgie",
    "25": "Fabrication de produits métalliques", "26": "Fabrication de produits informatiques, électroniques et optiques",
    "27": "Fabrication d'équipements électriques", "28": "Fabrication de machines et équipements n.c.a.",
    "29": "Industrie automobile", "30": "Fabrication d'autres matériels de transport", "31": "Fabrication de meubles",
    "32": "Autres industries manufacturières", "33": "Réparation et installation de machines et d'équipements",
    "35": "Production et distribution d'électricité, de gaz, de vapeur et d'air conditionné",
    "36": "Captage, traitement et distribution d'eau", "37": "Collecte et traitement des eaux usées",
    "38": "Collecte, traitement et élimination des déchets ; récupération",
    "39": "Dépollution et autres services de gestion des déchets", "41": "Construction de bâtiments",
    "42": "Génie civil", "43": "Travaux de construction spécialisés",
    "45": "Commerce et réparation d'automobiles et de motocycles",
    "46": "Commerce de gros, à l'exception des automobiles et des motocycles",
    "47": "Commerce de détail, à l'exception des automobiles et des motocycles",
    "49": "Transports terrestres et transport par conduites", "50": "Transports par eau", "51": "Transports aériens",
    "52": "Entreposage et services auxiliaires des transports", "53": "Activités de poste et de courrier",
    "55": "Hébergement", "56": "Restauration", "58": "Édition",
    "59": "Production de films, vidéo et programmes de télévision ; enregistrement sonore",
    "60": "Programmation et diffusion", "61": "Télécommunications",
    "62": "Programmation, conseil et autres activités informatiques", "63": "Services d'information",
    "64": "Activités des services financiers, hors assurance et caisses de retraite", "65": "Assurance",
    "66": "Activités auxiliaires de services financiers et d'assurance", "68": "Activités immobilières",
    "69": "Activités juridiques et comptables", "70": "Activités des sièges sociaux ; conseil de gestion",
    "71": "Activités d'architecture et d'ingénierie ; contrôle et analyses techniques",
    "72": "Recherche-développement scientifique", "73": "Publicité et études de marché",
    "74": "Autres activités spécialisées, scientifiques et techniques", "75": "Activités vétérinaires",
    "77": "Activités de location et location-bail", "78": "Activités liées à l'emploi",
    "79": "Activités des agences de voyage, voyagistes et services de réservation",
    "80": "Enquêtes et sécurité", "81": "Services relatifs aux bâtiments et aménagement paysager",
    "82": "Activités administratives et autres activités de soutien aux entreprises",
    "84": "Administration publique et défense ; sécurité sociale obligatoire", "85": "Enseignement",
    "86": "Activités pour la santé humaine", "87": "Hébergement médico-social et social",
    "88": "Action sociale sans hébergement", "90": "Activités créatives, artistiques et de spectacle",
    "91": "Bibliothèques, archives, musées et autres activités culturelles",
    "92": "Organisation de jeux de hasard et d'argent", "93": "Activités sportives, récréatives et de loisirs",
    "94": "Activités des organisations associatives", "95": "Réparation d'ordinateurs et de biens personnels et domestiques",
    "96": "Autres services personnels",
}

# Ordre de priorité (identique à la v1 : amenity avant shop), désormais explicite et étendu.
PRIORITY = ("amenity", "shop", "office", "craft", "industrial", "man_made", "power",
            "healthcare", "tourism", "leisure", "landuse", "building")


def section_of_division(div: str) -> Optional[str]:
    try:
        n = int(div)
    except (TypeError, ValueError):
        return None
    for lo, hi, s in _RANGES:
        if lo <= n <= hi:
            return s
    return None


class NafClassifier:
    """Ajoute les colonnes NAF à un DataFrame de sites.

    Colonnes ajoutées : naf_division_code, naf_division, naf_section_code, naf_section,
    naf_source (ex. « shop=bakery »), naf_confidence (high/medium/low).
    """

    def __init__(self, mapping_csv: Optional[Path] = None, extra_csv: Optional[Path] = None):
        path = Path(mapping_csv) if mapping_csv else DATA_DIR / "naf_mapping.csv"
        m = pd.read_csv(path, dtype=str, keep_default_na=False)
        if extra_csv and Path(extra_csv).exists():  # surcharges locales (validation métier)
            m = pd.concat([m, pd.read_csv(extra_csv, dtype=str, keep_default_na=False)])
            m = m.drop_duplicates(["osm_key", "osm_value"], keep="last")
        self.mapping = m.reset_index(drop=True)
        self._by_key: dict[str, pd.DataFrame] = {k: g.set_index("osm_value") for k, g in m.groupby("osm_key")}

    def _lookup(self, key: str, value) -> Optional[tuple[str, str, str, str]]:
        """(division, section, confiance, source) pour une valeur brute (gère « a;b »)."""
        tbl = self._by_key.get(key)
        if tbl is None or value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        parts = [p.strip() for p in str(value).split(";") if p.strip()] or [str(value)]
        for p in parts:
            if p in tbl.index:
                row = tbl.loc[p]
            elif "*" in tbl.index:
                row = tbl.loc["*"]
            else:
                continue
            div = row["naf_division"] or ""
            sec = section_of_division(div) if div else (row["naf_section_override"] or "")
            if row["confidence"] == "n/a":
                continue
            if not sec:
                continue
            return div, sec, row["confidence"], f"{key}={p}"
        return None

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        div = [None] * n
        sec = [None] * n
        conf = [None] * n
        src = [None] * n
        cols = [c for c in PRIORITY if c in out.columns]
        # cache par (clé, valeur) : peu de valeurs distinctes
        memo: dict[tuple[str, object], Optional[tuple[str, str, str, str]]] = {}
        arrays = {c: out[c].tolist() for c in cols}
        for i in range(n):
            for c in cols:
                v = arrays[c][i]
                mk = (c, v)
                if mk not in memo:
                    memo[mk] = self._lookup(c, v)
                r = memo[mk]
                if r:
                    div[i], sec[i], conf[i], src[i] = r
                    break
        out["naf_division_code"] = [d or None for d in div]
        out["naf_division"] = [DIVISIONS.get(d) if d else None for d in div]
        out["naf_section_code"] = sec
        out["naf_section"] = [SECTIONS.get(s) if s else None for s in sec]
        out["naf_source"] = src
        out["naf_confidence"] = conf
        return out

    def unmapped(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valeurs de tags présentes dans les sites mais absentes du mapping (à compléter)."""
        rows = []
        for c in PRIORITY:
            if c not in df.columns:
                continue
            known = self._by_key.get(c)
            wildcard = known is not None and "*" in known.index
            for v, cnt in df[c].dropna().astype(str).str.split(";").str[0].value_counts().items():
                if (known is None or v not in known.index) and not wildcard:
                    rows.append({"osm_key": c, "osm_value": v, "n_sites": int(cnt)})
        cols = ["osm_key", "osm_value", "n_sites"]
        res = pd.DataFrame(rows, columns=cols)
        # `building=yes` etc. sont des valeurs génériques non informatives
        res = res[~((res.osm_key == "building") & (res.osm_value.isin(["yes", "house", "apartments", "residential"])))]
        return res.sort_values("n_sites", ascending=False).reset_index(drop=True)
