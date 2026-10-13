"""Objets de données partagés entre les modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .names import clean_company_name, valid_wikidata


@dataclass(frozen=True)
class NameQuery:
    """Une entreprise à rechercher."""
    raw: str                       # nom tel que saisi (conservé pour le rapport)
    wikidata: Optional[str] = None  # identifiant brand:wikidata (recherche exacte, très fiable)

    @property
    def search_name(self) -> str:
        return clean_company_name(self.raw)

    @staticmethod
    def make(raw: str, wikidata: Optional[str] = None) -> "NameQuery":
        return NameQuery(raw=str(raw).strip(), wikidata=valid_wikidata(wikidata))


@dataclass(frozen=True)
class SearchOptions:
    """Paramètres de la recherche : ils déterminent le compromis rappel / précision."""
    include_operator: bool = False      # chercher aussi dans le tag operator (plus de bruit)
    include_relations: bool = True      # relations type=multipolygon (grands sites)
    require_business_tag: bool = True   # exige shop/amenity/office/craft/… (exclut le bruit)
    countries: tuple[str, ...] = ()     # codes ISO 3166-1 alpha-2 ; vide = monde entier
    bbox: Optional[tuple[float, float, float, float]] = None  # (sud, ouest, nord, est)
    strict_word_match: bool = True      # mot entier obligatoire (recommandé)
    min_score: float = 0.0              # score minimal conservé (0 = tout garder)
    dedup_radius_m: float = 25.0        # fusion node/way d'un même site ; 0 = désactivé

    def cache_signature(self) -> dict:
        return {
            "op": self.include_operator, "rel": self.include_relations,
            "biz": self.require_business_tag, "c": list(self.countries),
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass
class QueryReport:
    """Résultat de recherche pour UNE entreprise (corrige A6 : le rapport « sans résultat »
    se base sur le nom demandé, pas sur une comparaison fragile de chaînes)."""
    nom_demande: str
    nom_recherche: str
    wikidata: Optional[str] = None
    statut: str = "ok"               # ok | aucun résultat | partiel | erreur
    n_elements_osm: int = 0          # éléments renvoyés par Overpass
    n_rejetes_nom: int = 0           # rejetés par le contrôle « mot entier »
    n_sans_geometrie: int = 0
    n_doublons_fusionnes: int = 0
    n_sites: int = 0
    message: str = ""
    endpoint: str = ""
    snapshot: str = ""               # date de la base OSM interrogée
    depuis_cache: bool = False
    duree_s: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)
