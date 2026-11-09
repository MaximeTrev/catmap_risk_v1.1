"""Recherche d'entreprises par activité économique (NAF) sur un périmètre ou un pays.

Principe
--------
1. **Taxonomie harmonisée** : chaque activité (ex. « Composants électroniques ») relie des divisions NAF à des critères
   OpenStreetMap : tags exacts (``industrial=slaughterhouse``), mots du nom (``name~electronic|semiconductor``) combinés à un
   tag « porteur » (``industrial``, ``man_made=works``, ``craft``, ``office=company``…) et valeurs de ``product=*``.
   Les divisions NAF sans activité dédiée utilisent la table ``data/naf_mapping.csv`` en sens inverse (division -> tags).
2. **Langage naturel** : une analyse déterministe (mots-clés FR/EN + pays) fonctionne sans clé. Si la clé Linkup (déjà utilisée pour les fournisseurs) est
   présente dans les secrets, l'IA propose en plus activités, codes NAF, mots-clés multilingues et pays ; sa réponse est
   validée (liste blanche) avant toute requête.
3. **Requête Overpass** sur un cercle (``around``) ou un pays (zone ISO 3166-1), résultat au même format que la recherche
   par zone, avec la colonne ``activite`` et la raison de la sélection (``activite_critere``).

Limite assumée : OpenStreetMap décrit mal l'industrie. Les sites sans nom évocateur ni tag ``industrial``/``product`` ne sont pas
trouvés, et la sélection par mots du nom est une approximation (à contrôler via ``activite_critere``).
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from . import geo
from .area import AreaOptions, AreaQuery
from .config import Settings
from .countries import CountryIndex
from .models import QueryReport
from .naf import DIVISIONS, SECTIONS, NafClassifier, section_of_division
from .overpass import OverpassClient, OverpassPartialResult, OverpassQueryError, OverpassUnavailable
from .parsing import CLASS_TAGS, EXTRA_TAGS, dedupe_sites
from .pipeline import SITE_COLUMNS, GeolocationResult, empty_sites

log = logging.getLogger("catmap.activity")

# tags « porteurs » : un nom évocateur n'est retenu que si l'objet est de nature économique
INDUSTRY_CARRIERS = ('["industrial"]', '["man_made"="works"]', '["craft"]', '["office"="company"]',
                     '["landuse"="industrial"]', '["building"~"^(industrial|factory|warehouse)$"]')
WAREHOUSE_CARRIERS = ('["industrial"]', '["man_made"="works"]', '["office"~"^(company|logistics)$"]',
                      '["landuse"="industrial"]', '["building"~"^(industrial|warehouse|commercial)$"]',
                      '["shop"="wholesale"]')
CARRIER_SETS = {"industry": INDUSTRY_CARRIERS, "warehouse": WAREHOUSE_CARRIERS}
_CARRIER_TAGS = {
    "industry": (("industrial", None), ("man_made", "works"), ("craft", None), ("office", "company"),
                 ("landuse", "industrial"), ("building", "industrial|factory|warehouse")),
    "warehouse": (("industrial", None), ("man_made", "works"), ("office", "company|logistics"),
                  ("landuse", "industrial"), ("building", "industrial|warehouse|commercial"), ("shop", "wholesale")),
}


@dataclass(frozen=True)
class Activity:
    id: str
    label: str
    naf: tuple[str, ...]                      # divisions NAF
    kw: tuple[str, ...]                       # mots-clés de la demande (FR/EN)
    tags: tuple[tuple[str, str], ...] = ()    # tags OSM exacts
    names: tuple[str, ...] = ()               # fragments du nom (multilingue)
    products: tuple[str, ...] = ()            # fragments de product=*
    carriers: str = "industry"


def _a(id, label, naf, kw, tags=(), names=(), products=(), carriers="industry"):
    return Activity(id, label, tuple(naf), tuple(kw), tuple(tags), tuple(names), tuple(products), carriers)


ACTIVITIES: tuple[Activity, ...] = (
    _a("electronic_components", "Composants et produits électroniques", ["26"],
       ["composant electronique", "composants electroniques", "electronic component", "semi-conducteur", "semiconducteur",
        "semiconductor", "circuit imprime", "circuits imprimes", "pcb", "electronique", "electronics"],
       names=["electronic", "electronique", "elektronik", "electrónica", "eletrônica", "semiconductor", "semi-conducteur",
              "microelectronic", "circuit", "pcb", "capteur", "sensor"],
       products=["electronic", "semiconductor", "pcb", "circuit", "chip"]),
    _a("batteries", "Batteries et accumulateurs", ["27"],
       ["batterie", "batteries", "battery", "batteries", "accumulateur", "lithium", "pile electrique"],
       names=["batter", "accumul", "lithium", "akku", "bateri", "batería", "energy storage", "stockage d'energie"],
       products=["battery", "batteries", "lithium", "accumulator"]),
    _a("electrical_equipment", "Équipements électriques", ["27"],
       ["equipement electrique", "electrical equipment", "transformateur", "cable electrique", "moteur electrique", "electrical"],
       names=["electric", "elektro", "eléctric", "cable", "kabel", "transformer", "motor", "switchgear"],
       products=["electrical", "cable", "motor", "transformer"]),
    _a("building_materials", "Matériaux de construction", ["23", "16", "24", "25"],
       ["materiau de construction", "materiaux de construction", "building material", "construction material", "ciment",
        "cement", "beton", "concrete", "brique", "brick", "carrelage", "tuile", "platre", "gypse", "granulat", "baustoff"],
       tags=[("landuse", "quarry")],
       names=["cement", "ciment", "zement", "cemento", "cimento", "concrete", "béton", "beton", "brick", "brique", "ziegel",
              "tile", "fliesen", "carrelage", "gypsum", "plaster", "plâtre", "ceramic", "keramik", "baustoff",
              "building material", "matériaux", "materiales de construcción", "materiais de construção", "aggregate", "kies"],
       products=["cement", "concrete", "brick", "tile", "gypsum", "lime", "glass", "aggregate", "plaster"]),
    _a("food_processing", "Industrie agroalimentaire", ["10", "11"],
       ["agroalimentaire", "agro-alimentaire", "agro alimentaire", "industrie alimentaire", "food processing", "food industry",
        "transformation alimentaire", "agri-food", "agrifood", "agribusiness", "alimentaire"],
       tags=[("industrial", "slaughterhouse"), ("industrial", "dairy"), ("industrial", "bakery"),
             ("industrial", "brewery"), ("industrial", "distillery"), ("craft", "brewery"), ("craft", "winery"),
             ("craft", "distillery"), ("craft", "dairy")],
       names=["food", "alimentar", "alimentaire", "lebensmittel", "alimentos", "alimenticia", "meat", "viande", "fleisch",
              "dairy", "laitier", "milch", "lácteo", "laticín", "frigorífico", "abattoir", "slaughter", "juice", "bebida",
              "sugar", "açúcar", "azúcar", "flour", "farinha", "harina", "conserv", "cannery"],
       products=["food", "meat", "dairy", "flour", "sugar", "oil", "beverage", "juice", "canned", "milk"]),
    _a("beverages", "Boissons", ["11"], ["boisson", "beverage", "brasserie", "brewery", "distillerie", "vin", "winery", "embouteill"],
       tags=[("industrial", "brewery"), ("industrial", "distillery"), ("craft", "brewery"), ("craft", "winery"),
             ("craft", "distillery")],
       names=["brew", "brasserie", "brauerei", "cervec", "cervej", "distill", "winery", "bodega", "bottling", "embouteill"],
       products=["beer", "wine", "spirits", "beverage", "water", "soft drink"]),
    _a("cold_storage", "Entrepôts frigorifiques / chaîne du froid", ["52", "10"],
       ["entrepot frigorifique", "entrepots frigorifiques", "frigorifique", "chambre froide", "cold storage", "cold chain",
        "chaine du froid", "refrigerated", "logistique du froid", "froid"],
       names=["cold", "frigor", "frío", "frio ", "kühl", "tiefkühl", "refriger", "froid", "coldstore", "cold chain",
              "cool store", "friorífico", "congel", "frozen"],
       carriers="warehouse"),
    _a("logistics_warehousing", "Entreposage et logistique", ["52"],
       ["entrepot", "entreposage", "logistique", "warehouse", "warehousing", "logistics", "plateforme logistique", "fret"],
       tags=[("office", "logistics")],
       names=["logistic", "logistik", "logística", "logistique", "warehouse", "lager", "entrepôt", "almacén", "armazém",
              "distribution center", "centre de distribution", "freight", "cargo"],
       carriers="warehouse"),
    _a("metals", "Métallurgie et produits métalliques", ["24", "25"],
       ["metallurgie", "acier", "steel", "fonderie", "foundry", "aluminium", "metal", "forge", "siderurgie", "chaudronnerie"],
       tags=[("craft", "metal_construction"), ("craft", "blacksmith")],
       names=["steel", "acier", "stahl", "acero", "aço", "foundry", "fonderie", "gießerei", "fundición", "fundição",
              "aluminium", "aluminum", "metal", "forge", "iron", "inox"],
       products=["steel", "iron", "aluminium", "aluminum", "metal", "copper", "zinc"]),
    _a("chemicals", "Industrie chimique", ["20"],
       ["chimie", "chimique", "chemical", "chemistry", "engrais", "fertilizer", "peinture", "paint", "plasturgie chimique"],
       names=["chemic", "chimi", "chemie", "químic", "quimic", "fertili", "engrais", "dünger", "paint", "peinture", "farben",
              "pesticid", "petrochem"],
       products=["chemical", "fertilizer", "paint", "plastic", "petrochemical"]),
    _a("pharma", "Industrie pharmaceutique", ["21"],
       ["pharmaceutique", "pharmaceutical", "pharma", "medicament", "laboratoire pharmaceutique", "drug"],
       names=["pharma", "farmac", "laborator", "biotech", "medic", "arznei", "drug"],
       products=["pharmaceutical", "medicine", "drug", "vaccine"]),
    _a("plastics_rubber", "Plastiques et caoutchouc", ["22"],
       ["plastique", "plastic", "plasturgie", "caoutchouc", "rubber", "pneu", "tyre", "tire", "emballage plastique"],
       names=["plastic", "plástic", "kunststoff", "plasturg", "rubber", "caoutchouc", "gummi", "tyre", "tire", "pneu", "polymer"],
       products=["plastic", "rubber", "tyre", "tire", "polymer"]),
    _a("textiles", "Textile et habillement", ["13", "14", "15"],
       ["textile", "habillement", "vetement", "garment", "apparel", "confection", "tissu", "cuir", "leather", "chaussure",
        "footwear", "shoes"],
       tags=[("craft", "tailor"), ("craft", "dressmaker"), ("craft", "shoemaker")],
       names=["textile", "têxtil", "textil", "garment", "apparel", "clothing", "confection", "tissu", "fabric", "spinning",
              "weaving", "knit", "denim", "leather", "cuir", "leder", "couro", "cuero", "shoe", "footwear", "calçado",
              "calzado", "schuh", "may mặc", "dệt"],
       products=["textile", "clothes", "garment", "fabric", "leather", "shoes", "footwear"]),
    _a("wood_furniture", "Bois et meubles", ["16", "31"],
       ["bois", "scierie", "wood", "sawmill", "meuble", "furniture", "menuiserie industrielle", "parquet", "timber"],
       tags=[("craft", "sawmill"), ("craft", "cabinet_maker"), ("industrial", "sawmill")],
       names=["wood", "bois", "holz", "madera", "madeira", "timber", "sawmill", "scierie", "sägewerk", "furniture", "meuble",
              "möbel", "muebles", "móveis", "parquet", "plywood"],
       products=["wood", "timber", "furniture", "plywood", "lumber", "pellets"]),
    _a("paper_packaging", "Papier, carton et emballage", ["17", "18"],
       ["papier", "paper", "carton", "cardboard", "emballage", "packaging", "imprimerie", "printing", "cellulose", "pulp"],
       tags=[("craft", "print"), ("craft", "printer")],
       names=["paper", "papier", "papel", "cardboard", "carton", "packag", "emballage", "verpackung", "embalagem", "pulp",
              "cellulose", "print", "imprim", "druck", "gráfic"],
       products=["paper", "cardboard", "packaging", "pulp", "cellulose"]),
    _a("machinery", "Machines et équipements industriels", ["28", "33"],
       ["machine", "machines", "machinery", "equipement industriel", "industrial equipment", "outillage", "mecanique", "machine-outil"],
       names=["machin", "maschinen", "máquina", "maquina", "mechanic", "mécanique", "engineering", "équipement", "equipment",
              "tooling", "outillage", "pump", "pompe", "hydraulic", "compressor"],
       products=["machinery", "machine", "equipment", "pump", "tool"]),
    _a("automotive", "Industrie automobile", ["29", "45"],
       ["automobile", "automotive", "constructeur automobile", "equipementier", "auto parts", "pieces auto", "car manufacturer",
        "voiture", "vehicule"],
       names=["automotive", "automobil", "auto parts", "car ", "motor", "vehicle", "véhicule", "fahrzeug", "autopeças",
              "autopartes"],
       products=["car", "cars", "vehicle", "automotive", "auto parts", "engine"]),
    _a("transport_equipment", "Aéronautique, naval, ferroviaire", ["30"],
       ["aeronautique", "aerospace", "aviation", "naval", "chantier naval", "shipyard", "ferroviaire", "railway equipment", "avion"],
       tags=[("industrial", "shipyard")],
       names=["aero", "aviation", "aircraft", "shipyard", "chantier naval", "werft", "astillero", "estaleiro", "railway",
              "ferroviaire", "bahn", "airbus", "boeing"],
       products=["aircraft", "ship", "ships", "train", "trains", "aerospace"]),
    _a("solar_renewable", "Énergies renouvelables (production)", ["35", "27"],
       ["solaire", "solar", "photovoltaique", "photovoltaic", "eolien", "wind", "renouvelable", "renewable", "ferme solaire",
        "parc eolien"],
       tags=[("power", "plant"), ("power", "generator")],
       names=["solar", "solaire", "photovolta", "wind", "éolien", "windkraft", "eólic", "renewable", "renouvelable"],
       products=["solar", "wind", "photovoltaic"]),
    _a("power_generation", "Production d'électricité", ["35"],
       ["centrale electrique", "power plant", "centrale", "production d'electricite", "electricity generation", "nucleaire",
        "hydroelectrique"],
       tags=[("power", "plant"), ("power", "generator")],
       names=["power plant", "centrale", "kraftwerk", "central eléctrica", "usina", "thermal power", "hydroelectric"]),
    _a("oil_gas", "Pétrole, gaz et raffinage", ["06", "19"],
       ["petrole", "oil", "gaz", "gas", "raffinerie", "refinery", "hydrocarbure", "pipeline", "carburant"],
       tags=[("man_made", "petroleum_well"), ("industrial", "oil"), ("industrial", "refinery")],
       names=["refiner", "raffiner", "raffinerie", "petrol", "petróleo", "oil", "gas ", "lng"],
       products=["oil", "gas", "petroleum", "fuel", "lng", "lpg"]),
    _a("mining", "Mines et carrières", ["07", "08"],
       ["mine", "mines", "mining", "extraction", "carriere", "quarry", "minerai", "ore"],
       tags=[("landuse", "quarry"), ("industrial", "mine"), ("man_made", "mineshaft"), ("man_made", "adit")],
       names=["mine", "mining", "minería", "mineração", "bergbau", "carrière", "quarry", "steinbruch", "cantera", "pedreira"]),
    _a("waste_recycling", "Déchets et recyclage", ["38", "39"],
       ["dechet", "waste", "recyclage", "recycling", "ordure", "decharge", "landfill", "traitement des dechets"],
       tags=[("amenity", "recycling"), ("amenity", "waste_transfer_station"), ("landuse", "landfill")],
       names=["recycl", "waste", "déchet", "abfall", "entsorg", "residuos", "resíduos", "scrap", "ferraille", "landfill"]),
    _a("water_utilities", "Eau et assainissement", ["36", "37"],
       ["eau potable", "water supply", "assainissement", "wastewater", "station d'epuration", "water treatment"],
       tags=[("man_made", "water_works"), ("man_made", "wastewater_plant")],
       names=["water", "eau", "wasser", "agua", "água", "sanit", "assainissement", "wastewater"]),
    _a("construction_contractors", "Construction et travaux", ["41", "42", "43"],
       ["construction", "batiment", "btp", "building contractor", "travaux publics", "genie civil", "civil engineering", "entreprise de construction"],
       tags=[("craft", "builder"), ("craft", "roofer"), ("craft", "plumber"), ("craft", "electrician"), ("craft", "hvac")],
       names=["construction", "bau ", "baugesellschaft", "construcción", "construção", "constructora", "construtora",
              "btp", "civil", "contractor", "building"],
       carriers="industry"),
    _a("agriculture", "Agriculture et élevage", ["01"],
       ["agriculture", "agricole", "ferme", "farm", "elevage", "livestock", "culture", "plantation", "exploitation agricole"],
       tags=[("landuse", "farmyard"), ("landuse", "farmland"), ("man_made", "silo"), ("man_made", "storage_tank")],
       names=["farm", "ferme", "hof ", "granja", "fazenda", "finca", "plantation", "ranch", "agri", "cooperative"]),
    _a("fishing_aquaculture", "Pêche et aquaculture", ["03"],
       ["peche", "fishing", "aquaculture", "pisciculture", "fish farm", "crevette", "shrimp"],
       tags=[("man_made", "fish_farm"), ("landuse", "aquaculture")],
       names=["fish", "pêch", "aqua", "pesca", "shrimp", "crevett", "tôm", "cá "]),
    _a("data_it", "Informatique et numérique", ["62", "63"],
       ["informatique", "logiciel", "software", "datacenter", "data center", "numerique", "it services", "esn"],
       tags=[("office", "it")], names=["software", "logiciel", "data center", "datacenter", "informatik", "digital"],
       carriers="industry"),
)

ACTIVITY_BY_ID = {a.id: a for a in ACTIVITIES}

# tags dont on accepte la valeur dans une réponse IA (liste blanche) : évite l'injection dans la requête Overpass
ALLOWED_KEYS = frozenset({"industrial", "craft", "office", "man_made", "power", "shop", "amenity", "landuse", "healthcare",
                          "tourism", "leisure", "building", "product"})
_SAFE_VALUE = re.compile(r"^[a-z0-9_:;\-]{1,40}$")
_SAFE_WORD = re.compile(r"^[\w \-'&.À-ɏ]{2,40}$", re.UNICODE)
_DIV_RE = re.compile(r"^(\d{2})")


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# --------------------------------------------------------------------------------- critères
@dataclass(frozen=True)
class Criteria:
    """Critères de sélection (résultat de l'analyse d'une demande ou du choix NAF)."""
    activities: tuple[str, ...] = ()
    naf_divisions: tuple[str, ...] = ()
    extra_tags: tuple[tuple[str, str], ...] = ()
    extra_names: tuple[str, ...] = ()
    label: str = ""

    def is_empty(self) -> bool:
        return not (self.activities or self.naf_divisions or self.extra_tags or self.extra_names)

    def cache_signature(self) -> dict:
        return {"mode": "activite", "act": sorted(self.activities), "naf": sorted(self.naf_divisions),
                "tags": sorted(self.extra_tags), "names": sorted(self.extra_names)}


@dataclass(frozen=True)
class Filter:
    """Un critère élémentaire : ``tag`` (clé=valeur), ``name`` (regex + porteur) ou ``product`` (regex)."""
    kind: str
    key: str
    value: str
    activity: str            # id d'activité (ou « naf:26 »)
    carriers: str = "industry"

    @property
    def description(self) -> str:
        if self.kind == "tag":
            return f"{self.key}={self.value}"
        if self.kind == "product":
            return "product~" + self.value.replace("(^|;) *(", "").replace(") *(;|$)", "")
        return f"nom~{self.value}"


def normalize_divisions(values) -> tuple[str, ...]:
    """Codes NAF libres (« 26 », « 26.1 », « 2611Z », « C », « 26.11Z ») -> divisions à 2 chiffres, sans doublon."""
    out: list[str] = []
    for v in values or []:
        t = str(v).strip().upper().replace(" ", "")
        if not t:
            continue
        if len(t) == 1 and t.isalpha():  # section
            out += [d for d in DIVISIONS if section_of_division(d) == t]
            continue
        m = _DIV_RE.match(t.replace(".", ""))
        if m and m.group(1) in DIVISIONS:
            out.append(m.group(1))
    return tuple(dict.fromkeys(out))


def tags_for_division(naf: NafClassifier, division: str) -> list[tuple[str, str]]:
    """Tags OSM associés à une division dans ``naf_mapping.csv`` (sens inverse, sans jokers ni ``building``)."""
    m = naf.mapping
    sub = m[(m["naf_division"] == division) & (m["osm_value"] != "*") & (m["osm_key"] != "building")
            & (m["confidence"] != "n/a")]
    return [(r.osm_key, r.osm_value) for r in sub.itertuples() if _SAFE_VALUE.match(str(r.osm_value))]


def build_filters(criteria: Criteria, naf: Optional[NafClassifier] = None) -> list[Filter]:
    """Liste ordonnée et sans doublon des critères élémentaires."""
    out: list[Filter] = []
    seen: set = set()

    def add(f: Filter):
        k = (f.kind, f.key, f.value, f.carriers)
        if k not in seen:
            seen.add(k)
            out.append(f)

    acts = [ACTIVITY_BY_ID[a] for a in criteria.activities if a in ACTIVITY_BY_ID]
    for d in criteria.naf_divisions:  # activités curatées qui couvrent la division
        for a in ACTIVITIES:
            if d in a.naf and a not in acts and a.names:
                acts.append(a)
    for a in acts:
        for k, v in a.tags:
            add(Filter("tag", k, v, a.id))
        if a.names:
            add(Filter("name", "name", "|".join(a.names), a.id, a.carriers))
        if a.products:
            add(Filter("product", "product", "(^|;) *(" + "|".join(a.products) + ") *(;|$)", a.id))
    if naf is not None:
        for d in criteria.naf_divisions:
            for k, v in tags_for_division(naf, d):
                add(Filter("tag", k, v, f"naf:{d}"))
    for k, v in criteria.extra_tags:
        add(Filter("tag", k, v, "demande"))
    if criteria.extra_names:
        add(Filter("name", "name", "|".join(criteria.extra_names), "demande"))
    return out


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _statements(f: Filter, scope: str) -> list[str]:
    if f.kind == "tag":
        return [f'nwr{scope}["{f.key}"="{_esc(f.value)}"];']
    if f.kind == "product":
        return [f'nwr{scope}["product"~"{_esc(f.value)}",i];']
    return [f'nwr{scope}["name"~"{_esc(f.value)}",i]{c};' for c in CARRIER_SETS[f.carriers]]


# --------------------------------------------------------------------------------- périmètre
@dataclass(frozen=True)
class ActivityScope:
    kind: str                                  # "circle" | "country"
    area: Optional[AreaQuery] = None
    iso2: str = ""
    country: str = ""

    @property
    def label(self) -> str:
        if self.kind == "circle" and self.area:
            return self.area.label
        return f"Pays : {self.country or self.iso2}"

    def validate(self, max_radius_m: float) -> None:
        if self.kind == "circle":
            if self.area is None:
                raise ValueError("Cercle non défini.")
            self.area.validate(max_radius_m)
        elif self.kind == "country":
            if not re.fullmatch(r"[A-Z]{2}", self.iso2 or ""):
                raise ValueError("Pays invalide (code ISO à 2 lettres attendu).")
        else:
            raise ValueError("Type de périmètre inconnu.")


def build_activity_query(scope: ActivityScope, criteria: Criteria, settings: Settings,
                         naf: Optional[NafClassifier] = None) -> str:
    scope.validate(settings.area_max_radius_m)
    filters = build_filters(criteria, naf)
    if not filters:
        raise ValueError("Aucun critère OSM n'a pu être déduit de cette activité : précisez-la ou ajoutez des mots-clés.")
    timeout = int(settings.server_timeout_s if scope.kind == "country" else settings.area_timeout_s)
    head = f"[out:json][timeout:{timeout}]"
    if settings.maxsize_mb > 0:
        head += f"[maxsize:{int(settings.maxsize_mb) * 1024 * 1024}]"
    lines = [head + ";"]
    if scope.kind == "country":
        lines.append(f'area["ISO3166-1"="{scope.iso2}"]["boundary"="administrative"]->.a;')
        sc = "(area.a)"
    else:
        a = scope.area
        sc = f"(around:{int(round(a.radius_m))},{a.lat:.6f},{a.lon:.6f})"
    lines.append("(")
    for f in filters:
        lines += ["  " + s for s in _statements(f, sc)]
    lines += [");", f"out center tags {int(settings.area_max_sites) + 1};"]
    return "\n".join(lines)


def match_filter(tags: dict, filters: list[Filter]) -> Optional[Filter]:
    """Premier critère satisfait par les tags (même logique que la requête, pour tracer la raison de la sélection)."""
    for f in filters:
        if f.kind == "tag":
            vals = [p.strip() for p in str(tags.get(f.key, "")).split(";")]
            if f.value in vals:
                return f
        elif f.kind == "product":
            if re.search(f.value, str(tags.get("product", "")), re.I):
                return f
        else:
            if not re.search(f.value, str(tags.get("name", "")), re.I):
                continue
            for k, v in _CARRIER_TAGS[f.carriers]:
                tv = tags.get(k)
                if tv and (v is None or re.fullmatch(v, str(tv))):
                    return f
    return None


def _center(el: dict) -> Optional[tuple[float, float]]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center")
    if c and "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None


def parse_activity_elements(data: dict, scope: ActivityScope, criteria: Criteria, options: AreaOptions, max_sites: int,
                            naf: Optional[NafClassifier] = None) -> tuple[list[dict], dict]:
    filters = build_filters(criteria, naf)
    st = {"n_elements": 0, "n_sans_nom": 0, "n_sans_position": 0, "n_hors_critere": 0, "n_fusionnes": 0,
          "n_hors_cercle": 0, "tronque": False}
    records: list[dict] = []
    for el in data.get("elements", []):
        st["n_elements"] += 1
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if name.lower() == "unknown":
            name = ""
        display = name or (tags.get("brand") or "").strip()
        if not display and options.named_only:
            st["n_sans_nom"] += 1
            continue
        pos = _center(el)
        if pos is None:
            st["n_sans_position"] += 1
            continue
        flt = match_filter(tags, filters)
        if flt is None:
            st["n_hors_critere"] += 1
            continue
        lat, lon = pos
        dist = None
        if scope.kind == "circle" and scope.area is not None:
            dist = geo.haversine_m(scope.area.lon, scope.area.lat, lon, lat)
            if dist > scope.area.radius_m:
                st["n_hors_cercle"] += 1
                continue
        rec = {
            "nom_demande": criteria.label or scope.label, "osm_type": el["type"], "osm_id": int(el["id"]),
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "name": display or "(sans nom)", "lat": round(lat, 7), "long": round(lon, 7),
            "distance_centre_m": None if dist is None else round(dist, 1),
            "geom_kind": "point", "surface_m2": None, "surface_note": "", "match_field": "activité",
            "match_score": 1.0, "match_exact": True, "n_fusionnes": 0, "_outer": None,
            "activite": (ACTIVITY_BY_ID[flt.activity].label if flt.activity in ACTIVITY_BY_ID
                         else (f"NAF {flt.activity[4:]}" if flt.activity.startswith("naf:") else "Demande libre")),
            "activite_id": flt.activity, "activite_critere": flt.description,
        }
        for k in CLASS_TAGS:
            rec[k] = tags.get(k) or None
        for k, col in EXTRA_TAGS.items():
            rec[col] = tags.get(k) or None
        rec["product"] = tags.get("product") or None
        records.append(rec)
    if options.dedup_radius_m and options.dedup_radius_m > 0:
        records, st["n_fusionnes"] = dedupe_sites(records, options.dedup_radius_m)
    for r in records:
        r.pop("_outer", None)
    if len(records) > max_sites:
        records = records[:max_sites]
        st["tronque"] = True
    return records, st


def apply_activity_naf(sites: pd.DataFrame) -> pd.DataFrame:
    """Le classifieur par tag ne sait rien d'une usine « industrial=* » : on reprend alors la division de l'activité demandée."""
    if sites.empty or "activite_id" not in sites.columns:
        return sites
    out = sites.copy()
    for i, row in out.iterrows():
        a = ACTIVITY_BY_ID.get(row.get("activite_id"))
        if a is None or not a.naf:
            continue
        if row.get("naf_confidence") in (None, "low") or pd.isna(row.get("naf_confidence")):
            d = a.naf[0]
            out.at[i, "naf_division_code"] = d
            out.at[i, "naf_division"] = DIVISIONS.get(d)
            out.at[i, "naf_section_code"] = section_of_division(d)
            out.at[i, "naf_section"] = SECTIONS.get(section_of_division(d))
            out.at[i, "naf_source"] = f"activité={a.id}"
            out.at[i, "naf_confidence"] = "medium"
    return out


def locate_activity(client: OverpassClient, settings: Settings, scope: ActivityScope, criteria: Criteria,
                    options: AreaOptions, country_index: Optional[CountryIndex] = None,
                    naf: Optional[NafClassifier] = None, snap_km: float = 50.0,
                    on_event: Optional[Callable[[str], None]] = None) -> GeolocationResult:
    """Ne lève jamais : le statut du rapport porte l'erreur."""
    t0 = time.monotonic()
    rep = QueryReport(nom_demande=criteria.label or scope.label, nom_recherche=scope.label)
    records: list[dict] = []
    st: dict = {}
    try:
        query = build_activity_query(scope, criteria, settings, naf)
        print(f"[CatMap] Requête Overpass (activité) pour « {scope.label} » :\n{query}", flush=True)
        try:
            fr = client.fetch(query, on_event=on_event)
            data, rep.endpoint, rep.snapshot, rep.depuis_cache = fr.data, fr.endpoint, fr.snapshot, fr.from_cache
        except OverpassPartialResult as p:
            data, rep.endpoint = p.data, p.endpoint
            rep.snapshot = str((p.data.get("osm3s") or {}).get("timestamp_osm_base") or "")
            rep.statut = "partiel"
            rep.message = f"Résultat partiel (non mis en cache) : {p.remark}"
        records, st = parse_activity_elements(data, scope, criteria, options, settings.area_max_sites, naf)
        rep.n_elements_osm = st["n_elements"]
        rep.n_rejetes_nom = st["n_sans_nom"]
        rep.n_sans_geometrie, rep.n_doublons_fusionnes, rep.n_sites = st["n_sans_position"], st["n_fusionnes"], len(records)
        notes = []
        if st["tronque"] or st["n_elements"] > settings.area_max_sites:
            notes.append(f"réponse limitée à {settings.area_max_sites} objets : restreignez l'activité ou le périmètre")
        if st["n_sans_nom"]:
            notes.append(f"{st['n_sans_nom']} objet(s) sans nom écarté(s) (option « lieux nommés »)")
        if st["n_hors_cercle"]:
            notes.append(f"{st['n_hors_cercle']} objet(s) hors cercle écarté(s)")
        if notes:
            rep.message = (rep.message + " ; " if rep.message else "") + ", ".join(notes)
        if rep.statut != "partiel":
            rep.statut = "ok" if records else "aucun résultat"
    except OverpassQueryError as e:
        rep.statut, rep.message = "erreur", f"Requête refusée : {e}"
    except OverpassUnavailable as e:
        rep.statut, rep.message = "erreur", f"Serveurs Overpass indisponibles : {e}"
    except ValueError as e:
        rep.statut, rep.message = "erreur", str(e)
    except Exception as e:  # noqa: BLE001
        log.exception("Erreur inattendue (activité) %s", scope.label)
        rep.statut, rep.message = "erreur", f"{type(e).__name__}: {e}"
    rep.duree_s = round(time.monotonic() - t0, 2)
    print(f"[CatMap] activité « {scope.label} » : {rep.duree_s:.1f} s - statut : {rep.statut}, {rep.n_sites} site(s)", flush=True)

    for r in records:
        r["snapshot_osm"] = rep.snapshot
    if records:
        sites = pd.DataFrame(records)
        for c in SITE_COLUMNS:
            if c not in sites.columns:
                sites[c] = None
        if country_index is not None:
            geo_df = country_index.locate(sites["long"].to_numpy(float), sites["lat"].to_numpy(float), snap_km=snap_km)
            for c in geo_df.columns:
                sites[c] = geo_df[c].to_numpy()
        if naf is not None:
            sites = apply_activity_naf(naf.classify(sites))
        order = ["name", "activite", "activite_critere"] + [c for c in SITE_COLUMNS if c != "name"]
        if scope.kind == "circle":
            order.insert(order.index("long") + 1, "distance_centre_m")
        sites = sites[order + [c for c in sites.columns if c not in order]]
        if scope.kind != "circle" and "distance_centre_m" in sites.columns:
            sites = sites.drop(columns=["distance_centre_m"])
    else:
        sites = empty_sites()
    area_meta = ({"lat": scope.area.lat, "lon": scope.area.lon, "radius_m": scope.area.radius_m}
                 if scope.kind == "circle" and scope.area else {"country": scope.country or scope.iso2, "iso2": scope.iso2})
    meta = {"n_requetes": 1, "n_sites": int(len(sites)), "snapshots": [rep.snapshot] if rep.snapshot else [],
            "options": {**options.cache_signature(), **criteria.cache_signature()},
            "endpoints": [rep.endpoint] if rep.endpoint else [], "area": area_meta,
            "activity": {"activities": list(criteria.activities), "naf": list(criteria.naf_divisions),
                         "label": criteria.label}}
    return GeolocationResult(sites=sites, report=pd.DataFrame([rep.as_dict()]), meta=meta)


# --------------------------------------------------------------------------------- langage naturel
@dataclass
class Interpretation:
    criteria: Criteria
    country_iso2: str = ""
    country_name: str = ""
    source: str = "mots-clés"          # "mots-clés" | AI_SOURCE
    notes: list = field(default_factory=list)


COUNTRY_ALIASES = {"vietnam": "VN", "viet nam": "VN", "usa": "US", "etats unis": "US", "united states": "US", "uk": "GB",
                   "royaume uni": "GB", "angleterre": "GB", "coree du sud": "KR", "south korea": "KR", "russie": "RU",
                   "tchequie": "CZ", "republique tcheque": "CZ", "emirats arabes unis": "AE", "uae": "AE", "turquie": "TR",
                   "birmanie": "MM", "myanmar": "MM", "pays bas": "NL", "hollande": "NL", "holland": "NL"}


def detect_country(text: str, country_index: Optional[CountryIndex]) -> tuple[str, str]:
    """(iso2, nom FR) du premier pays reconnu dans le texte (noms FR/EN/natifs, le plus long d'abord)."""
    if country_index is None:
        return "", ""
    t = f" {_norm(text)} "
    present = {(a.get("iso2") or "").upper(): a.get("name_fr") or "" for a in country_index.attrs}
    for alias, iso in COUNTRY_ALIASES.items():
        if f" {alias} " in t and iso in present:
            return iso, present[iso]
    cands = []
    for a in country_index.attrs:
        iso2 = (a.get("iso2") or "").upper()
        if not iso2:
            continue
        for key in ("name_fr", "name_en", "name_native"):
            n = _norm(a.get(key) or "")
            if len(n) >= 3:
                cands.append((len(n), n, iso2, a.get("name_fr") or n))
    for _, n, iso2, nm in sorted(cands, reverse=True):
        if f" {n} " in t:
            return iso2, nm
    return "", ""


def match_activities(text: str, limit: int = 4) -> list[str]:
    """Activités dont les mots-clés apparaissent dans la demande (le mot-clé le plus long compte le plus)."""
    t = f" {_norm(text)} "
    scored = []
    for a in ACTIVITIES:
        score = 0
        for k in a.kw:
            kn = _norm(k)
            if kn and f" {kn} " in t or (kn and len(kn) >= 6 and f" {kn}" in t):
                score += len(kn)
        if score:
            scored.append((score, a.id))
    scored.sort(reverse=True)
    if not scored:
        return []
    best = scored[0][0]
    return [i for s, i in scored if s >= 0.5 * best][:limit]


# Termes français courants -> tag OSM (le mapping est en anglais) ; extensible sans toucher au reste du code.
FR_ALIASES = {
    "cafe": [("amenity", "cafe")], "bar": [("amenity", "bar")], "bistrot": [("amenity", "bar")], "brasserie": [("amenity", "restaurant")],
    "restaurant": [("amenity", "restaurant")], "restauration rapide": [("amenity", "fast_food")], "fast food": [("amenity", "fast_food")],
    "pharmacie": [("amenity", "pharmacy"), ("healthcare", "pharmacy")], "boulangerie": [("shop", "bakery")],
    "patisserie": [("shop", "pastry")], "boucherie": [("shop", "butcher")], "supermarche": [("shop", "supermarket")],
    "epicerie": [("shop", "convenience"), ("shop", "greengrocer")], "librairie": [("shop", "books")], "fleuriste": [("shop", "florist")],
    "coiffeur": [("shop", "hairdresser")], "coiffure": [("shop", "hairdresser")], "quincaillerie": [("shop", "doityourself"), ("shop", "hardware")],
    "bricolage": [("shop", "doityourself")], "concessionnaire": [("shop", "car")], "garage": [("shop", "car_repair")],
    "boutique de vetements": [("shop", "clothes")], "vetements": [("shop", "clothes")], "chaussures": [("shop", "shoes")],
    "banque": [("amenity", "bank")], "ecole": [("amenity", "school")], "hopital": [("amenity", "hospital")],
    "medecin": [("amenity", "doctors")], "dentiste": [("amenity", "dentist")], "hotel": [("tourism", "hotel")],
    "station service": [("amenity", "fuel")], "carburant": [("amenity", "fuel")], "cinema": [("amenity", "cinema")],
    "parking": [("amenity", "parking")], "poste": [("amenity", "post_office")], "bureau de poste": [("amenity", "post_office")],
    "entrepot": [("building", "warehouse")], "usine": [("man_made", "works")], "carriere": [("landuse", "quarry")],
}


def _stem(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _norm(text).split() if t]


def lookup_term(term: str, naf: NafClassifier) -> Interpretation:
    """Étape 1 (sans IA) : un mot ou une expression courte (« café », « commerce », « pharmacie ») est cherché dans
    ``naf_mapping.csv`` (valeurs de tags OSM), les alias français, les activités de la taxonomie, puis les libellés NAF
    (divisions et sections). ``source`` vaut « table NAF/OSM » ; ``criteria`` est vide si rien ne correspond."""
    q = _tokens(term)
    notes: list[str] = []
    if not q:
        return Interpretation(Criteria(), "", "", "table NAF/OSM", ["Saisissez un terme."])
    qset, qtxt = set(q), " ".join(q)
    tags: list[tuple[str, str]] = []
    m = naf.mapping
    ok = m[(m["osm_value"] != "*") & (m["osm_key"] != "building") & (m["confidence"] != "n/a")]
    for r in ok.itertuples():
        vt = _tokens(str(r.osm_value).replace("_", " "))
        if vt and set(vt) == qset:
            if _SAFE_VALUE.match(str(r.osm_value)):
                tags.append((r.osm_key, r.osm_value))
    for alias, pairs in FR_ALIASES.items():
        if " ".join(_tokens(alias)) == qtxt:
            tags += [p for p in pairs if p not in tags]
    acts = () if tags else tuple(match_activities(term))   # un tag exact prime sur la taxonomie industrielle
    divs: list[str] = []
    if not tags and not acts:
        for d, lab in DIVISIONS.items():
            if qset <= set(_tokens(lab)):
                divs.append(d)
        for sc, lab in SECTIONS.items():
            if qset <= set(_tokens(lab)):
                divs += [d for d in DIVISIONS if section_of_division(d) == sc and d not in divs]
    tags = list(dict.fromkeys(tags))
    if tags:
        notes.append("Tags OSM trouvés : " + ", ".join(f"{k}={v}" for k, v in tags[:12]) + (" …" if len(tags) > 12 else ""))
    if acts:
        notes.append("Activités de la taxonomie : " + ", ".join(ACTIVITY_BY_ID[a].label for a in acts))
    if divs:
        notes.append("Divisions NAF dont le libellé contient ce terme : " + ", ".join(f"{d} ({DIVISIONS[d]})" for d in divs[:6])
                     + (" …" if len(divs) > 6 else ""))
    if not (tags or acts or divs):
        notes.append("Aucune correspondance dans la table NAF/OSM.")
    return Interpretation(Criteria(acts, tuple(divs), tuple(tags), (), term.strip()[:80]), "", "", "table NAF/OSM", notes)


def interpret_keywords(text: str, country_index: Optional[CountryIndex] = None) -> Interpretation:
    iso2, nm = detect_country(text, country_index)
    acts = match_activities(text)
    notes = []
    if not acts:
        notes.append("Aucune activité reconnue par mots-clés : choisissez-en une dans la liste ou activez l'IA (clé Linkup).")
    return Interpretation(Criteria(activities=tuple(acts), label=text.strip()[:80]), iso2, nm, "mots-clés", notes)


AI_SOURCE = "IA (Linkup)"
COST_INTERPRET_USD = 0.055   # ordre de grandeur d'un appel Linkup « structured » (voir websearch.py)

AI_SCHEMA = {
    "type": "object",
    "properties": {
        "activity_ids": {"type": "array", "items": {"type": "string"},
                         "description": "Identifiants d'activités de la liste fournie dans la question (vide si aucune ne convient)."},
        "naf_divisions": {"type": "array", "items": {"type": "string"}, "description": "Codes NAF rév. 2 à 2 chiffres."},
        "osm_tags": {"type": "array", "description": "Tags OpenStreetMap [clé, valeur] parmi industrial, craft, office, man_made, "
                     "power, shop, amenity, landuse, healthcare, product ; valeurs en minuscules sans espace.",
                     "items": {"type": "array", "items": {"type": "string"}}},
        "name_keywords": {"type": "array", "items": {"type": "string"},
                          "description": "Mots courts susceptibles d'apparaître dans le NOM de ces entreprises, dans la langue du pays "
                                         "visé et en anglais (15 maximum)."},
        "country_iso2": {"type": "string", "description": "Code ISO 3166-1 alpha-2 du pays visé, vide si aucun."},
        "summary": {"type": "string", "description": "Libellé de l'activité, 8 mots maximum."},
    },
    "required": ["activity_ids", "naf_divisions", "osm_tags", "name_keywords", "country_iso2"],
}


def _ai_question(text: str) -> str:
    cat = "\n".join(f"- {a.id}: {a.label} (NAF {','.join(a.naf)})" for a in ACTIVITIES)
    return (
        "Convert this company-search request into OpenStreetMap + NAF rev. 2 criteria. This is a classification task: "
        "do not search for companies, only translate the request into criteria.\n"
        f"Request: \"{text.strip()[:400]}\"\n\nKnown activities (use their ids in activity_ids when they fit):\n{cat}\n\n"
        "Fill: naf_divisions (2-digit NAF rev. 2 codes), osm_tags (pairs [key, value] among industrial, craft, office, man_made, "
        "power, shop, amenity, landuse, healthcare, product), name_keywords (short words likely to appear in the NAME of such "
        "companies, in the local language of the target country AND in English, 15 max), country_iso2 (ISO 3166-1 alpha-2 of "
        "the target country, empty if none), summary (8 words max)."
    )


def validate_ai_output(raw: dict) -> Interpretation:
    """Liste blanche stricte : seuls des ids connus, des divisions NAF existantes, des clés/valeurs sûres ressortent."""
    acts = tuple(dict.fromkeys(a for a in (raw.get("activity_ids") or []) if isinstance(a, str) and a in ACTIVITY_BY_ID))
    divs = normalize_divisions([d for d in (raw.get("naf_divisions") or []) if isinstance(d, (str, int))
                                and not (isinstance(d, str) and len(d.strip()) == 1)])
    tags = []
    for pair in raw.get("osm_tags") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            k, v = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
            if k in ALLOWED_KEYS and _SAFE_VALUE.match(v):
                tags.append((k, v))
    names = [w.strip() for w in (raw.get("name_keywords") or []) if isinstance(w, str) and _SAFE_WORD.match(w.strip())]
    names = tuple(dict.fromkeys(n.lower() for n in names))[:15]
    iso2 = str(raw.get("country_iso2") or "").strip().upper()
    iso2 = iso2 if re.fullmatch(r"[A-Z]{2}", iso2) else ""
    label = re.sub(r"[^\w \-'&.,À-ɏ]", "", str(raw.get("summary") or ""))[:80]
    return Interpretation(Criteria(acts, divs, tuple(dict.fromkeys(tags)), names, label), iso2, "", AI_SOURCE)


def interpret_ai(text: str, api_key: str, session=None) -> Interpretation:
    """Interprétation par Linkup (le seul service d'IA de l'application, clé ``LINKUP_API_KEY``).
    Lève ``RuntimeError`` avec un message exploitable en cas d'échec."""
    if not api_key:
        raise RuntimeError("Clé Linkup absente (secret LINKUP_API_KEY).")
    from . import websearch as ws
    try:
        data = ws._call({"q": _ai_question(text), "depth": "standard", "outputType": "structured",
                         "structuredOutputSchema": json.dumps(AI_SCHEMA)}, api_key, session, timeout=90.0)
    except ws.SearchError as e:
        raise RuntimeError(str(e)) from e
    if isinstance(data, dict) and "activity_ids" not in data:
        for k in ("data", "result", "output", "structured"):
            if isinstance(data.get(k), dict):
                data = data[k]
                break
    if not isinstance(data, dict) or not any(k in data for k in ("activity_ids", "naf_divisions", "osm_tags", "name_keywords")):
        raise RuntimeError("Réponse Linkup inattendue pour l'interprétation de la demande.")
    return validate_ai_output(data)


def merge_interpretations(kw: Interpretation, ai: Optional[Interpretation], country_index=None) -> Interpretation:
    """Union des deux analyses : l'IA complète les mots-clés sans les écraser."""
    if ai is None:
        return kw
    c = Criteria(
        tuple(dict.fromkeys(kw.criteria.activities + ai.criteria.activities)),
        tuple(dict.fromkeys(kw.criteria.naf_divisions + ai.criteria.naf_divisions)),
        tuple(dict.fromkeys(kw.criteria.extra_tags + ai.criteria.extra_tags)),
        tuple(dict.fromkeys(kw.criteria.extra_names + ai.criteria.extra_names)),
        ai.criteria.label or kw.criteria.label)
    iso2, nm = kw.country_iso2, kw.country_name
    if not iso2 and ai.country_iso2:
        iso2 = ai.country_iso2
        if country_index is not None:
            for a in country_index.attrs:
                if (a.get("iso2") or "").upper() == iso2:
                    nm = a.get("name_fr") or ""
                    break
    return Interpretation(c, iso2, nm, "mots-clés + IA (Linkup)", kw.notes)
