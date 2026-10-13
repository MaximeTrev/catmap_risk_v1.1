#!/usr/bin/env python3
"""Génère ``data/naf_mapping.csv`` : correspondance tag OSM -> division / section NAF rév. 2.

Remplace ``amenity_with_naf.csv`` et ``shop_with_naf.csv`` de la v1, dont 91 vecteurs sur 105
étaient nuls et dont plusieurs lignes étaient incohérentes (cafe -> Construction,
pharmacy -> Eau/déchets, beverages -> Industries extractives, alcohol -> Restauration).

Principe (documenté, éditable, validable par le métier) :
  * ``shop=*`` -> commerce de détail (division 47) sauf exceptions listées ;
  * ``amenity=*`` -> activité de l'équipement (56 restauration, 86 santé, 85 enseignement…) ;
  * les objets de mobilier urbain (bancs, poubelles…) ne sont **pas** des établissements :
    division vide, ``confidence = n/a``.
  * ``confidence`` : high (correspondance directe), medium (probable), low (ambigu, à valider),
    n/a (non classable).
  * valeur ``*`` : joker (toute valeur non listée de la clé).

Relancer :  python scripts/build_naf_mapping.py
"""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "naf_mapping.csv"

H, M, L, NA = "high", "medium", "low", "n/a"

# --------------------------------------------------------------------------
# amenity=*   valeur -> (division, confiance, commentaire)
# --------------------------------------------------------------------------
AMENITY = {
    # Restauration / débits de boissons -> 56
    **{k: ("56", H, "") for k in ("restaurant", "fast_food", "cafe", "bar", "pub", "food_court", "canteen", "biergarten")},
    "ice_cream": ("56", M, "Glacier : 56.10 ; peut relever de 10.52 (fabrication)"),
    "nightclub": ("56", M, "Discothèque : 56.30 ou 93.29 selon l'activité principale"),
    # Enseignement -> 85
    **{k: ("85", H, "") for k in ("school", "college", "university", "kindergarten")},
    "driving_school": ("85", H, "85.53 Enseignement de la conduite"),
    "language_school": ("85", H, ""), "music_school": ("85", H, ""),
    "driver_training": ("85", M, ""),
    "library": ("91", M, "91.01 Bibliothèques"), "toy_library": ("91", L, ""),
    "traffic_park": ("93", L, "Parc de sécurité routière"),
    # Transports / mobilité
    "bus_station": ("49", H, ""), "taxi": ("49", M, "49.32 Transports de voyageurs par taxis"),
    "ferry_terminal": ("50", M, ""),
    "parking": ("52", M, "52.21 Services auxiliaires des transports terrestres"),
    "parking_entrance": ("52", L, ""), "parking_space": ("52", L, ""),
    "bicycle_parking": ("52", L, ""), "motorcycle_parking": ("52", L, ""),
    "weighbridge": ("52", L, ""),
    "car_rental": ("77", H, "77.11"), "bicycle_rental": ("77", M, ""), "boat_rental": ("77", M, ""),
    "car_sharing": ("77", M, ""), "boat_sharing": ("77", L, ""),
    "vehicle_inspection": ("71", H, "71.20 Analyses, essais et inspections techniques"),
    "fuel": ("47", H, "47.30 Commerce de détail de carburants"),
    "charging_station": ("47", L, "Recharge électrique : 47.30 ou 35.14 selon l'opérateur"),
    "compressed_air": ("47", L, "Service annexe de station"),
    "car_wash": ("45", M, "45.20 Entretien et réparation de véhicules"),
    "bicycle_repair_station": ("95", L, ""),
    # Finance
    "bank": ("64", H, "64.19"), "atm": ("64", H, ""), "bureau_de_change": ("64", H, ""),
    "payment_centre": ("66", L, ""),
    # Santé / social
    **{k: ("86", H, "") for k in ("hospital", "clinic", "doctors", "dentist")},
    "nursing_home": ("87", H, ""), "social_facility": ("88", M, "87 ou 88 selon hébergement"),
    "baby_hatch": ("88", L, ""),
    "pharmacy": ("47", H, "47.73 Commerce de détail de produits pharmaceutiques"),
    "veterinary": ("75", H, ""),
    "childcare": ("88", H, "88.91 Accueil de jeunes enfants"),
    # Culture / loisirs
    "theatre": ("90", H, ""), "arts_centre": ("90", M, ""), "stripclub": ("90", L, ""),
    "cinema": ("59", M, "59.14 Projection de films cinématographiques (section J)"),
    "casino": ("92", H, ""), "planetarium": ("91", M, ""),
    "community_centre": ("93", L, "94.99 ou 93.29 selon le gestionnaire"),
    "social_centre": ("94", L, ""), "events_venue": ("82", L, "82.30 Organisation de salons et congrès"),
    "brothel": ("96", L, ""), "swingerclub": ("96", L, ""), "love_hotel": ("55", L, ""),
    # Administration
    **{k: ("84", H, "") for k in ("police", "fire_station", "townhall", "courthouse", "prison", "embassy", "ranger_station")},
    # Postes
    "post_office": ("53", H, ""), "post_depot": ("53", M, ""), "parcel_locker": ("53", L, ""),
    # Eau, déchets, telecom
    "recycling": ("38", M, ""), "waste_disposal": ("38", M, ""),
    "drinking_water": ("36", L, ""), "water_point": ("36", L, ""),
    "telephone": ("61", L, ""),
    "sanitary_dump_station": ("37", L, ""),
    # Services divers
    "marketplace": ("47", M, "47.8 Commerce de détail sur éventaires et marchés"),
    "vending_machine": ("47", M, "47.99 Autres commerces de détail hors magasin"),
    "photo_booth": ("74", L, ""),
    "animal_boarding": ("96", M, "96.09"), "animal_shelter": ("96", L, ""), "animal_breeding": ("01", M, ""),
    "crematorium": ("96", H, "96.03 Services funéraires"), "mortuary": ("96", M, ""),
    "grave_yard": ("96", L, ""), "place_of_mourning": ("96", L, ""),
    "place_of_worship": ("94", H, "94.91 Activités des organisations religieuses"),
    "monastery": ("94", M, ""), "public_bath": ("96", M, "96.04 Entretien corporel"),
    # Mobilier urbain / non-établissements
    **{k: (None, NA, "Mobilier / équipement, pas un établissement") for k in (
        "bench", "bbq", "shelter", "toilets", "dog_toilet", "dressing_room", "give_box", "mailroom",
        "public_bookcase", "fountain", "waste_basket", "watering_place", "baking_oven", "hunting_stand",
        "lounger", "grit_bin", "post_box")},
}

# --------------------------------------------------------------------------
# shop=*   par défaut : 47 (commerce de détail), sauf exceptions
# --------------------------------------------------------------------------
SHOP_DEFAULT = ("47", M, "Commerce de détail par défaut")
SHOP_HIGH_RETAIL = {
    "supermarket", "convenience", "department_store", "clothes", "shoes", "jewelry", "bakery", "butcher",
    "greengrocer", "seafood", "cheese", "dairy", "deli", "wine", "alcohol", "beverages", "florist",
    "garden_centre", "hardware", "doityourself", "furniture", "electronics", "computer", "mobile_phone",
    "optician", "hearing_aids", "medical_supply", "cosmetics", "perfumery", "books", "stationery",
    "toys", "sports", "bicycle", "pet", "second_hand", "tobacco", "newsagent", "kiosk", "chemist",
    "variety_store", "general", "fabric", "bag", "watches", "shoe", "appliance", "kitchen", "paint",
    "electrical", "hifi", "video_games", "music", "musical_instrument", "camera", "art", "gift",
}
SHOP_EXCEPTIONS = {
    "car": ("45", H, "45.11 Commerce de voitures"), "car_parts": ("45", H, "45.3"),
    "car_repair": ("45", H, "45.20"), "tyres": ("45", H, ""), "motorcycle": ("45", H, "45.40"),
    "motorcycle_repair": ("45", H, ""), "scooter": ("45", M, ""), "truck": ("45", M, ""),
    "caravan": ("45", M, ""), "atv": ("45", L, ""), "trailer": ("45", L, ""),
    "boat": ("47", L, "Vente de bateaux : 46.14 / 47.64"), "fuel": ("47", L, "Combustibles de chauffage"),
    "wholesale": ("46", H, "Commerce de gros"), "trade": ("46", L, "Magasin pour professionnels"),
    "agrarian": ("46", L, ""), "hairdresser_supply": ("46", M, ""),
    "hairdresser": ("96", H, "96.02 Coiffure"), "beauty": ("96", H, "96.02 Soins de beauté"),
    "massage": ("96", H, "96.04"), "tattoo": ("96", H, "96.09"),
    "laundry": ("96", H, "96.01"), "dry_cleaning": ("96", H, "96.01"),
    "funeral_directors": ("96", H, "96.03"), "pet_grooming": ("96", H, "96.09"),
    "shoe_repair": ("95", H, "95.23 Réparation de chaussures"), "tailor": ("14", M, "Couture sur mesure"),
    "money_lender": ("64", H, "64.92"), "pawnbroker": ("64", H, "64.92"),
    "bookmaker": ("92", H, ""), "lottery": ("92", M, ""),
    "travel_agency": ("79", H, "79.11"), "ticket": ("79", M, "79.90 Réservations"),
    "estate_agent": ("68", H, "68.31"),
    "copyshop": ("82", H, "82.19"), "pest_control": ("81", H, "81.29"),
    "security": ("80", M, "80.20"), "locksmith": ("95", L, "à valider"),
    "tool_hire": ("77", H, "77.32"), "rental": ("77", H, ""), "storage_rental": ("52", H, "52.10"),
    "telecommunication": ("61", L, "Agence d'opérateur : 61.20 ou 47.42"),
    "ice_cream": ("56", M, ""), "glaziery": ("43", M, "43.34"), "groundskeeping": ("81", M, "81.30"),
    "mall": ("68", L, "Centre commercial : gestion immobilière 68.20 ; commerces à l'intérieur = 47"),
    "outpost": ("47", L, ""), "party": ("47", L, ""), "cannabis": ("47", L, ""), "charity": ("47", L, "88 selon statut"),
    "farm": ("47", L, "Vente directe à la ferme : 47 ou 01"),
    "energy": ("47", L, ""), "gas": ("47", L, ""), "scuba_diving": ("47", L, "Ou 93.29"),
    "vacant": (None, NA, "Local vacant : pas un établissement"), "yes": (None, NA, "Valeur non renseignée"),
}
SHOP_KEYS = (
    "alcohol bakery beverages brewing_supplies butcher cheese chocolate coffee confectionery convenience dairy deli "
    "farm food frozen_food greengrocer health_food ice_cream nuts pasta pastry seafood spices tea tortilla water wine "
    "department_store general kiosk mall supermarket wholesale baby_goods bag boutique clothes fabric fashion "
    "fashion_accessories jewelry leather sewing shoes shoe_repair tailor watches wool charity second_hand variety_store "
    "beauty chemist cosmetics erotic hairdresser hairdresser_supply hearing_aids herbalist massage medical_supply "
    "nutrition_supplements optician perfumery tattoo agrarian appliance bathroom_furnishing country_store doityourself "
    "electrical energy fireplace florist garden_centre garden_furniture gas glaziery groundskeeping hardware houseware "
    "locksmith paint pottery security tool_hire trade antiques bed candles carpet curtain doors flooring furniture "
    "household_linen interior_decoration kitchen lighting tiles window_blind computer electronics hifi mobile_phone "
    "printer_ink radiotechnics telecommunication vacuum_cleaner atv bicycle boat car car_parts car_repair caravan fishing "
    "fuel golf hunting military_surplus motorcycle motorcycle_repair outdoor scooter scuba_diving ski snowmobile sports "
    "surf swimming_pool trailer truck tyres art camera collector craft frame games model music musical_instrument photo "
    "trophy video video_games anime books gift lottery newsagent stationery ticket bookmaker cannabis copyshop "
    "dry_cleaning e-cigarette funeral_directors laundry money_lender outpost party pawnbroker pest_control pet "
    "pet_grooming pyrotechnics religion rental storage_rental tobacco toys travel_agency vacant weapons yes"
).split()

# --------------------------------------------------------------------------
# Autres clés OSM  (clé, valeur) -> (division, confiance, commentaire)
# Division vide + section renseignée : classement au niveau section seulement.
# --------------------------------------------------------------------------
OFFICE = {
    "accountant": ("69", H, ""), "lawyer": ("69", H, ""), "notary": ("69", H, ""), "tax_advisor": ("69", H, ""),
    "insurance": ("65", H, ""), "estate_agent": ("68", H, ""), "financial": ("64", M, "64 ou 66"),
    "financial_advisor": ("66", M, ""), "architect": ("71", H, ""), "engineer": ("71", H, ""),
    "surveyor": ("71", H, ""), "it": ("62", H, ""), "employment_agency": ("78", H, ""),
    "travel_agent": ("79", H, ""), "telecommunication": ("61", M, ""), "government": ("84", H, ""),
    "educational_institution": ("85", M, ""), "research": ("72", M, ""), "advertising_agency": ("73", H, ""),
    "logistics": ("52", M, ""), "coworking": ("68", L, ""), "association": ("94", M, ""), "ngo": ("94", M, ""),
    "political_party": ("94", M, ""), "religion": ("94", M, ""), "therapist": ("86", M, ""),
    "energy_supplier": ("35", M, ""), "water_utility": ("36", M, ""), "consulting": ("70", M, ""),
    "company": ("70", L, "Siège / bureau d'entreprise : activité réelle inconnue"),
    "diplomatic": ("84", M, ""), "foundation": ("94", L, ""), "guide": ("79", L, ""), "publisher": ("58", M, ""),
    "newspaper": ("58", M, ""), "quango": ("84", L, ""), "administrative": ("84", M, ""),
}
CRAFT = {
    **{k: ("43", H, "") for k in ("electrician", "plumber", "painter", "roofer", "hvac", "plasterer", "tiler",
                                   "glaziery", "carpenter", "builder", "stonemason", "insulation", "floorer")},
    "brewery": ("11", H, ""), "winery": ("11", H, ""), "distillery": ("11", H, ""), "bakery": ("10", H, ""),
    "confectionery": ("10", H, ""), "caterer": ("56", H, ""), "gardener": ("81", H, ""),
    "photographer": ("74", H, ""), "tailor": ("14", M, ""), "shoemaker": ("15", M, ""), "sculptor": ("90", M, ""),
    "handicraft": ("32", L, ""), "jeweller": ("32", M, ""), "metal_construction": ("25", M, ""),
    "sawmill": ("16", H, ""), "cabinet_maker": ("31", M, ""), "upholsterer": ("95", M, ""),
    "electronics_repair": ("95", H, ""), "car_repair": ("45", H, ""), "key_cutter": ("95", L, ""),
    "print": ("18", M, ""), "printer": ("18", M, ""), "cleaning": ("81", M, ""), "carpet_layer": ("43", M, ""),
    "clockmaker": ("95", M, ""), "dressmaker": ("14", M, ""), "pottery": ("23", M, ""), "blacksmith": ("25", M, ""),
    "*": (None, L, "Artisanat : section C par défaut (à préciser)"),
}
POWER = {
    "plant": ("35", H, ""), "generator": ("35", M, ""), "substation": ("35", M, "Distribution d'électricité"),
    "transformer": ("35", L, ""), "*": ("35", L, "power=* : section D"),
}
HEALTHCARE = {
    "hospital": ("86", H, ""), "clinic": ("86", H, ""), "doctor": ("86", H, ""), "dentist": ("86", H, ""),
    "laboratory": ("86", H, ""), "physiotherapist": ("86", H, ""), "pharmacy": ("47", H, ""),
    "nursing_home": ("87", H, ""), "*": ("86", L, "healthcare=* : section Q"),
}
TOURISM = {
    **{k: ("55", H, "") for k in ("hotel", "motel", "hostel", "guest_house", "apartment", "chalet", "camp_site", "caravan_site")},
    "museum": ("91", H, ""), "gallery": ("90", M, ""), "zoo": ("91", M, "91.04"), "theme_park": ("93", H, ""),
    "attraction": ("93", L, ""), "information": ("79", L, ""),
}
LEISURE = {
    **{k: ("93", H, "") for k in ("fitness_centre", "sports_centre", "swimming_pool", "stadium", "golf_course",
                                   "amusement_arcade", "bowling_alley", "ice_rink", "water_park", "dance")},
    "marina": ("52", L, ""),
}
LANDUSE = {
    "industrial": ("", L, "Zone industrielle : section C par défaut"), "retail": ("47", L, ""),
    "quarry": ("08", M, ""), "farmland": ("01", M, ""), "farmyard": ("01", M, ""), "orchard": ("01", M, ""),
    "vineyard": ("01", M, ""), "commercial": ("", L, "Zone commerciale : non classable précisément"),
}
BUILDING = {
    "industrial": ("", L, "Bâtiment industriel : section C"), "retail": ("47", L, ""), "supermarket": ("47", M, ""),
    "hospital": ("86", M, ""), "school": ("85", M, ""), "university": ("85", M, ""), "college": ("85", M, ""),
    "hotel": ("55", M, ""), "warehouse": ("52", L, ""),
    **{k: ("94", L, "Lieu de culte") for k in ("church", "cathedral", "chapel", "mosque", "temple", "synagogue")},
    **{k: ("84", L, "") for k in ("civic", "government", "public")},
}
INDUSTRIAL = {"*": ("", L, "industrial=* : section C par défaut (à préciser)")}
MAN_MADE = {
    "works": ("", L, "Usine : section C par défaut"), "wastewater_plant": ("37", M, ""),
    "water_works": ("36", M, ""), "petroleum_well": ("06", M, ""), "storage_tank": ("52", L, ""),
}

SECTION_C_DEFAULTS = {("craft", "*"), ("industrial", "*"), ("landuse", "industrial"), ("building", "industrial"),
                      ("man_made", "works")}
SECTION_D_DEFAULTS = {("power", "*")}


def rows():
    for v, (div, conf, com) in AMENITY.items():
        yield "amenity", v, div, conf, com
    for v in SHOP_KEYS:
        if v in SHOP_EXCEPTIONS:
            div, conf, com = SHOP_EXCEPTIONS[v]
        elif v in SHOP_HIGH_RETAIL:
            div, conf, com = "47", H, ""
        else:
            div, conf, com = SHOP_DEFAULT
        yield "shop", v, div, conf, com
    for key, table in (("office", OFFICE), ("craft", CRAFT), ("power", POWER), ("healthcare", HEALTHCARE),
                       ("tourism", TOURISM), ("leisure", LEISURE), ("landuse", LANDUSE), ("building", BUILDING),
                       ("industrial", INDUSTRIAL), ("man_made", MAN_MADE)):
        for v, (div, conf, com) in table.items():
            yield key, v, div, conf, com


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["osm_key", "osm_value", "naf_division", "naf_section_override", "confidence", "comment"])
        for key, val, div, conf, com in rows():
            if (key, val) in seen:
                raise SystemExit(f"doublon : {key}={val}")
            seen.add((key, val))
            override = ""
            if not div:
                if (key, val) in SECTION_C_DEFAULTS:
                    override = "C"
                elif (key, val) in SECTION_D_DEFAULTS:
                    override = "D"
            w.writerow([key, val, div or "", override, conf, com])
    print(f"{len(seen)} règles écrites dans {OUT}")


if __name__ == "__main__":
    main()
