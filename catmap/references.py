"""Références scientifiques des méthodes, affichées dans les infobulles (format auteurs, année, titre, revue ou éditeur, DOI si connu).

Seules des références dont les éléments sont connus avec certitude sont citées ; pour les jeux de données sans article de référence stable,
le producteur et l'identifiant du jeu sont donnés. À faire relire avant tout usage dans une publication."""

REFS = {
    "gev": ["Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. "
            "*Journal of the Royal Statistical Society, Series B*, 52(2), 105–124.",
            "Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer, London."],
    "gpd": ["Pickands, J. (1975). Statistical inference using extreme order statistics. *The Annals of Statistics*, 3(1), 119–131.",
            "Davison, A. C., & Smith, R. L. (1990). Models for exceedances over high thresholds. *Journal of the Royal Statistical Society, Series B*, "
            "52(3), 393–442.",
            "Hosking, J. R. M. (1990), op. cit. (estimation par moments L)."],
    "era5": ["Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. "
             "doi:10.1002/qj.3803."],
    "tracks": ["Hodges, K. I. (1994). A general method for tracking analysis and its application to meteorological data. *Monthly Weather Review*, "
               "122(11), 2573–2586.",
               "Ullrich, P. A., et al. (2021). TempestExtremes v2.1: a community framework for feature detection, tracking, and analysis in large "
               "datasets. *Geoscientific Model Development*, 14, 5023–5048.",
               "Roberts, J. F., et al. (2014). The XWS open access catalogue of extreme European windstorms from 1979 to 2012. "
               "*Natural Hazards and Earth System Sciences*, 14, 2487–2501.",
               "Jeu de données : Copernicus Climate Change Service (C3S), « European windstorm reanalysis: tracks and footprints » "
               "(sis-european-wind-storm-reanalysis)."],
    "emanuel": ["Emanuel, K. (2011). Global warming effects on U.S. hurricane damage. *Weather, Climate, and Society*, 3(4), 261–268."],
    "jrc_flood": ["Baugh, C., et al. (2024). Global river flood hazard maps (GloFAS, v2.1.2). European Commission, Joint Research Centre (JRC Data "
                  "Catalogue)."],
    "schmid_hail": ["Schmid, T., Portmann, R., Villiger, L., Schröer, K., & Bresch, D. N. (2024). An open-source radar-based hail damage model for "
                    "buildings and cars. *Natural Hazards and Earth System Sciences*, 24, 847-872. doi:10.5194/nhess-24-847-2024 (Table 2 ; forme "
                    "sigmoïdale d'après Emanuel, K. (2011), *Weather, Climate, and Society*, 3, 261-268).",
                    "Approche simplifiée par CatMap : taux de dommage moyen évalué à 50 mm × fréquence annuelle des événements ≥ 5 cm (Battaglioli "
                    "et al., 2025)."],
    "aqueduct": ["Ward, P. J., et al. (2020). *Aqueduct Floods Methodology*. Technical Note, World Resources Institute.",
                 "Muis, S., Verlaan, M., Winsemius, H. C., Aerts, J. C. J. H., & Ward, P. J. (2016). A global reanalysis of storm surges and extreme "
                 "sea levels. *Nature Communications*, 7, 11969.",
                 "Gorelick, N., et al. (2017). Google Earth Engine: planetary-scale geospatial analysis for everyone. *Remote Sensing of "
                 "Environment*, 202, 18-27."],
    "jrc_damage": ["Huizinga, J., de Moel, H., & Szewczyk, W. (2017). *Global flood depth-damage functions: methodology and the database with "
                   "guidelines*. EUR 28552 EN, Publications Office of the European Union, JRC105688. doi:10.2760/16510."],
    "hazus": ["FEMA. *Hazus Inland Flood Model Technical Manual*. Federal Emergency Management Agency, Washington, D.C. "
              "(fonctions profondeur-dommage utilisées : jeu publié sur Zenodo, enregistrement 10027236)."],
    "hazus_eq": ["FEMA. *Hazus Earthquake Model Technical Manual* (Hazus 6.1). Federal Emergency Management Agency, Washington, D.C. : courbes de fragilité "
                 "lognormales par état de dommage (chapitre 5) ; § 5.4.3 et tableaux 5-37 à 5-40 : fragilité structurelle en PGA équivalente "
                 "(β = 0,64) ; pertes économiques : chapitre 11."],
    "gem": ["Pagani, M., et al. (2020). The 2018 version of the Global Earthquake Model: hazard component. *Earthquake Spectra*, 36(1_suppl), 226–251.",
            "GEM Foundation (2023). Global Seismic Hazard Map, version 2023.1. Zenodo, doi:10.5281/zenodo.8409647."],
    "eshm20": ["Danciu, L., et al. (2021). *The 2020 update of the European Seismic Hazard Model: model overview*. EFEHR Technical Report 001, "
               "v1.0.0. doi:10.12686/a15."],
    "grele": ["Battaglioli, F., Taszarek, M., Groenemeijer, P., Púčik, T., & Rädler, A. T. (2025). Contrasting trends in very large hail events and "
              "related economic losses across the globe. *Nature Geoscience*. doi:10.1038/s41561-025-01868-0.",
              "Jeu de données : ESSL, « Contrasting trends in very large hail events… » (grêle ≥ 5 cm, ERA5, 0,25°, 1950–2023), Zenodo, "
              "doi:10.5281/zenodo.17379738 (licence CC BY-NC-ND 4.0). Données de reproduction de l'article : doi:10.5281/zenodo.17064885.",
              "Modèle de l'article : régression logistique AR-CHaMo appliquée à ERA5 (probabilité d'éclair × probabilité conditionnelle de grêle ≥ 5 cm, "
              "pas de 3 h)."],
    "weibull": ["Makkonen, L. (2006). Plotting positions in extreme value analysis. *Journal of Applied Meteorology and Climatology*, 45(2), 334–340 "
                "(positions de Weibull i/(n+1) utilisées pour le quantile empirique)."],
    "irip": ["Breil, P., Lagadec, L.-R., & González-Sosa, E. (2016). *Inondation pluviale et développement périurbain*. NOVATECH 2016 "
             "(méthode IRIP : trois cartes production, transfert, accumulation ; facteurs notés 0 ou 1).",
             "Dehotin, J., & Breil, P. (2011). Note technique de la méthode IRIP (Indicateur de Ruissellement Intense Pluvial), Irstea / INRAE "
             "(note citée par les publications de la méthode ; non consultée par CatMap).",
             "DRIEAT Île-de-France. *Guide méthodologique ruissellement* (2023) : description de l'IRIP (cartes qualitatives, sans débit ni hauteur d'eau)."],
    "irip_braud": ["Braud, I., et al. (2020). A method to use proxy data of runoff-related impacts for the evaluation of a model mapping intense storm "
                   "runoff hazard: application to the railway context. *Natural Hazards and Earth System Sciences*, 20, 947–962 (application de l'IRIP : "
                   "seuils de pente et d'indice topographique calculés par zone d'étude par classification en 2 classes ; aire drainée 2,5 ha ; Ks < 1e-6 m/s)."],
    "gally": ["Gally, T. (2018). *Validation du modèle de ruissellement intense (IRIP)*. Mémoire de M2 Hydroressources, Université Grenoble Alpes. "
              "hal-03787493 (cité par le porteur du projet comme source du seuil d'aire drainée ; non consulté par CatMap)."],
    "barjot": ["Barjot, G., Breil, P., & Braud, I. (2023). Substitution des indicateurs topographiques par une approche à l'aide de géomorphons dans la "
               "méthodologie IRIP. *NOVATECH 2023*, Lyon. hal-04176937 (score de géomorphons de 0 à 3)."],
    "geomorphons": ["Jasiewicz, J., & Stepinski, T. F. (2013). Geomorphons, a pattern recognition approach to classification and mapping of landforms. "
                    "*Geomorphology*, 182, 147–156 (référence citée dans Barjot et al., 2023 ; Stepinski & Jasiewicz, 2011, également citée par ces auteurs)."],
    "amatulli": ["Amatulli, G., et al. (2020). Geomorpho90m, empirical evaluation and accuracy assessment of global high-resolution geomorphometric "
                 "layers. *Scientific Data*, 7, 162."],
    "hihydrosoil": ["Simons, G. W. H., Koster, R., & Droogers, P. (2020). *HiHydroSoil v2.0: high resolution soil maps of global hydraulic properties*. "
                    "FutureWater Report 213 (groupe hydrologique de sol, conductivité hydraulique à saturation ; 250 m)."],
    "hydrobasins": ["Lehner, B., & Grill, G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's "
                    "large river systems. *Hydrological Processes*, 27(15), 2171–2186 (HydroSHEDS / HydroBASINS, niveaux 1 à 12)."],
    "zt1987": ["Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land surface topography. *Earth Surface Processes and Landforms*, "
               "12, 47–56 (courbure du profil ; rupture de pente)."],
    "rubin": ["Rubin, J. M. (1967). Optimal classification into groups: an approach for solving the taxonomy problem. *Journal of Theoretical Biology*, "
              "15, 103–144 (classification optimale en 2 classes)."],
    "twi": ["Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences "
            "Bulletin*, 24(1), 43–69 (indice topographique ln(a / tan β))."],
    "merit": ["Yamazaki, D., et al. (2019). MERIT Hydro: a high-resolution global hydrography map based on latest topography datasets. *Water Resources "
              "Research*, 55, 5053–5073 (altitude et aire drainée amont, 90 m)."],
    "worldcover": ["Zanaga, D., et al. (2022). *ESA WorldCover 10 m 2021 v200*. doi:10.5281/zenodo.7254221 (CC BY 4.0)."],
    "gcn250": ["Jaafar, H. H., Ahmad, F. A., & El Beyrouthy, N. (2019). GCN250, new global gridded curve numbers for hydrologic modeling and design. "
               "*Scientific Data*, 6, 145 (numéros de courbe SCS à 250 m ; licence CC BY 4.0)."],
    "gee": ["Gorelick, N., et al. (2017). Google Earth Engine: planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment*, "
            "202, 18-27."],
    "aal": ["Principe de la perte annuelle moyenne : intégration de la perte par rapport à la probabilité annuelle de dépassement "
            "(méthode des trapèzes entre les périodes de retour retenues, sans extrapolation) ; voir par exemple Grossi, P., & Kunreuther, H. (2005), "
            "*Catastrophe Modeling: A New Approach to Managing Risk*, Springer."],
}


def refs_md(*keys: str, title: str = "Références") -> str:
    """Bloc Markdown « Références » pour une infobulle (une ligne par référence, sans doublon)."""
    seen, lines = set(), []
    for k in keys:
        for r in REFS.get(k, []):
            if r not in seen:
                seen.add(r)
                lines.append("• " + r)
    return f"**{title}**\n\n" + "\n\n".join(lines) if lines else ""
