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
    "aqueduct": ["Ward, P. J., et al. (2020). *Aqueduct Floods Methodology*. Technical Note, World Resources Institute.",
                 "Muis, S., Verlaan, M., Winsemius, H. C., Aerts, J. C. J. H., & Ward, P. J. (2016). A global reanalysis of storm surges and extreme "
                 "sea levels. *Nature Communications*, 7, 11969."],
    "jrc_damage": ["Huizinga, J., de Moel, H., & Szewczyk, W. (2017). *Global flood depth-damage functions: methodology and the database with "
                   "guidelines*. EUR 28552 EN, Publications Office of the European Union, JRC105688. doi:10.2760/16510."],
    "hazus": ["FEMA. *Hazus Inland Flood Model Technical Manual*. Federal Emergency Management Agency, Washington, D.C. "
              "(fonctions profondeur-dommage utilisées : jeu publié sur Zenodo, enregistrement 10027236)."],
    "gem": ["Pagani, M., et al. (2020). The 2018 version of the Global Earthquake Model: hazard component. *Earthquake Spectra*, 36(1_suppl), 226–251.",
            "GEM Foundation (2023). Global Seismic Hazard Map, version 2023.1. Zenodo, doi:10.5281/zenodo.8409647."],
    "eshm20": ["Danciu, L., et al. (2021). *The 2020 update of the European Seismic Hazard Model: model overview*. EFEHR Technical Report 001, "
               "v1.0.0. doi:10.12686/a15."],
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
