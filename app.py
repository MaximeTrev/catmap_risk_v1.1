"""CatMap Risk - application Streamlit.

Lancer :  streamlit run app.py

Toute la logique métier est dans le paquet ``catmap`` (testé sans Streamlit) ; ce fichier ne
contient que l'interface. Points clés par rapport à la v1 :
- la recherche part d'un bouton « Lancer » et le résultat est mémorisé (``st.session_state``) :
  changer un filtre ne relance plus jamais Overpass (A20) ;
- un résultat vide ou une erreur n'est plus une exception (A6) ;
- tous les textes issus d'OSM sont échappés avant d'entrer dans une popup HTML ;
- carte avec regroupement (MarkerCluster) et plafond d'affichage (A13) ;
- chemins basés sur ``__file__`` : lancement possible depuis n'importe quel dossier.
"""
from __future__ import annotations

import html
import json
import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from catmap.area import AreaOptions, AreaQuery, locate_area
from catmap.circle_picker import circle_picker
from catmap.analysis import completeness, summary_by_company, top_with_other
from catmap.countries import get_country_index
from catmap.inputs import InputError, parse_names_text, read_company_list
from catmap.models import NameQuery, SearchOptions
from catmap.naf import NafClassifier
from catmap.overpass import OverpassClient
from catmap.pipeline import geolocate
from catmap.config import load_settings
from catmap import cyclone_map as cm
from catmap import cyclone_risk as cr
from catmap import cyclones as cy
from catmap import vigilance as vg
from catmap import meteoalarm as ma
from catmap import events as evt
from catmap import supply as sp
from catmap import websearch as ws
from catmap import activity as act
from catmap import georisques as gr
from catmap import floodhazard as fhz
import numpy as np
from catmap import damage as dmg, wind, seismic as sm, hail as hl
from catmap.references import refs_md
from catmap import coastalhazard as coast
from catmap import copernicus as cop
from catmap import runoff as rs

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAX_MARKERS = 5000
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
DISPLAY_COLUMNS = ["nom_demande", "name", "pays", "naf_section", "naf_division", "surface_m2", "match_score",
                   "match_field", "geom_kind", "lat", "long", "naf_source", "naf_confidence", "addr_street",
                   "addr_postcode", "addr_city", "brand", "osm_url"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# --------------------------------------------------------------------------
# Ressources partagées (chargées une seule fois par processus)
# --------------------------------------------------------------------------
def _flatten_secrets(m, out: dict) -> dict:
    """Aplati les sections TOML : une clé placée après un en-tête [section] reste utilisable."""
    for k, v in m.items():
        if hasattr(v, "items"):
            _flatten_secrets(v, out)
        else:
            out.setdefault(str(k), str(v))
    return out


def _secrets() -> dict:
    try:
        return _flatten_secrets(st.secrets, {})
    except Exception:  # noqa: BLE001 - pas de fichier secrets.toml : on s'appuie sur l'environnement
        return {}


def _ee_credentials():
    """(projet, compte de service) lus dans les secrets Streamlit ; ``(None, None)`` s'ils sont absents. Jamais affichés."""
    try:
        sec = st.secrets
        acc = sec["EE_SERVICE_ACCOUNT"] if "EE_SERVICE_ACCOUNT" in sec else None
        proj = sec["EE_PROJECT"] if "EE_PROJECT" in sec else None
        return (str(proj) if proj else None), (dict(acc) if hasattr(acc, "items") else acc)
    except Exception:  # noqa: BLE001 - pas de fichier secrets.toml
        return None, None


def get_settings():
    """Réglages relus à chaque exécution (peu coûteux) : un changement de code ou de secrets est pris en compte
    sans redémarrer le processus. Un cache ici figeait les anciennes valeurs après une mise à jour."""
    return load_settings(secrets=_secrets())


def _settings_sig() -> str:
    import hashlib
    return hashlib.sha1(repr(get_settings()).encode()).hexdigest()


@st.cache_resource
def _client_for(sig: str):
    return OverpassClient(get_settings())


def get_client():
    return _client_for(_settings_sig())


@st.cache_resource
def get_countries():
    return get_country_index()


@st.cache_resource
def get_naf():
    return NafClassifier()


@st.cache_resource
def _departements_for(sig: str):
    return vg.load_departements(get_settings())


def get_departements():
    return _departements_for(_settings_sig())


# --------------------------------------------------------------------------
# Aides d'affichage (compatibles avec plusieurs versions de Streamlit)
# --------------------------------------------------------------------------
def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _supply_popup_lines(row: pd.Series) -> str:
    """Lignes supplémentaires du popup pour la carte supply chain (rôle, relation, confiance, source)."""
    def g(k):
        v = row.get(k)
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else html.escape(str(v))
    out = []
    if g("role"):
        out.append(f"<b>Rôle :</b> {g('role')}" + (f" (client : {g('client')})" if g("role") == "fournisseur" and g("client") else ""))
    if g("type_relation"):
        out.append(f"<b>Relation :</b> {g('type_relation')}" + (f", rang {g('rang')}" if g("rang") else ""))
    if g("confiance"):
        out.append(f"<b>Confiance :</b> {g('confiance')}")
    if g("source_url"):
        out.append(f"<a href='{g('source_url')}' target='_blank' rel='noopener'>Source du recensement</a>")
    return ("<br>".join(out) + "<br>") if out else ""


def _popup_html(row: pd.Series) -> str:
    def esc(v):
        return "" if pd.isna(v) else html.escape(str(v))
    lat, lon = float(row["lat"]), float(row["long"])
    gmaps = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
    streetview = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"
    surf = "" if pd.isna(row.get("surface_m2")) else f"<br>Surface : {row['surface_m2']:,.0f} m²".replace(",", " ")
    return (f"<div style='font:12px sans-serif;min-width:180px'><b>{esc(row['name'])}</b><br>"
            f"{esc(row['nom_demande'])}<br>{esc(row.get('pays'))}<br>{esc(row.get('naf_section'))}{surf}<br>"
            f"Score : {esc(row.get('match_score'))}<br>{_supply_popup_lines(row)}"
            f"<a href='{esc(row['osm_url'])}' target='_blank' rel='noopener'>Voir sur OpenStreetMap</a><br>"
            f"<a href='{gmaps}' target='_blank' rel='noopener'>Voir sur Google Maps</a><br>"
            f"<a href='{streetview}' target='_blank' rel='noopener'>Voir sur Street View</a></div>")


_COLOR_CHOICES = {"Entreprise": "nom_demande", "Pays": "pays", "Secteur NAF": "naf_section", "Aucune (une seule couleur)": None,
                  "Rôle (client / fournisseurs)": "role"}


def _add_basemaps(m):
    """Fonds de carte communs à toutes les cartes de l'application (même jeu, même ordre, même défaut)."""
    import folium
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer("OpenStreetMap", name="Plan (OpenStreetMap)").add_to(m)
    folium.TileLayer(tiles=esri.format("World_Light_Gray_Base"), attr="Esri", name="Fond clair", max_native_zoom=16).add_to(m)
    folium.TileLayer(tiles=esri.format("World_Dark_Gray_Base"), attr="Esri", name="Fond sombre", max_native_zoom=16).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)


def _site_map(df: pd.DataFrame, area: dict | None = None, kp: str = "", default_color: str | None = None):
    """Carte interactive des sites du tableau : fonds de carte au choix, regroupement, popups, plein écran."""
    import folium
    from folium.plugins import Fullscreen, MarkerCluster
    from streamlit_folium import st_folium

    d = df.dropna(subset=["lat", "long"])
    if d.empty:
        st.info("Aucun site à afficher.")
        return
    choices = {k: v for k, v in _COLOR_CHOICES.items() if v != "role" or "role" in d.columns}
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Colorer les points par", list(choices), key=f"{kp}map_color",
                          index=list(choices).index(default_color) if default_color in choices else
                          (list(choices).index("Secteur NAF") if area else 0))
    cluster_on = c2.checkbox("Regrouper les points proches", True, key=f"{kp}map_cluster")
    color_col = choices[choice]

    if len(d) > MAX_MARKERS:
        st.warning(f"Carte limitée à {MAX_MARKERS} sites sur {len(d)} (les mieux appariés). "
                   "Filtrez par entreprise ou par pays pour tout voir.")
        d = d.sort_values("match_score", ascending=False).head(MAX_MARKERS)

    m = folium.Map(location=[float(d["lat"].mean()), float(d["long"].mean())], zoom_start=5,
                   control_scale=True, tiles=None)
    _add_basemaps(m)
    Fullscreen().add_to(m)
    m.fit_bounds([[float(d["lat"].min()), float(d["long"].min())], [float(d["lat"].max()), float(d["long"].max())]])

    if color_col:
        keys = sorted(d[color_col].fillna("n/d").astype(str).unique())
        colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
        if color_col == "role":                      # client en or, fournisseurs en bleu
            colors.update({"client": "#bd8e43", "fournisseur": "#1f77b4"})
    else:
        keys, colors = [], {}
    layer = MarkerCluster(name="Sites").add_to(m) if cluster_on else folium.FeatureGroup(name="Sites").add_to(m)
    for _, r in d.iterrows():
        col = colors.get(str(r[color_col]) if pd.notna(r[color_col]) else "n/d", "#1f77b4") if color_col else "#bd8e43"
        folium.CircleMarker(
            [float(r["lat"]), float(r["long"])], radius=6, color=col, fill=True, fill_opacity=0.75,
            popup=folium.Popup(_popup_html(r), max_width=320), tooltip=html.escape(str(r["name"])),
        ).add_to(layer)
    if area:
        folium.Circle([area["lat"], area["lon"]], radius=area["radius_m"], color="#bd8e43", weight=3, fill=True,
                      fill_opacity=0.06, tooltip=f"Périmètre : {area['radius_m']:.0f} m").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    if keys:
        shown = keys[:12]
        st.markdown(" &nbsp; ".join(
            f"<span style='color:#000'><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
            f"background:{colors[k]};margin-right:4px'></span>{html.escape(k)}</span>" for k in shown)
            + (" …" if len(keys) > len(shown) else ""), unsafe_allow_html=True)
    key = f"{kp}map_{color_col}_{cluster_on}_{len(d)}_{round(float(d['lat'].sum()), 4)}_{bool(area)}"
    try:
        st_folium(m, height=620, returned_objects=[], key=key, use_container_width=True)
    except TypeError:  # anciennes versions de streamlit-folium
        st_folium(m, height=620, width=1400, returned_objects=[], key=key)
    st.caption(f"{len(d)} site(s) affiché(s) : ceux du tableau ci-dessus (filtres Entreprises / Pays appliqués). "
               "Cliquez sur un point pour le détail ; le sélecteur en haut à droite change le fond de carte.")


def _pie(df: pd.DataFrame, col: str, title: str, empty_label: str):
    """Camembert : 10 modalités maximum, le reste regroupé dans « Autres » ; valeurs manquantes comptées à part."""
    import plotly.express as px
    t = top_with_other(df[col].fillna(empty_label), top_n=10)
    if t.empty:
        st.info(f"Aucune donnée pour « {title.lower()} » sur cette sélection.")
        return
    fig = px.pie(t, names="modalite", values="n", hole=0.3, title=title)
    fig.update_traces(textinfo="percent", texttemplate="%{percent:.0%}", sort=False,
                      hovertemplate="%{label}<br>%{value} site(s) (%{percent})<extra></extra>")
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#000000",
                      title_font_color="#000000", legend_font_color="#000000", margin=dict(t=60, b=20, l=10, r=10))
    _stretch(st.plotly_chart, fig)


# --------------------------------------------------------------------------
# Page « Géolocalisation »
# --------------------------------------------------------------------------
def _collect_queries(mode: str) -> tuple[list[NameQuery], list[str]]:
    if mode == "Un nom d'entreprise":
        name = st.text_input("Nom de l'entreprise", placeholder="Ex. IKEA", key="in_name")
        qid = st.text_input("Identifiant Wikidata (facultatif, recherche très précise)", placeholder="Ex. Q54078", key="in_qid")
        if qid.strip():
            st.info(f"Identifiant Wikidata renseigné ({qid.strip().upper()}) : il prime sur le nom et peut ramener une autre "
                    "entreprise que celle saisie. Videz ce champ pour chercher uniquement par nom.")
        return ([NameQuery.make(name, qid)] if name.strip() else []), []
    if mode == "Liste (une entreprise par ligne)":
        txt = st.text_area("Une entreprise par ligne ; identifiant Wikidata facultatif après « ; »",
                           placeholder="IKEA;Q54078\nLeroy Merlin", height=140, key="in_text")
        if not txt.strip():
            return [], []
        try:
            return parse_names_text(txt)
        except InputError as e:
            st.error(str(e))
            return [], []
    up = st.file_uploader("Fichier CSV ou Excel (nom d'entreprise en première colonne)", type=["csv", "txt", "xlsx"], key="in_file")
    header = st.radio("Première ligne", ["Détecter automatiquement", "C'est un en-tête", "C'est une entreprise"],
                      horizontal=True, key="in_header")
    if up is None:
        return [], []
    flag = {"Détecter automatiquement": None, "C'est un en-tête": True, "C'est une entreprise": False}[header]
    try:
        qs, warns = read_company_list(up.getvalue(), up.name, flag)
    except InputError as e:
        st.error(str(e))
        return [], []
    st.caption(f"{len(qs)} entreprise(s) lue(s).")
    return qs, warns


def _options_ui() -> SearchOptions:
    s = get_settings()
    with st.expander("Options de recherche (compromis rappel / précision)"):
        c1, c2 = st.columns(2)
        with c1:
            by_name = st.checkbox("Chercher dans « name » (nom du lieu)", True,
                                  help="Interroge le tag name d'OpenStreetMap par regex insensible à la casse.")
            by_brand = st.checkbox("Chercher dans « brand » (marque)", False,
                                   help="Interroge aussi le tag brand par regex. Plus de résultats (enseignes dont le nom du lieu diffère "
                                        "de la marque), mais requête plus lente et plus de bruit.")
            strict = st.checkbox(
                "Nom en mot entier (recommandé)", True,
                help="**Principe** : la regex envoyée à Overpass trouve toute chaîne qui *contient* le nom saisi, "
                     "y compris « Pikea » ou « xxIKEAxx ». Cette case décide de ce qu'on garde dans la réponse.\n\n"
                     "**Cochée** : on ne garde que les lieux où le nom apparaît comme mot entier "
                     "(« IKEA », « IKEA Paris Nord »). Les faux positifs sont écartés.\n\n"
                     "**Décochée** : on garde tout ce que renvoie la regex. Plus de résultats, mais plus de bruit ; "
                     "les lieux non conformes reçoivent un score faible (0,30).\n\n"
                     "**Effet sur la requête** : aucun, le tri se fait après coup.")
            biz = st.checkbox(
                "Établissements uniquement", False,
                help="**Principe** : OpenStreetMap contient aussi des objets qui portent un nom sans être un établissement "
                     "(parking, arrêt, monument, parcelle…). Cette case filtre sur la nature de l'objet.\n\n"
                     "**Cochée** : on ne garde que les objets ayant au moins un tag *shop, amenity, office, craft, industrial, "
                     "man_made, power, healthcare, tourism, leisure, landuse* ou *building*. Moins de bruit, mais on peut perdre "
                     "un site mal renseigné dans OSM.\n\n"
                     "**Décochée** : tout objet portant le nom est gardé (plus de résultats, plus de bruit). "
                     "Les limites administratives (tag boundary) sont toujours écartées.\n\n"
                     "**Effet sur la requête** : aucun, le tri se fait après la réponse d'Overpass.")
            rel = st.checkbox(
                "Inclure les grands sites (relations)", False,
                help="**Principe** : un grand site (campus, usine, centre commercial avec cour intérieure) est parfois décrit "
                     "dans OpenStreetMap par une *relation* multipolygone plutôt que par un point ou un contour simple.\n\n"
                     "**Cochée** : la requête interroge aussi les relations. On trouve ces grands sites et leur surface, "
                     "au prix d'une requête plus lourde.\n\n"
                     "**Décochée** : seuls les points (nodes) et les contours simples (ways) sont interrogés. "
                     "Plus rapide, mais les sites décrits par une relation sont manqués.\n\n"
                     "**Effet sur la requête** : oui, la ligne `rel[type=multipolygon]…` est ajoutée ou retirée.")
            op = st.checkbox(
                "Chercher aussi dans « operator »", False,
                help="**Principe** : le tag *operator* désigne l'exploitant d'un lieu, souvent une société différente de "
                     "l'enseigne (ex. un franchisé).\n\n"
                     "**Cochée** : la requête cherche aussi le nom dans operator. Plus de résultats, plus de bruit ; "
                     "le score de ces correspondances est diminué de 0,20.\n\n"
                     "**Décochée** : seuls name et/ou brand (selon vos cases) sont interrogés.\n\n"
                     "**Effet sur la requête** : oui, une condition `operator` est ajoutée.")
        with c2:
            idx = get_countries()
            labels = sorted(f"{a['name_fr']} ({a['iso2']})" for a in idx.attrs if a.get("iso2") and a.get("name_fr"))
            picked = st.multiselect(
                "Limiter à des pays (recommandé pour un grand périmètre)", labels,
                help="**Principe** : restreint la recherche aux pays choisis.\n\n"
                     "**Vide** : le monde entier est interrogé (plus lent, plus de résultats).\n\n"
                     "**Rempli** : la requête est limitée à ces pays (`area[\"ISO3166-1\"=…]`), donc plus rapide "
                     "et plus ciblée.\n\n**Effet sur la requête** : oui.")
            min_score = st.slider(
                "Score minimal conservé", 0.0, 1.0, 0.0, 0.05,
                help="**Principe** : chaque lieu trouvé reçoit un score de 0 à 1 mesurant la qualité de la correspondance "
                     "entre le nom saisi et le nom OpenStreetMap (calculé après la réponse d'Overpass, sur les mots entiers, "
                     "sans tenir compte de la casse, des accents ni des séparateurs).\n\n"
                     "**Calcul** :\n\n"
                     "- **1,00** : identifiant Wikidata identique ;\n"
                     "- **0,90** : nom identique (« IKEA » pour « IKEA ») ;\n"
                     "- **0,75** : le nom saisi est le début du nom OSM (« IKEA Lyon ») ;\n"
                     "- **0,60** : le nom saisi est contenu en mots entiers (« Magasin IKEA ») ;\n"
                     "- **moins 0,20** si la correspondance vient du tag operator ;\n"
                     "- **0,30** si « Nom en mot entier » est décoché et que le nom n'apparaît pas en mot entier.\n\n"
                     "**Ce curseur** : tout lieu dont le score est inférieur au seuil est écarté. "
                     "0 = on garde tout ; 0,75 = on garde les noms identiques ou commençant par le nom saisi.\n\n"
                     "**Effet sur la requête** : aucun, le tri se fait après coup.")
            dedup = st.slider(
                "Fusion des doublons node/way (mètres)", 0, 100, 25,
                help="**Principe** : un même site peut exister plusieurs fois dans OpenStreetMap (un point posé dans le bâtiment "
                     "et le contour du bâtiment).\n\n"
                     "**Valeur > 0** : un point situé dans un contour du même nom, ou à moins de cette distance, "
                     "est fusionné avec lui ; on garde le contour et on reprend les tags du point.\n\n"
                     "**0** : aucune fusion, tous les objets sont gardés.\n\n"
                     "**Effet sur la requête** : aucun, la fusion se fait après coup.")
            fast = st.checkbox(
                "Mode rapide (noms exacts uniquement)", False,
                help="**Principe** : par défaut, Overpass compare le nom à une regex insensible à la casse. "
                     "Cela parcourt tous les lieux nommés du périmètre : c'est complet mais lent (plusieurs minutes "
                     "sur le monde entier). Le mode rapide remplace la regex par des égalités exactes sur quelques "
                     "graphies (IKEA, ikea, Ikea, sans accents, avec tirets…), que le serveur retrouve instantanément "
                     "grâce à son index.\n\n"
                     "**Coché** : réponse en quelques secondes, mais seuls les lieux dont le champ vaut exactement "
                     "l'une de ces graphies sont trouvés. « IKEA » est trouvé, « IKEA Paris Nord » ne l'est pas.\n\n"
                     "**Décoché** : regex complète, qui trouve aussi « IKEA Paris Nord », « Magasin IKEA », etc. "
                     "Plus de résultats, mais plus lent.\n\n"
                     "**Effet sur la requête** : oui. La regex est remplacée par une liste d'égalités exactes "
                     "(les champs name / brand cochés restent respectés). Timeout serveur de 60 s au lieu de 280 s.")
    if not (by_name or by_brand):
        st.warning("Cochez « name » et/ou « brand » : sans champ de recherche, seule la recherche par identifiant Wikidata fonctionne.")
    if s.uses_public_overpass:
        st.caption("Serveurs Overpass **publics** : usage raisonnable demandé. "
                   + ("" if s.contact else "Renseignez `CATMAP_CONTACT` (e-mail ou URL) pour vous identifier. ")
                   + "Pour de gros volumes, voir « Overpass local » dans le README.")
    else:
        st.caption(f"Instance Overpass **locale** : {s.overpass_urls[0]}")
    return SearchOptions(search_name=by_name, search_brand=by_brand, include_operator=op, include_relations=rel, require_business_tag=biz,
                         countries=tuple(p.rsplit("(", 1)[1].rstrip(")") for p in picked),
                         strict_word_match=strict, min_score=min_score, dedup_radius_m=float(dedup),
                         search_mode="exact" if fast else "regex")


def _run_search(queries: list[NameQuery], options: SearchOptions):
    box = st.empty()
    bar = box.progress(0.0, text="Démarrage…")

    def cb(done: int, total: int, name: str):
        bar.progress(done / total, text=f"{done}/{total} — {name}")

    status = st.empty()
    res = geolocate(queries, options, get_client(), get_settings(), get_countries(), get_naf(), progress=cb,
                    on_event=(lambda m: status.caption(m)) if get_settings().max_workers == 1 else None)
    status.empty()
    box.empty()
    st.session_state["result"] = res
    st.session_state["result_options"] = options
    st.session_state["result_queries"] = [(q.raw.casefold(), q.wikidata) for q in queries]


def _status_messages(report: pd.DataFrame):
    for statut, fn, label in (("erreur", st.error, "en erreur"), ("partiel", st.warning, "avec résultat partiel"),
                              ("aucun résultat", st.info, "sans résultat")):
        sub = report[report["statut"] == statut]
        if sub.empty:
            continue
        lines = [f"- **{r.nom_demande}**" + (f" : {r.message}" if r.message else "") for r in sub.itertuples()]
        fn(f"{len(sub)} entreprise(s) {label} :\n\n" + "\n".join(lines[:30]))


def _render_result(res):
    sites, report = res.sites, res.report
    n_req = len(report)
    n_found = int((report["statut"].isin(["ok", "partiel"]) & (report["n_sites"] > 0)).sum())
    comp = completeness(sites)
    area = res.meta.get("area")
    c = st.columns(5)
    c[0].metric("Sites", f"{len(sites):,}".replace(",", " "))
    if area and area.get("radius_m"):
        c[1].metric("Rayon du périmètre", f"{area['radius_m']:,.0f} m".replace(",", " "))
    elif area:
        c[1].metric("Zone", str(area.get("country") or area.get("iso2") or "pays"))
    else:
        c[1].metric("Entreprises trouvées", f"{n_found} / {n_req}")
    c[2].metric("Pays renseigné", f"{comp['pays']:.0f} %")
    c[3].metric("Secteur NAF renseigné", f"{comp['naf_section']:.0f} %")
    c[4].metric("Base OSM", (res.meta["snapshots"][0][:10] if res.meta["snapshots"] else "n/d"),
                help="Date de la base Overpass interrogée (traçabilité).")
    _status_messages(report)
    if sites.empty:
        if area:
            st.info("Aucun site à afficher sur ce périmètre. Essayez d'élargir le rayon, de décocher « Uniquement les lieux "
                    "nommés » ou d'inclure les relations.")
        else:
            st.info("Aucun site à afficher. Essayez de désactiver « Nom en mot entier » ou « Établissements uniquement », "
                    "ou de renseigner un identifiant Wikidata.")
        _render_report_tab(res)
        return

    companies = sorted(sites["nom_demande"].unique())
    countries = sorted(sites["pays"].dropna().unique())
    if area:   # un seul « périmètre » : le filtre par entreprise n'a pas de sens
        pick_c = companies
        pick_p = st.multiselect("Pays", countries, key="f_country", help="Vide = tous (y compris sans pays)")
    else:
        f1, f2 = st.columns(2)
        pick_c = f1.multiselect("Entreprises", companies, default=companies, key="f_company")
        pick_p = f2.multiselect("Pays", countries, key="f_country", help="Vide = tous (y compris sans pays)")
    view = sites[sites["nom_demande"].isin(pick_c)]
    if pick_p:
        view = view[view["pays"].isin(pick_p)]

    t1, t4, t5 = st.tabs(["Tableau, graphiques et carte", "Rapport par entreprise", "Qualité"])
    with t1:
        cols = [c for c in DISPLAY_COLUMNS if c in view.columns]
        if area:   # « name » en premier, puis la distance au centre après la position
            cols = ["name"] + [c for c in cols if c not in ("name", "nom_demande")]
            if "distance_centre_m" in view.columns:
                cols.insert(cols.index("long") + 1 if "long" in cols else len(cols), "distance_centre_m")
        _stretch(st.dataframe, view[cols], hide_index=True,
                 column_config={"osm_url": st.column_config.LinkColumn("OSM", display_text="ouvrir")})
        d1, d2 = st.columns(2)
        d1.download_button("Télécharger (CSV)", _csv_bytes(view), "catmap_sites.csv", "text/csv")
        d2.download_button("Télécharger (Excel)", _xlsx_bytes({"sites": view, "rapport": report}), "catmap_sites.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.subheader("Répartitions")
        g_naf, g_pays = st.columns(2)
        with g_naf:
            if area:
                naf_company = "Toutes"
                naf_country = st.selectbox("Pays (secteur NAF)", ["Tous"] + sorted(view["pays"].dropna().unique()),
                                           key="pie_naf_country")
            else:
                fc1, fc2 = st.columns(2)
                naf_company = fc1.selectbox("Entreprise (secteur NAF)", ["Toutes"] + companies, key="pie_naf_company")
                naf_country = fc2.selectbox("Pays (secteur NAF)", ["Tous"] + sorted(view["pays"].dropna().unique()),
                                            key="pie_naf_country")
            dn = view
            if naf_company != "Toutes":
                dn = dn[dn["nom_demande"] == naf_company]
            if naf_country != "Tous":
                dn = dn[dn["pays"] == naf_country]
            _pie(dn, "naf_section", "Répartition par secteur NAF", "Non classé")
        with g_pays:
            pays_company = "Toutes" if area else st.selectbox("Entreprise (pays)", ["Toutes"] + companies, key="pie_pays_company")
            dp = view if pays_company == "Toutes" else view[view["nom_demande"] == pays_company]
            _pie(dp, "pays", "Répartition par pays", "Pays non renseigné")
        st.subheader("Carte interactive")
        _site_map(view, area)
    with t4:
        _render_report_tab(res)
    with t5:
        _render_quality_tab(view, area)


def _render_report_tab(res):
    st.subheader("Synthèse par entreprise")
    _stretch(st.dataframe, summary_by_company(res.sites, res.report), hide_index=True)
    st.subheader("Détail de la recherche")
    _stretch(st.dataframe, res.report, hide_index=True)
    st.caption(f"Serveur(s) : {', '.join(res.meta['endpoints']) or 'cache'} — options : {res.meta['options']}")


def _render_quality_tab(view: pd.DataFrame, area: dict | None = None):
    if not area:   # pas de score d'appariement du nom pour une recherche par périmètre
        st.markdown("**Fiabilité de l'appariement du nom** (plus le score est élevé, plus le site est sûr)")
        bins = pd.cut(view["match_score"], [0, .5, .7, .8, .95, 1.0], labels=["≤0,5", "0,5-0,7", "0,7-0,8", "0,8-0,95", "1,0"])
        _stretch(st.dataframe, bins.value_counts().sort_index().rename("sites").reset_index().rename(columns={"match_score": "score"}), hide_index=True)
    st.markdown("**Fiabilité de la classification NAF**")
    _stretch(st.dataframe, view["naf_confidence"].fillna("non classé").value_counts().rename("sites").reset_index(), hide_index=True)
    un = get_naf().unmapped(view)
    if not un.empty:
        st.markdown("**Valeurs de tags OSM absentes de `data/naf_mapping.csv`** (à compléter)")
        _stretch(st.dataframe, un.head(50), hide_index=True)
    snapped = int((view["pays_methode"] == "nearest").sum())
    if snapped:
        st.caption(f"{snapped} site(s) rattaché(s) au pays le plus proche (point côtier ou frontalier).")


def _clear_all():
    """Vide le cache disque Overpass, les résultats affichés, les filtres et la saisie (appelé par le bouton)."""
    n = 0
    cache_dir = Path(get_settings().cache_dir)
    if cache_dir.is_dir():
        for f in list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.tmp")):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    for k in ("result", "result_options", "result_queries", "f_company", "f_country", "pie_naf_company", "pie_naf_country", "pie_pays_company",
              "map_color", "map_cluster", "area_pick_n"):
        st.session_state.pop(k, None)
    for k in ("in_name", "in_qid", "in_text"):
        if k in st.session_state:
            st.session_state[k] = ""
    try:
        st.cache_data.clear()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["_cleared"] = n


def _area_new(kp="area_", result_key="result"):
    """Bouton « Nouvelle prise » : nouvelle instance de la carte (cercle effacé) et résultats du périmètre retirés."""
    st.session_state[f"{kp}pick_n"] = st.session_state.get(f"{kp}pick_n", 0) + 1
    res = st.session_state.get(result_key)
    if res is not None and res.meta.get("area"):
        st.session_state.pop(result_key, None)


def _page_area(kp="area_", result_key="result", center=(46.6, 2.5), zoom=6):
    s = get_settings()
    with st.expander("Options du périmètre"):
        rel = st.checkbox(
            "Inclure les grands sites (relations)", False, key=f"{kp}rel",
            help="**Principe** : un grand site (campus, usine, centre commercial) est parfois décrit par une *relation* "
                 "multipolygone plutôt que par un point ou un contour simple.\n\n"
                 "**Cochée** : les relations sont interrogées aussi (requête plus lourde).\n\n"
                 "**Décochée** : seuls les points (nodes) et les contours (ways) sont interrogés.")
        named = st.checkbox(
            "Uniquement les lieux nommés (name ou brand)", True, key=f"{kp}named",
            help="**Principe** : les objets *shop*, *amenity* ou *brand* incluent aussi du mobilier sans nom (bancs, "
                 "poubelles, parkings à vélos…).\n\n"
                 "**Cochée** : on garde les objets qui portent un name ou un brand.\n\n"
                 "**Décochée** : tout est gardé (beaucoup plus de bruit).")
        dedup = st.slider(
            "Fusion des doublons node/way (mètres)", 0, 100, 25, key=f"{kp}dedup",
            help="Un même site peut exister en point et en contour dans OpenStreetMap. Valeur > 0 : on fusionne ceux de même "
                 "nom situés à moins de cette distance (ou un point dans un contour). 0 : aucune fusion.")
    st.caption(f"Rayon maximal : **{s.area_max_radius_m:,} m**".replace(",", " ") + f" (au-delà, la recherche serait trop lourde) ; "
               f"au plus {s.area_max_sites:,} sites sont conservés (les plus proches du centre).".replace(",", " "))
    n = st.session_state.get(f"{kp}pick_n", 0)
    picked = circle_picker(center=center, zoom=zoom, max_radius_m=s.area_max_radius_m, key=f"{kp}picker_{n}")
    area = None
    if picked:
        try:
            area = AreaQuery.from_value(picked)
            area.validate(s.area_max_radius_m)
        except ValueError as e:
            st.error(str(e))
            area = None
    if area:
        st.markdown(f"**Périmètre validé** : centre ({area.lat:.5f}, {area.lon:.5f}), rayon **{area.radius_m:,.0f} m**".replace(",", " "))
        b1, b2 = st.columns(2)
        b1.button("Nouvelle prise", key=f"{kp}new", on_click=_area_new, args=(kp, result_key))
        go = b2.button("Recherche", type="primary", key=f"{kp}go")
        if go:
            _run_area(area, AreaOptions(include_relations=rel, named_only=named, dedup_radius_m=float(dedup)),
                      result_key=result_key)
    res = st.session_state.get(result_key)
    if res is not None and res.meta.get("area"):
        st.divider()
        _render_result(res)


def _run_area(area: AreaQuery, options: AreaOptions, result_key="result"):
    status = st.empty()
    with st.spinner("Interrogation d'Overpass sur le périmètre…"):
        res = locate_area(get_client(), get_settings(), area, options, get_countries(), get_naf(),
                          on_event=lambda m: status.caption(m))
    status.empty()
    st.session_state[result_key] = res
    st.session_state["result_options"] = options


ZONE_MODES = ["Toutes les entreprises d'une zone", "Un secteur d'activité (code NAF)",
              "Une activité économique (langage naturel)"]


def _page_zone():
    mode = st.radio("Que cherchez-vous ?", ZONE_MODES, horizontal=True, key="zone_mode")
    if mode == ZONE_MODES[0]:
        st.markdown("**Toutes les entreprises situées dans une zone** : cliquez sur la carte pour fixer le centre, déplacez la souris "
                    "pour régler le rayon, cliquez de nouveau pour valider, puis lancez la recherche. Les objets portant un tag "
                    "*shop*, *amenity* ou *brand* sont retenus.")
        _page_area()
    else:
        _page_activity(mode == ZONE_MODES[2])


def _country_choices():
    out = {}
    for a in get_countries().attrs:
        iso = (a.get("iso2") or "").upper()
        if iso:
            out[f"{a.get('name_fr') or a.get('name_en')} ({iso})"] = (iso, a.get("name_fr") or a.get("name_en") or iso)
    return dict(sorted(out.items()))


def _activity_label(aid: str) -> str:
    a = act.ACTIVITY_BY_ID.get(aid)
    return f"{a.label} (NAF {', '.join(a.naf)})" if a else aid


def _page_activity(natural: bool):
    s = get_settings()
    kp = "act_"
    countries = _country_choices()
    detected_country = None
    if natural:
        st.markdown("**Partir d'une activité économique** : saisissez un terme court (« café », « commerce », « pharmacie », "
                    "« composants électroniques »), sans lieu : le périmètre se choisit plus bas. Le terme est d'abord cherché dans "
                    "la table de correspondance NAF/OSM de l'application ; l'IA (Linkup) n'est interrogée que si rien ne correspond.")
        text = st.text_input("Activité", key=f"{kp}text", placeholder="café")
        key = _search_key()
        use_ai = st.checkbox("Si aucune correspondance, interroger l'IA (Linkup)", value=bool(key), key=f"{kp}use_ai",
                             disabled=not key)
        if not key:
            st.caption("IA indisponible : renseignez `LINKUP_API_KEY` dans les secrets Streamlit pour l'activer.")
        if st.button("Chercher le terme", key=f"{kp}analyse", disabled=not text.strip()):
            interp = act.lookup_term(text, get_naf())
            ai_err = ""
            if interp.criteria.is_empty() and use_ai and key:
                try:
                    interp = act.interpret_ai(text, key)
                    interp.notes.append("Aucune correspondance dans la table : critères proposés par l'IA (à vérifier).")
                except RuntimeError as e:
                    ai_err = str(e)
            st.session_state[f"{kp}interp"] = (text, interp, ai_err)
            st.session_state[f"{kp}acts"] = list(interp.criteria.activities)
            st.session_state[f"{kp}kw"] = ", ".join(interp.criteria.extra_names)
            st.session_state[f"{kp}divs"] = list(interp.criteria.naf_divisions)
            st.session_state[f"{kp}tags"] = [f"{k}={v}" for k, v in interp.criteria.extra_tags]
            st.session_state.pop(f"{kp}result", None)
        stored = st.session_state.get(f"{kp}interp")
        if stored and stored[0] == text:
            _, interp, ai_err = stored
            st.caption(f"Source : **{interp.source}**")
            if ai_err:
                st.warning(f"IA indisponible : {ai_err}")
            for n in interp.notes:
                st.caption(n)
        elif stored:
            st.caption("Le terme a changé : cliquez sur « Chercher le terme ».")
        tag_opts = list(dict.fromkeys(st.session_state.get(f"{kp}tags", [])))
        chosen_tags = st.multiselect("Tags OSM retenus (modifiables)", tag_opts, key=f"{kp}tags")
        chosen = st.multiselect("Activités de la taxonomie", list(act.ACTIVITY_BY_ID), key=f"{kp}acts",
                                format_func=_activity_label)
        divs = st.multiselect("Divisions NAF", list(act.DIVISIONS), key=f"{kp}divs",
                              format_func=lambda d: f"{d} - {act.DIVISIONS[d]}")
        kw = st.text_input("Mots du nom à ajouter (séparés par des virgules)", key=f"{kp}kw")
        words = [w.strip().lower() for w in kw.split(",") if act._SAFE_WORD.match(w.strip())]
        pairs = tuple((t.split("=", 1)[0], t.split("=", 1)[1]) for t in chosen_tags if "=" in t)
        criteria = act.Criteria(tuple(chosen), act.normalize_divisions(divs), pairs, tuple(dict.fromkeys(words)),
                                (text or "").strip()[:80])
    else:
        st.markdown("**Rechercher un secteur d'activité (NAF)** : choisissez des divisions NAF rév. 2 (ou saisissez des codes comme "
                    "`26.11Z`, `C`). Les tags OSM correspondants viennent de la table de correspondance de l'application, "
                    "complétée par la taxonomie des activités industrielles.")
        divs = st.multiselect("Divisions NAF", list(act.DIVISIONS), key=f"{kp}naf_divs",
                              format_func=lambda d: f"{d} - {act.DIVISIONS[d]}")
        free = st.text_input("Ou codes NAF libres (séparés par des virgules)", key=f"{kp}naf_free", placeholder="26.11Z, 10, C")
        codes = act.normalize_divisions(list(divs) + [c for c in free.split(",") if c.strip()])
        if free.strip() and not act.normalize_divisions([c for c in free.split(",") if c.strip()]):
            st.warning("Aucun code NAF reconnu dans la saisie libre.")
        st.caption("Seule la division (2 premiers chiffres) est utilisée : OpenStreetMap ne permet pas d'aller plus fin de façon fiable.")
        label = "NAF " + ", ".join(codes) if codes else ""
        criteria = act.Criteria(naf_divisions=codes, label=label[:80])

    filters = act.build_filters(criteria, get_naf()) if not criteria.is_empty() else []
    if filters:
        with st.expander(f"Critères OpenStreetMap utilisés ({len(filters)})"):
            _stretch(st.dataframe, pd.DataFrame([{"activité": f.activity, "type": f.kind, "critère": f.description}
                                                 for f in filters]), hide_index=True)
    with st.expander("Options"):
        named = st.checkbox("Uniquement les lieux nommés (name ou brand)", True, key=f"{kp}named")
        dedup = st.slider("Fusion des doublons (mètres)", 0, 100, 25, key=f"{kp}dedup")
    options = AreaOptions(include_relations=False, named_only=named, dedup_radius_m=float(dedup))

    scope_kind = st.radio("Périmètre", ["Un pays", "Un cercle sur la carte"], horizontal=True, key=f"{kp}scope")
    scope = None
    if scope_kind == "Un pays":
        names = list(countries)
        default = next((i for i, n in enumerate(names) if detected_country and n.endswith(f"({detected_country})")), 0)
        sel = st.selectbox("Pays", names, index=default, key=f"{kp}country_{detected_country or 'x'}")
        iso, nm = countries[sel]
        scope = act.ActivityScope("country", iso2=iso, country=nm)
        st.caption("Recherche sur tout le territoire du pays (zone administrative OpenStreetMap). Les grands pays ou les critères "
                   f"très larges peuvent être longs ; au plus {s.area_max_sites:,} objets sont lus.".replace(",", " "))
        if any(f.kind == "tag" and f.key in ("shop", "amenity") for f in filters):
            st.warning("Ces critères incluent des commerces ou services très nombreux : à l'échelle d'un pays la requête peut "
                       "dépasser la limite ; préférez un cercle.")
        go_row = st.container()
    else:
        n = st.session_state.get(f"{kp}pick_n", 0)
        picked = circle_picker(center=(46.6, 2.5), zoom=6, max_radius_m=s.area_max_radius_m, key=f"{kp}picker_{n}")
        go_row = st.container()
        if picked:
            try:
                area = AreaQuery.from_value(picked)
                area.validate(s.area_max_radius_m)
                scope = act.ActivityScope("circle", area=area)
                st.markdown(f"**Périmètre validé** : centre ({area.lat:.5f}, {area.lon:.5f}), rayon **{area.radius_m:,.0f} m**"
                            .replace(",", " "))
                go_row.button("Nouvelle prise", key=f"{kp}new", on_click=_area_new, args=(kp, f"{kp}result"))
            except ValueError as e:
                st.error(str(e))
    if go_row.button("Recherche", type="primary", key=f"{kp}go{'_nl' if natural else '_naf'}",
                     disabled=(scope is None or criteria.is_empty())):
        status = st.empty()
        with st.spinner("Interrogation d'Overpass…"):
            res = act.locate_activity(get_client(), s, scope, criteria, options, get_countries(), get_naf(),
                                      on_event=lambda m: status.caption(m))
        status.empty()
        st.session_state[f"{kp}result"] = res
    if criteria.is_empty():
        st.caption("Choisissez au moins une activité ou un code NAF pour lancer la recherche.")
    res = st.session_state.get(f"{kp}result")
    if res is not None and res.meta.get("activity") is not None:
        st.divider()
        _render_result(res)


def page_geolocalisation():
    st.title("Géolocalisation")
    t_ent, t_sc, t_zone = st.tabs(["Géolocalisation des entreprises", "Géolocalisation supply chain", "Entreprises dans une zone"])
    with t_ent:
        _tab_entreprises()
    with t_sc:
        _tab_supply_chain()
    with t_zone:
        _page_zone()


def _tab_entreprises():
    mode = st.radio("Mode de saisie", ["Un nom d'entreprise", "Liste (une entreprise par ligne)", "Fichier CSV / Excel"],
                    horizontal=True)
    queries, warns = _collect_queries(mode)
    for w in warns[:5]:
        st.warning(w)
    options = _options_ui()
    with st.expander("Tester la connexion Overpass"):
        if st.button("Lancer le test"):
            with st.spinner("Test des serveurs…"):
                rows = get_client().ping()
            for r in rows:
                (st.success if r["ok"] else st.error)(f"{r['endpoint']} — {r['latence_s']} s — {r['message']}")
            if not any(r["ok"] for r in rows):
                st.info("Aucun serveur joignable depuis cet hébergement. Voir le README : instance Overpass "
                        "payante/auto-hébergée accessible depuis internet, variable `OVERPASS_URLS`.")
    b1, b2 = st.columns([1, 1])
    run = b1.button("Lancer la recherche", type="primary", disabled=not queries)
    b2.button("Vider le cache et les résultats", on_click=_clear_all)
    b2.caption("Supprime les réponses Overpass mémorisées, les résultats affichés et la saisie. "
               "La prochaine recherche interrogera de nouveau Overpass.")
    if "_cleared" in st.session_state:
        st.success(f"Cache vidé ({st.session_state.pop('_cleared')} fichier(s) supprimé(s)). Saisie et résultats réinitialisés.")
    if run:
        _run_search(queries, options)
    res = st.session_state.get("result")
    if res is not None and not res.meta.get("area"):
        st.divider()
        shown = sorted(res.report["nom_demande"].astype(str).unique()) if "nom_demande" in res.report else []
        if shown:
            st.markdown("**Résultats pour :** " + ", ".join(shown[:10]) + (" …" if len(shown) > 10 else ""))
        current = [(q.raw.casefold(), q.wikidata) for q in queries]
        if current and current != st.session_state.get("result_queries"):
            st.warning("Ces résultats correspondent à la recherche précédente, pas à la saisie actuelle. "
                       "Cliquez sur « Lancer la recherche » pour les mettre à jour.")
        _render_result(res)


# --------------------------------------------------------------------------
# Onglet « Géolocalisation supply chain »
# --------------------------------------------------------------------------
SC_MAX_SUPPLIERS = 300
CONFIDENCE_HELP = (
    "**Niveau de confiance** : solidité de la preuve que l'entreprise est bien un fournisseur du client.\n\n"
    "**A** : liste publiée par l'entreprise elle-même (document fourni).\n\n"
    "**B** : site de production rattaché au client dans une base ouverte (réservé, pas encore utilisé).\n\n"
    "**C** : proposé par l'IA avec une source, et la citation a été retrouvée mot pour mot dans la page source.\n\n"
    "**D** : proposé par l'IA ; citation non retrouvée ou page illisible : à vérifier avant usage.\n\n"
    "**n/d** : saisie manuelle, non documentée.\n\n"
    "Une citation retrouvée prouve que la phrase existe dans la page, pas que l'IA l'a bien interprétée.")


def _sc_add(df: pd.DataFrame):
    cur = st.session_state.get("sc_cands")
    both = df if cur is None or cur.empty else pd.concat([cur, df], ignore_index=True)
    st.session_state["sc_cands"] = sp.consolidate(both)
    st.session_state.pop("sc_result", None)


def _sc_table_source(src: str):
    """Charge un document (adresse web ou fichier) et mémorise les tableaux détectés."""
    if src == "Adresse web (page ou PDF)":
        url = st.text_input("Adresse du document", placeholder="https://… (page web ou PDF contenant la liste des fournisseurs)", key="sc_url")
        if st.button("Lire le document", key="sc_read_url", disabled=not url.strip()):
            try:
                with st.spinner("Téléchargement et lecture…"):
                    data, mime = sp.fetch_url(url.strip())
                    tabs = sp.tables_from_bytes(data, urlparse_name(url), mime)
                st.session_state["sc_tables"] = {"label": url.strip(), "url": url.strip(), "tables": tabs}
            except sp.SupplyError as e:
                st.session_state.pop("sc_tables", None)
                st.error(str(e))
    else:
        up = st.file_uploader("Fichier de fournisseurs (CSV, Excel, PDF ou page HTML enregistrée)",
                              type=["csv", "txt", "xlsx", "xlsm", "pdf", "html", "htm"], key="sc_file")
        if up is not None and st.session_state.get("sc_tables", {}).get("label") != up.name:
            try:
                tabs = sp.tables_from_bytes(up.getvalue(), up.name, up.type or "")
                st.session_state["sc_tables"] = {"label": up.name, "url": "", "tables": tabs}
            except sp.SupplyError as e:
                st.session_state.pop("sc_tables", None)
                st.error(str(e))
    info = st.session_state.get("sc_tables")
    if info:
        _sc_pick_table(info, "sc")


def _sc_pick_table(info: dict, kp: str, official_default: bool = True):
    """Choix d'un tableau détecté, de ses colonnes, puis ajout à la liste des fournisseurs."""
    if not info["tables"]:
        st.warning("Aucun tableau détecté dans ce document. Si la liste est dans du texte libre, utilisez le mode assisté.")
        return
    labels = [t[0] for t in info["tables"]]
    pick = st.selectbox("Tableau à utiliser", labels, key=f"{kp}_table_pick")
    df = info["tables"][labels.index(pick)][1]
    _stretch(st.dataframe, df.head(8), hide_index=True)
    g_name, g_country = sp.guess_columns(df)
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    name_col = c1.selectbox("Colonne du nom du fournisseur", cols, index=cols.index(g_name) if g_name in cols else 0,
                            key=f"{kp}_namecol_{pick}")
    country_col = c2.selectbox("Colonne du pays (facultatif)", ["(aucune)"] + cols,
                               index=(cols.index(g_country) + 1) if g_country in cols else 0, key=f"{kp}_ctrcol_{pick}")
    official = st.checkbox("Ce document est la liste de fournisseurs publiée par l'entreprise elle-même", official_default,
                           key=f"{kp}_official", help="Coché : confiance A. Décoché : confiance C (source tierce).")
    if st.button("Ajouter ces fournisseurs à la liste", key=f"{kp}_add_tbl"):
        _sc_add(sp.candidates_from_table(df, name_col, None if country_col == "(aucune)" else country_col,
                                         source=info["label"][:120], source_url=info["url"], confiance="A" if official else "C"))


def _search_key() -> str:
    import os
    sec = _secrets()
    return (os.environ.get("LINKUP_API_KEY") or str(sec.get("LINKUP_API_KEY") or sec.get("linkup_api_key") or "")).strip()


def _sc_auto_search(client: str):
    key = _search_key()
    if not key:
        st.info("La recherche automatique utilise Linkup (linkup.so). Créez un compte, copiez votre clé d'API et ajoutez-la dans "
                "les secrets Streamlit sous `LINKUP_API_KEY`. D'après leur documentation, les comptes créés avec une adresse "
                "professionnelle reçoivent 20 $ de crédit par mois (à vérifier). En attendant, utilisez l'adresse web, le "
                "fichier ou le mode assisté.")
        return
    if not client:
        st.info("Renseignez d'abord le nom du client.")
        return
    use_docs = _checkbox_with_help(
        "Recherche de documents",
        "**Ce qui est extrait** : des adresses de documents publics (pages web, PDF) susceptibles de contenir une liste de "
        "fournisseurs, puis les tableaux qu'ils contiennent (nom du fournisseur, pays).\n\n"
        "**Comment** : 4 recherches Linkup (liste de fournisseurs, usines, version française, rapport de durabilité). "
        "L'application télécharge les 8 meilleurs résultats et détecte leurs tableaux, sans modèle de langage. Les documents "
        "sans tableau ne sont pas analysés.\n\n"
        "**Coût** : environ 0,02 $ par recherche (4 requêtes à 0,005 $), prélevé sur vos crédits Linkup.",
        key="sc_use_docs", value=True, wide=True)
    use_ai = _checkbox_with_help(
        "Extraction par IA",
        "Utilisation de Linkup : consomme des crédits (environ 0,06 $ par recherche)\n\n"
        "**Ce qui est extrait** : les fournisseurs, fabricants et sous-traitants déclarés publiquement, avec pays, type de "
        "relation, rang, adresse de la source et citation.\n\n"
        "**Comment** : un modèle de langage de Linkup cherche sur le web (plusieurs langues), lit les documents, y compris en "
        "texte libre (rapports RSE, rapports annuels), et renvoie une liste structurée. L'application vérifie ensuite chaque "
        "citation dans la page source : retrouvée mot pour mot = confiance C, sinon D (à vérifier).\n\n"
        "**Coût** : environ 0,055 $ par recherche (profondeur « deep »).",
        key="sc_use_ai", value=False, wide=True)
    focus = "production"
    if use_ai:
        focus_label = st.selectbox("Fournisseurs recherchés par l'IA", list(ws.FOCUS.values()), key="sc_focus")
        focus = next(k for k, v in ws.FOCUS.items() if v == focus_label)
    if st.button("Lancer la recherche automatique", type="primary", key="sc_auto_run", disabled=not (use_docs or use_ai)):
        st.session_state.pop("sc_ai", None)
        st.session_state.pop("sc_found", None)
        try:
            if use_docs:
                with st.spinner("Recherche des documents…"):
                    found = ws.rank(ws.search_documents(ws.build_queries(client), key), client)
                with st.spinner(f"Lecture de {len(found)} document(s)…"):
                    res = sp.analyse_urls([f["url"] for f in found])
                by_url = {r["url"]: r for r in res}
                st.session_state["sc_found"] = {"client": client, "docs": [dict(f, **{k: by_url[f["url"]][k] for k in ("tables", "error")}) for f in found]}
            if use_ai:
                with st.spinner("Extraction par l'IA de Linkup (jusqu'à 1 à 2 minutes)…"):
                    items = ws.extract_suppliers(client, key, focus=focus)
                with st.spinner("Vérification des citations…"):
                    st.session_state["sc_ai"] = {"client": client, "df": sp.candidates_from_ai(items)}
        except ws.SearchError as e:
            st.error(str(e))
    ai = st.session_state.get("sc_ai")
    if ai and ai["client"] == client:
        df = ai["df"]
        _help_line(f"<b>Propositions de l'IA : {len(df)} fournisseur(s)</b> "
                   f"({int((df['confiance'] == 'C').sum())} avec citation vérifiée, les autres à vérifier).",
                   CONFIDENCE_HELP, wide=True)
        if df.empty:
            st.info("L'IA n'a trouvé aucun fournisseur déclaré publiquement.")
        else:
            sel_all = st.checkbox("Tout sélectionner", False, key="sc_ai_all")
            ed = df.copy()
            ed.insert(0, "garder", True if sel_all else (df["confiance"] == "C"))
            edited = st.data_editor(ed, hide_index=True, key=f"sc_ai_editor_{int(sel_all)}",
                                    disabled=[c for c in ed.columns if c != "garder"],
                                    column_config={"source_url": st.column_config.LinkColumn("source_url")})
            if st.button("Ajouter la sélection à la liste", key="sc_ai_add"):
                _sc_add(df[edited["garder"].to_numpy()])
                st.success("Ajouté à la liste ci-dessous.")
    fd = st.session_state.get("sc_found")
    if not fd or fd["client"] != client:
        return
    docs = fd["docs"]
    if not docs:
        st.warning("Aucun document trouvé.")
        return
    with_tables = [d for d in docs if d["tables"]]
    st.markdown(f"**{len(docs)} document(s) trouvé(s), dont {len(with_tables)} avec une liste exploitable.**")
    for i, d in enumerate(docs):
        title = f"{'✅' if d['tables'] else '⚪'} {d['title']}"
        with st.expander(title, expanded=bool(d["tables"]) and i == 0):
            st.markdown(f"[{d['url']}]({d['url']})")
            if d["snippet"]:
                st.caption(d["snippet"])
            if d["error"]:
                st.warning(f"Lecture impossible : {d['error']}")
            elif not d["tables"]:
                st.caption("Aucun tableau de fournisseurs détecté dans ce document.")
            else:
                _sc_pick_table({"label": d["title"], "url": d["url"], "tables": d["tables"]}, f"sc_auto{i}", official_default=False)


def urlparse_name(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).path.rsplit("/", 1)[-1]


def _tab_supply_chain():
    st.caption("Seuls les fournisseurs déclarés publiquement peuvent être recensés ; une chaîne d'approvisionnement complète n'est "
               "jamais publique. Les usines hors d'Europe sont souvent absentes d'OpenStreetMap : elles figurent alors dans la liste « non trouvés ».")
    cc1, cc2 = st.columns([5, 2])
    client = cc1.text_input("Nom du client (entreprise dont on cherche les fournisseurs)", placeholder="Ex. IKEA", key="sc_client")
    with cc2:
        _help_line("Wikidata du client (facultatif)",
                   "**Identifiant Wikidata** : code unique d'une entreprise sur wikidata.org (exemple : Q54078 pour IKEA).\n\n"
                   "**Renseigné** : il prime sur le nom et cherche, dans OpenStreetMap, les lieux rattachés à cette marque. "
                   "Recherche très précise, qui évite les homonymes.\n\n"
                   "**Vide** : recherche du client par son nom, comme pour les fournisseurs.", left=True, wide=True,
                   label_style="font-size:0.875rem;white-space:nowrap")
        client_qid = st.text_input("Wikidata du client", placeholder="Ex. Q54078", key="sc_client_qid", label_visibility="collapsed")
    src = st.radio("Source des noms de fournisseurs",
                   ["Recherche automatique sur internet", "Adresse web (page ou PDF)", "Fichier",
                    "Mode assisté (Claude / ChatGPT)", "Saisie manuelle"],
                   horizontal=True, key="sc_src")
    if src == "Recherche automatique sur internet":
        _sc_auto_search(client.strip())
    elif src in ("Adresse web (page ou PDF)", "Fichier"):
        _sc_table_source(src)
    elif src == "Mode assisté (Claude / ChatGPT)":
        st.caption("Gratuit : l'application prépare la consigne, vous la collez dans Claude ou ChatGPT (avec recherche web), puis "
                   "vous collez ici leur réponse. Les lignes sont marquées « à vérifier » (confiance C ou D).")
        if not client.strip():
            st.info("Renseignez d'abord le nom du client.")
        else:
            st.markdown("**1. Copiez cette consigne** (bouton en haut à droite du cadre) :")
            st.code(sp.build_prompt(client.strip()), language=None)
            st.markdown("**2. Collez ici la réponse de l'assistant :**")
            reply = st.text_area("Réponse de l'assistant", height=180, key="sc_reply", label_visibility="collapsed")
            if st.button("Lire la réponse", key="sc_read_reply", disabled=not reply.strip()):
                got = sp.parse_assisted_reply(reply)
                if got.empty:
                    st.warning("Aucun fournisseur lu dans cette réponse (ou « aucun fournisseur trouvé »).")
                else:
                    _sc_add(got)
                    st.success(f"{len(got)} ligne(s) lue(s).")
    else:
        txt = st.text_area("Un fournisseur par ligne (pays facultatif après « ; »)", placeholder="Acme Furniture;Poland\nBeta Textiles",
                           height=140, key="sc_manual")
        if st.button("Ajouter à la liste", key="sc_add_manual", disabled=not txt.strip()):
            _sc_add(sp.candidates_from_text(txt))

    cands = st.session_state.get("sc_cands")
    if cands is None or cands.empty:
        return
    st.divider()
    _help_line(f"<b>Fournisseurs recensés : {len(cands)}</b> (doublons fusionnés). Décochez ceux à exclure avant la recherche.",
               CONFIDENCE_HELP, icon_first=True, wide=True)
    all_on = st.checkbox("Tout sélectionner", True, key="sc_all")
    ed = cands.copy()
    ed.insert(0, "garder", bool(all_on))
    edited = st.data_editor(ed, hide_index=True, key=f"sc_editor_{int(all_on)}_{len(cands)}",
                            disabled=[c for c in ed.columns if c != "garder"],
                            column_config={"source_url": st.column_config.LinkColumn("source_url")})
    kept = cands[edited["garder"].to_numpy()] if len(edited) == len(cands) else cands
    with st.expander("Options de recherche OpenStreetMap"):
        o_brand = st.checkbox("Chercher aussi dans « brand »", False, key="sc_o_brand")
        o_biz = st.checkbox("Établissements uniquement (plus précis, moins de résultats)", False, key="sc_o_biz")
    too_many = len(kept) > SC_MAX_SUPPLIERS
    if too_many:
        st.warning(f"{len(kept)} fournisseurs : limitez à {SC_MAX_SUPPLIERS} par recherche (décochez des lignes).")
    if not client.strip():
        st.info("Renseignez le nom du client pour lancer la recherche.")
    b1, b3, b2 = st.columns([1, 1, 1])
    dis = (not client.strip()) or kept.empty or too_many
    run_sup = b1.button("Géolocaliser les fournisseurs", type="primary", key="sc_run", disabled=dis)
    run_all = b3.button("Géolocalisation client & fournisseurs", type="primary", key="sc_run_all", disabled=dis)
    with b2:
        if st.button("Vider la liste", key="sc_clear"):
            for k in ("sc_cands", "sc_result", "sc_tables"):
                st.session_state.pop(k, None)
            st.rerun()
    if run_sup or run_all:
        queries = [NameQuery.make(n) for n in kept["fournisseur"]]
        if run_all:
            queries = [NameQuery.make(client.strip(), client_qid)] + queries
        options = SearchOptions(search_brand=o_brand, require_business_tag=o_biz, include_relations=False)
        box = st.empty()
        bar = box.progress(0.0, text="Démarrage…")

        def cb(done: int, total: int, name: str):
            bar.progress(done / total, text=f"{done}/{total} — {name}")

        res = geolocate(queries, options, get_client(), get_settings(), get_countries(), get_naf(), progress=cb)
        box.empty()
        st.session_state["sc_result"] = {"client": client.strip(), "res": res, "cands": kept.copy(), "with_client": bool(run_all)}
    out = st.session_state.get("sc_result")
    if out:
        _sc_render(out)


def _sc_render(out: dict):
    res, cands, client = out["res"], out["cands"], out["client"]
    with_client = bool(out.get("with_client"))
    final = sp.build_output(client, res.sites, cands, include_client=with_client)
    miss = sp.not_found(res.report, cands, client)
    st.divider()
    sup = final[final["role"] == "fournisseur"] if len(final) else final
    found = int(sup["nom_demande"].nunique()) if len(sup) else 0
    m = st.columns(4 if with_client else 3)
    m[0].metric("Fournisseurs recherchés", len(cands))
    m[1].metric("Trouvés sur OpenStreetMap", f"{found} / {len(cands)}")
    m[2].metric("Sites fournisseurs", len(sup))
    if with_client:
        m[3].metric("Sites du client", int((final["role"] == "client").sum()) if len(final) else 0)
    _status_messages(res.report)
    if final.empty:
        st.info("Aucun site trouvé pour ces fournisseurs.")
    else:
        view = final
        if with_client:
            pick = st.multiselect("Afficher", ["Client", "Fournisseurs"], default=["Client", "Fournisseurs"], key="sc_show",
                                  help="Filtre le tableau et la carte.")
            roles = [{"Client": "client", "Fournisseurs": "fournisseur"}[x] for x in pick]
            view = final[final["role"].isin(roles)]
        show = [c for c in ["client", "role", "nom_demande", "name", "pays", "addr_city", "naf_section", "confiance",
                            "type_relation", "rang", "lat", "long", "osm_url", "source_url"] if c in view.columns]
        _stretch(st.dataframe, view[show], hide_index=True,
                 column_config={"osm_url": st.column_config.LinkColumn("OSM", display_text="ouvrir"),
                                "source_url": st.column_config.LinkColumn("source_url")})
        _site_map(view, kp="sc_", default_color="Rôle (client / fournisseurs)" if with_client else "Entreprise")
    d1, d2 = st.columns(2)
    d1.download_button("Télécharger « suply » (CSV)", _csv_bytes(final), "suply.csv", "text/csv", key="sc_dl_csv")
    d2.download_button("Télécharger « suply » (Excel)", _xlsx_bytes({"suply": final, "non_trouves": miss, "fournisseurs": cands}),
                       "suply.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="sc_dl_xlsx")
    if len(miss):
        with st.expander(f"{len(miss)} entreprise(s) sans site trouvé sur OpenStreetMap"):
            _stretch(st.dataframe, miss, hide_index=True)


# --------------------------------------------------------------------------
# Pages « Temps réel » et « Cartographies avancées »
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Interrogation de Météo-France…")
def _fetch_vigilance_df() -> pd.DataFrame:
    s = get_settings()
    return vg.parse_vigilance(vg.fetch_vigilance(vg.MeteoFranceAuth(s), s))


def _load_vigilance():
    s = get_settings()
    if not vg.MeteoFranceAuth(s).configured:
        st.info("Identifiants Météo-France absents. Définissez `METEOFRANCE_API_KEY` (ou `METEOFRANCE_CLIENT_ID` et "
                "`METEOFRANCE_CLIENT_SECRET`) dans l'environnement ou `.streamlit/secrets.toml`.\n\n"
                f"Noms de secrets détectés par l'application (valeurs jamais affichées) : {', '.join(sorted(_secrets())) or 'aucun'}.")
        return None
    if st.button("Relancer la recherche", key="vg_refresh"):
        _fetch_vigilance_df.clear()
        _fetch_om_json.clear()
        _fetch_om_zip.clear()
    try:
        return _fetch_vigilance_df()
    except vg.VigilanceError as e:
        st.error(str(e))
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_cyclones(with_rain: bool = True, _v: int = 2):
    """Cyclones actifs + trajectoire + (facultatif) précipitations. Mis en cache 10 minutes."""
    storms = cy.fetch_storms()
    radii = cy.fetch_wind_radii()
    rain = cy.attach_precipitation(storms) if with_rain else False
    rows = []
    for s in storms:
        c = s.current
        rows.append({"Nom": s.name, "Bassin": s.basin, "Nature": cy.stage_label(c.wind_kt, s.basin),
                     "Maximum Sustained Winds": cy.wind_text(c.wind_kt), "Max Wind Gusts": cy.wind_text(c.gust_kt),
                     "Pressure": cy.pressure_text(c.mslp_hpa), "Movement": cy.movement_text(c.direction_deg, c.speed_kt),
                     "Bulletin (UTC)": s.advisory_time.strftime("%d/%m %H:%M") if s.advisory_time else "",
                     "_wind": c.wind_kt or 0, "_cat": c.category})
    return cm.build_payload(storms, radii, rain_available=rain), rows, storms


def _help_html(text: str) -> str:
    """Texte d'infobulle -> HTML sûr : échappé, puis **gras**, *italique* et sauts de ligne."""
    import re
    s = html.escape(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s.replace("\n\n", "<br><br>").replace("\n", "<br>")


def _help_icon(text: str, left: bool = False, wide: bool = False) -> None:
    """Pastille « ? » dont le texte n'apparaît qu'au survol (CSS pur, voir .cy-help dans assets/styles.css)."""
    cls = "cy-help" + (" cy-left" if left else "") + (" cy-wide" if wide else "")
    tip = _help_html(text).replace("$", "&#36;")          # « $ » déclencherait le rendu des formules mathématiques
    st.markdown(f'<span class="{cls}"><span class="cy-tip">{tip}</span></span>', unsafe_allow_html=True)


def _help_line(label_html: str, text: str, *, icon_first: bool = False, left: bool = False, wide: bool = False,
               label_style: str = "") -> None:
    """Ligne de texte + « ? » sur une même ligne (HTML en ligne : fonctionne dans une colonne étroite)."""
    cls = "cy-help" + (" cy-left" if left else "") + (" cy-wide" if wide else "")
    tip = _help_html(text).replace("$", "&#36;")
    icon = f'<span class="{cls}"><span class="cy-tip">{tip}</span></span>'
    label = f'<span style="{label_style}">{label_html}</span>'

    inner = (icon + label) if icon_first else (label + icon)
    st.markdown(f'<div style="display:flex;align-items:center;gap:.45rem;flex-wrap:nowrap">{inner}</div>', unsafe_allow_html=True)


def _num_help(container, label: str, help_text: str, **kw):
    """``number_input`` dont l'infobulle s'affiche au survol (pastille « ? » en CSS pur, comme ailleurs dans l'app) :
    l'infobulle native de Streamlit ne réagit pas sur ces champs."""
    with container if container is not st else _NullCtx():
        _help_line(html.escape(label), help_text, label_style="font-size:.875rem")
        return container.number_input(label, label_visibility="collapsed", **kw)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _checkbox_with_help(label: str, help_text: str, *, key: str, value: bool = False, left: bool = False, wide: bool = False) -> bool:
    """Case à cocher + « ? » collé au texte (infobulle au survol uniquement)."""
    try:
        row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    except TypeError:
        row = st.container()
    with row:
        v = st.checkbox(label, value, key=key)
        _help_icon(help_text, left=left, wide=wide)
    return v


def _cyclone_section():
    import streamlit.components.v1 as components
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Actualiser les cyclones", key="cy_refresh"):
            _load_cyclones.clear()
    with c2:
        try:                                   # Streamlit récent : ligne horizontale, l'icône colle au texte
            row = st.container(horizontal=True, vertical_alignment="center", gap="small")
        except TypeError:                      # anciennes versions : l'icône passe sous la case
            row = st.container()
        with row:
            with_rain = st.checkbox("Calculer les précipitations attendues (Open-Meteo)", value=True, key="cy_rain")
            _help_icon("Cumul de pluie prévu sur 24 h au centre de la tempête, à chaque échéance de la trajectoire. "
                       "Décochez pour accélérer l'affichage.")
    try:
        with st.spinner("Chargement des cyclones actifs…"):
            payload, rows, storms = _load_cyclones(with_rain)
    except cy.CycloneError as e:
        st.error(f"Données cyclones indisponibles : {e}")
        return
    n = len(rows)
    strong = sum(1 for r in rows if r["_cat"] >= 1)
    m = st.columns(3)
    m[0].metric("Systèmes tropicaux actifs", n)
    m[1].metric("Ouragans / typhons / cyclones (cat. 1 à 5)", strong)
    m[2].metric("Vent soutenu max", f"{max((r['_wind'] for r in rows), default=0):.0f} kt" if rows else "—")
    map_box = st.container()
    st.caption("Cliquez sur une icône pour afficher ses caractéristiques. Les couches (trajectoire, vents, précipitations, fonds de carte) "
               "se sélectionnent directement sur la carte, en haut à droite.")
    if rows:
        _stretch(st.dataframe, pd.DataFrame(rows).drop(columns=["_wind", "_cat"]), hide_index=True)
    with st.expander("Sources et limites"):
        st.markdown(
            "- **Positions, vents, rafales, pression, déplacement, trajectoire** : NHC (Atlantique, Pacifique Est/Centre) et JTWC "
            "(Pacifique Ouest, océan Indien, hémisphère Sud) via le service public Esri « Active Hurricanes ». Prévision officielle (trajectoire la plus probable).\n"
            "- **Catégorie** : calculée à partir du vent soutenu maximal (échelle de Saffir-Simpson, nœuds) pour tous les bassins ; "
            "les dépressions et tempêtes tropicales utilisent l'icône sans numéro.\n"
            "- **Précipitations attendues** : modèle global Open-Meteo, valeur ponctuelle au centre (sous-estime les maxima des bandes pluvieuses). "
            "**Précipitations observées** : NASA GIBS / IMERG (raster, mise à jour environ toutes les 30 min).\n"
            "- Les données sont informatives et ne remplacent pas les bulletins officiels des centres de prévision.")
    pts = _cyclone_risk_section(storms)
    if pts is not None:
        df_p, lat_c, lon_c, res = pts
        src = res if res is not None else df_p
        points, truncated = cm.build_points(src, lat_c, lon_c, storms, analysed=res is not None)
        payload["points"] = points
        payload["points_analysed"] = res is not None
        if truncated:
            st.caption("Carte : seuls les 5 000 premiers points sont tracés (les points impactés d'abord). Le CSV téléchargé les contient tous.")
    with map_box:
        components.html(cm.render_map_html(payload), height=720, scrolling=False)
    _cyclone_tau_diagnostic(storms)
    _cyclone_area_section(storms)


def _cyclone_tau_diagnostic(storms):
    with st.expander("Diagnostic des pas de temps (échéances de prévision)"):
        ver = getattr(cy, "VERSION", 1)
        if ver < 2 or not hasattr(cm, "step_label"):
            st.error("Les modules cyclones en mémoire sont anciens (le serveur n'a pas rechargé les fichiers mis à jour). "
                     "Redémarrez l'application : « Manage app » > menu ⋮ > « Reboot app ».")
        st.caption(f"Version du module cyclones : {ver}")
        rows = [{"Tempête": s.name, "Bassin": s.basin, "Bulletin": s.advisory, "Points de prévision": len(s.forecast),
                 "TAU lus": ", ".join("—" if t is None else str(t) for t in getattr(s, "tau_raw", [])),
                 "Échéances utilisées": ", ".join(str(p.tau) for p in s.forecast) if getattr(s, "tau_ok", True) else "indisponibles",
                 "Source": getattr(s, "tau_source", "?")} for s in storms]
        if rows:
            _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
        st.caption("Si une tempête indique « indisponibles », le service source ne fournit pas d'échéance distincte : "
                   "les points sont alors étiquetés « prév. n » (rang) au lieu de « +0 h ».")


def _cyclone_area_section(storms):
    st.divider()
    st.subheader("Entreprises sur un périmètre autour d'un cyclone")
    st.caption("Délimitez un cercle sur la carte (par exemple sur la zone d'impact prévue) pour extraire toutes les entreprises "
               "situées dedans, comme dans l'onglet Géolocalisation.")
    center, zoom = (20.0, -60.0), 3
    names = [s.name for s in storms if s.current is not None]
    if names:
        pick = st.selectbox("Centrer la carte sur", ["—"] + names, key="cy_area_center")
        st_ = next((s for s in storms if s.name == pick and s.current is not None), None)
        if st_:
            center, zoom = (st_.current.lat, st_.current.lon), 6
    _page_area(kp="cy_area_", result_key="cy_area_result", center=center, zoom=zoom)


def _cross_table(res, new_cols, mask, key, mask_label="Seulement les points touchés", default_only=False, max_base=8):
    """Tableau d'un croisement : colonnes d'origine (au choix) + TOUTES les colonnes ajoutées par le croisement, avec option de
    ne garder que les points touchés. Le CSV téléchargé par l'appelant reste complet."""
    new_cols = [c for c in new_cols if c in res.columns]
    base_cols = [c for c in res.columns if c not in new_cols]
    c1, c2 = st.columns([3, 2])
    with c1:
        shown_base = st.multiselect("Colonnes d'origine affichées", base_cols, default=base_cols[:max_base], key=f"{key}_basecols")
    with c2:
        only = st.checkbox(mask_label, default_only, key=f"{key}_only")
    view = res[mask.fillna(False).to_numpy(bool)] if (only and mask is not None) else res
    st.caption(f"{len(view)} ligne(s) · {len(new_cols)} colonne(s) ajoutée(s) par le croisement : " + (", ".join(new_cols) or "aucune"))
    if view.empty:
        st.caption("Aucune ligne ne correspond.")
        return view
    _stretch(st.dataframe, view[shown_base + new_cols], hide_index=True)
    return view


def _cyclone_risk_section(storms):
    st.subheader("Croiser vos points avec les cyclones")
    up = st.file_uploader("CSV de points (colonnes « lat » et « long », par exemple l'export de l'onglet Géolocalisation)",
                          type=["csv", "txt", "xlsx"], key="cy_csv")
    if up is None:
        st.caption("Chargez un fichier pour lancer l'analyse. Séparateur « ; » ou « , » accepté.")
        return None
    try:
        df, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name)
    except InputError as e:
        st.error(str(e))
        return None
    st.caption(f"{len(df)} point(s) lus (colonnes {lat_c} / {lon_c}).")
    radius = st.slider("Distance maximale à la trajectoire (km)", 10, 500, 150, step=10, key="cy_radius")
    st.caption("Un point est « sur la trajectoire » s'il est à moins de cette distance de la ligne de trajectoire prévue "
               "(position actuelle comprise). Ce n'est pas le rayon réel des vents, qui varie selon le cyclone.")
    if not storms:
        st.info("Aucun cyclone actif : tous les points auraient un impact de 0.")
    if st.button("Analyser le risque", key="cy_analyse"):
        with st.spinner("Croisement en cours…"):
            st.session_state["cy_risk"] = cr.assess_points(df, lat_c, lon_c, storms, radius_km=float(radius))
            st.session_state["cy_risk_sig"] = (up.name, len(df))
    res = st.session_state.get("cy_risk")
    if res is None or st.session_state.get("cy_risk_sig") != (up.name, len(df)):
        return df, lat_c, lon_c, None
    hit = res[res["impact"] > 0]
    m = st.columns(3)
    m[0].metric("Points analysés", len(res))
    m[1].metric("Points sur la trajectoire", len(hit))
    m[2].metric("Impact le plus proche", f"{int(hit['impact'].min())} h" if len(hit) else "—")
    _cross_table(res, cr.NEW_COLUMNS, res["impact"] > 0, "cy_risk_tbl", "Seulement les points sur la trajectoire")
    st.caption("impact : 0 = hors trajectoire, sinon échéance (12, 24… h) à laquelle le cyclone atteint le point. "
               "max_sustained_winds et max_wind_gusts en km/h, total_precipitation en mm (cumul prévu au point jusqu'à 12 h après l'impact). "
               "None = hors trajectoire ou donnée indisponible.")
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(res), "catmap_risque_cyclones.csv", "text/csv")
    return df, lat_c, lon_c, res


def page_temps_reel():
    st.title("Temps réel")
    t_cy, t_vg, t_eu, t_ev = st.tabs(["Cyclones tropicaux", "Vigilance Météo-France", "Vigilance Europe",
                                      "Suivi évènements climatiques"])
    with t_ev:
        _safe_tab(_events_section, "Suivi évènements climatiques")
    with t_cy:
        _safe_tab(_cyclone_section, "Cyclones tropicaux")
    with t_vg:
        _safe_tab(_vigilance_section, "Vigilance Météo-France")
    with t_eu:
        _safe_tab(_europe_section, "Vigilance Europe")


# --------------------------------------------------------------------------- Suivi évènements climatiques
@st.cache_data(ttl=900, show_spinner=False)
def _ev_activations():
    return evt.fetch_activations()


@st.cache_data(ttl=3600, show_spinner=False)
def _ev_discover(kind: str):
    urls = {"effis": evt.EFFIS_WMS, "efas": evt.EFAS_WMS, "glofas": evt.GLOFAS_WMS}[kind]
    return evt.discover_wms(urls)


@st.cache_data(ttl=900, show_spinner=False)
def _ev_burnt(bbox: tuple):
    return evt.burnt_areas(bbox, max_features=5000)


@st.cache_data(ttl=3600, show_spinner=False)
def _ev_details(url: str):
    return evt.fetch_layer_details(url)


@st.cache_data(ttl=900, show_spinner=False)
def _ev_ems_detail(code: str):
    return evt.fetch_activation_detail(code)


@st.cache_data(ttl=900, show_spinner=False)
def _ev_ems_geojson(url: str):
    return evt.fetch_geojson(url)


@st.cache_data(ttl=3600, show_spinner=False)
def _ev_ems_rules(url: str):
    return evt.parse_sld(evt.fetch_text(url)) if url else []


@st.cache_data(ttl=3600, show_spinner=False)
def _ev_legend(url: str, layer: str, legends: tuple, styles: tuple):
    return evt.fetch_legend(url, layer, legends, styles)


@st.cache_data(ttl=3600, show_spinner=False)
def _ev_style_legend(url: str, layer: str):
    return evt.fetch_style_legend(url, layer)


def _ev_preview_html(url, chosen, times, details, center, zoom):
    """Page HTML autonome : fond OpenStreetMap, couches WMS choisies et légendes du service (images chargées par le navigateur)."""
    import folium
    # OpenStreetMap refuse les tuiles aux pages ouvertes depuis un fichier local (pas de « Referer » : erreur 403) : fonds Esri à la place.
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(tiles=esri.format("World_Street_Map"), attr="Esri, HERE, Garmin, OpenStreetMap contributors", name="Plan").add_to(m)
    folium.TileLayer(tiles=esri.format("Canvas/World_Light_Gray_Base"), attr="Esri", name="Fond clair", max_native_zoom=16).add_to(m)
    folium.TileLayer(tiles=esri.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    for n in chosen:
        t = times.get(n)
        kw = {"time": t} if t else {}
        folium.raster_layers.WmsTileLayer(url=url, layers=n, fmt="image/png", transparent=True, version="1.3.0", name=n,
                                          overlay=True, control=True, **kw).add_to(m)
    parts = []
    for n in chosen:
        data, ctype, _d = _ev_legend(url, n, tuple(details.get(n, {}).get("legends") or ()), tuple(details.get(n, {}).get("styles") or ()))
        if data:                                              # image intégrée au fichier : ne dépend plus du service
            import base64
            body = f"<img src='data:{ctype or 'image/png'};base64,{base64.b64encode(data).decode()}' alt='légende' style='max-width:260px'>"
        else:
            items, _s = _ev_style_legend(url, n)
            body = "<br>".join(f"<span style='display:inline-block;width:12px;height:12px;background:{it['couleur']};"
                               f"border:2px solid {it['contour']};margin-right:4px'></span>{html.escape(it['titre'])}" for it in items) \
                or "<i style='font-size:11px'>légende non fournie par le service</i>"
        parts.append(f"<div style='margin-bottom:6px'><b style='font-size:11px'>{html.escape(n)}</b><br>{body}</div>")
    imgs = "".join(parts)
    box = ("<div style='position:fixed;bottom:20px;left:10px;z-index:9999;background:rgba(255,255,255,.92);padding:8px 10px;"
           f"border:1px solid #888;border-radius:6px;max-height:50vh;overflow:auto;font-family:sans-serif'>{imgs}</div>")
    m.get_root().html.add_child(folium.Element(box))
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render().encode("utf-8")


def _ev_test_links(url, chosen, tv, times, details, viewer, center, zoom, key):
    """Liens de test d'une couche : image brute, visionneuse officielle du service, prévisualisation autonome (fond + légende)."""
    if not chosen:
        return
    first = chosen[0]
    st.markdown(f"[Tester la première couche dans le navigateur (image brute du service)]({evt.getmap_url(url, first, time_value=times.get(first) or tv)})")
    st.caption("Si ce lien affiche une erreur XML (jeton exigé, date hors plage, couche inconnue), elle explique pourquoi la carte reste vide.")
    st.markdown(f"[Ouvrir la visionneuse officielle du service]({viewer})")
    st.caption("Je ne connais pas d'adresse qui ouvre directement la visionneuse sur une couche donnée : choisissez-la dans son menu.")
    st.download_button("Télécharger une prévisualisation (page HTML : fond de carte, couches et légendes)",
                       _ev_preview_html(url, chosen, times, details, center, zoom), f"previsualisation_{key}.html", "text/html", key=f"{key}_preview")
    st.caption("Ouvrez le fichier téléchargé dans votre navigateur : les couches et leurs légendes sont chargées directement depuis le service.")


def _ev_show_legend(url, layer, details):
    """Légende d'une couche : image lue côté serveur (variantes d'adresses essayées) ; sinon lien et diagnostic."""
    d = details.get(layer, {})
    data, ctype, diag = _ev_legend(url, layer, tuple(d.get("legends") or ()), tuple(d.get("styles") or ()))
    if data:
        st.image(data)
        return
    items, d_style = _ev_style_legend(url, layer)
    if items:
        st.caption("Le service ne publie pas d'image de légende (404) : légende reconstituée à partir du style de la couche.")
        st.markdown(" &nbsp; ".join(
            f"<span style='display:inline-block;width:14px;height:14px;background:{it['couleur']};border:2px solid {it['contour']};"
            f"vertical-align:middle;margin-right:4px'></span>{html.escape(it['titre'])}" for it in items), unsafe_allow_html=True)
    else:
        st.caption("Légende non fournie par le service pour cette couche (ni image, ni style exploitable).")
    st.markdown(f"[Essayer d'ouvrir la légende dans le navigateur]({evt.legend_url(url, layer)})")
    with st.expander(f"Diagnostic de la légende · {layer}"):
        _stretch(st.dataframe, pd.DataFrame(diag + d_style), hide_index=True)


@st.cache_data(ttl=600, show_spinner=False)
def _ev_info(url: str, layer: str, lat: float, lon: float, zoom: float, time_value):
    return evt.get_feature_info(url, layer, lat, lon, zoom, time_value=time_value, exhaustive=True)


def _ev_diag(diag, title="Diagnostic des appels"):
    with st.expander(title):
        if diag:
            _stretch(st.dataframe, pd.DataFrame(diag), hide_index=True)
        else:
            st.caption("Aucun appel effectué.")


def _ev_show_map(m, key, track_click=False):
    """Affiche la carte. Avec track_click, retourne le dernier clic ({'lat','lng'}) et le zoom courant."""
    import folium
    from streamlit_folium import st_folium
    folium.LayerControl(collapsed=True).add_to(m)
    objs = ["last_clicked", "zoom"] if track_click else []
    try:
        out = st_folium(m, height=580, returned_objects=objs, key=key, use_container_width=True)
    except TypeError:
        out = st_folium(m, height=580, width=1400, returned_objects=objs, key=key)
    return out if isinstance(out, dict) else {}


def _ev_base_map(center, zoom):
    import folium
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    _add_basemaps(m)
    return m


def _ev_add_wms(m, base_url, layers, time_value=None, timed=None):
    """Superpose des couches WMS ; `time_value` n'est envoyé qu'aux couches à dimension temporelle (`timed`, None = toutes)."""
    import folium
    for name in layers:
        kw = {"time": time_value} if (time_value and (timed is None or name in timed)) else {}
        folium.raster_layers.WmsTileLayer(
            url=base_url, layers=name, fmt="image/png", transparent=True, version="1.3.0", name=f"WMS · {name}",
            overlay=True, control=True, show=True, **kw).add_to(m)


def _safe_tab(fn, label, *args):
    """Exécute un onglet ; une erreur s'affiche dans cet onglet sans empêcher les autres de fonctionner."""
    try:
        fn(*args)
    except Exception as e:                                       # noqa: BLE001 - l'affichage de l'erreur est le but
        st.error(f"{label} : {type(e).__name__} — {str(e)[:300]}")
        if isinstance(e, AttributeError):
            st.caption("Cette erreur indique en général que le serveur utilise d'anciens fichiers en mémoire : redémarrez l'application "
                       "(« Manage app » > menu ⋮ > « Reboot app ») après avoir remplacé tous les fichiers mis à jour.")


def _events_section():
    st.subheader("Suivi évènements climatiques")
    st.caption("Inondations et incendies en temps quasi réel à partir des services publics Copernicus (EMS, EFFIS, EFAS, GloFAS). "
               "Aucune clé n'est nécessaire. Les adresses des services n'ont pas pu être testées depuis le serveur de développement : "
               "chaque onglet fournit un diagnostic des appels.")
    t_act, t_fire, t_flood = st.tabs(["Activations Copernicus EMS", "Incendies (EFFIS)", "Crues (EFAS / GloFAS)"])
    if getattr(evt, "VERSION", 1) < 3:
        st.error("Le module `catmap/events.py` en mémoire est ancien. Remplacez le fichier dans GitHub puis redémarrez l'application "
                 "(« Manage app » > menu ⋮ > « Reboot app »).")
    with t_act:
        _safe_tab(_ev_activations_tab, "Activations Copernicus EMS")
    with t_fire:
        _safe_tab(_ev_fire_tab, "Incendies")
    with t_flood:
        _safe_tab(_ev_flood_tab, "Crues")


def _ev_activations_tab():
    st.markdown("**Évènements pour lesquels le service de cartographie rapide Copernicus EMS a été activé** "
                "(inondations, incendies, etc.), avec lien vers les cartes officielles.")
    if st.button("Actualiser", key="ev_act_refresh"):
        _ev_activations.clear()
    with st.spinner("Lecture des activations Copernicus EMS…"):
        df, diag = _ev_activations()
    c1, c2, c3 = st.columns([3, 2, 2])
    labels = {"flood": "Inondations", "fire": "Incendies", "other": "Autres"}
    kinds = c1.multiselect("Type d'évènement", list(labels), default=["flood", "fire"],
                           format_func=lambda k: labels[k], key="ev_act_kinds")
    periods = {"30 derniers jours": 30, "90 derniers jours": 90, "12 derniers mois": 365, "Tout": 0}
    per = c2.selectbox("Période (date d'activation)", list(periods), key="ev_act_period")
    only_open = c3.checkbox("Activations non clôturées", False, key="ev_act_open")
    if df.empty:
        st.warning("Aucune activation lue. Consultez le diagnostic ci-dessous (adresse refusée ou format inattendu).")
        _ev_diag(diag)
        return
    view = evt.filter_activations(df, kinds, periods[per] or None, only_open)
    m1, m2, m3 = st.columns(3)
    m1.metric("Activations affichées", len(view))
    m2.metric("Inondations", int((view["genre"] == "flood").sum()))
    m3.metric("Incendies", int((view["genre"] == "fire").sum()))
    if view.empty:
        st.caption("Aucune activation pour ces critères : élargissez la période ou les types.")
    else:
        import folium
        geo = view.dropna(subset=["lat", "long"])
        colors = {"flood": "#1f6fb5", "fire": "#d7191c", "other": "#6b6b6b"}
        center = (float(geo["lat"].mean()), float(geo["long"].mean())) if len(geo) else (30.0, 10.0)
        m = _ev_base_map(center, 3 if len(geo) > 1 else 6)
        for r in geo.itertuples():
            when = r.activation.strftime("%d/%m/%Y") if pd.notna(r.activation) else "n/d"
            pop = (f"<b>{html.escape(str(r.nom))}</b><br>{html.escape(evt.KIND_LABEL[r.genre])} · {html.escape(str(r.pays))}"
                   f"<br>Activé le {when} · {'clôturé' if r.cloture else 'en cours'}<br>"
                   f"<a href='{html.escape(r.lien, quote=True)}' target='_blank' rel='noopener noreferrer'>{html.escape(r.code)}</a>")
            folium.CircleMarker([r.lat, r.long], radius=8, color="#222", weight=1, fill=True, fill_color=colors[r.genre],
                                fill_opacity=0.85, tooltip=f"{r.code} — {r.nom}",
                                popup=folium.Popup(pop, max_width=320)).add_to(m)
        _ev_show_map(m, "ev_act_map")
        shown = view.assign(activation=view["activation"].dt.strftime("%Y-%m-%d"), maj=view["maj"].dt.strftime("%Y-%m-%d"))
        shown = shown[["code", "nom", "genre", "type", "pays", "activation", "maj", "cloture", "cartes", "lat", "long", "lien"]]
        _stretch(st.dataframe, shown, hide_index=True,
                 column_config={"lien": st.column_config.LinkColumn("Cartes", display_text="ouvrir")})
        st.download_button("Télécharger (CSV)", _csv_bytes(shown), "activations_copernicus_ems.csv", "text/csv", key="ev_act_dl")
        if len(geo) < len(view):
            st.caption(f"{len(view) - len(geo)} activation(s) sans position : visibles dans le tableau uniquement.")
        _ev_contours_section(view)
    _ev_diag(diag)


def _ev_contours_section(view):
    st.divider()
    st.subheader("Contours d'une activation")
    st.caption("Zones cartographiées par Copernicus EMS (étendue de l'évènement, bâti, dégâts…), avec les couleurs du service. "
               "Cliquez sur un contour pour lire ses attributs.")
    opts = {f"{r.code} — {r.nom}": r.code for r in view.head(200).itertuples()}
    pick = st.selectbox("Activation", list(opts), key="ev_ct_pick")
    code = opts[pick]
    with st.spinner("Lecture du détail de l'activation…"):
        detail, d_det = _ev_ems_detail(code)
    page = evt.EMS_PAGE.format(code=code)
    layers = evt.iter_layers(detail) if detail else []
    if not layers:
        st.warning("Aucune couche vectorielle exploitable pour cette activation (cartes non encore publiées, ou structure de réponse "
                   "inattendue : voir le diagnostic).")
        st.markdown(f"[Ouvrir l'activation sur le site Copernicus EMS]({page})")
        _ev_diag(d_det, "Diagnostic du détail de l'activation")
        return
    labels = {i: f"{l['aoi'] or 'AOI'} · {evt.PRODUCT_LABEL.get(l['type'], l['type'] or 'produit')}"
                 f"{'' if l['numero'] is None else ' n°' + str(l['numero'])} · {l['nom']}" for i, l in enumerate(layers)}
    with st.expander(f"{len(layers)} couche(s) disponible(s)"):
        _stretch(st.dataframe, pd.DataFrame([{"Zone (AOI)": l["aoi"], "Produit": evt.PRODUCT_LABEL.get(l["type"], l["type"]),
                                              "N°": l["numero"], "Suivi": l["suivi"], "Couche": l["nom"],
                                              "Style fourni": "oui" if l["sld"] else "non"} for l in layers]), hide_index=True)
    keys = ("observed", "event", "flood", "burnt", "fire", "aoi")
    default = [i for i, l in enumerate(layers) if any(k in l["nom"].lower() for k in keys)][:3] or list(range(min(3, len(layers))))
    sel = st.multiselect("Couches à afficher", list(labels), default=default, format_func=lambda i: labels[i], key=f"ev_ct_layers_{code}")
    if not sel:
        st.caption("Sélectionnez au moins une couche.")
        return
    import folium
    m = _ev_base_map((30.0, 10.0), 3)
    bounds, legend, problems, total, loaded = None, [], [], 0, {}
    for i in sel:
        l = layers[i]
        with st.spinner(f"Lecture de « {l['nom']} »…"):
            gj, err = _ev_ems_geojson(l["geojson"])
        if gj is None:
            problems.append(f"{l['nom']} : {err}")
            continue
        rules = _ev_ems_rules(l["sld"])
        if not rules:
            rules = [evt.default_rule(l["nom"])]
            styled = False
        else:
            styled = True
        total += len(gj["features"])
        b = evt.geojson_bounds(gj)
        if b:
            bounds = b if bounds is None else (min(bounds[0], b[0]), min(bounds[1], b[1]), max(bounds[2], b[2]), max(bounds[3], b[3]))
        props = [k for k in ((gj["features"][0].get("properties") or {}) if gj["features"] else {})][:10]
        folium.GeoJson(gj, name=l["nom"], style_function=lambda f, r=rules: evt.style_for(f.get("properties") or {}, r),
                       highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.7},
                       tooltip=folium.GeoJsonTooltip(fields=props[:3]) if props else None,
                       popup=folium.GeoJsonPopup(fields=props, labels=True, max_width=360) if props else None).add_to(m)
        legend.append((l["nom"], evt.legend_items(rules, l["nom"]), styled, len(gj["features"])))
        lname = l["nom"] if sum(1 for x in layers if x["nom"] == l["nom"]) == 1 else f"{l['nom']} ({l['aoi']})"
        loaded[lname] = gj
    for pbm in problems:
        st.warning(f"Couche non lue : {pbm}")
    if bounds:
        m.fit_bounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
        _ev_show_map(m, f"ev_ct_map_{code}")
        st.markdown("**Légende**")
        for name, items, styled, n in legend:
            sw = " &nbsp; ".join(
                f"<span style='display:inline-block;width:14px;height:14px;background:{it['couleur']};border:2px solid {it['contour']};"
                f"vertical-align:middle;margin-right:4px'></span>{html.escape(it['titre'])}" for it in items)
            st.markdown(f"**{html.escape(name)}** ({n} objet(s)){'' if styled else ' — style par défaut (fichier de style du service non lu)'}<br>{sw}",
                        unsafe_allow_html=True)
        st.caption(f"{total} objet(s) affichés. Couleurs lues dans les fichiers de style (SLD) du service lorsqu'ils sont fournis.")
    st.markdown(f"[Ouvrir l'activation sur le site Copernicus EMS]({page})")
    _ev_contours_cross(code, loaded)
    _ev_diag(d_det, "Diagnostic du détail de l'activation")


def _ev_contours_cross(code, loaded):
    """Croise un CSV de points avec les contours des couches affichées : les colonnes ajoutées suivent les couches sélectionnées."""
    if not loaded:
        return
    st.divider()
    st.subheader("Croiser vos points avec ces contours")
    up = st.file_uploader("CSV de points (colonnes « lat » et « long », par exemple l'export de l'onglet Géolocalisation)",
                          type=["csv", "txt", "xlsx"], key=f"ev_ct_csv_{code}")
    if up is None:
        st.caption("Chargez un fichier pour savoir quels points sont dans les contours des couches affichées ci-dessus.")
        return
    try:
        df, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name, reserved_check=False)
    except InputError as e:
        st.error(str(e))
        return
    st.caption(f"{len(df)} point(s) lus (colonnes {lat_c} / {lon_c}).")
    snap = st.slider("Distance maximale à un contour (km)", 0, 50, 0, key=f"ev_ct_snap_{code}")
    sig = (up.name, len(df), snap, tuple(loaded))
    if st.button("Croiser", type="primary", key=f"ev_ct_go_{code}"):
        with st.spinner("Croisement en cours…"):
            res = df.copy()
            for name, gj in loaded.items():
                for col, vals in evt.cross_with_layer(df, lat_c, lon_c, gj, name, snap_km=float(snap)).items():
                    res[col] = vals
            st.session_state[f"ev_ct_cross_{code}"] = (sig, res)
    saved = st.session_state.get(f"ev_ct_cross_{code}")
    if not saved or saved[0] != sig:
        return
    res = saved[1]
    stat_cols = [c for c in res.columns if c.endswith(".statut")]
    touched = res[stat_cols].apply(lambda col: col != "Hors zone").any(axis=1) if stat_cols else pd.Series(False, index=res.index)
    cols = st.columns(min(4, len(stat_cols) + 1))
    cols[0].metric("Points touchés (au moins une couche)", int(touched.sum()))
    for c_, sc in zip(cols[1:], stat_cols):
        c_.metric(sc[:-7][:28], int((res[sc] != "Hors zone").sum()))
    new_cols = [c for c in res.columns if any(c.startswith(n + ".") for n in loaded)]
    _cross_table(res, new_cols, touched, f"ev_ct_tbl_{code}")
    st.download_button("Télécharger le croisement (CSV)", _csv_bytes(res), f"croisement_{code}.csv", "text/csv", key=f"ev_ct_dl_{code}")


FWI_COLORS = {0: "#9e9e9e", 1: "#2e9e5b", 2: "#a6d96a", 3: "#fee08b", 4: "#fdae61", 5: "#f46d43", 6: "#a50026"}


def _ev_fire_tab():
    st.markdown("**Incendies de forêt (EFFIS, Copernicus)** : zones brûlées détectées par satellite, danger météo d'incendie "
                "et foyers, sur une zone et une période au choix. **Cliquez sur la carte** pour lire la valeur des couches au point cliqué.")
    c1, c2 = st.columns([2, 3])
    zone = c1.selectbox("Zone", list(evt.BBOX_PRESETS), key="ev_fire_zone")
    bbox = evt.BBOX_PRESETS[zone]
    with c2:
        _help_line("Période", "**Toutes les dates** : toutes les zones brûlées lues.\n\n**En cours** : zones dont la dernière mise à jour "
                   "date de moins de N jours (on ne peut pas savoir si un feu est réellement éteint : c'est une estimation).\n\n"
                   "**Une date** : zones actives ce jour-là.\n\n**Période** : zones dont l'activité chevauche la période.",
                   label_style="font-size:.875rem")
        mode_lbl = st.radio("Période", ["Toutes les dates", "En cours", "Une date", "Période"], horizontal=True,
                            key="ev_fire_mode", label_visibility="collapsed")
    mode = {"Toutes les dates": "tout", "En cours": "en cours", "Une date": "date", "Période": "période"}[mode_lbl]
    today = pd.Timestamp.now(tz="UTC").date()
    d1 = d2 = None
    active_days = 3
    if mode == "en cours":
        active_days = int(st.slider("Considérer un feu « en cours » s'il a été mis à jour depuis moins de (jours)", 1, 30, 3, key="ev_fire_active"))
    elif mode == "date":
        d1 = st.date_input("Date", value=today, key="ev_fire_d1")
    elif mode == "période":
        rng = st.date_input("Période", value=(today - pd.Timedelta(days=30), today), key="ev_fire_rng")
        if isinstance(rng, (list, tuple)):
            d1 = rng[0] if rng else None
            d2 = rng[1] if len(rng) > 1 else rng[0] if rng else None
        else:
            d1 = d2 = rng
        if d1 is None:
            st.caption("Choisissez la date de début et la date de fin.")
            mode = "tout"

    url, layers, d_cap = _ev_discover("effis")
    details = {d["name"]: d for d in _ev_details(url)} if url else {}
    timed = {n for n, d in details.items() if d.get("time")}
    tv = None
    if mode == "date" and d1:
        tv = pd.Timestamp(d1).strftime("%Y-%m-%d")
    elif mode == "période" and d1:
        tv = f"{pd.Timestamp(d1):%Y-%m-%d}/{pd.Timestamp(d2 or d1):%Y-%m-%d}"
    chosen = []
    if layers:
        _help_line("Couches cartographiques EFFIS (WMS)",
                   "Couches d'images fournies par le service EFFIS, à superposer à la carte. La liste est lue dans le catalogue du service.\n\n"
                   "**fwi** : danger météo d'incendie (indice Forêt-Météo).\n\n**ba** : zones brûlées détectées par satellite.\n\n"
                   "**hs / hotspot** : foyers actifs.\n\nLe détail de chaque couche est dans « À quoi servent les couches ? » ci-dessous.",
                   label_style="font-size:.875rem")
        default = evt.pick_layers(layers, evt.FIRE_LAYER_WORDS)[:3]
        chosen = st.multiselect("Couches cartographiques EFFIS (WMS)", [n for n, _ in layers], default=default,
                                format_func=lambda n: dict(layers).get(n, n) + f"  [{n}]", key="ev_fire_layers",
                                label_visibility="collapsed")
        with st.expander("À quoi servent les couches ?"):
            rows = [{"Couche": n, "Titre": t, "Rôle (d'après son nom)": evt.layer_note(n, t),
                     "Description officielle": (details.get(n, {}).get("abstract") or "")[:300],
                     "Temporelle": "oui" if n in timed else "non",
                     "Interrogeable au clic": "oui" if details.get(n, {}).get("queryable") else "?"} for n, t in layers]
            _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
            st.caption("« Zones brûlées (EFFIS) » sur la carte n'est pas une couche d'image mais les polygones lus directement "
                       "(service WFS) : ils sont cliquables et filtrables par date. En basse saison, les couches de zones brûlées "
                       "peuvent être presque vides : élargissez la période.")
    else:
        st.warning("Catalogue de couches EFFIS indisponible (voir le diagnostic). Les périmètres brûlés restent affichés s'ils répondent.")
    show_burnt = st.checkbox("Zones brûlées (polygones cliquables)", True, key="ev_fire_burnt")

    gj_all, d_ba = (None, [])
    if show_burnt:
        with st.spinner("Lecture des zones brûlées…"):
            gj_all, d_ba = _ev_burnt(bbox)
    tbl_all = evt.burnt_table(gj_all)
    date_cols = evt.detect_date_columns(tbl_all) if not tbl_all.empty else []
    start_col, end_col = evt.default_date_columns(date_cols)
    if date_cols:
        with st.expander("Colonnes de date des zones brûlées (avancé)"):
            start_col = st.selectbox("Début d'activité", date_cols, index=date_cols.index(start_col) if start_col in date_cols else 0,
                                     key="ev_fire_c1")
            opts = ["(aucune)"] + date_cols
            e = st.selectbox("Fin d'activité / dernière mise à jour", opts, index=opts.index(end_col) if end_col in opts else 0, key="ev_fire_c2")
            end_col = None if e == "(aucune)" else e
    elif mode != "tout" and not tbl_all.empty:
        st.warning("Aucune colonne de date reconnue dans les zones brûlées : le filtre de période ne peut pas s'appliquer.")
    gj = evt.filter_burnt(gj_all, mode, start_col, end_col, d1, d2, active_days)
    tbl = evt.burnt_table(gj)

    import folium
    m = _ev_base_map(((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2), 4 if zone == "Europe" else 6)
    if url and chosen:
        _ev_add_wms(m, url, chosen, tv, timed)
    times = {n: (tv or details.get(n, {}).get("time_default") or None) for n in timed}       # date par défaut si aucune choisie
    if gj and gj.get("features"):
        cols = [c for c in tbl.columns if c != "geometry"][:12]
        folium.GeoJson(
            gj, name="Zones brûlées (EFFIS)",
            style_function=lambda f: {"color": "#8b1a1a", "weight": 1.2, "fillColor": "#d7191c", "fillOpacity": 0.5},
            highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.75},
            tooltip=folium.GeoJsonTooltip(fields=cols[:4]) if cols else None,
            popup=folium.GeoJsonPopup(fields=cols, labels=True, max_width=380) if cols else None).add_to(m)
    k1, k2, k3 = st.columns(3)
    k1.metric("Zones brûlées lues", len(gj_all["features"]) if gj_all else 0)
    k2.metric("Après filtre de période", len(gj["features"]) if gj else 0)
    k3.metric("Couches WMS affichées", len(chosen))
    out = _ev_show_map(m, "ev_fire_map", track_click=True)
    st.caption("Zones rouges : polygones de zones brûlées EFFIS (clic = tous les attributs). Autres couleurs : voir les légendes ci-dessous.")
    click = out.get("last_clicked") if out else None
    if click and url and chosen:
        _ev_click_values(url, chosen, click, out.get("zoom") or 6, tv, times)
    if url and chosen:
        with st.expander("Légendes des couches affichées"):
            for n in chosen:
                st.markdown(f"**{dict(layers).get(n, n)}** · `{n}`")
                _ev_show_legend(url, n, details)
            st.caption("Légendes fournies par le service EFFIS. Classes de l'indice Forêt-Météo utilisées pour le croisement : "
                       + " · ".join(f"{lbl} (< {hi:g})" if hi != float("inf") else f"{lbl} (≥ 50)" for hi, lbl in evt.FWI_CLASSES) + ".")
    if not tbl.empty:
        _stretch(st.dataframe, tbl.head(1000), hide_index=True)
        st.download_button("Télécharger les zones brûlées affichées (CSV)", _csv_bytes(tbl), "effis_zones_brulees.csv", "text/csv",
                           key="ev_fire_dl")
    st.caption("Le service ne filtre pas par date : le filtrage est fait ici sur les colonnes de date, parmi les 5 000 zones les plus "
               "récentes lues dans la zone choisie.")
    _ev_fire_cross(url, chosen, details, gj_all, start_col, end_col, mode, d1, d2, active_days, tv, times)
    _ev_diag(d_cap + d_ba)


def _ev_click_values(url, chosen, click, zoom, tv, timed):
    st.markdown(f"**Valeurs au point cliqué** ({click['lat']:.4f}, {click['lng']:.4f})")
    rows, raw = [], {}
    for n in chosen:
        txt, _ = _ev_info(url, n, float(click["lat"]), float(click["lng"]), float(zoom), timed.get(n))
        attrs = evt.info_attributes(txt)
        name, val = evt.primary_value(attrs) if attrs else (None, None)
        if val is None and not attrs:
            val = evt.first_number(txt)
        present = evt.info_has_feature(txt)
        rank, label = evt.fwi_class(val) if "fwi" in n.lower() else (0, "")
        others = "; ".join(f"{k} = {v}" for k, v in attrs.items() if k != name and v not in ("", None))[:300]
        rows.append({"Couche": n, "Objet à cet endroit": "oui" if present else "non (hors zone)",
                     "Valeur": val if val is not None else "—", "Attribut": name or "—", "Niveau": label or "—",
                     "Autres attributs": others or "—"})
        raw[n] = txt
    _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
    with st.expander("Réponse brute du service"):
        for n, t in raw.items():
            st.markdown(f"`{n}`")
            st.code(t or "(réponse vide : couche non interrogeable ou aucune donnée à cet endroit)")


def _ev_sample_cross(url, chosen, decide, rule_mode, thr, times, details, lat, lon, step, max_cells, prefix, res):
    """Lit chaque couche WMS choisie aux points et ajoute à `res` ses colonnes (valeur, statut de lecture, attributs) ; les colonnes
    dépendent donc des couches sélectionnées. rule_mode : 0 = « dans une zone », 1 = valeur > seuil, 2 = valeur <= seuil.
    Retourne (touché, interrogé, remarques, exemples de réponses brutes)."""
    import numpy as np
    hit = np.zeros(len(res), dtype=bool)
    covered = np.ones(len(res), dtype=bool)
    notes, examples = [], []
    for n in chosen:
        vals, info = evt.sample_layer(url, n, lat, lon, time_value=times.get(n), step=step, max_cells=max_cells,
                                      bbox=details.get(n, {}).get("bbox"))
        pres, attrs = info["presence"], info["attributs"]
        interroge = info.get("interroge", np.ones(len(res), dtype=bool))
        res[f"{prefix}{n}"] = [round(float(v), 3) if v == v else ("présent" if p_ else None) for v, p_ in zip(vals, pres)]
        res[f"{prefix}{n}.statut"] = list(info.get("statut", ["n/d"] * len(res)))
        keys = []
        for a_ in attrs:
            for k in (a_ or {}):
                if k.lower() not in evt.ID_LIKE and k not in keys:
                    keys.append(k)
        for k in keys[:8]:
            res[f"{prefix}{n}.{k}"] = [(a_ or {}).get(k) for a_ in attrs]
        if n in decide:
            covered &= interroge
            if rule_mode == 0:
                hit |= pres
            elif rule_mode == 1:
                hit |= np.nan_to_num(vals, nan=-np.inf) > thr
            else:
                hit |= np.nan_to_num(vals, nan=np.inf) <= thr
        examples += [{"couche": n, **ex} for ex in info.get("exemples", [])]
        if info["hors_emprise"]:
            notes.append(f"Couche « {n} » : {info['hors_emprise']} point(s) hors de l'emprise de la couche, non interrogés.")
        if info.get("erreurs"):
            notes.append(f"Couche « {n} » : {info['erreurs']} des {info['mailles']} mailles ont reçu une erreur du service "
                         f"(« {str(info.get('erreur_exemple', ''))[:160]} »). Le service ne sait pas répondre à cette requête pour cette couche "
                         "(couche non interrogeable, ou date invalide) : les valeurs ne peuvent pas être lues point par point.")
        if info["mailles"] == 0:
            notes.append(f"Couche « {n} » : aucun point dans l'emprise de la couche.")
        elif info["objets"] == 0 and not info.get("erreurs"):
            notes.append(f"Couche « {n} » : aucun objet trouvé aux {info['mailles']} mailles interrogées (aucun point dans une zone de la "
                         "couche, ou couche non interrogeable : voir le diagnostic de lecture).")
        elif info["objets"] > 0 and info["valeurs_lues"] == 0:
            notes.append(f"Couche « {n} » : objets trouvés mais sans valeur numérique (colonnes d'attributs ajoutées).")
        if info["tronque"]:
            notes.append(f"Couche « {n} » : seules les {info['mailles']} premières mailles ont été interrogées (limite) ; "
                         "augmentez le maximum de mailles ou le regroupement.")
    return hit, covered, notes, examples


def _ev_fire_cross(url, chosen, details, gj_all, start_col, end_col, mode, d1, d2, active_days, tv, times):
    st.divider()
    st.subheader("Croiser vos points avec les incendies")
    up = st.file_uploader("CSV de points (colonnes « lat » et « long », par exemple l'export de l'onglet Géolocalisation)",
                          type=["csv", "txt", "xlsx"], key="ev_fire_csv")
    if up is None:
        st.caption("Chargez un fichier pour savoir quels points sont touchés par une zone brûlée (dans la période choisie ci-dessus) "
                   "et quel est leur niveau de danger météo d'incendie (FWI).")
        return
    try:
        df, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name, reserved_check=False)
    except InputError as e:
        st.error(str(e))
        return
    st.caption(f"{len(df)} point(s) lus (colonnes {lat_c} / {lon_c}).")
    snap = st.slider("Distance maximale à une zone brûlée (km)", 0, 50, 5, key="ev_fire_snap")
    st.caption("Les colonnes ajoutées dépendent des couches sélectionnées plus haut : chaque couche WMS cochée est lue à la position des "
               "points (valeur, statut de lecture, attributs) ; si une couche FWI est cochée, son niveau de danger est ajouté.")
    sig = (up.name, len(df), snap, mode, str(d1), str(d2), active_days, tuple(chosen))
    if st.button("Croiser", type="primary", key="ev_fire_cross_go"):
        with st.spinner("Croisement en cours…"):
            lat = pd.to_numeric(df[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            lon = pd.to_numeric(df[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            ok = lat.notna() & lon.notna()
            bb = (float(lon[ok].min()) - 0.5, float(lat[ok].min()) - 0.5, float(lon[ok].max()) + 0.5, float(lat[ok].max()) + 0.5) if ok.any() else None
            notes = []
            res = df.copy()
            if bb:
                gj_pts, _ = _ev_burnt(tuple(round(v, 3) for v in bb))
                tb = evt.burnt_table(gj_pts)
                cols_d = evt.detect_date_columns(tb) if not tb.empty else []
                sc, ec = (start_col, end_col) if start_col in cols_d else evt.default_date_columns(cols_d)
                gj_f = evt.filter_burnt(gj_pts, mode, sc, ec, d1, d2, active_days)
                if gj_pts is None:
                    notes.append("Zones brûlées indisponibles sur l'emprise de vos points (voir le diagnostic).")
                elif len(gj_pts.get("features", [])) >= 5000:
                    notes.append("Limite de 5 000 zones atteinte sur l'emprise de vos points : résultat possiblement incomplet.")
                res = evt.cross_with_burnt(df, lat_c, lon_c, gj_f, snap_km=float(snap), date_cols=tuple(c for c in (sc, ec) if c))
            if chosen and url:
                latv, lonv = lat.to_numpy(), lon.to_numpy()
                decide = [n for n in chosen if "fwi" not in n.lower()]
                hit, _cov, more, examples = _ev_sample_cross(url, chosen, decide, 0, 0.0, times, details, latv, lonv, 0.01, 1500, "feu_", res)
                notes += more
                res["feu_touche"] = ["Oui" if h else "Non" for h in hit]
                fwi_cols = [f"feu_{n}" for n in chosen if "fwi" in n.lower()]
                if fwi_cols:
                    v = pd.to_numeric(res[fwi_cols[0]], errors="coerce")
                    cls = [evt.fwi_class(x) for x in v]
                    res["fwi_valeur"] = [None if x != x else round(float(x), 1) for x in v]
                    res["fwi_rang"] = [c[0] for c in cls]
                    res["fwi_niveau"] = [c[1] or "n/d" for c in cls]
                st.session_state["ev_fire_examples"] = examples
            st.session_state["ev_fire_cross"] = (sig, res, notes)
    saved = st.session_state.get("ev_fire_cross")
    if not saved or saved[0] != sig:
        return
    _, res, notes = saved
    for n in notes:
        st.warning(n)
    touched = res["incendie_statut"].ne("Non touché") if "incendie_statut" in res else pd.Series(False, index=res.index)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Points analysés", len(res))
    m2.metric("Dans une zone brûlée", int((res.get("incendie_statut") == "Dans une zone brûlée").sum()) if "incendie_statut" in res else 0)
    m3.metric("À proximité", int(touched.sum() - (res.get("incendie_statut") == "Dans une zone brûlée").sum()) if "incendie_statut" in res else 0)
    m4.metric("FWI élevé ou plus", int((res["fwi_rang"] >= 4).sum()) if "fwi_rang" in res else 0)
    import folium
    lat = pd.to_numeric(res[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    lon = pd.to_numeric(res[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    ok = lat.notna() & lon.notna()
    if ok.any():
        m = _ev_base_map((float(lat[ok].mean()), float(lon[ok].mean())), 5)
        for i in res.index[ok][:5000]:
            r = res.loc[i]
            rank = int(r["fwi_rang"]) if "fwi_rang" in res and pd.notna(r.get("fwi_rang")) else 0
            hit = "incendie_statut" in res and r["incendie_statut"] != "Non touché"
            lines = [str(r.get("name") or r.get("nom_demande") or "")]
            if "incendie_statut" in res:
                lines.append(f"Incendie : {r['incendie_statut']}")
            if "fwi_niveau" in res:
                lines.append(f"FWI : {r.get('fwi_valeur')} ({r['fwi_niveau']})")
            folium.CircleMarker([lat[i], lon[i]], radius=9 if hit else 6, color="#000" if hit else "#555", weight=2 if hit else 1,
                                fill=True, fill_color=FWI_COLORS.get(rank, "#9e9e9e"), fill_opacity=0.9,
                                tooltip=" · ".join(x for x in lines if x)).add_to(m)
        _ev_show_map(m, "ev_fire_cross_map")
        st.markdown("Couleur : niveau de danger météo (FWI) — "
                   + " · ".join(f"<span style='color:{FWI_COLORS[i + 1]}'>■</span> {lbl}" for i, (_, lbl) in enumerate(evt.FWI_CLASSES))
                   + " · <span style='color:#9e9e9e'>■</span> n/d. Contour noir épais : point touché par une zone brûlée.", unsafe_allow_html=True)
    show = [c for c in res.columns if c.startswith(("incendie_", "fwi_", "feu_"))]
    mask = ((res["incendie_statut"] != "Non touché") if "incendie_statut" in res else pd.Series(False, index=res.index)) | \
           ((res["fwi_rang"] >= 4) if "fwi_rang" in res else pd.Series(False, index=res.index)) | \
           ((res["feu_touche"] == "Oui") if "feu_touche" in res else pd.Series(False, index=res.index))
    _cross_table(res, show, mask, "ev_fire_tbl", "Seulement les points touchés ou à danger élevé (FWI ≥ Élevé)")
    st.download_button("Télécharger le croisement (CSV)", _csv_bytes(res), "croisement_incendies.csv", "text/csv", key="ev_fire_cross_dl")


def _ev_flood_tab():
    st.markdown("**Prévision de crues (Copernicus EMS, EFAS pour l'Europe, GloFAS pour le monde)** : couches de prévision et "
                "d'alerte. Ce sont des prévisions de débit des cours d'eau, pas la cartographie de l'inondation constatée "
                "(celle-ci est dans l'onglet « Activations »). **Cliquez sur la carte** pour lire la valeur des couches au point cliqué.")
    import streamlit.components.v1 as components
    src = st.radio("Service", ["EFAS (Europe)", "GloFAS (monde)"], horizontal=True, key="ev_flood_src")
    kind = "efas" if src.startswith("EFAS") else "glofas"
    url, layers, d_cap = _ev_discover(kind)
    if not layers:
        st.warning("Catalogue de couches indisponible (voir le diagnostic). L'accès à EFAS peut exiger un compte gratuit sur le site du service.")
        _ev_diag(d_cap)
        return
    details = {d["name"]: d for d in _ev_details(url)}
    timed = {n for n, d in details.items() if d.get("time")}
    default = evt.pick_layers(layers, evt.FLOOD_LAYER_WORDS)[:2]
    _help_line("Couches", "Couches d'images fournies par le service ; la liste est lue dans son catalogue. Le rôle de chaque couche est "
               "décrit dans « À quoi servent les couches ? » ci-dessous, et sa légende dans « Légendes ».", label_style="font-size:.875rem")
    chosen = st.multiselect("Couches", [n for n, _ in layers], default=default,
                            format_func=lambda n: dict(layers).get(n, n) + f"  [{n}]", key=f"ev_flood_layers_{kind}",
                            label_visibility="collapsed")
    _help_line("Date de prévision (facultatif)", "Laissez vide pour la dernière prévision disponible.", label_style="font-size:.875rem")
    when = st.date_input("Date de prévision (facultatif)", value=None, key=f"ev_flood_date_{kind}", label_visibility="collapsed")
    tv = None
    if when:
        tv = f"{when.isoformat()}T00:00:00" if kind == "glofas" else f"{when.isoformat()}-00-00"
    simple = st.checkbox("Affichage simplifié (sans lecture au clic)", False, key=f"ev_flood_simple_{kind}")
    st.caption("À cocher si la carte interactive ne se charge pas (message « trouble loading the streamlit_folium.st_folium component »).")
    m = _ev_base_map((48.0, 8.0) if kind == "efas" else (20.0, 10.0), 4 if kind == "efas" else 2)
    _ev_add_wms(m, url, chosen, tv, timed)
    times = {n: (tv or details.get(n, {}).get("time_default") or None) for n in timed}       # date par défaut si aucune choisie
    if simple:
        import folium
        folium.LayerControl(collapsed=True).add_to(m)
        components.html(m.get_root().render(), height=580, scrolling=False)
        out = {}
    else:
        out = _ev_show_map(m, f"ev_flood_map_{kind}", track_click=True)
    click = out.get("last_clicked") if out else None
    if click and chosen:
        _ev_click_values(url, chosen, click, out.get("zoom") or 4, tv, times)
    elif chosen and not simple:
        st.caption("Cliquez sur la carte pour lire la valeur des couches sélectionnées à cet endroit.")
    with st.expander("À quoi servent les couches ?"):
        rows = [{"Couche": n, "Titre": t, "Rôle (d'après son nom)": evt.layer_note(n, t),
                 "Description officielle": (details.get(n, {}).get("abstract") or "")[:400],
                 "Dates disponibles": details.get(n, {}).get("time", "")[:120],
                 "Interrogeable au clic": "oui" if details.get(n, {}).get("queryable") else "?"} for n, t in layers]
        _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
    if chosen:
        with st.expander("Légendes des couches affichées"):
            for n in chosen:
                d = details.get(n, {})
                st.markdown(f"**{dict(layers).get(n, n)}** · `{n}`")
                note = evt.layer_note(n, dict(layers).get(n, ""))
                if note:
                    st.caption(note)
                if d.get("abstract"):
                    st.caption("Description officielle : " + d["abstract"][:400])
                _ev_show_legend(url, n, details)
            st.caption("Légendes fournies par le service. Si une image est vide ou cassée, le service ne publie pas de légende pour cette couche.")
        _ev_test_links(url, chosen, tv, times, details,
                       "https://european-flood.emergency.copernicus.eu/" if kind == "efas" else "https://global-flood.emergency.copernicus.eu/",
                       (48.0, 8.0) if kind == "efas" else (20.0, 10.0), 4 if kind == "efas" else 2, f"crues_{kind}")
    st.caption(f"Service interrogé : {url}")
    _ev_flood_cross(url, chosen, tv, times, kind, details)
    _ev_diag(d_cap)


def _ev_flood_cross(url, chosen, tv, times, kind, details):
    import numpy as np
    st.divider()
    st.subheader("Croiser vos points avec les crues")
    up = st.file_uploader("CSV de points (colonnes « lat » et « long », par exemple l'export de l'onglet Géolocalisation)",
                          type=["csv", "txt", "xlsx"], key=f"ev_flood_csv_{kind}")
    if up is None:
        st.caption("Chargez un fichier pour lire, pour chaque point, la valeur des couches sélectionnées ci-dessus et savoir s'il est touché.")
        return
    if not chosen:
        st.info("Sélectionnez au moins une couche ci-dessus.")
        return
    try:
        df, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name, reserved_check=False)
    except InputError as e:
        st.error(str(e))
        return
    st.caption(f"{len(df)} point(s) lus (colonnes {lat_c} / {lon_c}).")
    c1, c2 = st.columns(2)
    with c1:
        _help_line("Seuil de la valeur", "Utilisé par les règles « Valeur supérieure… » et « Valeur inférieure ou égale… » (ci-dessous). Exemples : 50 pour un cumul "
                   "de pluie en mm, 100 pour une période de retour en années.",
                   label_style="font-size:.875rem")
        thr = st.slider("Seuil de la valeur", 0.0, 500.0, 0.0, key=f"ev_flood_thr_{kind}", label_visibility="collapsed")
    prec = {"Position exacte (≈ 10 m, recommandé pour les zones inondables)": 0.0001, "1 km environ": 0.01, "5 km environ": 0.05}
    pk = c2.selectbox("Regroupement des points", list(prec), key=f"ev_flood_prec_{kind}")
    with c1:
        _help_line("Mailles interrogées (maximum)", "Les points proches sont regroupés par maille ; chaque maille correspond à un appel au service "
                   "(environ 0,3 s, 12 en parallèle). Les points au-delà de ce maximum ne sont pas interrogés et restent « n/d » : "
                   "augmentez-le, ou choisissez un regroupement plus large (moins précis : une zone inondable étroite peut être manquée).", label_style="font-size:.875rem")
        max_cells = int(st.slider("Mailles interrogées (maximum)", 100, 5000, 1500, step=100, key=f"ev_flood_cells_{kind}",
                                  label_visibility="collapsed"))
    rules = ["Dans une zone (un objet existe au point)", "Valeur supérieure au seuil", "Valeur inférieure ou égale au seuil (ex. période de retour ≤ N ans)"]
    with c1:
        _help_line("Règle « touché »", "**Dans une zone** : le point est touché s'il se trouve sur un objet de la couche (par exemple une zone "
                   "inondable d'un scénario de crue).\n\n**Valeur supérieure au seuil** : pour un cumul de pluie ou une hauteur d'eau.\n\n"
                   "**Valeur inférieure ou égale au seuil** : pour une période de retour (une crue rare = grande valeur ; « ≤ 100 » retient les "
                   "crues au moins centennales).", label_style="font-size:.875rem")
    rule = c2.selectbox("Règle « touché »", rules, key=f"ev_flood_rule_{kind}", label_visibility="collapsed")
    decide_default = [n for n in chosen if "protect" not in n.lower()] or chosen
    with c1:
        _help_line("Couches qui décident « touché »", "Les autres couches (par exemple les niveaux de protection) ajoutent seulement des colonnes d'information. "
                   "Un niveau de protection n'indique pas une inondation : il est exclu par défaut.", label_style="font-size:.875rem")
    decide = st.multiselect("Couches qui décident « touché »", chosen, default=decide_default, key=f"ev_flood_decide_{kind}",
                            label_visibility="collapsed")
    sig = (up.name, len(df), tuple(chosen), thr, prec[pk], tv, max_cells, rule, tuple(decide))
    if st.button("Croiser", type="primary", key=f"ev_flood_go_{kind}"):
        with st.spinner("Lecture des valeurs aux points…"):
            lat = pd.to_numeric(df[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
            lon = pd.to_numeric(df[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
            res = df.copy()
            rule_mode = rules.index(rule)
            hit, covered, notes, examples = _ev_sample_cross(url, chosen, decide, rule_mode, thr, times, details, lat, lon,
                                                             prec[pk], max_cells, "crue_", res)
            if not decide:
                covered[:] = False
            res["crue_touche"] = ["Oui" if h else ("Non" if c_ else "n/d") for h, c_ in zip(hit, covered)]
            st.session_state[f"ev_flood_cross_{kind}"] = (sig, res, notes, examples)
    saved = st.session_state.get(f"ev_flood_cross_{kind}")
    if not saved or saved[0] != sig:
        return
    _, res, notes, examples = saved
    for n in notes:
        st.warning(n)
    with st.expander("Diagnostic de lecture (réponses brutes du service sur quelques mailles)"):
        if examples:
            _stretch(st.dataframe, pd.DataFrame(examples), hide_index=True)
        else:
            st.caption("Aucune maille interrogée.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Points analysés", len(res))
    m2.metric("Points touchés", int((res["crue_touche"] == "Oui").sum()))
    m3.metric("Sans valeur (n/d)", int((res["crue_touche"] == "n/d").sum()))
    import folium
    lat = pd.to_numeric(res[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    lon = pd.to_numeric(res[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    ok = lat.notna() & lon.notna()
    if ok.any():
        mp = _ev_base_map((float(lat[ok].mean()), float(lon[ok].mean())), 5)
        colors = {"Oui": "#d7191c", "Non": "#2e9e5b", "n/d": "#9e9e9e"}
        vcols = [c for c in res.columns if c.startswith("crue_") and c != "crue_touche"]
        for i in res.index[ok][:5000]:
            r = res.loc[i]
            tip = " · ".join(x for x in [str(r.get("name") or r.get("nom_demande") or ""), f"Crue : {r['crue_touche']}"]
                             + [f"{c[5:]} = {r[c]}" for c in vcols if r[c] is not None and r[c] == r[c]] if x)
            folium.CircleMarker([lat[i], lon[i]], radius=8, color="#222", weight=1, fill=True, fill_color=colors[r["crue_touche"]],
                                fill_opacity=0.9, tooltip=tip).add_to(mp)
        _ev_show_map(mp, f"ev_flood_cross_map_{kind}")
        st.markdown("<span style='color:#d7191c'>■</span> touché · <span style='color:#2e9e5b'>■</span> non touché · "
                    "<span style='color:#9e9e9e'>■</span> n/d (point non interrogé : hors emprise de la couche ou au-delà du maximum de mailles)", unsafe_allow_html=True)
    show = [c for c in res.columns if c.startswith("crue_")]
    _cross_table(res, show, res["crue_touche"] == "Oui", f"ev_flood_tbl_{kind}")
    st.download_button("Télécharger le croisement (CSV)", _csv_bytes(res), f"croisement_crues_{kind}.csv", "text/csv",
                       key=f"ev_flood_dl_{kind}")


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_om_json(path: str):
    s = get_settings()
    return vg.fetch_path(vg.MeteoFranceAuth(s), s, path)


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_om_zip(path: str):
    s = get_settings()
    return vg.fetch_path(vg.MeteoFranceAuth(s), s, path, binary=True)


def _overseas_tables() -> tuple[pd.DataFrame, list[str]]:
    """Table longue unique (colonne « territoire ») : Polynésie, Nouvelle-Calédonie et flux ZIP des autres DROM."""
    frames, problems = [], []
    for name, info in vg.OM_TERRITORIES.items():
        try:
            t = vg.parse_vigilance(_fetch_om_json(info["path"]), phenomena=vg.OM_PHENOMENA, include_domain_max=True)
        except (vg.VigilanceError, vg.VigilanceFormatError) as e:
            problems.append(f"{name} : {e}")
            continue
        t["territoire"] = name
        frames.append(t)
    try:
        z = vg.parse_drom_zip(_fetch_om_zip(vg.OM_DROM_PATH))
        if len(z):
            frames.append(z)
        else:
            problems.append("Autres DROM : aucune donnée exploitable dans le flux.")
    except vg.VigilanceError as e:
        problems.append(f"Autres DROM : {e}")
    if not frames:
        return pd.DataFrame(columns=vg.COLUMNS + ["territoire"]), problems
    return pd.concat(frames, ignore_index=True), problems


def _overseas_section():
    """Vigilance outre-mer : un indicateur et un repère coloré par territoire, détail par zone."""
    import folium
    from streamlit_folium import st_folium

    st.subheader("Outre-mer")
    table, problems = _overseas_tables()
    for p in problems:
        st.warning(p)
    if table.empty:
        return
    summary = []
    for terr, g in table.groupby("territoire"):
        w = vg.worst_by_domain(g, None)
        mx = int(w["couleur_id"].max()) if len(w) and w["couleur_id"].notna().any() else 0
        summary.append((terr, mx, w))
    summary.sort(key=lambda x: (-x[1], x[0]))
    for i in range(0, len(summary), 5):
        cols = st.columns(5)
        for col, (terr, mx, _) in zip(cols, summary[i:i + 5]):
            col.metric(terr, vg.COLORS.get(mx, "n/d"))

    centers = dict(vg.OM_CENTERS)
    view = st.selectbox("Zoom de la carte", list(centers), key="om_view")
    zoom = vg.OM_TERRITORIES[view]["zoom"] if view in vg.OM_TERRITORIES else vg.OM_VIEWS.get(view, (None, 10))[1]
    m = folium.Map(location=list(centers[view]), zoom_start=zoom, tiles=None, control_scale=True)
    _add_basemaps(m)
    for terr, mx, w in summary:
        if terr not in centers:
            continue
        zones = "<br>".join(f"{html.escape(str(r.domaine))} : {html.escape(str(r.couleur))}"
                            + (f" — {html.escape(r.phenomenes)}" if r.phenomenes else "") for r in w.itertuples())
        folium.CircleMarker(list(centers[terr]), radius=16, color="#222", weight=2, fill=True, fill_opacity=0.85,
                            fill_color=vg.COLOR_HEX.get(mx, "#999999"),
                            popup=folium.Popup(f"<b>{html.escape(terr)}</b><br>{zones}", max_width=340),
                            tooltip=f"{terr} : {vg.COLORS.get(mx, 'n/d')}").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    try:
        st_folium(m, height=420, returned_objects=[], key=f"om_map_{view}", use_container_width=True)
    except TypeError:
        st_folium(m, height=420, width=1400, returned_objects=[], key=f"om_map_{view}")
    with st.expander("Détail par zone"):
        detail = pd.concat([w.assign(territoire=terr) for terr, _, w in summary], ignore_index=True)
        detail = detail[["territoire", "domaine", "couleur", "phenomenes"]].rename(
            columns={"domaine": "zone", "phenomenes": "phénomènes en alerte"})
        _stretch(st.dataframe, detail, hide_index=True)


def _vigilance_section():
    st.subheader("Vigilance Météo-France")
    df = _load_vigilance()
    if df is None:
        return
    periodes = sorted(p for p in df["periode"].unique() if p) or [""]
    per = st.selectbox("Échéance", periodes) if len(periodes) > 1 else periodes[0]
    worst = vg.worst_by_domain(df, per or None)
    n = worst["couleur_id"].value_counts()
    c = st.columns(4)
    for i, (cid, name) in enumerate(vg.COLORS.items()):
        c[i].metric(name, int(n.get(cid, 0)))
    try:
        deps = get_departements()
    except vg.VigilanceError as e:
        st.warning(f"Carte indisponible : {e}")
        deps = None
    if deps is not None:
        import folium
        from streamlit_folium import st_folium
        by_code = {}
        for r in worst.itertuples():
            by_code[str(r.domaine).strip().upper()] = r
        def color_for(code: str):
            for cand in vg._domain_candidates(code):
                if cand in by_code:
                    return vg.COLOR_HEX.get(int(by_code[cand].couleur_id), "#999999")
            return "#cccccc"
        m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles=None, control_scale=True)
        _add_basemaps(m)
        folium.GeoJson(deps.geojson, style_function=lambda f: {
            "fillColor": color_for(str((f.get("properties") or {}).get("code", ""))),
            "color": "#333", "weight": 0.6, "fillOpacity": 0.65},
            tooltip=folium.GeoJsonTooltip(fields=[k for k in ("nom", "code") if k in (deps.geojson["features"][0].get("properties") or {})])
        ).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        try:
            st_folium(m, height=560, returned_objects=[], key="vigilance_map", use_container_width=True)
        except TypeError:  # anciennes versions de streamlit-folium
            st_folium(m, height=560, width=1400, returned_objects=[], key="vigilance_map")
    _stretch(st.dataframe, worst[worst["couleur_id"] >= 2].sort_values("couleur_id", ascending=False), hide_index=True)
    with st.expander("Données brutes analysées"):
        _stretch(st.dataframe, df, hide_index=True)
    st.divider()
    _overseas_section()
    st.divider()
    _vigilance_sites_section(df)


def _vigilance_input(key: str = "vg_upload"):
    """Source des sites : fichier chargé (export de la page Géolocalisation) sinon dernière géolocalisation de la session.
    Retourne (df de travail, colonnes d'origine, est_un_fichier) ou None."""
    up = st.file_uploader("Fichier de sites (CSV ou Excel téléchargé depuis la page Géolocalisation, recherche classique ou par périmètre)",
                          type=["csv", "txt", "xlsx"], key=key)
    res = st.session_state.get("result")
    if up is None:
        if res is not None and not res.sites.empty:
            st.caption("Source : dernière géolocalisation de cette session. Chargez un fichier pour en utiliser un autre.")
            return res.sites, list(res.sites.columns), False
        st.info("Chargez un fichier de sites (export de la page « Géolocalisation ») ou lancez une géolocalisation : "
                "les sites seront croisés avec la vigilance.")
        return None
    try:
        raw, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name, reserved_check=False)
    except InputError as e:
        st.error(str(e))
        return None
    work = raw.copy()
    work["lat"] = cr._to_float(raw[lat_c])
    work["long"] = cr._to_float(raw[lon_c])
    bad = int((work["lat"].isna() | work["long"].isna()).sum())
    work = work.dropna(subset=["lat", "long"])
    if bad:
        st.warning(f"{bad} ligne(s) sans coordonnées valides ignorée(s).")
    if work.empty:
        st.error("Aucun point exploitable dans ce fichier.")
        return None
    cols = [c for c in raw.columns if c not in ("lat", "long")] if (lat_c, lon_c) != ("lat", "long") else list(raw.columns)
    cols = list(raw.columns)
    for c in ("name", "nom_demande", "pays", "naf_section"):          # colonnes attendues par la carte
        if c not in work.columns:
            work[c] = None
    if work["name"].isna().all():
        guess = next((c for c in raw.columns if c.lower() in cm.NAME_COLUMNS), None)
        work["name"] = work[guess] if guess else ""
    if "osm_url" not in work.columns:
        work["osm_url"] = [f"https://www.openstreetmap.org/?mlat={la:.6f}&mlon={lo:.6f}#map=17/{la:.6f}/{lo:.6f}"
                           for la, lo in zip(work["lat"], work["long"])]
    if "match_score" not in work.columns:
        work["match_score"] = None
    st.caption(f"Source : fichier « {up.name} » ({len(work)} point(s) ; colonnes {lat_c} / {lon_c}).")
    return work, cols, True


def _vigilance_sites_section(df):
    """Croise un fichier de sites (ou la dernière géolocalisation) avec la vigilance déjà chargée (``df``)."""
    st.subheader("Sites exposés à la vigilance en cours")
    src = _vigilance_input()
    if src is None:
        return
    sites, orig_cols, from_file = src
    try:
        exp = vg.expose_sites(sites, df, get_departements())
    except vg.VigilanceError as e:
        st.error(str(e))
        return
    if "pays_iso2" in exp.columns and exp["pays_iso2"].notna().any():
        fr = exp[exp["pays_iso2"] == "FR"]
    else:                                  # fichier sans pays : on garde ce qui tombe dans un département français
        fr = exp[exp["departement_code"].notna()]
    alert = fr[fr["vigilance_couleur_id"].fillna(0) >= 3]
    c = st.columns(3)
    c[0].metric("Sites en France", len(fr))
    c[1].metric("En vigilance orange ou rouge", len(alert))
    c[2].metric("Hors France (non traités)", len(exp) - len(fr))
    vcols = ["departement_code", "departement", "vigilance_couleur_id", "vigilance_couleur", "vigilance_phenomenes"]
    if from_file:
        full_cols = [c for c in orig_cols if c in exp.columns and c not in vcols] + vcols
        cols = full_cols
    else:
        full_cols = [c for c in exp.columns]
        cols = ["nom_demande", "name", "departement", "vigilance_couleur", "vigilance_phenomenes", "naf_section", "surface_m2", "osm_url"]
    if alert.empty:
        st.success("Aucun site en vigilance orange ou rouge.")
    fr_s = fr.sort_values("vigilance_couleur_id", ascending=False, na_position="last")
    _cross_table(fr_s[full_cols], vcols, fr_s["vigilance_couleur_id"].fillna(0) >= 3, "vg_tbl",
                 "Seulement les vigilances orange ou rouge", default_only=True)
    if not alert.empty:
        _site_map(alert, kp="vg_")
        st.download_button("Télécharger les sites en vigilance orange ou rouge (CSV)", _csv_bytes(alert[cols]),
                           "catmap_sites_vigilance.csv", "text/csv")
    st.download_button("Télécharger tous les sites avec leur vigilance (CSV)", _csv_bytes(exp[full_cols]),
                       "catmap_sites_vigilance_complet.csv", "text/csv", key="vg_dl_all")


@st.cache_data(ttl=86400, show_spinner=False)
def _geocodes_cached(url: str, local: str, uploaded: bytes | None):
    """Contours des zones : fichier chargé > fichier du dépôt > adresse (secret) > couche GeoSphere. Les erreurs ne sont pas mises en cache."""
    if uploaded:
        return ma.load_geocodes_bytes(uploaded)
    if local:
        return ma.load_geocodes_bytes(Path(local).read_bytes())
    if url and url.lower().split("?")[0].endswith(".json"):
        return ma.load_geocodes_url(url)
    return ma.load_geocodes(url=url or ma.GEOCODES_URL)


def _load_geocodes(url: str, uploaded: bytes | None = None):
    local = next((str(p) for p in (ASSETS.parent / "data" / "meteoalarm_geocodes_light.json.gz",
                                   ASSETS.parent / "data" / "meteoalarm_geocodes.json",
                                   ASSETS.parent / "data" / "meteoalarm_geocodes.json.gz") if p.exists()), "")
    try:
        return _geocodes_cached(url, local, uploaded), None
    except (ma.MeteoAlarmError, OSError) as e:
        return None, str(e)


@st.cache_data(ttl=300, show_spinner=False)
def _load_europe(slugs: tuple, geo_url: str, uploaded: bytes | None = None, aliases_raw: bytes | None = None, _v: int = 7):
    alerts, rings, errors = ma.fetch_all(slugs)
    geo, geo_err = _load_geocodes(geo_url, uploaded)
    if not alerts.empty:
        rings, _ = ma.attach_geometry(alerts, rings, geo, ma.load_aliases(aliases_raw) if aliases_raw else None)
    info = {}
    if geo:
        for code, (ctry, name, _r) in geo["by_code"].items():
            e = info.setdefault(str(ctry), [0, []])
            e[0] += 1
            if len(e[1]) < 4:
                e[1].append(f"{code} ({name})")
    geo_info = pd.DataFrame([{"pays (fichier)": k, "zones": v[0], "exemples de codes": ", ".join(v[1])}
                             for k, v in sorted(info.items())])
    return alerts, rings, errors, geo_err, (len(geo["by_code"]) if geo else 0), geo_info


def _aliases_bytes(uploaded) -> bytes | None:
    if uploaded is not None:
        return uploaded.getvalue()
    p = ASSETS.parent / "data" / "meteoalarm_geocodes_aliases.csv"
    try:
        return p.read_bytes() if p.exists() else None
    except OSError:
        return None


def _alert_popup(a) -> str:
    def esc(x):
        return html.escape(str(x or ""))
    def fmt(d):
        return pd.Timestamp(d).tz_convert("Europe/Paris").strftime("%d/%m/%Y %H:%M (Paris)") if d is not None and pd.notna(d) else "—"
    col = ma.LEVEL_HEX.get(int(a["niveau_id"]), "#999") if pd.notna(a["niveau_id"]) else "#999"
    desc = esc(a["description"])[:700]
    cons = esc(a["consigne"])[:500]
    return (f"<div style='font:13px/1.4 sans-serif;max-width:340px'>"
            f"<div style='background:{col};color:#fff;padding:4px 8px;border-radius:4px;font-weight:600'>"
            f"{esc(a['type'])} · {esc(a['niveau'])}</div>"
            f"<b>{esc(a['pays'])} — {esc(a['zone'])}</b><br>"
            f"Début : {fmt(a['debut'])}<br>Fin : {fmt(a['fin'])}<br>"
            f"Sévérité : {esc(a['severite'])} · Urgence : {esc(a['urgence'])} · Certitude : {esc(a['certitude'])}<br>"
            + (f"<i>{esc(a['titre'])}</i><br>" if a["titre"] else "")
            + (f"{desc}<br>" if desc else "") + (f"<b>Consigne :</b> {cons}<br>" if cons else "")
            + f"<small>Émetteur : {esc(a['emetteur'])}</small></div>")


def _europe_map(alerts, rings):
    import folium
    from streamlit_folium import st_folium
    m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles=None, control_scale=True)
    _add_basemaps(m)
    grp = folium.FeatureGroup(name="Alertes par zone")
    zones: dict = {}
    for i in rings:                                   # une seule géométrie par zone (évite une page énorme)
        a = alerts.iloc[i]
        zones.setdefault((a["pays"], a["zone"]), []).append(i)
    for idxs in zones.values():
        idxs.sort(key=lambda i: -(alerts.iloc[i]["niveau_id"] or 0))
        a = alerts.iloc[idxs[0]]
        col = ma.LEVEL_HEX.get(int(a["niveau_id"]), "#999") if pd.notna(a["niveau_id"]) else "#999"
        tip = f"{a['pays']} · {a['zone']} · " + ", ".join(f"{alerts.iloc[i]['type']} ({alerts.iloc[i]['niveau']})" for i in idxs[:4])
        body = "<hr style='margin:6px 0'>".join(_alert_popup(alerts.iloc[i]) for i in idxs[:6])
        geom = {"type": "MultiPolygon", "coordinates": [[[[round(float(x), 3), round(float(y), 3)] for x, y in r]] for r in rings[idxs[0]]]}
        folium.GeoJson({"type": "Feature", "properties": {}, "geometry": geom},
                       style_function=lambda f, c=col: {"fillColor": c, "color": "#222", "weight": 1, "fillOpacity": 0.55},
                       tooltip=tip, popup=folium.Popup(folium.Html(body, script=True), max_width=380, max_height=380)).add_to(grp)
    grp.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    try:
        st_folium(m, height=600, returned_objects=[], key="europe_map", use_container_width=True)
    except TypeError:
        st_folium(m, height=600, width=1400, returned_objects=[], key="europe_map")


def _europe_section():
    st.subheader("Vigilance Europe (MeteoAlarm)")
    st.caption("Alertes des services météo nationaux agrégées par MeteoAlarm (EUMETNET), licence CC BY 4.0. "
               "Niveaux : jaune, orange, rouge. Les flux sont actualisés toutes les 3 minutes environ.")
    if not hasattr(ma, "COUNTRIES") or not hasattr(ma, "diagnostic"):
        st.error("Le fichier `catmap/meteoalarm.py` du dépôt n'est pas la dernière version (il manque `COUNTRIES` ou `diagnostic`). "
                 "Remplacez-le par la version la plus récente envoyée, puis redéployez.")
        return
    labels = {k: v[0] for k, v in ma.COUNTRIES.items()}
    default = ["france", "belgium", "luxembourg", "germany", "switzerland", "italy", "spain"]
    chosen = list(labels)                  # tous les pays MeteoAlarm, à chaque fois
    if st.button("Relancer la recherche", key="eu_refresh"):
        _load_europe.clear()
        _geocodes_cached.clear()
    with st.expander("Contours des zones (avancé)"):
        st.caption("Par défaut : couche publique GeoSphere. Pour utiliser le fichier officiel « Geocodes » (.json) du Redistribution Hub "
                   "de MeteoAlarm, chargez-le ici, ou placez-le dans le dépôt sous `data/meteoalarm_geocodes_light.json.gz` (version allégée, voir `tools/slim_geocodes.py`).")
        up_geo = st.file_uploader("Fichier Geocodes MeteoAlarm (.json)", type=["json"], key="eu_geo_up")
        up_al = st.file_uploader("Fichier Geocodes Aliases (.csv) — sinon `data/meteoalarm_geocodes_aliases.csv` du dépôt",
                                 type=["csv", "txt"], key="eu_al_up")
    if not chosen:
        st.info("Sélectionnez au moins un pays.")
        return
    with st.spinner("Lecture des flux MeteoAlarm…"):
        alerts, rings, errors, geo_err, n_geo, geo_info = _load_europe(tuple(sorted(chosen)), str(_secrets().get("meteoalarm_geocodes_url", "")),
                                                      up_geo.getvalue() if up_geo is not None else None, _aliases_bytes(up_al))
    no_feed = {k: v for k, v in errors.items() if "404" in v}
    real_err = {k: v for k, v in errors.items() if k not in no_feed}
    if real_err:
        st.warning("Pays indisponibles : " + " ; ".join(f"{k} ({v})" for k, v in real_err.items()))
    if no_feed:
        st.caption("Pas de flux publié (en général : aucune alerte en cours) : " + ", ".join(no_feed) + ".")
    with st.expander("Diagnostic par pays (flux, alertes lues, contours)"):
        st.caption(f"Zones de contours chargées : {n_geo}.")
        _stretch(st.dataframe, ma.diagnostic(alerts, chosen, errors, None if not n_geo else {}), hide_index=True)
        st.markdown("**Codes de zones présents dans le fichier de contours**")
        _stretch(st.dataframe, geo_info, hide_index=True)
    if alerts.empty:
        if not real_err:
            st.success("Aucune alerte en cours.")
        return
    c1, c2, c3 = st.columns(3)
    types = sorted(set(alerts["type"].dropna()))
    sel_types = c1.multiselect("Types d'alerte", types, default=types, key="eu_types")
    min_level = c2.selectbox("Niveau minimal", [1, 2, 3, 4], index=1, format_func=lambda i: ma.LEVELS[i], key="eu_level")
    d0 = pd.Timestamp.now(tz="Europe/Paris").normalize()
    jours = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    labels_p = {"Maintenant": "Maintenant", "Prochaines 24h": "Prochaines 24h", "J": "Aujourd'hui"}
    for k in range(1, 5):
        dd = d0 + pd.Timedelta(days=k)
        labels_p[f"J+{k}"] = f"J+{k} ({jours[dd.weekday()]} {dd:%d/%m})"
    labels_p["Toutes"] = "Toutes"
    day = c3.selectbox("Période", list(labels_p), index=2, format_func=lambda k: labels_p[k], key="eu_day2")
    now = pd.Timestamp.now(tz="Europe/Paris")
    if day == "Toutes":
        win = None
    elif day == "Maintenant":
        win = (now, now)
    elif day == "Prochaines 24h":
        win = (now, now + pd.Timedelta(hours=24))
    else:
        k = 0 if day == "J" else int(day[2:])
        win = (d0 + pd.Timedelta(days=k), d0 + pd.Timedelta(days=k + 1))
    if win is None:
        in_win = pd.Series(True, index=alerts.index)
    else:                                  # l'alerte chevauche la période (heure de Paris) ; fin inconnue = toujours valable
        deb = pd.to_datetime(alerts["debut"], utc=True, errors="coerce")
        fin = pd.to_datetime(alerts["fin"], utc=True, errors="coerce")
        in_win = (deb.isna() | (deb < win[1])) & (fin.isna() | (fin >= win[0]))
    base = alerts[alerts["type"].isin(sel_types) & in_win]
    view = base[base["niveau_id"].fillna(0) >= min_level]
    n = base["niveau_id"].value_counts()
    st.caption("Heures affichées : Paris. La carte montre, pour chaque zone, l'alerte la plus forte de la période choisie.")
    c = st.columns(4)
    for i, name in ma.LEVELS.items():
        c[i - 1].metric(name, int(n.get(i, 0)))
    keep_idx = set(view.index)
    shown = {i: r for i, r in rings.items() if i in keep_idx}
    _europe_map(alerts, shown)
    missing = view[~view.index.isin(shown)]
    if geo_err:
        st.warning(f"Contours des zones indisponibles ({geo_err}) : seules les alertes dont le flux contient déjà un polygone sont dessinées.")
    if len(missing):
        with st.expander(f"{len(missing)} alerte(s) non dessinée(s) (contour de zone introuvable)"):
            _stretch(st.dataframe, ma.coverage(alerts), hide_index=True)
            _stretch(st.dataframe, missing[["pays", "zone", "codes_zone", "niveau", "type"]], hide_index=True)
    view = view.copy()
    for c_ in ("debut", "fin", "emis"):
        view[c_] = pd.to_datetime(view[c_], utc=True, errors="coerce").dt.tz_convert("Europe/Paris").dt.strftime("%d/%m/%Y %H:%M")
    show = view.sort_values(["niveau_id", "pays"], ascending=[False, True])
    cols = ["pays", "niveau", "niveau_brut", "type", "debut", "fin", "zone", "severite", "urgence", "certitude", "titre", "emetteur"]
    st.markdown("**Alertes en cours**")
    cols = [c_ for c_ in cols if c_ in show.columns]
    _stretch(st.dataframe, show[cols].rename(columns={"debut": "début", "severite": "sévérité", "certitude": "certitude"}), hide_index=True)
    with st.expander("Détail des consignes et données brutes analysées"):
        _stretch(st.dataframe, view.drop(columns=["niveau_id", "type_id"]), hide_index=True)
    st.download_button("Télécharger les alertes (CSV)", _csv_bytes(view), "catmap_vigilance_europe.csv", "text/csv", key="eu_dl")
    st.divider()
    _europe_sites_section(alerts, rings, set(base.index))


def _europe_sites_section(alerts, rings, active=None):
    st.subheader("Sites exposés aux alertes MeteoAlarm")
    src = _vigilance_input(key="eu_upload")
    if src is None:
        return
    sites, orig_cols, from_file = src
    exp = sites.reset_index(drop=True).copy()
    rings = {i: r for i, r in rings.items() if active is None or i in active}
    res = ma.expose_points(exp["lat"].to_numpy(float), exp["long"].to_numpy(float), alerts, rings, min_level=2)
    for col in res.columns:
        exp[col] = res[col].values
    hit = exp[exp["europe_niveau_id"].notna()]
    orange = hit[hit["europe_niveau_id"] >= 3]
    nocontour = sorted(set(alerts[~alerts["avec_contour"]]["pays"]))
    c = st.columns(3)
    c[0].metric("Sites analysés", len(exp))
    c[1].metric("En alerte jaune ou plus", len(hit))
    c[2].metric("En alerte orange ou rouge", len(orange))
    if nocontour:
        st.caption("Pays dont le flux ne fournit pas de contour (non croisés avec vos sites) : " + ", ".join(nocontour) + ".")
    ecols = list(res.columns)
    base = [x for x in orig_cols if x in exp.columns] if from_file else [x for x in ("nom_demande", "name", "naf_section", "surface_m2", "osm_url") if x in exp.columns]
    cols = base + ecols
    if orange.empty:
        st.success("Aucun site en alerte orange ou rouge.")
    exp_s = exp.sort_values("europe_niveau_id", ascending=False, na_position="last")
    _cross_table(exp_s[cols], ecols, exp_s["europe_niveau_id"].fillna(0) >= 3, "eu_tbl",
                 "Seulement les alertes orange ou rouge", default_only=True)
    if not orange.empty:
        _site_map(orange, kp="eu_")
    st.download_button("Télécharger tous les sites avec leurs alertes (CSV)", _csv_bytes(exp[cols]),
                       "catmap_sites_vigilance_europe.csv", "text/csv", key="eu_dl_sites")


# --------------------------------------------------------------------------
def _points_input(kp: str):
    """Points à analyser : fichier CSV / Excel (colonnes lat et long) ou dernier résultat de géolocalisation.
    Retourne (df, colonne lat, colonne lon, nom) ou (None, None, None, "")."""
    src = st.radio("Points à analyser", ["Fichier CSV / Excel", "Dernier résultat de géolocalisation"], horizontal=True, key=f"{kp}src")
    df = lat_c = lon_c = None
    name = ""
    if src == "Fichier CSV / Excel":
        up = st.file_uploader("CSV de points (colonnes « lat » et « long », par exemple l'export de l'onglet Géolocalisation)",
                              type=["csv", "txt", "xlsx"], key=f"{kp}csv_{st.session_state.get(f'{kp}up_n', 0)}")
        if up is None:
            st.caption("Chargez un fichier pour lancer l'analyse. Séparateur « ; » ou « , » accepté.")
        else:
            try:
                df, lat_c, lon_c = cr.read_points_csv(up.getvalue(), up.name)
                name = up.name
            except InputError as e:
                st.error(str(e))
    else:
        last = st.session_state.get("result") or st.session_state.get("area_result") or st.session_state.get("act_result")
        if last is None or last.sites.empty:
            st.caption("Aucun résultat de géolocalisation en mémoire : lancez une recherche dans l'onglet Géolocalisation ou chargez un fichier.")
        else:
            df, lat_c, lon_c, name = last.sites.drop(columns=[c for c in ("_outer",) if c in last.sites.columns]), "lat", "long", "géolocalisation"
    return df, lat_c, lon_c, name


def _georisques_section():
    st.subheader("Risques naturels et technologiques en France (Géorisques)")
    st.markdown("Interroge l'API publique **Géorisques** (BRGM) pour chacun de vos points GPS et ajoute des colonnes `gr_…` : "
                "risques par adresse, sismicité, radon, argiles, mouvements de terrain, cavités, sols pollués, ICPE, arrêtés CatNat… "
                "Valable pour la France uniquement.")
    df, lat_c, lon_c, name = _points_input("gr_")
    if df is None:
        return
    st.caption(f"{len(df)} point(s) lus (colonnes {lat_c} / {lon_c}).")
    keys = list(gr.BY_KEY)
    chosen = st.multiselect("Jeux de données interrogés", keys, default=[d.key for d in gr.DATASETS if d.default], key="gr_sets",
                            format_func=lambda k: gr.BY_KEY[k].label)
    c1, c2 = st.columns(2)
    radius = c1.slider("Rayon de recherche autour du point (m)", 100, 5000, 1000, step=100, key="gr_radius",
                       help="Utilisé pour compter les objets proches (cavités, mouvements de terrain, ICPE, SIS…).")
    max_pts = c2.number_input("Nombre maximal de points interrogés", 1, 5000, 300, step=50, key="gr_max")
    n_pts = min(len(df), int(max_pts))
    n_other = n_pts * (len([k for k in chosen if k != "rapport"]) + (1 if any(gr.BY_KEY[k].by_insee for k in chosen) else 0))
    n_rap = n_pts if "rapport" in chosen else 0
    est = max(n_other / gr.RATE_PER_MINUTE["default"], n_rap / gr.RATE_PER_MINUTE["rapport"])
    st.caption(f"≈ {n_other + n_rap} appel(s) à l'API, soit environ {max(1, round(est))} min (quotas : 5 appels/s, et 1 appel/s pour le "
               "rapport de risques, dont l'API est instable : décoché par défaut). Les points identiques ne sont interrogés qu'une fois.")
    with st.expander("Tester sur un point (diagnostic)"):
        t1, t2 = st.columns(2)
        tlat = t1.number_input("Latitude", -90.0, 90.0, 48.8566, format="%.6f", key="gr_tlat")
        tlon = t2.number_input("Longitude", -180.0, 180.0, 2.3522, format="%.6f", key="gr_tlon")
        if st.button("Tester", key="gr_probe", disabled=not chosen):
            with st.spinner("Interrogation de Géorisques…"):
                st.session_state["gr_probe_rows"] = gr.probe(float(tlat), float(tlon), chosen, int(radius))
        rows = st.session_state.get("gr_probe_rows")
        if rows:
            _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
            st.caption("Si un jeu de données est en « chemin introuvable » ou ne renvoie rien d'attendu, copiez la ligne ci-dessus : "
                       "elle suffit à corriger le chemin ou le nom du champ.")
    if st.button("Interroger Géorisques", type="primary", key="gr_go", disabled=not chosen):
        bar = st.progress(0.0)
        with st.spinner("Interrogation de Géorisques…"):
            res, summ = gr.enrich_points(df, lat_c, lon_c, chosen, int(radius), int(max_pts),
                                         on_progress=lambda i, n: bar.progress(min(1.0, i / max(1, n))))
        bar.empty()
        st.session_state["gr_result"] = (name, len(df), res, summ, chosen)
    stored = st.session_state.get("gr_result")
    if not stored or stored[0] != name or stored[1] != len(df):
        return
    _, _, res, summ, used = stored
    m = st.columns(4)
    m[0].metric("Points", summ["points"])
    m[1].metric("Interrogés", summ["interroges"])
    m[2].metric("Réponses complètes", int((res["gr_statut"] == "ok").sum()))
    m[3].metric("En erreur / partiel", int(res["gr_statut"].isin(["échec", "partiel"]).sum()))
    if summ["non_interroges"]:
        st.warning(f"{summ['non_interroges']} point(s) au-delà du maximum n'ont pas été interrogés : augmentez le maximum.")
    if summ["invalides"]:
        st.warning(f"{summ['invalides']} point(s) sans coordonnées valides.")
    for msg, n in sorted(summ["erreurs"].items(), key=lambda x: -x[1])[:6]:
        st.warning(f"{n} point(s) : {msg}")
    new_cols = [c for c in res.columns if c.startswith("gr_")]
    flag = [c for c in new_cols if c.endswith("_nb")]
    mask = (res[flag].fillna(0).astype(float) > 0).any(axis=1) if flag else None
    _cross_table(res, new_cols, mask, "gr_tbl", "Seulement les points avec au moins un objet proche")
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(res), "catmap_georisques.csv", "text/csv")
    st.caption("Source : Géorisques (BRGM / ministère de la Transition écologique), données publiques ; vérifiez les conditions de "
               "réutilisation sur georisques.gouv.fr. `_nb` = nombre d'objets dans le rayon choisi ; « Oui / Non » = risque présent à l'adresse "
               "selon le rapport de risques. Un risque absent de la base n'est pas un risque nul.")


@st.cache_data(ttl=86400, show_spinner=False)
def _fh_catalog():
    return fhz.load_catalog()


def _fh_popup(row, cols, kinds_of):
    import html
    lines = [f"<b>{html.escape(str(row.get('name') or row.get('nom') or 'Point'))}</b>"]
    for c in cols:
        v = row.get(c)
        k = kinds_of.get(c)
        if k == "raw":
            txt = "n/d" if v is None or pd.isna(v) else f"{float(v):.2f} m"
        elif k == "reclass":
            txt = "n/d" if v is None or pd.isna(v) else f"classe {int(v)} ({fhz.CLASS_LABELS.get(int(v), '')})"
        elif k == "ratio":
            txt = "n/d" if v is None or pd.isna(v) else f"{float(v) * 100:.1f} %"
        elif k == "eur":
            txt = "n/d" if v is None or pd.isna(v) else f"{float(v):,.0f} €".replace(",", " ")
        else:
            txt = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        lines.append(f"{html.escape(c)} : <b>{html.escape(txt)}</b>")
    return "<br>".join(lines)


@st.cache_data(show_spinner=False, max_entries=12)
def _fh_preview(path: str, kind: str, size: int):
    """Image RGBA d'une tuile (mise en cache ; ``size`` invalide le cache si le fichier est retéléchargé)."""
    img, bounds = fhz.render_preview(Path(path), kind)
    return img, bounds


def _fh_map(res, cols_by_kind, color_col, tiles=None, overlays=None, damage=None, cells=None):
    import folium
    pts = res.dropna(subset=["lat_f", "lon_f"])
    center = (float(pts["lat_f"].mean()), float(pts["lon_f"].mean())) if len(pts) else (46.6, 2.5)
    m = _ev_base_map(center, 6)
    shown = pts.head(3000)
    kinds_of = {c: k for k, cs in cols_by_kind.items() for c in cs}
    dmg_cols = list((damage or {}).get("kinds", {}))
    kinds_of.update((damage or {}).get("kinds", {}))
    allc = ([c for cs in cols_by_kind.values() for c in cs] + [c for c in ("debordement_tuile", "debordement_statut") if c in res.columns]
            + [c for c in ("dommage_classe", "dommage_valeur_exposee_eur") if c in res.columns] + [c for c in dmg_cols if c in res.columns])
    kind = kinds_of.get(color_col, "raw")
    is_dmg = kind in ("ratio", "eur")
    edges = (damage or {}).get("edges", {}).get(color_col, [0.1, 0.3, 0.6])
    for img, (south, west, north, east) in overlays or []:      # valeurs de la tuile, mêmes couleurs que la légende
        folium.raster_layers.ImageOverlay(image=img, bounds=[[south, west], [north, east]], opacity=0.75, interactive=False,
                                          zindex=2).add_to(m)
    if cells:                                               # pixels de submersion autour des points (un seul objet GeoJSON : léger)
        import json as _json
        h = coast.CELL_HALF_DEG
        feats = [{"type": "Feature", "properties": {"d": v, "c": fhz.CLASS_COLORS[fhz.flood_class(v, "raw")]},
                  "geometry": {"type": "Polygon", "coordinates": [[[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h], [x - h, y - h]]]}}
                 for x, y, v in cells[:6000]]
        folium.GeoJson({"type": "FeatureCollection", "features": feats}, style_function=lambda f: {"fillColor": f["properties"]["c"], "color": "none",
                       "weight": 0, "fillOpacity": 0.6}, tooltip=folium.GeoJsonTooltip(fields=["d"], aliases=["Hauteur (m)"]), zindex=2).add_to(m)
    for t in tiles or []:                                   # emprise des tuiles réellement lues (vert : lue, rouge : en erreur)
        ok = t["statut"] == "ok"
        tip = f"Tuile ID{t['id']} : {t['points']} point(s) ; " + ("lue" if ok else "ERREUR : " + " | ".join(t["erreurs"])[:200])
        folium.Rectangle([[t["south"], t["west"]], [t["north"], t["east"]]], color="#2e7d32" if ok else "#c62828", weight=2,
                         fill=True, fill_opacity=0.06, tooltip=html_escape(tip)).add_to(m)
    if tiles and not len(shown):
        m.fit_bounds([[min(t["south"] for t in tiles), min(t["west"] for t in tiles)], [max(t["north"] for t in tiles), max(t["east"] for t in tiles)]])
    for _, r in shown.iterrows():
        cls = dmg.damage_class(r.get(color_col), edges) if is_dmg else fhz.flood_class(r.get(color_col), kind)
        folium.CircleMarker([r["lat_f"], r["lon_f"]], radius=7, color="#222", weight=1, fill=True,
                            fill_color=(dmg.DMG_COLORS if is_dmg else fhz.CLASS_COLORS)[cls],
                            fill_opacity=0.9, tooltip=str(r.get("name") or ""),
                            popup=folium.Popup(_fh_popup(r, allc, kinds_of), max_width=420)).add_to(m)
    if tiles:
        m.fit_bounds([[min(t["south"] for t in tiles), min(t["west"] for t in tiles)], [max(t["north"] for t in tiles), max(t["east"] for t in tiles)]])
    elif len(shown) > 1:
        m.fit_bounds([[shown["lat_f"].min(), shown["lon_f"].min()], [shown["lat_f"].max(), shown["lon_f"].max()]])
    pal = dmg.DMG_COLORS if is_dmg else fhz.CLASS_COLORS
    title = "Dommages" if is_dmg else "Hauteur d'eau"
    labels = dmg.legend_labels(edges, kind) if is_dmg else fhz.CLASS_LABELS
    legend = "".join(f'<div><span style="background:{pal[k]};display:inline-block;width:12px;height:12px;border:1px solid #444;'
                     f'margin-right:6px"></span>{lab}</div>' for k, lab in labels.items())
    legend += (f'<div><span style="background:{pal[None]};display:inline-block;width:12px;height:12px;border:1px solid #444;'
               f'margin-right:6px"></span>{"Pas de donnée" if is_dmg else "Hors tuile / erreur"}</div>')
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;color:#222;padding:8px 10px;border:1px solid #888;'
        f'border-radius:4px;font-size:12px"><b>{html_escape(title)} ({html_escape(color_col)})</b>{legend}'
        + ('<div style="margin-top:4px;color:#555">Raster : zones sans inondation transparentes</div>' if overlays else "") + '</div>'))
    if len(pts) > len(shown):
        st.caption(f"Carte limitée aux {len(shown)} premiers points sur {len(pts)} (le tableau et le CSV contiennent tout).")
    _ev_show_map(m, "fh_map")


def html_escape(x):
    import html
    return html.escape(str(x))


@st.cache_data(ttl=3600, show_spinner=False)
def _dmg_jrc_local():
    try:
        return dmg.load_jrc_local(), ""
    except dmg.DamageError as e:
        return None, str(e)


@st.cache_data(show_spinner=False, max_entries=8)
def _dmg_parse_jrc(data: bytes, filename: str):
    return dmg.parse_jrc(data, filename)


_DMG_VALUE_MODES = {
    "col": "Valeur de chaque point : colonne du CSV (en €)",
    "area": "Surface (colonne, m²) × valeur de remplacement saisie (€/m²)",
    "const": "Même valeur exposée pour tous les points (€)",
    "jrc_area": "Surface (colonne, m²) × valeur de remplacement JRC du pays (recommandé si la valeur n'est pas connue)",
}


def _fh_damage_hazus_ui():
    """Charge les fonctions Hazus (dossier data/flood, téléchargement Zenodo à la demande, ou fichier téléversé)."""
    hz = st.session_state.get("fh_hz")
    if hz is None:
        local = dmg.local_files(dmg.HAZUS_FILES)
        if "haz_fl_dept.csv" in local:
            try:
                hz = dmg.parse_hazus(local["haz_fl_dept.csv"], local.get("riverine_ids.csv"), local.get("haz_fl_occ.csv"))
                st.session_state["fh_hz"] = hz
            except dmg.DamageError as e:
                st.error(str(e))
    if hz is None:
        c1, c2 = st.columns([1, 2])
        if c1.button("Télécharger les fonctions Hazus (Zenodo)", key="fh_hz_dl"):
            with st.spinner("Téléchargement depuis Zenodo…"):
                try:
                    got = dmg.fetch_hazus()
                    st.session_state["fh_hz"] = dmg.parse_hazus(got["haz_fl_dept.csv"], got.get("riverine_ids.csv"), got.get("haz_fl_occ.csv"))
                    st.rerun()
                except dmg.DamageError as e:
                    st.error(str(e))
        ups = c2.file_uploader("… ou téléversez haz_fl_dept.csv (facultatifs : riverine_ids.csv, haz_fl_occ.csv)", type=["csv"],
                               accept_multiple_files=True, key="fh_hz_up")
        if ups:
            by = {u.name.lower(): u.getvalue() for u in ups}
            if "haz_fl_dept.csv" not in by:
                st.warning("Le fichier haz_fl_dept.csv est indispensable.")
            else:
                try:
                    st.session_state["fh_hz"] = dmg.parse_hazus(by["haz_fl_dept.csv"], by.get("riverine_ids.csv"), by.get("haz_fl_occ.csv"))
                    st.rerun()
                except dmg.DamageError as e:
                    st.error(str(e))
    return st.session_state.get("fh_hz")


def _fh_damage_jrc_ui():
    parts = []
    loc, err = _dmg_jrc_local()
    if err:
        st.error(err)
    if loc:
        parts.append(loc)
    ups = st.file_uploader("Fichier JRC (Huizinga et al. 2017) : classeur Excel officiel ou CSV au format simple "
                           "(facultatif si déjà dans data/flood/)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="fh_jrc_up")
    for u in ups or []:
        try:
            parts.append(_dmg_parse_jrc(u.getvalue(), u.name))
        except dmg.DamageError as e:
            st.error(str(e))
    jrc = dmg.merge_jrc(parts) if parts else None
    if jrc is None:
        st.caption("Aucun fichier JRC chargé. " + dmg.JRC_TEMPLATE)
    else:
        for n in jrc["notes"][:4]:
            st.warning(n)
        st.caption(f"{len(jrc['curves'])} courbe(s) lue(s) ; {len(jrc['maxdmg'])} valeur(s) de remplacement par pays et catégorie.")
    return jrc


def _guess_col(cols, words, default_none=True):
    """Index (dans ["(aucune)"] + cols) de la 1re colonne dont le nom contient un des mots ; 0 = « (aucune) »."""
    for w in words:                                      # priorité à l'ordre des mots : le 1er mot reconnu gagne
        for i, c in enumerate(cols):
            if w in str(c).lower():
                return i + 1
    return 0 if default_none else 1


def _kpi(col, label: str, text: str):
    """Indicateur à valeur entière visible (retour à la ligne si besoin), contrairement à ``st.metric`` qui tronque les grands montants."""
    col.markdown(f'<div style="line-height:1.25;margin-bottom:.6rem"><div style="font-size:.85rem;opacity:.75">{label}</div>'
                 f'<div style="font-size:1.5rem;font-weight:600;white-space:normal;overflow-wrap:anywhere">{text}</div></div>',
                 unsafe_allow_html=True)


def _eur(v) -> str:
    return f"{float(v):,.0f} €".replace(",", "\u202f")


def _fh_damage_panel(df, name, hazards, ensure):
    """Bloc « Estimer les dommages » commun à tous les aléas sélectionnés : mêmes paramètres d'actif et de valeur pour chaque aléa.
    ``hazards`` : liste de ``{"name", "label", "rps", "col", "result"}``. Si les hauteurs (raw) d'un aléa ne sont pas calculées,
    ``ensure([noms])`` les estime d'abord. Renvoie ``(nom, n, {aléa: (colonnes, résumé)}, courbes, avertissements)`` ou ``None``."""
    base_cols = [c for c in df.columns if not c.startswith(("debordement_", "submersion_", "dommage_")) and c not in ("lat_f", "lon_f")]
    if st.session_state.get("ex_dmg_open"):
        with st.container(border=True):
            st.markdown("**Estimation des dommages** : perte = taux de dommage(hauteur d'eau) × valeur exposée, pour chaque scénario. "
                        "Les hauteurs d'eau (raw) des scénarios choisis sont estimées automatiquement si elles ne le sont pas déjà.")
            _help_line("Approche : méthode et références", "**JRC** : courbes profondeur-dommage par continent et catégorie d'actif, avec valeurs de "
                       "remplacement par pays (Huizinga et al. 2017). **FEMA Hazus** : fonctions profondeur-dommage riveraines américaines "
                       "(structure seule). Dans les deux cas : perte = taux(hauteur d'eau) × valeur exposée.\n\n" + refs_md("jrc_damage", "hazus", "aal"),
                       wide=True, label_style="font-size:.875rem")
            method = st.radio("Approche", ["JRC – courbes mondiales par continent (Huizinga et al. 2017)",
                                           "FEMA Hazus – fonctions riverines (États-Unis)"], key="fh_dmg_method")
            is_jrc = method.startswith("JRC")
            with st.expander("Fichier JRC (courbes et valeurs de remplacement par pays)", expanded=is_jrc):
                jrc = _fh_damage_jrc_ui()
            hz = None if is_jrc else _fh_damage_hazus_ui()
            c1, c2, c3 = st.columns(3)
            cls_main = list(dmg.CLASSES) if is_jrc else ["industrial", "commercial", "residential"]
            default_cls = c1.selectbox("Type d'actif par défaut", cls_main, key="fh_dmg_cls", index=cls_main.index("commercial"), format_func=lambda k: dmg.CLASS_LABELS[k])
            type_col = c2.selectbox("Colonne décrivant le type d'actif (facultatif)", ["(aucune)"] + base_cols, key="fh_dmg_typecol",
                                    index=_guess_col(base_cols, ("type", "categ", "activ", "naf")),
                                    help="Libellés reconnus : industriel / usine, commercial / magasin / bureau, résidentiel / logement… "
                                         "Les autres lignes reçoivent le type par défaut.")
            country_col = c3.selectbox("Colonne pays (code ISO 2/3 lettres ou nom anglais)", ["(aucune)"] + base_cols, key="fh_dmg_ctrycol",
                                       index=_guess_col(base_cols, ("pays_iso2", "pays_iso", "iso2", "iso3", "country", "pays")),
                                       help="Sert à choisir la courbe du bon continent (approche JRC) et la valeur de remplacement du pays.")
            type_col = None if type_col == "(aucune)" else type_col
            country_col = None if country_col == "(aucune)" else country_col
            jrc_val_ok = bool(jrc and jrc["maxdmg"] and country_col)
            if jrc and jrc["maxdmg"] and not country_col:
                st.caption("Choisissez une colonne pays pour utiliser les valeurs de remplacement JRC par pays.")
            modes = (["jrc_area"] if jrc_val_ok else []) + ["area", "col", "const"]
            mode = st.radio("Valeur exposée", modes, key="fh_dmg_mode", format_func=lambda m: _DMG_VALUE_MODES[m])
            v1, v2 = st.columns(2)
            value_col = area_col = None
            component = "total"
            unit_value = const = 0.0
            if mode == "col":
                value_col = v1.selectbox("Colonne de valeur (€)", base_cols, key="fh_dmg_valcol")
            elif mode == "const":
                const = v1.number_input("Valeur exposée par point (€)", min_value=0.0, value=0.0, step=100000.0, key="fh_dmg_const")
            else:
                area_idx = _guess_col(base_cols, ("surface", "area", "m2", "m²"), default_none=False) - 1
                area_col = v1.selectbox("Colonne de surface (m²)", base_cols, key="fh_dmg_areacol", index=max(0, area_idx))
                if mode == "area":
                    unit_value = v2.number_input("Valeur de remplacement (€/m²)", min_value=0.0, value=0.0, step=50.0, key="fh_dmg_unit")
                else:
                    comp_lab = v2.radio("Valeur de remplacement JRC", ["Structure + contenu (total)", "Structure seule"], key="fh_dmg_comp",
                                        help="Prix 2010, par pays et par catégorie (résidentiel, commercial, industriel).")
                    component = "struct" if comp_lab.startswith("Structure seule") else "total"
            continent = occ = None
            floor_h = 0.0
            if is_jrc:
                conts = sorted(set(dmg.CONTINENTS.values()))
                continent = st.selectbox("Continent de la courbe si le pays n'est pas reconnu" if country_col else
                                         "Continent de la courbe (appliqué à tous les points : choisissez une colonne pays pour le déduire)",
                                         conts, index=conts.index("Europe"), key="fh_dmg_cont",
                                         help="Avec une colonne pays, chaque point reçoit la courbe de son continent ; « Global » sert de repli "
                                              "quand le continent n'a pas de courbe pour la catégorie.")
            elif hz is not None:
                occs = hz["occupancies"]
                occ = {}
                cols = st.columns(3)
                for col, (cls, pref) in zip(cols, (("industrial", "IND"), ("commercial", "COM"), ("residential", "RES"))):
                    opts = [o for o in occs if o.startswith(pref)]
                    if opts:
                        occ[cls] = col.selectbox(f"Occupation Hazus – {dmg.CLASS_LABELS[cls].lower()}", opts, key=f"fh_hz_occ_{cls}",
                                                 format_func=lambda o: f"{o} – {occs[o]}" if occs.get(o) else o)
                floor_h = _num_help(st, "Hauteur du plancher au-dessus du terrain (m)", "Les fonctions Hazus sont relatives au plancher du "
                                    "rez-de-chaussée ; 0 = plancher au niveau du terrain.", min_value=0.0, max_value=3.0, value=0.0, step=0.1,
                                    key="fh_hz_floor")
            ready = (jrc is not None and bool(jrc["curves"])) if is_jrc else hz is not None
            if st.button("Calculer les dommages", type="primary", key="fh_dmg_calc", disabled=not ready):
                def _depths_ok(h):
                    prev = st.session_state.get(h["result"])
                    return bool(prev and prev[0] == name and prev[1] == len(df) and "raw" in prev[5] and set(h["rps"]) <= set(prev[4]))
                missing = [h["name"] for h in hazards if not _depths_ok(h)]
                if missing:
                    ensure(missing)
                base = df.reset_index(drop=True)
                classes = dmg.asset_class_series(base, type_col, default_cls)
                curves, problems = {}, []
                conts_pt = None
                if is_jrc:
                    conts_pt, n_unk = dmg.continent_series(base, country_col, jrc, continent)
                    if n_unk:
                        problems.append(f"{n_unk} point(s) : pays non reconnu, courbe « {continent} » utilisée.")
                    for cont, cls in sorted(set(zip(conts_pt, classes))):
                        try:
                            curves[f"{cont}|{cls}"] = dmg.jrc_curve_for(jrc, cont, cls)
                        except dmg.DamageError as e:
                            problems.append(str(e))
                else:
                    for cls in sorted(classes.dropna().unique()):
                        try:
                            if occ.get(cls):
                                curves[cls] = dmg.hazus_curve(hz, occ[cls], floor_h)
                            else:
                                problems.append(f"Hazus : pas d'occupation pour la classe « {dmg.CLASS_LABELS.get(cls, cls)} ».")
                        except dmg.DamageError as e:
                            problems.append(str(e))
                try:
                    values = dmg.exposed_value(base, mode, value_col, area_col, unit_value, const, country_col, classes, jrc, component)
                    per, fallback = {}, 0
                    for h in hazards:
                        res_h = st.session_state[h["result"]][2]
                        depth_cols = {r: h["col"](r) for r in h["rps"]}
                        out, summ = dmg.estimate_damages(res_h, depth_cols, curves, classes, values,
                                                         "JRC" if is_jrc else "FEMA Hazus", conts_pt, h["name"])
                        summ["valeur_nulle"] = bool((values.fillna(0) <= 0).all())
                        summ["points_exposes"] = int((res_h[list(depth_cols.values())].apply(pd.to_numeric, errors="coerce").fillna(0) > 0)
                                                     .any(axis=1).sum())
                        summ["points_lus"] = int(res_h[list(depth_cols.values())].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).sum())
                        fallback = max(fallback, summ.get("repli_global", 0))
                        per[h["name"]] = (out, summ)
                    if fallback:
                        problems.append(f"{fallback} point(s) : pas de courbe du continent pour cette catégorie, courbe « Global » utilisée.")
                    st.session_state["ex_dmg_result"] = (name, len(df), per, curves, problems)
                except dmg.DamageError as e:
                    st.error(str(e))
    stored = st.session_state.get("ex_dmg_result")
    if not stored or stored[0] != name or stored[1] != len(df):
        return None
    _, _, per, curves, problems = stored
    for p in problems:
        st.warning(p)
    shown = [h for h in hazards if h["name"] in per]
    for h in shown:
        _, summ = per[h["name"]]
        st.markdown(f"**Dommages : {h['label']}**")
        cols = st.columns(4)
        no_data = summ.get("points_lus") == 0
        for i, (c, v) in enumerate(list(summ["pertes_totales"].items())[:3]):
            _kpi(cols[i], f"Perte totale {__import__('re').search(r'RP' + chr(92) + 'd+', c).group(0)}", "n/d (aucune hauteur lue)" if no_data else _eur(v))
        if summ["pam_totale"] is not None and not no_data:
            _kpi(cols[3], "Perte annuelle moyenne (approx.)", _eur(summ["pam_totale"]))
        n_exp = summ.get("points_exposes")
        if no_data:
            st.error(f"{h['label']} : aucune hauteur d'eau n'a pu être lue (fichier introuvable ou lecture en échec : voir « Scénarios en échec » "
                     "dans le détail ci-dessous). Les pertes ne sont donc PAS calculées : cela ne signifie pas qu'il n'y a pas de risque.")
        elif n_exp is not None:
            st.caption(f"{n_exp} point(s) sur {summ['points']} avec une hauteur d'eau > 0 dans au moins un scénario.")
            if n_exp == 0:
                st.warning(f"{h['label']} : aucun point n'a de hauteur d'eau > 0, donc les pertes sont nulles. Soit vos points ne sont pas "
                           "dans une zone exposée à cet aléa (ex. points loin de la côte pour la submersion), soit la lecture de la carte est à "
                           "contrôler avec le « Diagnostic des sources ».")
        if summ["classes_sans_courbe"]:
            st.warning(f"{h['label']} : pas de courbe pour " + ", ".join(summ["classes_sans_courbe"]) + " (pertes laissées vides).")
    if shown:
        summ0 = per[shown[0]["name"]][1]
        if summ0["sans_valeur"]:
            st.warning(f"{summ0['sans_valeur']} point(s) sans valeur exposée exploitable : pas de perte calculée pour eux.")
        if summ0.get("valeur_nulle"):
            st.warning("La valeur exposée est nulle ou absente pour tous les points : les pertes en € sont vides ou nulles (les taux restent valables).")
        st.caption("Perte annuelle moyenne : intégrale trapézoïdale entre la plus petite et la plus grande période de retour choisies, sans "
                   "extrapolation hors de cet intervalle. Les pertes par scénario ne s'additionnent pas d'un aléa à l'autre.")
    with st.expander("Courbes utilisées"):
        for key, cv in curves.items():
            cont_, _, cls = key.rpartition("|")
            st.markdown(f"**{(cont_ + ' / ') if cont_ else ''}{dmg.CLASS_LABELS.get(cls, cls)}** : {cv.label}")
            st.line_chart(cv.as_frame().set_index("hauteur (m)"), height=180)
    return stored


_HAZARDS = {
    "debordement": {"label": "Débordement de cours d'eau", "rps": fhz.RPS, "result": "fh_result", "col": lambda r: fhz.column_name(r, "raw")},
    "submersion": {"label": "Submersion marine", "rps": coast.RPS, "result": "cs_result", "col": lambda r: coast.column_name(r)},
}


def _rp_fmt(r):
    return f"RP{r}"


def _exposition_section():
    st.subheader("Débordement et submersion")
    st.markdown("Sélectionnez un ou plusieurs aléas : les points sont chargés une seule fois, les scénarios peuvent être communs ou propres à "
                "chaque aléa, et l'estimation des dommages utilise les mêmes paramètres d'actif et de valeur pour tous les aléas choisis.")
    _help_line("Méthodes et références des aléas et des dommages",
               "**Débordement de cours d'eau** : hauteur d'eau lue dans les cartes mondiales JRC / GloFAS (aucun ajustement statistique par CatMap).\n\n"
               "**Submersion marine** : hauteur d'eau lue dans les cartes WRI Aqueduct (idem).\n\n**Dommages** : taux de dommage = courbe profondeur-dommage "
               "(JRC par continent, ou FEMA Hazus) × valeur exposée ; perte annuelle moyenne par intégration sur les périodes de retour.\n\n"
               + refs_md("jrc_flood", "aqueduct", "jrc_damage", "hazus", "aal"), wide=True, label_style="font-size:.875rem")
    sel = st.multiselect("Aléas à analyser", list(_HAZARDS), default=["debordement"], key="ex_hazards", format_func=lambda k: _HAZARDS[k]["label"])
    if not sel:
        st.caption("Choisissez au moins un aléa.")
        return
    df, lat_c, lon_c, name = _points_input("ex_")
    if df is None:
        return
    common = _checkbox_with_help("Mêmes périodes de retour pour tous les aléas sélectionnés",
                                 "Décochez pour choisir des périodes de retour différentes pour chaque aléa.", key="ex_common_rp", value=True)
    union = sorted({r for k in sel for r in _HAZARDS[k]["rps"]})
    rps_common = None
    if common:
        rps_common = st.multiselect("Périodes de retour (années)", union, default=[100] if 100 in union else union[:1], key="ex_rps",
                                    format_func=_rp_fmt)
    params = {}
    for k in sel:
        H = _HAZARDS[k]
        with st.expander(f"Paramètres : {H['label']}", expanded=True):
            _meth = {"debordement": ("**Méthode** : aucune loi n'est ajustée par CatMap. Hauteur d'eau lue au point dans les cartes de débordement fluvial "
                                     "JRC / GloFAS (simulation hydrodynamique, 90 m) précalculées par période de retour. Les petits cours d'eau "
                                     "(bassins < 500 km²) ne sont pas modélisés.\n\n" + refs_md("jrc_flood")),
                     "submersion": ("**Méthode** : aucune loi n'est ajustée par CatMap. Hauteur d'eau lue au point (bande `inundation_depth`, ≈ 1 km) dans "
                                    "la collection WRI Aqueduct Floods v2 de submersion côtière (niveaux marins extrêmes issus d'une réanalyse de "
                                    "surcotes), précalculée par période de retour et scénario climatique, via Google Earth Engine (collection "
                                    "`WRI/Aqueduct_Flood_Hazard_Maps/V2`). Niveau d'eau extrême sans vagues ni protections locales ; un point à "
                                    "l'intérieur des terres vaut 0.\n\n**Source** : World Resources Institute, Aqueduct Floods (à citer en cas de "
                                    "réutilisation).\n\n**Conditions d'utilisation** : accès à Earth Engine en utilisation non commerciale au regard de "
                                    "l'abonnement du projet ; Google demande de revérifier périodiquement l'éligibilité non commerciale. Toute "
                                    "utilisation commerciale nécessite un enregistrement commercial.\n\n" + refs_md("aqueduct"))}[k]
            _help_line("Méthode et références", _meth, wide=True, label_style="font-size:.875rem")
            if common:
                rps_k = [r for r in rps_common if r in H["rps"]]
                skipped = [r for r in rps_common if r not in H["rps"]]
                if skipped:
                    st.caption("Non disponibles pour cet aléa : " + ", ".join(_rp_fmt(r) for r in skipped) + ".")
            else:
                rps_k = st.multiselect("Périodes de retour (années)", list(H["rps"]), default=[100], key=f"ex_rps_{k}", format_func=_rp_fmt)
            p = {"rps": rps_k}
            if k == "debordement":
                kind_lab = st.radio("Type de donnée", ["raw (hauteur d'eau en m)", "reclass (classe 1 à 4)", "les deux"], horizontal=True, key="fh_kind")
                p["kinds"] = ["raw"] if kind_lab.startswith("raw") else ["reclass"] if kind_lab.startswith("reclass") else ["raw", "reclass"]
            else:
                c1, c2, c3 = st.columns(3)
                p["scen"] = c1.selectbox("Scénario climatique", list(coast.SCENARIOS), key="cs_scen", format_func=lambda x: coast.SCENARIOS[x])
                p["year"] = c2.selectbox("Horizon", list(coast.YEARS[p["scen"]]), key="cs_year", format_func=lambda y: "Historique" if y == "hist" else y)
                p["sub"] = c3.selectbox("Subsidence", list(coast.SUBSIDENCE), key="cs_sub", format_func=lambda x: coast.SUBSIDENCE[x])
                p["radius"] = float(st.selectbox("Rayon de lecture autour du point (km)", [0, 1, 2, 5], index=0, key="cs_radius",
                                                 format_func=lambda x: "0 (pixel du point)" if x == 0 else f"{x} km (maximum alentour)"))
                if p["radius"] > 0:
                    st.caption("Lecture prudente : hauteur maximale des pixels (≈ 1 km) dans le rayon choisi. À utiliser car la carte est grossière : "
                               "un site proche de la côte peut tomber sur un pixel non inondé voisin d'un pixel inondé.")
                if p["scen"] == "historical" and p["year"] != "hist":
                    st.caption("Climat historique avec subsidence projetée : n'existe que pour l'option « Avec subsidence ».")
                p["proj"] = "50"
                if p["scen"] != "historical":
                    p["proj"] = st.selectbox("Percentile de projection du niveau de la mer", ["5", "50", "95"], index=1, key="cs_proj",
                                             format_func=lambda x: {"5": "5 % (bas)", "50": "50 % (médian)", "95": "95 % (haut)"}[x])
            params[k] = p
    with st.expander("Diagnostic des sources (à lancer en cas de problème)"):
        if "debordement" in sel:
            st.markdown("**JRC / GloFAS (débordement)**")
            rp_t = st.selectbox("Période de retour testée", list(fhz.RPS), index=4, key="fh_diag_rp", format_func=_rp_fmt)
            if st.button("Tester l'accès aux tuiles JRC", key="fh_diag_go"):
                with st.spinner("Contrôle du catalogue, des noms de fichiers et du téléchargement…"):
                    st.session_state["fh_diag"] = fhz.diagnose(int(rp_t))
            if st.session_state.get("fh_diag"):
                _stretch(st.dataframe, pd.DataFrame(st.session_state["fh_diag"]), hide_index=True)
        if "submersion" in sel:
            st.markdown("**WRI Aqueduct via Earth Engine (submersion marine)**")
            rp_c = st.selectbox("Période de retour testée", list(coast.RPS), index=5, key="cs_diag_rp", format_func=_rp_fmt)
            if st.button("Tester l'accès à Earth Engine", key="cs_diag_go"):
                pc = params["submersion"]
                ee_p, ee_a = _ee_credentials()
                with st.spinner("Secrets, authentification, collection et lecture de points de contrôle…"):
                    st.session_state["cs_diag"] = coast.diagnose_ee(ee_p, ee_a, pc["scen"], pc["sub"], pc["year"], int(rp_c), pc["proj"])
            if st.session_state.get("cs_diag"):
                _stretch(st.dataframe, pd.DataFrame(st.session_state["cs_diag"]), hide_index=True)
        st.caption("En cas d'ÉCHEC, copiez le tableau (il ne contient jamais la clé) : il suffit à corriger le réglage ou le format de lecture.")
    active = [k for k in sel if params[k]["rps"]]
    if not active:
        st.caption("Choisissez au moins une période de retour.")
        return
    catalog = None
    if "debordement" in active:
        try:
            with st.spinner("Lecture du catalogue des tuiles JRC (la première fois, jusqu'à une minute si le serveur est lent)…"):
                catalog = _fh_catalog()
        except fhz.TileError as e:
            st.error(f"Catalogue des tuiles JRC indisponible : {e} (le débordement est ignoré).")
            active.remove("debordement")
            if not active:
                return
    notes = [f"{len(df)} point(s) lus."]
    if "debordement" in active:
        pf = params["debordement"]
        pl = fhz.plan(df, lat_c, lon_c, catalog)
        n_t = len(pl["tile_ids"])
        mb = n_t * len(pf["rps"]) * ((30 if "raw" in pf["kinds"] else 0) + (3 if "reclass" in pf["kinds"] else 0))
        notes.append(f"Débordement : {n_t} tuile(s), {n_t * len(pf['rps']) * len(pf['kinds'])} fichier(s), au plus ≈ {mb} Mo la première fois.")
    if "submersion" in active:
        notes.append(f"Submersion : {len(params['submersion']['rps'])} période(s) de retour lue(s) dans Earth Engine (aucun téléchargement).")
    st.caption(" ".join(notes))
    b1, b2, _sp = st.columns([1, 1, 2])
    go = b1.button("Estimer la hauteur d'eau", type="primary", key="ex_go")
    if b2.button("Estimer les dommages", type="primary", key="ex_dmg_go"):
        st.session_state["ex_dmg_open"] = True

    def run_hazard(k, force_raw=False):
        p = params[k]
        bar = st.progress(0.0)
        prog = lambda i, n: bar.progress(min(1.0, i / max(1, n)))  # noqa: E731
        with st.spinner(f"{_HAZARDS[k]['label']} : téléchargement et lecture…"):
            if k == "debordement":
                kinds_run = sorted(set(p["kinds"]) | ({"raw"} if force_raw else set()))
                res_, summ_ = fhz.estimate_depths(df, lat_c, lon_c, p["rps"], kinds_run, catalog, on_progress=prog)
            else:
                kinds_run = ["raw"]
                ee_p, ee_a = _ee_credentials()
                try:
                    res_, summ_ = coast.estimate_depths_ee(df, lat_c, lon_c, p["rps"], p["scen"], p["sub"], p["year"], projection=p["proj"], radius_km=p["radius"],
                                                           ee_project=ee_p, ee_account=ee_a, on_progress=prog)
                except coast.TileError as e:
                    bar.empty()
                    st.error(f"Submersion marine : {e}  (utilisez « Diagnostic des sources » ci-dessus pour localiser le problème).")
                    return
        bar.empty()
        res_["lat_f"] = pd.to_numeric(res_[lat_c], errors="coerce")
        res_["lon_f"] = pd.to_numeric(res_[lon_c], errors="coerce")
        st.session_state[_HAZARDS[k]["result"]] = (name, len(df), res_, summ_, list(p["rps"]), list(kinds_run))
        st.session_state.pop("ex_dmg_result", None)               # les dommages dépendent des hauteurs : à recalculer

    def ensure(names):
        for k in names:
            run_hazard(k, force_raw=True)

    if go:
        for k in active:
            run_hazard(k)
    hazards = [{"name": k, "label": _HAZARDS[k]["label"], "rps": params[k]["rps"], "col": _HAZARDS[k]["col"], "result": _HAZARDS[k]["result"]}
               for k in active]
    dmg_stored = _fh_damage_panel(df, name, hazards, ensure)

    have = {}
    for k in active:
        r = st.session_state.get(_HAZARDS[k]["result"])
        if r and r[0] == name and r[1] == len(df):
            have[k] = r
    if not have:
        return
    merged = None
    cols_by_kind: dict[str, list] = {}
    depth_cols_all: list[str] = []
    status_cols: list[str] = []
    for k, (_, _, res_k, summ_k, rps_k, kinds_k) in have.items():
        new = [c for c in res_k.columns if merged is None or c not in merged.columns]
        merged = res_k.copy() if merged is None else pd.concat([merged, res_k[new]], axis=1)
        if k == "debordement":
            for kind in kinds_k:
                cs = [fhz.column_name(r, kind) for r in rps_k]
                cols_by_kind.setdefault(kind, []).extend(cs)
                depth_cols_all += cs
            status_cols += ["debordement_tuile", "debordement_statut"]
        else:
            cs = [coast.column_name(r) for r in rps_k]
            cols_by_kind.setdefault("raw", []).extend(cs)
            depth_cols_all += cs
            status_cols.append("submersion_statut")
    with st.expander("Détail par aléa", expanded=False):
        for k, (_, _, res_k, summ_k, rps_k, kinds_k) in have.items():
            st.markdown(f"**{_HAZARDS[k]['label']}**")
            m = st.columns(4)
            m[0].metric("Points", summ_k["points"])
            if k == "debordement":
                m[1].metric("Dans une tuile", summ_k["points"] - summ_k["hors_tuile"])
                m[2].metric("Hors tuile", summ_k["hors_tuile"])
                m[3].metric("Points partiels", int(res_k["debordement_statut"].str.startswith("partiel").sum()))
                for kk, v in list(summ_k["problemes"].items())[:6]:
                    st.warning(f"{kk} : {v[0]}")
            else:
                m[1].metric("Coordonnées invalides", summ_k["invalides"])
                m[2].metric("Hors raster", summ_k.get("hors_raster", 0))
                m[3].metric("Scénarios en échec", len(summ_k["problemes"]))
                pos = summ_k.get("points_positifs", {})
                if pos:
                    st.caption("Points avec hauteur > 0 : " + " ; ".join(f"{k} : {v}" for k, v in pos.items()))
                for kk, v in list(summ_k["problemes"].items())[:6]:
                    st.warning(f"{kk} : {v}")
    damage = None
    color_opts = list(depth_cols_all)
    if dmg_stored is not None:
        kinds_d: dict[str, str] = {}
        pam_cols = []
        for k, (out_h, _s) in dmg_stored[2].items():
            if k not in have:
                continue
            new = [c for c in out_h.columns if c not in merged.columns]
            merged = pd.concat([merged, out_h[new]], axis=1)
            for c in out_h.columns:
                if c.startswith(f"dommage_{k}_RP") or c == f"dommage_{k}_PAM_eur":
                    kinds_d[c] = "ratio" if c.endswith("_ratio") else "eur"
            if f"dommage_{k}_PAM_eur" in out_h.columns:
                pam_cols.append(f"dommage_{k}_PAM_eur")
        if len(pam_cols) >= 2:
            merged["dommage_total_PAM_eur"] = merged[pam_cols].sum(axis=1, min_count=1)
            kinds_d["dommage_total_PAM_eur"] = "eur"
            _kpi(st, "Perte annuelle moyenne cumulée, tous aléas (approx.)", _eur(merged["dommage_total_PAM_eur"].sum()))
        damage = {"kinds": kinds_d, "edges": {c: dmg.class_edges(merged[c], kd) for c, kd in kinds_d.items()}}
        color_opts += list(kinds_d)
    flood_cols = [c for kd in ("raw", "reclass") for c in cols_by_kind.get(kd, []) if c.startswith("debordement_")]
    c_a, c_b, c_c = st.columns([2, 1, 1])
    color_col = c_a.selectbox("Colorer la carte selon", color_opts, key="ex_color")
    is_flood_col = color_col in flood_cols
    overlays = []
    show_tiles = False
    if "debordement" in have:
        summ = have["debordement"][3]
        show_tiles = c_b.checkbox("Afficher les tuiles extraites", value=False, key="ex_show_tiles")
        show_vals = c_c.checkbox("… avec leurs valeurs (raster)", value=False, key="ex_show_vals", disabled=not (show_tiles and is_flood_col),
                                 help="Colore chaque pixel de la tuile selon la légende, pour une colonne de débordement.")
        if show_tiles and show_vals and not is_flood_col:
            st.caption("Le raster des tuiles représente des hauteurs d'eau du débordement : choisissez une colonne de ce type.")
        if show_tiles and show_vals and is_flood_col:
            kind_sel = "reclass" if color_col.endswith("_reclass") else "raw"
            ok_tiles = [t for t in summ.get("tuiles_detail", []) if t["statut"] == "ok" or color_col in
                        [kk.split("|")[0] for kk in summ.get("fichiers_lus", {}) if kk.endswith(f"|{t['id']}")]]
            for t in ok_tiles[:6]:
                pth = summ.get("fichiers_lus", {}).get(f"{color_col}|{t['id']}")
                if not pth or not Path(pth).exists():
                    st.caption(f"Tuile ID{t['id']} : fichier absent du cache (cache vidé ?) ; relancez le calcul pour l'afficher.")
                    continue
                try:
                    with st.spinner(f"Image de la tuile ID{t['id']}…"):
                        overlays.append(_fh_preview(pth, kind_sel, Path(pth).stat().st_size))
                except fhz.TileError as e:
                    st.warning(f"Tuile ID{t['id']} : {e}")
            if len(ok_tiles) > 6:
                st.caption(f"Valeurs affichées pour 6 tuiles sur {len(ok_tiles)} (limite d'affichage de la carte).")
    cells = None
    if "submersion" in have and color_col.startswith("submersion_RP"):
        if st.checkbox("Afficher les zones de submersion à 5 km autour des points", value=False, key="ex_show_cells"):
            img_id = have["submersion"][3].get("fichiers_lus", {}).get(color_col)
            res_c = have["submersion"][2]
            cache_c = st.session_state.setdefault("cs_cells", {})
            ckey = (color_col, name, len(df), str(img_id))
            if not img_id:
                st.caption("Image de submersion non lue pour cette période de retour : rien à afficher.")
            else:
                if ckey not in cache_c:
                    try:
                        ok_c = res_c.dropna(subset=["lat_f", "lon_f"])
                        with st.spinner("Lecture des pixels de submersion dans un rayon de 5 km (Earth Engine)…"):
                            cache_c[ckey] = coast.flood_cells(img_id, ok_c["lon_f"].to_numpy(float), ok_c["lat_f"].to_numpy(float), 5.0)
                    except coast.TileError as e:
                        st.error(f"Pixels de submersion indisponibles : {e}")
                cells = cache_c.get(ckey)
                if cells is not None:
                    st.caption(f"{len(cells)} pixel(s) de ≈ 1 km avec hauteur > 0 dans un rayon de 5 km autour des points, pour {color_col}."
                               + (" Affichage limité aux 6000 premiers." if len(cells) > 6000 else "")
                               + (" Aucune submersion modélisée dans ce rayon." if not cells else ""))
    tiles = have["debordement"][3].get("tuiles_detail") if (show_tiles and "debordement" in have) else None
    _fh_map(merged, cols_by_kind, color_col, tiles, overlays, damage, cells)
    out = merged.drop(columns=["lat_f", "lon_f"])
    exposed = (out[depth_cols_all].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1)
    dmg_cols_out = [c for c in out.columns if c.startswith("dommage_")]
    _cross_table(out, depth_cols_all + dmg_cols_out + status_cols, exposed, "ex_tbl", "Seulement les points exposés")
    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button("Télécharger le résultat (CSV)", _csv_bytes(out), "catmap_exposition.csv", "text/csv")
    with d2:
        n_c, mb_c = fhz.cache_info()
        if st.button(f"Vider le cache et le CSV ({n_c} fichier(s) en cache, {mb_c} Mo)", key="ex_clear_cache", type="primary"):
            n_del, mb_del = fhz.clear_cache()
            st.session_state["ex_up_n"] = st.session_state.get("ex_up_n", 0) + 1
            for kk in ("fh_result", "cs_result", "fh_diag", "cs_diag", "ex_color", "ex_show_tiles", "ex_show_vals", "ex_dmg_result", "ex_dmg_open", "cs_cells"):
                st.session_state.pop(kk, None)
            st.session_state["ex_cache_msg"] = (f"Cache vidé : {n_del} fichier(s), {mb_del} Mo libérés. Le fichier CSV chargé et les résultats "
                                                "ont aussi été retirés ; les fichiers seront retéléchargés au prochain calcul.")
            st.rerun()
        if st.session_state.get("ex_cache_msg"):
            st.caption(st.session_state.pop("ex_cache_msg"))
    st.caption("0 = aucun aléa modélisé à ce point (pas une garantie d'absence de risque). Débordement : JRC / Copernicus EMS GloFAS, « Global "
               "river flood hazard maps » v2.1.2, licence ODbL v1.0, cours d'eau de bassin inférieur à 500 km² non modélisés. Submersion : WRI "
               "Aqueduct Floods v2 via Google Earth Engine (utilisation non commerciale au regard de l'abonnement), environ 1 km, niveau extrême "
               "d'eau sans vagues ni protections locales. Vérifiez les conditions de réutilisation."
               )

def _runoff_map(res, col):
    import folium
    pts = res.dropna(subset=["lat_f", "lon_f"])
    pts = pts[pd.to_numeric(pts[col], errors="coerce").notna()]
    center = (float(pts["lat_f"].mean()), float(pts["lon_f"].mean())) if len(pts) else (46.6, 2.5)
    m = _ev_base_map(center, 11 if len(pts) == 1 else 6)
    shown = pts.head(3000)
    for _, r in shown.iterrows():
        v = float(r[col])
        if col == rs.COLUMNS["classe"]:
            cls, tip = int(v), f"{r.get('name') or ''} : {rs.LEVELS[int(v)]}"
        elif col == rs.COLUMNS["irip"]:
            cls = 1 + min(4, int(v))
            tip = f"{r.get('name') or ''} : score {v:.1f} / 5" + (" (≥ 3)" if v >= 3 else "")
        else:
            cls = 1 + min(4, int(v * 5))
            tip = f"{r.get('name') or ''} : {rs.LEVELS[cls]} ({v:.2f})"
        folium.CircleMarker([r["lat_f"], r["lon_f"]], radius=7, color="#222", weight=1, fill=True, fill_color=rs.COLORS[cls], fill_opacity=0.9,
                            tooltip=html_escape(tip)).add_to(m)
    if len(shown) > 1:
        m.fit_bounds([[shown["lat_f"].min(), shown["lon_f"].min()], [shown["lat_f"].max(), shown["lon_f"].max()]])
    _ev_show_map(m, "rs_map")
    legend = "".join(f'<span style="display:inline-block;width:.8rem;height:.8rem;background:{rs.COLORS[k]};border:1px solid #222;margin:0 .3rem 0 .8rem">'
                     f'</span>{rs.LEVELS[k]}' for k in sorted(rs.COLORS))
    st.markdown(f"<div style='font-size:.8rem'>Classe de susceptibilité :{legend}</div>", unsafe_allow_html=True)
    if len(pts) > 3000:
        st.caption(f"Carte limitée aux 3000 premiers points sur {len(pts)} ; le tableau et le CSV contiennent tous les points.")


def _select_help(label: str, help_text: str, options, left: bool = False, **kw):
    """``selectbox`` précédée d'une ligne de titre avec infobulle « ? » au survol."""
    _help_line(html.escape(label), help_text, wide=True, left=left, label_style="font-size:.875rem")
    return st.selectbox(label, options, label_visibility="collapsed", **kw)


def _ruissellement_section():
    st.subheader("Ruissellement : susceptibilité par point")
    st.markdown("Indice de **susceptibilité** au ruissellement pluvial de chaque point, croisant trois cartes de même poids : **production**, "
                "**transfert** et **accumulation**, calculées sur la topographie mondiale (≈ 90 m), l'occupation du sol et le sol. C'est un indice "
                "qualitatif, sans pluie ni hauteur d'eau (étapes suivantes).")
    _help_line("Méthode, hypothèses et références",
               "**Simplification de la méthode IRIP** (Indicateur de Ruissellement Intense Pluvial, Irstea / INRAE) : trois cartes (production, transfert, "
               "accumulation) de facteurs notés 0 (défavorable) ou 1 (favorable). L'IRIP original compte 5 facteurs par carte (dont épaisseur et "
               "érodabilité du sol, compacité, axes linéaires) que CatMap ne reprend pas ; il ne fixe pas de seuils numériques universels : les seuils de "
               "pente et d'indice topographique sont calculés pour chaque zone d'étude par une classification en 2 classes (Braud et al., 2020).\n\n"
               "**Données mondiales libres lues dans Earth Engine** : altitude et aire drainée amont (MERIT Hydro, 90 m) ; occupation du sol (ESA WorldCover, "
               "10 m) ; sol au choix (GCN250, HiHydroSoil) ; géomorphons (Geomorpho90m) en option ; bassins HydroBASINS.\n\n"
               "**Seuils par zone** : pour chaque zone (bassin ou fenêtre), CatMap sépare les pixels en 2 classes en minimisant la variance intra-classe "
               "(en 1 dimension, même optimum qu'un k-means à 2 classes) ; le seuil est la frontière des deux classes. Cela s'applique à la pente, à "
               "l'indice topographique ln(a/tan β), au logarithme de l'aire drainée, à l'amplitude de courbure du profil (rupture de pente) et, "
               "selon l'option, au numéro de courbe. Une zone trop petite ou quasi uniforme n'a pas de seuil : aucun facteur « fort » n'y est retenu.\n\n"
               "**Facteurs** : *production* = topographie (pente > T1 ou indice topographique > T2), sol cultivé ou bâti, sol peu perméable ; *transfert* = "
               "production amont, pente forte, rupture de pente convexe, aire drainée forte ; *accumulation* = production amont, aire drainée forte, pente "
               "faible, rupture de pente concave, indice topographique fort (ou, en option, score de géomorphons). Production amont ≈ production du pixel "
               "du point (≥ 0,6) : l'amont n'est pas agrégé. Chaque carte est ramenée à [0, 1] (facteurs favorables / disponibles) ; la susceptibilité "
               "globale est la **moyenne des trois cartes à poids égaux**, en 5 classes égales de 0,2, et en **score équivalent IRIP de 0 à 5** "
               "(= globale × 5, repère « ≥ 3 »). Ce score n'est pas le score officiel de l'IRIP : il en est une approximation.\n\n"
               "**Limites** : seuils relatifs à la zone (un terrain plat partout n'a pas de pixel « fort » ; deux zones peuvent donner des seuils très "
               "différents) ; MNT de 90 m qui ne voit ni le bâti ni le réseau pluvial ; points en mer ou hors 60°S–90°N sans valeur ; sources des jeux "
               "à vérifier par le diagnostic. L'IRIP est qualitatif : ni débit, ni hauteur d'eau.\n\n"
               "**Conditions** : accès à Earth Engine en utilisation non commerciale au regard de l'abonnement du projet (revérification périodique "
               "demandée par Google).\n\n" + refs_md("irip", "irip_braud", "rubin", "twi", "merit", "worldcover", "zt1987", "gee"),
               wide=True, label_style="font-size:.875rem")
    df, lat_c, lon_c, name = _points_input("rs_")
    if df is None:
        return
    kind = st.radio("Zone d'analyse de chaque point (où sont calculés les seuils)", ["basin", "window"], horizontal=True, key="rs_zone",
                    format_func=lambda k: {"basin": "Bassin versant (unité HydroBASINS)", "window": "Fenêtre locale autour du point"}[k])
    if kind == "basin":
        level = int(_select_help(
            "Niveau de bassin HydroBASINS",
            "**HydroBASINS** découpe les continents en unités de drainage emboîtées, du niveau 1 (très grandes) au niveau 12 (les plus petites). Plus le niveau est élevé, plus la zone est locale : les seuils reflètent le relief et les sols "
            "voisins ; plus il est bas, plus la zone est grande et hétérogène (plaine et montagne dans la même zone), donc des seuils moins "
            "discriminants. Défaut : 11. CatMap propose 9 à 12.\n\n"
            "**Approximation** : le bassin versant d'un point est l'ensemble de la surface qui draine vers ce point. CatMap le remplace par "
            "l'**unité de drainage HydroBASINS qui contient le point**. Elle peut différer : un point au milieu d'une unité n'en draine qu'une partie ; "
            "un point en bordure peut recevoir de l'eau d'unités amont non comprises ; une unité peut inclure du terrain qui ne draine pas vers le "
            "point. Les seuils décrivent donc le contexte de l'unité, pas l'amont exact. Les unités sont délimitées pour de grands cours d'eau : "
            "en zone très plate ou urbaine, la limite peut être imprécise.\n\n" + refs_md("hydrobasins"),
            list(rs.BASIN_LEVELS), index=list(rs.BASIN_LEVELS).index(rs.DEFAULT_LEVEL), key="rs_level"))
        zone = {"kind": "basin", "level": level}
    else:
        radius = float(_select_help(
            "Rayon de la fenêtre locale (km)",
            "Disque centré sur le point : les seuils sont calculés sur les pixels de ce disque. 1 ou 2 km : très local (seuils sensibles à un "
            "petit relief) ; 5 ou 10 km : contexte plus large. Le disque ne suit pas la topographie : contrairement au bassin, il mélange des "
            "pixels de versants voisins. Calcul plus long que par bassin quand les points sont nombreux et dispersés, car chaque point a sa fenêtre.\n\n"
            + refs_md("irip_braud"),
            list(rs.WINDOW_KM), index=1, key="rs_radius", format_func=lambda x: f"{x:g} km"))
        zone = {"kind": "window", "radius_km": radius}
    c1, c2, c3 = st.columns(3)
    with c1:
        area = _select_help(
            "Seuil d'aire drainée",
            "L'aire drainée amont (MERIT Hydro) repère les pixels où l'écoulement se concentre. **Local** (défaut) : seuil de log10(aire) calculé par "
            "classification en 2 classes dans la zone. **Fixe** : 2,5 ha (0,025 km²) partout, valeur d'application de l'IRIP citée par Braud et al. "
            "(2020) et rattaché à Gally (2018). Sur une "
            "grille de 90 m, un pixel fait 0,81 ha : le seuil de 2,5 ha représente environ 3 pixels, ce qui le rend peu discriminant.\n\n"
            + refs_md("irip_braud", "gally"),
            list(rs.AREA_METHODS), key="rs_area", format_func=lambda k: rs.AREA_METHODS[k])
    with c2:
        soil = _select_help(
            "Facteur sol",
            "**Numéro de courbe** (défaut) : GCN250 (SCS-CN, 250 m, combine occupation du sol et groupe hydrologique) ; favorable si supérieur au seuil "
            "de la zone (classification en 2 classes), comme la méthode statistique par bassin de l'IRIP. **Groupe hydrologique** : HiHydroSoil v2.0 "
            "(250 m) ; favorable si groupe D ou double groupe (A/D, B/D, C/D : sol peu perméable à l'état naturel), sans seuil à calculer. "
            "**Ks** : conductivité hydraulique à saturation de HiHydroSoil v2.0 ; favorable si inférieure à 1e-6 m/s (8,64 cm/j), seuil d'application "
            "de Braud et al. (2020). Les jeux sont à 250 m : un pixel de sol couvre une trentaine de pixels de relief.\n\n"
            + refs_md("gcn250", "hihydrosoil", "irip_braud"),
            list(rs.SOIL_METHODS), key="rs_soil", format_func=lambda k: rs.SOIL_METHODS[k])
    with c3:
        accum = _select_help(
            "Carte d'accumulation",
            "**Classique** : production amont, aire drainée, pente faible, rupture de pente concave, indice topographique de Beven et Kirkby. "
            "**Géomorphons** : Barjot, Breil et Braud (2023) remplacent pente, rupture de pente et indice topographique par un indicateur unique "
            "de forme du relief (10 classes, Geomorpho90m, 90 m), noté 0 à 3 : crête, pic, pente = 0 ; plat, éperon, épaulement = 1 ; creux, "
            "bas de versant = 2 ; vallée, fosse = 3. CatMap pondère ce score ×3 contre production amont et aire drainée (poids 1 chacune). "
            "La distance de recherche des géomorphons est celle du jeu Geomorpho90m, non réglable ici.\n\n"
            + refs_md("barjot", "geomorphons", "amatulli"),
            list(rs.ACCUM_METHODS), left=True, key="rs_accum", format_func=lambda k: rs.ACCUM_METHODS[k])
    st.caption(f"{len(df)} point(s) lus. Les points voisins du même pixel sont calculés une fois ; par bassin, les seuils d'une unité sont calculés une "
               "seule fois pour tous ses points. Pour des centaines de milliers de points, comptez de longues minutes ; le niveau 11 est le plus "
               "économe en calcul avec des points regroupés.")
    with st.expander("Diagnostic des sources (à lancer en cas de problème)"):
        if st.button("Tester Earth Engine et les jeux de données", key="rs_diag_go"):
            ee_p, ee_a = _ee_credentials()
            with st.spinner("Authentification, jeux de données et lieux de contrôle…"):
                st.session_state["rs_diag"] = rs.diagnose(ee_p, ee_a, zone, soil, accum, area)
        if st.session_state.get("rs_diag"):
            _stretch(st.dataframe, pd.DataFrame(st.session_state["rs_diag"]), hide_index=True)
        st.caption("Le tableau n'affiche jamais la clé du compte de service.")
    if st.button("Estimer la susceptibilité", type="primary", key="rs_go"):
        ee_p, ee_a = _ee_credentials()
        bar = st.progress(0.0)
        try:
            with st.spinner("Lecture des cartes dans Earth Engine…"):
                res_, summ_ = rs.estimate(df, lat_c, lon_c, zone, ee_p, ee_a, on_progress=lambda i, n: bar.progress(min(1.0, i / max(1, n))),
                                          soil=soil, accum=accum, area=area)
            st.session_state["rs_result"] = (name, len(df), res_, summ_)
        except (rs.TileError, ValueError) as e:
            st.error(f"Ruissellement : {e}  (utilisez « Diagnostic des sources » pour localiser le problème).")
        bar.empty()
    got = st.session_state.get("rs_result")
    if not got or got[0] != name or got[1] != len(df):
        return
    _, _, res_, summ_ = got
    m = st.columns(5)
    m[0].metric("Points", summ_["points"])
    m[1].metric("Pixels calculés", summ_["pixels_uniques"])
    m[2].metric("Zones distinctes", summ_.get("zones", 0))
    m[3].metric("Hors terre / couverture", summ_["hors_terre"])
    m[4].metric("Lots en échec", summ_["echecs"])
    if summ_.get("jeux_absents"):
        st.warning("Jeu(x) de données illisible(s), facteur(s) écarté(s) du score : " + ", ".join(summ_["jeux_absents"]))
    if summ_["repartition"]:
        st.caption("Répartition des classes : " + " ; ".join(f"{k} : {v}" for k, v in summ_["repartition"].items()))
    res_ = res_.copy()
    res_["lat_f"] = pd.to_numeric(res_[lat_c], errors="coerce")
    res_["lon_f"] = pd.to_numeric(res_[lon_c], errors="coerce")
    labels = {"classe": "Classe de susceptibilité (1 à 5)", "globale": "Susceptibilité globale (0 à 1)", "irip": "Score équivalent IRIP (0 à 5)",
              "prod": "Production", "transfert": "Transfert", "accum": "Accumulation"}
    pick = st.selectbox("Colorer la carte selon", list(labels), key="rs_color", format_func=lambda k: labels[k])
    _runoff_map(res_, rs.COLUMNS[pick])
    out = res_.drop(columns=["lat_f", "lon_f"])
    new_cols = list(rs.COLUMNS.values()) + list(rs.ZONE.values()) + list(rs.RAW.values()) + [rs.STATUS]
    exposed = pd.to_numeric(out[rs.COLUMNS["classe"]], errors="coerce") >= 4
    _cross_table(out, new_cols, exposed, "rs_tbl", "Seulement les classes forte et très forte")
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(out), "catmap_ruissellement.csv", "text/csv", key="rs_dl")
    st.caption("Indice qualitatif de susceptibilité, relatif à la zone d'analyse (bassin approché par l'unité de drainage HydroBASINS, ou fenêtre), pas une "
               "probabilité ni une hauteur d'eau. Sources : MERIT Hydro, ESA WorldCover, GCN250, HiHydroSoil, Geomorpho90m, HydroBASINS via Google Earth "
               "Engine (utilisation non commerciale au regard de l'abonnement). Vérifiez leurs conditions de réutilisation.")


def _wind_catalogue_reader():
    """Lecture facultative des descriptions du catalogue Copernicus (aucune clé requise)."""
    if st.button("Lire le catalogue Copernicus", key="cop_form_go", type="primary"):
        res = {}
        with st.spinner("Lecture des descriptions…"):
            for key, ds in cop.DATASETS.items():
                try:
                    res[key] = {"form": cop.fetch_form(ds)}
                except cop.CopernicusError as e:
                    res[key] = {"error": str(e)}
        st.session_state["cop_forms"] = res
    titles = {"timeseries": "ERA5 horaire en série temporelle (vent : toutes rafales, monde)",
              "storms": "Tempêtes européennes : trajectoires et empreintes de rafales"}
    for key, item in (st.session_state.get("cop_forms") or {}).items():
        st.markdown(f"**{titles.get(key, key)}** (`{cop.DATASETS[key]}`)")
        if "error" in item:
            st.error(f"Lecture impossible : {item['error']}")
            continue
        rows = cop.summarize_form(item["form"])
        if rows:
            _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
        else:
            st.warning("Structure de formulaire non reconnue : téléchargez le JSON brut ci-dessous et envoyez-le.")
        st.download_button("Télécharger la description brute (JSON)", cop.pretty(item["form"]).encode("utf-8"),
                           f"copernicus_{key}_form.json", "application/json", key=f"cop_dl_{key}")


def _wind_map(res, col, cls_fn=None, palette=None, labels=None, popup_suffix="_gust", key="wd_map"):
    import folium
    pts = res.dropna(subset=["lat_f", "lon_f"])
    center = (float(pts["lat_f"].mean()), float(pts["lon_f"].mean())) if len(pts) else (46.6, 2.5)
    m = _ev_base_map(center, 5)
    shown = pts.head(3000)
    is_dmg = col.startswith("dommage_")
    kind = "ratio" if col.endswith("_ratio") else "eur"
    edges = dmg.class_edges(res[col], kind) if is_dmg else None
    cls_fn = cls_fn or wind.gust_class
    pal = dmg.DMG_COLORS if is_dmg else (palette or wind.GUST_COLORS)
    for _, r in shown.iterrows():
        v = pd.to_numeric(r.get(col), errors="coerce")
        cls = dmg.damage_class(v, edges) if is_dmg else cls_fn(v)
        txt = "".join(f"<div>{html_escape(c)} : {html_escape(r.get(c))}</div>" for c in [col] + [k for k in res.columns if k.endswith(popup_suffix)][:8])
        folium.CircleMarker([r["lat_f"], r["lon_f"]], radius=7, color="#222", weight=1, fill=True, fill_color=pal[cls], fill_opacity=0.9,
                            tooltip=str(r.get("name") or ""),
                            popup=folium.Popup(f"<b>{html_escape(r.get('name') or '')}</b>{txt}", max_width=420)).add_to(m)
    if len(shown) > 1:
        m.fit_bounds([[shown["lat_f"].min(), shown["lon_f"].min()], [shown["lat_f"].max(), shown["lon_f"].max()]])
    labels = dmg.legend_labels(edges, kind) if is_dmg else (labels or wind.GUST_LABELS)
    legend = "".join(f'<div><span style="background:{pal[k]};display:inline-block;width:12px;height:12px;border:1px solid #444;margin-right:6px">'
                     f'</span>{lab}</div>' for k, lab in labels.items())
    legend += (f'<div><span style="background:{pal[None]};display:inline-block;width:12px;height:12px;border:1px solid #444;margin-right:6px">'
               f'</span>Pas de donnée</div>')
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;color:#222;padding:8px 10px;border:1px solid #888;'
        f'border-radius:4px;font-size:12px"><b>{html_escape(col)}</b>{legend}</div>'))
    _ev_show_map(m, key)


_WIND_HAZ = {"vent": "Vent (toutes rafales, monde)", "tempete": "Tempête (événements européens)"}


def _wind_compute(df, lat_c, lon_c, name, sel, rps, y0, y1, algo, max_cells, budget):
    key = cop.get_key(_secrets(), dict(__import__("os").environ))
    client = wind.make_client(key)
    pc = wind.cells_of_points(df[lat_c], df[lon_c])
    cells = list(dict.fromkeys(c for c in pc if c is not None))
    if len(cells) > max_cells:
        st.warning(f"{len(cells)} cellules ERA5 distinctes : seules les {max_cells} premières sont traitées (augmentez la limite ci-dessus).")
        cells = cells[:max_cells]
    bar = st.progress(0.0)
    with st.spinner("Requêtes Copernicus (file d'attente possible, reprise automatique au prochain clic)…"):
        got = wind.ensure_cells(cells, y0, y1, client, budget_s=budget, on_progress=lambda i, n: bar.progress(min(1.0, i / max(1, n))))
    bar.empty()
    daily = {c: wind.load_daily(wind.cell_path(c, y0, y1)) for c in got["done"]}
    out = df.copy().reset_index(drop=True)
    out["lat_f"] = pd.to_numeric(out[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["lon_f"] = pd.to_numeric(out[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    out["vent_cellule_era5"] = [f"{c[0]:+.2f};{c[1]:+.2f}" if c else None for c in pc]
    info = {"cells": len(cells), "done": len(got["done"]), "pending": len(got["pending"]), "queue_full": got.get("queue_full", False), "errors": {f"{c[0]},{c[1]}": m for c, m in got["errors"].items()}}
    storms = peaks = None
    if "vent" in sel and daily:
        lev = wind.compute_vent(daily, rps)
        for r in rps:
            out[wind.column_name("vent", r)] = [lev.get(c, {}).get(r) if c else None for c in pc]
    if "tempete" in sel and daily:
        try:
            with st.spinner("Trajectoires de tempêtes…"):
                tracks = wind.fetch_tracks(client, algo, y0, y1)
            info["tracks"] = wind.tracks_summary(tracks)
            storms = wind.extract_storms(tracks, list(cells))
            if storms.empty:
                info["tempete_msg"] = "Aucune tempête ne passe à proximité des points (ou aucune dans le domaine 25°O–35°E, 30–70°N)."
            else:
                lev, peaks = wind.compute_tempete(storms, daily, y1 - y0 + 1, rps)
                for r in rps:
                    out[wind.column_name("tempete", r)] = [lev.get(c, {}).get(r) if c else None for c in pc]
                info["n_storms"] = len(storms)
        except wind.WindError as e:
            info["tempete_msg"] = f"Tempêtes : {e}"
    fitdata = {}
    for c in daily:
        ann = wind.annual_maxima(daily[c])
        fitdata[c] = {"annual": ann, "peaks": (peaks or {}).get(c)}
    st.session_state["wd_result"] = {"name": name, "n": len(df), "df": out, "rps": list(rps), "sel": list(sel), "cells": pc, "storms": storms,
                                     "peaks": peaks, "info": info, "y": (y0, y1), "fitdata": fitdata, "n_years": y1 - y0 + 1}
    st.session_state.pop("wd_fit_table", None)
    st.session_state.pop("wd_dmg", None)


def _wind_damage(res, df_cols, key_ok_jrc):
    """Dommages vent : même fonction de dommage (Emanuel) pour le vent et la tempête ; perte par tempête pour le classement."""
    base_cols = [c for c in df_cols if not c.endswith("_gust") and not c.startswith(("dommage_", "vent_")) and c not in ("lat_f", "lon_f")]
    out_df = res["df"]
    with st.container(border=True):
        _help_line("Fonction de dommage : méthode et référence", "**Emanuel (2011)** : taux de dommage = x³ / (1 + x³), avec x = (v − v_seuil) / "
                   "(v_mi − v_seuil) pour v > v_seuil ; calée à l'origine sur les cyclones américains, ici utilisée comme forme générique pour la "
                   "rafale ; paramètres à recaler pour l'Europe.\n\n" + refs_md("emanuel"), wide=True, label_style="font-size:.875rem")
        st.markdown("**Estimation des dommages vent** : perte = taux(rafale) × valeur exposée, avec la fonction d'Emanuel (2011) "
                    "`x³/(1+x³)`. Même courbe pour le vent et la tempête ; seuls changent les niveaux de rafale par période de retour.")
        c1, c2, c3 = st.columns(3)
        v_t = _num_help(c1, "Seuil de dommage (m/s)", "Rafale en dessous de laquelle le taux de dommage est nul. Valeur par défaut = calibration "
                        "CLIMADA États-Unis pour les cyclones ; à ajuster pour l'Europe.", min_value=5.0, max_value=60.0, value=25.7, step=0.5, key="wd_vt")
        v_h = _num_help(c2, "Rafale de mi-dommage (m/s)", "Rafale pour laquelle 50 % de la valeur exposée est détruite (fonction d'Emanuel).",
                        min_value=20.0, max_value=150.0, value=74.7, step=0.5, key="wd_vh")
        default_cls = c3.selectbox("Type d'actif par défaut", list(dmg.CLASSES), key="wd_cls", index=list(dmg.CLASSES).index("commercial"), format_func=lambda k: dmg.CLASS_LABELS[k])
        d1, d2 = st.columns(2)
        type_col = d1.selectbox("Colonne du type d'actif (facultatif)", ["(aucune)"] + base_cols, key="wd_typecol",
                                index=_guess_col(base_cols, ("type", "categ", "activ", "naf")))
        country_col = d2.selectbox("Colonne pays (ISO ou nom anglais)", ["(aucune)"] + base_cols, key="wd_ctry",
                                   index=_guess_col(base_cols, ("pays_iso2", "pays_iso", "iso2", "iso3", "country", "pays")))
        type_col = None if type_col == "(aucune)" else type_col
        country_col = None if country_col == "(aucune)" else country_col
        jrc, jerr = _dmg_jrc_local()
        jrc_ok = bool(jrc and jrc["maxdmg"] and country_col)
        modes = (["jrc_area"] if jrc_ok else []) + ["area", "col", "const"]
        mode = st.radio("Valeur exposée", modes, key="wd_mode", format_func=lambda m: _DMG_VALUE_MODES[m])
        if not jrc_ok:
            st.caption("Valeurs de remplacement JRC par pays : placez le classeur JRC dans data/flood/ et choisissez une colonne pays.")
        value_col = area_col = None
        unit_value = const = 0.0
        if mode == "col":
            value_col = st.selectbox("Colonne de valeur (€)", base_cols, key="wd_valcol")
        elif mode == "const":
            const = st.number_input("Valeur exposée par point (€)", min_value=0.0, value=0.0, step=100000.0, key="wd_const")
        else:
            area_col = st.selectbox("Colonne de surface (m²)", base_cols, key="wd_area", index=max(0, _guess_col(base_cols, ("surface", "area", "m2", "m²"), False) - 1))
            if mode == "area":
                unit_value = st.number_input("Valeur de remplacement (€/m²)", min_value=0.0, value=0.0, step=50.0, key="wd_unit")
        if st.button("Calculer les dommages", type="primary", key="wd_dmg_calc"):
            try:
                curve = dmg.emanuel_curve(v_t, v_h)
                base = out_df.reset_index(drop=True)
                classes = dmg.asset_class_series(base, type_col, default_cls)
                values = dmg.exposed_value(base, mode, value_col, area_col, unit_value, const, country_col, classes, jrc, "total")
                curves = {c: curve for c in classes.dropna().unique()}
                per = {}
                for hz in res["sel"]:
                    cols = {r: wind.column_name(hz, r) for r in res["rps"] if wind.column_name(hz, r) in base.columns}
                    if not cols:
                        continue
                    o, sm = dmg.estimate_damages(base, cols, curves, classes, values, "Emanuel (2011)", None, hz)
                    sm["valeur_nulle"] = bool((values.fillna(0) <= 0).all())
                    per[hz] = (o, sm)
                ranking = None
                if res.get("storms") is not None and len(res["storms"]) and res.get("peaks"):
                    cells = res["cells"]
                    G = np.column_stack([res["peaks"].get(c, np.full(len(res["storms"]), np.nan)) if c else np.full(len(res["storms"]), np.nan)
                                         for c in cells])
                    loss = wind.event_losses(G, classes.to_numpy(), values.to_numpy(), curves)
                    st_df = res["storms"].copy()
                    st_df.insert(1, "nom_usuel", [wind.known_name(t) for t in st_df["peak_time"]])
                    st_df["rafale_max_portefeuille_ms"] = np.nanmax(np.where(np.isfinite(G), G, -np.inf), axis=1) if G.size else np.nan
                    st_df["rafale_max_portefeuille_ms"] = st_df["rafale_max_portefeuille_ms"].replace(-np.inf, np.nan)
                    st_df["perte_portefeuille_eur"] = loss
                    ranking = st_df.sort_values("perte_portefeuille_eur", ascending=False).reset_index(drop=True)
                st.session_state["wd_dmg"] = {"per": per, "ranking": ranking, "curve": curve, "name": res["name"], "n": res["n"]}
            except dmg.DamageError as e:
                st.error(str(e))


def _auto_resume(prefix: str, pending: int, done: int, queue_full: bool = False, max_stall: int = 20, wait_queue: int = 45) -> bool:
    """Reprise automatique d'un calcul par lots : renvoie True si un rerun a été demandé. Garde-fous : s'arrête si rien n'avance pendant
    ``max_stall`` relances consécutives. Le travail déjà fait est conservé (cache disque), donc l'arrêt ne perd rien."""
    import time
    ss = st.session_state
    if pending <= 0:
        ss[f"{prefix}_job"] = False
        ss[f"{prefix}_stall"] = 0
        return False
    prev = ss.get(f"{prefix}_prev", -1)
    ss[f"{prefix}_stall"] = 0 if done > prev else ss.get(f"{prefix}_stall", 0) + 1
    ss[f"{prefix}_prev"] = done
    if ss[f"{prefix}_stall"] >= max_stall:
        ss[f"{prefix}_job"] = False
        st.warning(f"Reprise automatique arrêtée : aucune progression depuis {max_stall} tentatives. Vérifiez les erreurs ci-dessous, puis relancez.")
        return False
    ss[f"{prefix}_job"] = True
    if queue_full:
        st.caption(f"File d'attente du service pleine : nouvelle tentative automatique dans {wait_queue} s… ({pending} restant(s))")
        time.sleep(wait_queue)
    else:
        st.caption(f"Reprise automatique du lot suivant… ({pending} restant(s))")
        time.sleep(1)
    st.rerun()
    return True


def _fit_step(step: int, n: int):
    cur = st.session_state.get("wd_fit_idx", 0)
    st.session_state["wd_fit_idx"] = (cur + step) % max(1, n)


def _fit_chart(fit, unit="m/s", title=""):
    import altair as alt
    pts = pd.DataFrame({"T": fit["T_emp"], "rafale": fit["x"]})
    cur = pd.DataFrame({"T": fit["T"], "rafale": fit["curve"]})
    xs = alt.X("T:Q", scale=alt.Scale(type="log"), title="Période de retour (années)")
    ys = alt.Y("rafale:Q", scale=alt.Scale(zero=False), title=f"Rafale ({unit})")
    layers = []
    if fit.get("lo") is not None:
        band = pd.DataFrame({"T": fit["T"], "lo": fit["lo"], "hi": fit["hi"]})
        layers.append(alt.Chart(band).mark_area(opacity=0.2).encode(x=xs, y=alt.Y("lo:Q", scale=alt.Scale(zero=False)), y2="hi:Q"))
    layers.append(alt.Chart(cur).mark_line().encode(x=xs, y=ys))
    layers.append(alt.Chart(pts).mark_point(filled=True, size=45).encode(x=xs, y=ys, tooltip=["T", "rafale"]))
    try:
        st.altair_chart(alt.layer(*layers).properties(height=320, title=title), use_container_width=True)
    except Exception as e:                                       # noqa: BLE001  (repli : courbe seule, sans échelle logarithmique)
        st.warning(f"Tracé avancé indisponible ({type(e).__name__}) : courbe simplifiée.")
        st.line_chart(cur.set_index("T")["rafale"])


def _wind_fit_viewer(res, algo):
    """Contrôle visuel des lois ajustées : une loi par maille ERA5 (ajustement indépendant maille par maille)."""
    fd = res.get("fitdata") or {}
    if not fd:
        return
    if st.button("Visualiser les lois ajustées", key="wd_fit_go", type="primary"):
        st.session_state["wd_fit_open"] = not st.session_state.get("wd_fit_open", False)
    if not st.session_state.get("wd_fit_open"):
        return
    kinds = [k for k in ("vent", "tempete") if k in res["sel"] and (k == "vent" or res.get("storms") is not None and len(res["storms"]))]
    if not kinds:
        st.caption("Aucun ajustement à montrer.")
        return
    kind = st.radio("Loi à contrôler", kinds, key="wd_fit_kind", horizontal=True,
                    format_func=lambda k: "Vent : GEV sur les maxima annuels" if k == "vent" else "Tempête : GPD sur les rafales de tempêtes")
    tkey = (kind, res["name"], res["n"], res["n_years"], len(fd))
    cached = st.session_state.get("wd_fit_table")
    if not cached or cached[0] != tkey:
        rows, fits = [], {}
        for c, d in fd.items():
            try:
                f = wind.fit_annual(d["annual"], boot=0) if kind == "vent" else wind.fit_storms(d["peaks"], res["n_years"], boot=0)
            except (wind.WindError, TypeError, ValueError) as e:
                rows.append({"maille": f"{c[0]:+.2f} ; {c[1]:+.2f}", "points": 0, "n": 0, "k": None, "corrélation": None, "écart (m/s)": None,
                             "avis": "Non évaluable", "alertes": str(e)})
                continue
            fits[c] = f
            rows.append({"maille": f"{c[0]:+.2f} ; {c[1]:+.2f}", "points": 0, "n": f["n"], "k": round(f["params"]["k"], 2), "corrélation": round(f["ppcc"], 3),
                         "écart (m/s)": round(f["rmse"], 2), "avis": f["verdict"], "alertes": " ; ".join(f["flags"])})
        order = {"Peu fiable": 0, "Non évaluable": 1, "À examiner": 2, "Satisfaisant": 3}
        cells_pts = pd.Series([f"{c[0]:+.2f} ; {c[1]:+.2f}" if c else None for c in res["cells"]]).value_counts()
        tab = pd.DataFrame(rows)
        tab["points"] = tab["maille"].map(cells_pts).fillna(0).astype(int)
        tab = tab.sort_values("avis", key=lambda s_: s_.map(order)).reset_index(drop=True)
        st.session_state["wd_fit_table"] = (tkey, tab, fits)
        cached = st.session_state["wd_fit_table"]
    _, tab, fits = cached
    st.markdown("**Revue de toutes les mailles** (classées des moins fiables aux plus fiables). Une loi est ajustée **indépendamment sur chaque maille "
                "ERA5 de 0,25°**, sans mise en commun spatiale : avec 30 à 35 ans de données, le paramètre de forme k est très incertain.")
    cnt = tab["avis"].value_counts().to_dict()
    st.caption(" · ".join(f"{k} : {v}" for k, v in cnt.items()))
    _stretch(st.dataframe, tab, hide_index=True)
    labels = list(tab["maille"])
    idx = min(st.session_state.get("wd_fit_idx", 0), len(labels) - 1)
    b1, b2, b3 = st.columns([1, 1, 4])
    b1.button("◀ Précédente", key="wd_fit_prev", on_click=_fit_step, args=(-1, len(labels)))
    b2.button("Suivante ▶", key="wd_fit_next", on_click=_fit_step, args=(1, len(labels)))
    st.session_state["wd_fit_idx"] = idx
    label = st.selectbox("Maille à examiner", labels, index=idx, key="wd_fit_sel")
    if label != labels[idx]:
        st.session_state["wd_fit_idx"] = idx = labels.index(label)
    row = tab.iloc[idx]
    lat_s, lon_s = (float(v) for v in row["maille"].split(";"))
    cell = (round(lat_s, 2), round(lon_s, 2))
    if cell not in fits:
        st.warning(f"Maille {row['maille']} : {row['alertes']}")
        return
    d = fd[cell]
    with st.spinner("Intervalle de confiance (bootstrap)…"):
        f = wind.fit_annual(d["annual"]) if kind == "vent" else wind.fit_storms(d["peaks"], res["n_years"])
    pts_here = [str(n_) for n_, c in zip(res["df"].get("name", res["df"].index), res["cells"]) if c == cell][:15]
    p = f["params"]
    m = st.columns(4)
    m[0].metric("Avis indicatif", f["verdict"])
    m[1].metric("Observations", f"{f['n']}")
    m[2].metric("Forme k", f"{p['k']:.2f}")
    m[3].metric("Corrélation points/loi", f"{f['ppcc']:.3f}")
    if kind == "vent":
        st.caption(f"GEV : position {p['xi']:.1f} m/s, échelle {p['alpha']:.1f} m/s, forme {p['k']:.2f} ; {f['n']} maxima annuels.")
    else:
        st.caption(f"GPD : seuil {p['u']:.1f} m/s, {p['lambda']:.2f} tempêtes dépassantes par an, échelle {p['alpha']:.1f} m/s, forme {p['k']:.2f}.")
    _fit_chart(f, title=f"Maille {row['maille']} : points observés (périodes de retour empiriques), loi ajustée et intervalle à 90 %")
    if f["flags"]:
        st.warning("Points d'attention : " + " ; ".join(f["flags"]) + ".")
    else:
        st.caption("Aucun point d'attention automatique : points alignés sur la loi, forme stable, intervalle raisonnable.")
    st.caption("Points du portefeuille dans cette maille : " + (", ".join(pts_here) if pts_here else "aucun") + ". Les seuils de l'avis (corrélation 0,97 ; "
               "|k| < 0,4 ; ≥ 30 observations ; intervalle à 90 % < ± 25 % à 100 ans) sont des repères heuristiques de CatMap, pas des critères normatifs : "
               "à croiser avec votre jugement. Au-delà de la durée d'observation, la courbe est une extrapolation.")


def _wind_section():
    st.subheader("Vent et tempête")
    st.markdown("**Vent** : rafale (m/s) par période de retour, ajustée sur les maxima annuels de toutes les rafales (ERA5, monde). **Tempête** : mêmes "
                "variables et périodes de retour, calculées uniquement sur les rafales des tempêtes identifiées (trajectoires Copernicus, Europe), avec "
                "le classement des tempêtes passées. Données Copernicus gratuites (licence CC-BY) ; même fonction de dommage vent pour les deux.")
    import os
    ok_key = cop.key_present(_secrets(), dict(os.environ))
    st.caption("Clé d'API Copernicus dans les secrets Streamlit (`CDSAPI_KEY`) : " + ("présente." if ok_key else
               "absente. Créez un compte gratuit sur cds.climate.copernicus.eu, copiez votre clé d'API (page de votre profil), acceptez les licences "
               "des deux jeux de données sur leurs pages, puis ajoutez la clé dans les secrets."))
    with st.expander("Catalogue Copernicus (contrôle des paramètres, sans clé)"):
        _wind_catalogue_reader()
    df, lat_c, lon_c, name = _points_input("wd_")
    if df is None:
        return
    _help_line("Méthodes et références (vent, tempête, dommages)",
               "**Vent** : rafale maximale annuelle en chaque maille ERA5 (0,25°), ajustée par une loi des valeurs extrêmes généralisée (GEV, moments L) "
               "→ rafale par période de retour. Ajustement indépendant maille par maille.\n\n**Tempête** : rafale maximale de chaque tempête "
               "identifiée par les trajectoires ; dépassements d'un seuil ajustés par une loi de Pareto généralisée (GPD, moments L) avec le nombre "
               "moyen de tempêtes par an → rafale de tempête par période de retour.\n\n**Dommages** : fonction d'Emanuel (2011), même courbe pour les deux.\n\n"
               + refs_md("era5", "gev", "gpd", "tracks", "emanuel"), wide=True, label_style="font-size:.875rem")
    sel = st.multiselect("Aléas à analyser", list(_WIND_HAZ), default=["vent"], key="wd_sel", format_func=lambda k: _WIND_HAZ[k])
    c1, c2, c3, c4 = st.columns(4)
    rps = c1.multiselect("Périodes de retour (années)", list(wind.RPS), default=[100], key="wd_rps", format_func=_rp_fmt)
    y0 = int(_num_help(c2, "Première année", "Plus la période est longue, plus les périodes de retour élevées sont fiables (≥ 30 ans requis).",
                       min_value=1940, max_value=2020, value=1990, step=1, key="wd_y0"))
    y1 = int(_num_help(c3, "Dernière année", "Dernière année de la période d'observation utilisée (ERA5 et trajectoires de tempêtes).",
                       min_value=1950, max_value=2025, value=2023, step=1, key="wd_y1"))
    with c4:
        _help_line("Algo. tempête", "**Méthode de suivi des tempêtes** dans la réanalyse ERA5 (trajectoires Copernicus).\n\n"
                   "**tempestextremes** (TempestExtremes) : suit les minima de pression au niveau de la mer. *Hypothèse* : une tempête est un "
                   "minimum de pression fermé qui se déplace. *Avantage* : critère physique simple et robuste, peu de faux systèmes. "
                   "*Inconvénient* : peut manquer les tempêtes faibles, rapides ou peu creusées ; dépend des seuils retenus.\n\n"
                   "**hodges** (TRACK) : suit les maxima de tourbillon relatif à 850 hPa. *Hypothèse* : une tempête est un maximum de rotation "
                   "dans la basse atmosphère. *Avantage* : détecte tôt les tempêtes en formation et les systèmes petits ou rapides. "
                   "*Inconvénient* : plus sensible au bruit, avec davantage de systèmes faibles, donc plus de tempêtes à trier.\n\n"
                   "Les deux listes se recoupent largement pour les grandes tempêtes ; l'écart se joue sur les événements moyens.\n\n" + refs_md("tracks"), wide=True,
                   label_style="font-size:.875rem")
        algo = st.selectbox("Algo. tempête", list(wind.ALGORITHMS), key="wd_algo", label_visibility="collapsed")
    cells_n = len({c for c in wind.cells_of_points(df[lat_c], df[lon_c]) if c})
    max_cells = int(_num_help(st, "Nombre maximal de cellules ERA5 (0,25°) à traiter", "Les points d'une même maille de 0,25° × 0,25° sont regroupés : "
                              "une seule requête Copernicus par maille (environ 1 à 2 minutes chacune), et les mailles déjà téléchargées sont "
                              "réutilisées. Par défaut, toutes les mailles de votre fichier ; diminuez pour limiter la durée.",
                              min_value=1, max_value=max(500, cells_n), value=max(1, cells_n), step=5, key=f"wd_maxc_{cells_n}"))
    if not sel or not rps:
        st.caption("Choisissez au moins un aléa et une période de retour.")
        return
    if y1 - y0 + 1 < 30:
        st.warning("Moins de 30 ans de données : l'ajustement du vent sera refusé (résultats vides).")
    n_cached = sum(wind.cell_path(c, y0, y1).exists() for c in {c for c in wind.cells_of_points(df[lat_c], df[lon_c]) if c})
    st.caption(f"{len(df)} point(s) regroupés en {cells_n} maille(s) ERA5 de 0,25° ({n_cached} déjà en cache, {cells_n - n_cached} à télécharger). Les requêtes sont en file d'attente chez Copernicus : un lot dure au plus "
               "4 minutes ; s'il en reste, recliquez sur le bouton, les cellules déjà obtenues sont conservées.")
    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button("Tester une requête (1 cellule, 1 an)", key="wd_test"):
        try:
            client = wind.make_client(cop.get_key(_secrets(), dict(os.environ)))
            cell = next(c for c in wind.cells_of_points(df[lat_c], df[lon_c]) if c)
            t0 = __import__("time").time()
            data = wind._retrieve(client, wind.TS_DATASET, wind.build_timeseries_request(cell[0], cell[1], y1, y1),
                                  wind.CACHE_DIR / "tmp_test.dat")
            ser = wind.parse_gust_series(data)
            st.success(f"Requête OK en {__import__('time').time() - t0:.0f} s : {len(ser)} valeurs horaires, rafale max {ser.max():.1f} m/s.")
        except (wind.WindError, cop.CopernicusError, StopIteration) as e:
            st.error(f"Test en échec : {e}")
    auto = _checkbox_with_help("Poursuivre automatiquement jusqu'à la fin", "Enchaîne les lots sans cliquer à nouveau (un lot dure au plus 4 minutes) ; "
                               "en cas de file d'attente pleine chez Copernicus, patiente puis réessaie. La page doit rester ouverte ; si vous la fermez, "
                               "relancez plus tard : les mailles déjà obtenues sont conservées. Le bouton « Arrêter » interrompt la reprise.",
                               key="wd_auto", value=True)
    job = bool(st.session_state.get("wd_job"))
    if job and st.button("Arrêter la reprise automatique", key="wd_stop"):
        st.session_state["wd_job"] = False
        job = False
    clicked = b2.button("Lancer / reprendre le calcul", type="primary", key="wd_go")
    if clicked:
        st.session_state["wd_stall"], st.session_state["wd_prev"] = 0, -1
    if clicked or (job and auto):
        try:
            _wind_compute(df, lat_c, lon_c, name, sel, rps, y0, y1, algo, max_cells, 240.0)
            inf_ = st.session_state["wd_result"]["info"]
            if auto and _auto_resume("wd", inf_["pending"], inf_["done"], inf_.get("queue_full", False)):
                return
            st.session_state["wd_job"] = False
        except (wind.WindError, cop.CopernicusError) as e:
            st.session_state["wd_job"] = False
            st.error(str(e))
    if b3.button("Vider le cache vent", key="wd_clear"):
        n = wind.clear_cache()
        st.session_state.pop("wd_result", None)
        st.session_state.pop("wd_dmg", None)
        st.caption(f"{n} fichier(s) supprimé(s).")
    res = st.session_state.get("wd_result")
    if not res or res["name"] != name or res["n"] != len(df):
        return
    info = res["info"]
    st.caption(f"Cellules obtenues : {info['done']} / {info['cells']}." + (f" {info['pending']} en attente : relancez le calcul." if info["pending"] else ""))
    for k, m in list(info["errors"].items())[:5]:
        st.warning(f"Cellule {k} : {m}")
    if info.get("queue_full"):
        st.warning("La file d'attente Copernicus est pleine (des requêtes précédentes sont encore en cours chez eux) : attendez quelques minutes, "
                   "puis relancez le calcul. Les requêtes sont envoyées une par une.")
    if info.get("tempete_msg"):
        st.warning(info["tempete_msg"])
    if info.get("tracks"):
        with st.expander("Diagnostic des trajectoires de tempêtes"):
            _stretch(st.dataframe, pd.DataFrame({"élément": list(info["tracks"]), "valeur": [str(v) for v in info["tracks"].values()]}), hide_index=True)
    out = res["df"]
    gust_cols = [wind.column_name(h, r) for h in res["sel"] for r in res["rps"] if wind.column_name(h, r) in out.columns]
    if not gust_cols:
        st.caption("Aucune valeur de rafale calculée pour le moment.")
        return
    if st.button("Estimer les dommages", type="primary", key="wd_dmg_go"):
        st.session_state["wd_dmg_open"] = True
    if st.session_state.get("wd_dmg_open"):
        _wind_damage(res, list(out.columns), None)
    dm = st.session_state.get("wd_dmg")
    merged = out.copy()
    col_opts = list(gust_cols)
    if dm and dm["name"] == name and dm["n"] == len(df):
        for hz, (o, sm) in dm["per"].items():
            new = [c for c in o.columns if c not in merged.columns]
            merged = pd.concat([merged, o[new]], axis=1)
            col_opts += [c for c in o.columns if c.startswith(f"dommage_{hz}_RP") or c == f"dommage_{hz}_PAM_eur"]
            st.markdown(f"**Dommages : {_WIND_HAZ[hz]}**")
            cols = st.columns(4)
            for i, (c, v) in enumerate(list(sm["pertes_totales"].items())[:3]):
                _kpi(cols[i], f"Perte totale RP{__import__('re').search(r'RP(' + chr(92) + 'd+)', c).group(1)}", _eur(v))
            if sm["pam_totale"] is not None:
                _kpi(cols[3], "Perte annuelle moyenne (approx.)", _eur(sm["pam_totale"]))
            if sm.get("valeur_nulle"):
                st.warning("La valeur exposée est nulle ou absente : les pertes en € sont nulles (les taux restent valables).")
    color_col = st.selectbox("Colorer la carte selon", col_opts, key="wd_color")
    _wind_map(merged, color_col)
    show = merged.drop(columns=["lat_f", "lon_f"])
    _stretch(st.dataframe, show, hide_index=True)
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(show), "catmap_vent_tempete.csv", "text/csv", key="wd_csv")
    if res.get("storms") is not None and len(res["storms"]):
        st.markdown(f"**Tempêtes passées près du portefeuille** ({len(res['storms'])}, algorithme {algo})")
        rk = dm["ranking"] if dm and dm.get("ranking") is not None else res["storms"].assign(nom_usuel=[wind.known_name(t) for t in res["storms"]["peak_time"]])
        _stretch(st.dataframe, rk, hide_index=True)
        st.download_button("Télécharger le classement (CSV)", _csv_bytes(rk), "catmap_tempetes.csv", "text/csv", key="wd_csv_storms")
        st.caption("Le nom usuel (Lothar, Xynthia…) est rapproché par la date du pic, à ± 1 jour ; le jeu Copernicus ne fournit que des identifiants.")
    _wind_fit_viewer(res, algo)
    st.caption("Rafales ERA5 à 0,25° (≈ 25 km) : lissées, elles sous-estiment souvent les extrêmes locaux. Périodes de retour supérieures à la durée "
               "d'observation = extrapolation. Fonction de dommage : Emanuel (2011), paramètres modifiables, à calibrer pour l'Europe.")


def _seisme_section():
    st.subheader("Séisme")
    st.markdown("Accélération maximale du sol (PGA, en g) par période de retour, lue à l'emplacement de chaque point, puis dommages par une "
                "fonction de vulnérabilité. Même logique que les autres onglets : colonnes `seisme_RP475_pga`, `dommage_seisme_RP475_eur`.")
    df, lat_c, lon_c, name = _points_input("sm_")
    if df is None:
        return
    _help_line("Jeu de données", "**GEM Global Seismic Hazard Map v2023.1** (GEM Foundation, licence CC BY-NC-SA 4.0 : usage non commercial). "
               "*Périmètre* : monde entier, résolution 3 minutes d'arc (≈ 5 km). *Contenu* : un seul niveau, la PGA à 10 % de probabilité de "
               "dépassement en 50 ans (période de retour 475 ans), sur sol rocheux. *Limites* : pas d'autre période de retour, pas d'effet de site "
               "(sol meuble), résolution grossière.\n\n**ESHM20** (EFEHR / ETH Zurich, licence CC BY 4.0). *Périmètre* : Europe et pourtour "
               "méditerranéen. *Contenu* : modèle européen 2020, courbes d'aléa permettant plusieurs périodes de retour (50 à 2475 ans proposées). *Limites* : hors Europe, "
               "aucune donnée. Une requête par maille de 0,05° (≈ 5 km), via le service web EFEHR : plus lent que GEM.\n\n" + refs_md("gem", "eshm20"), wide=True,
               label_style="font-size:.875rem")
    ds = st.selectbox("Jeu de données", list(sm.DATASETS), key="sm_ds", label_visibility="collapsed", format_func=lambda k: sm.DATASETS[k]["label"])
    raw_sm = False
    if ds == "gem":
        raw_sm = _checkbox_with_help("Données brutes", "Décoché (par défaut) : lecture du fichier précalculé data/seisme/gem_pga475_compact.npz (quelques Mo, "
                                     "instantané), s'il est présent. Coché : ignore ce fichier et lit le raster GEM d'origine (35 Mo à télécharger "
                                     "depuis Zenodo, puis 173 Mo extraits) : plus lent ; utile pour contrôler le fichier précalculé. Sans effet pour "
                                     "ESHM20, toujours interrogé en ligne.", key="sm_raw", value=False, wide=True)
        st.caption("Source : " + ("raster GEM brut (Zenodo)." if raw_sm or not sm.compact_path() else
                                  "fichier précalculé data/seisme/gem_pga475_compact.npz (aucun téléchargement)."))
    avail = list(sm.DATASETS[ds]["rps"])
    all_rps = _checkbox_with_help("Toutes les périodes de retour disponibles", "Coche l'ensemble des périodes de retour que le jeu de données propose "
                                  "(GEM : 475 ans uniquement ; ESHM20 : 50, 100, 200, 475, 975 et 2475 ans, dans la limite de la courbe d'aléa du point).",
                                  key="sm_all", value=True)
    rps = avail if all_rps else st.multiselect("Périodes de retour (années)", avail, default=avail, key="sm_rps", format_func=_rp_fmt)
    with st.expander("Diagnostic de la source (à lancer en cas de problème)"):
        if ds == "gem":
            if st.button("Tester le fichier GEM", key="sm_diag_go"):
                with st.spinner("Lecture du fichier précalculé ou, à défaut, téléchargement (≈ 35 Mo), extraction et lecture…"):
                    st.session_state["sm_diag"] = sm.diagnose(raw=raw_sm)
        else:
            if st.button("Tester le service ESHM20 (point de Genève)", key="sm_diag_eshm_go"):
                with st.spinner("Interrogation du service EFEHR…"):
                    st.session_state["sm_diag"] = sm.diagnose_eshm()
        if st.session_state.get("sm_diag"):
            _stretch(st.dataframe, pd.DataFrame(st.session_state["sm_diag"]), hide_index=True)
        st.caption("En cas d'ÉCHEC, copiez le tableau : il suffit à corriger l'adresse ou le format de lecture. Pour GEM, vous pouvez aussi "
                   "déposer le raster dans data/seisme/ (si votre dépôt accepte les gros fichiers).")
    with st.expander("Description du service web EFEHR (ESHM20)"):
        if st.button("Lire la description du service", key="sm_openapi_go", type="primary"):
            try:
                st.session_state["sm_openapi"] = sm.fetch_openapi()
            except sm.SeismicError as e:
                st.error(str(e))
        spec = st.session_state.get("sm_openapi")
        if spec:
            rows_ = sm.summarize_openapi(spec)
            if rows_:
                _stretch(st.dataframe, pd.DataFrame(rows_), hide_index=True)
    if not rps:
        st.caption("Choisissez au moins une période de retour.")
        return
    c1, c2, _sp = st.columns([1, 1, 2])
    auto_sm = _checkbox_with_help("Poursuivre automatiquement jusqu'à la fin (ESHM20)", "Enchaîne les lots de requêtes au service ESHM20 sans "
                                  "cliquer à nouveau. La page doit rester ouverte ; les mailles déjà obtenues sont conservées.", key="sm_auto", value=True)
    job_sm = bool(st.session_state.get("sm_job"))
    if job_sm and st.button("Arrêter la reprise automatique", key="sm_stop"):
        st.session_state["sm_job"] = False
        job_sm = False
    go_sm = c1.button("Estimer l'aléa sismique", type="primary", key="sm_go")
    if go_sm:
        st.session_state["sm_stall"], st.session_state["sm_prev"] = 0, -1
    if go_sm or (job_sm and auto_sm and ds == "eshm20"):
        try:
            with st.spinner("Lecture de l'aléa sismique (GEM : téléchargement la première fois ; ESHM20 : une requête par maille, reprise au clic suivant)…"):
                res_, summ_ = sm.estimate_pga(df, lat_c, lon_c, ds, rps, use_compact=not raw_sm)
            if summ_.get("en_attente"):
                st.warning(f"{summ_['en_attente']} point(s) en attente : relancez le calcul (les mailles déjà obtenues sont conservées).")
            for k_, m_ in list((summ_.get("erreurs") or {}).items())[:3]:
                st.warning(f"Maille {k_} : {m_}")
            res_["lat_f"] = pd.to_numeric(res_[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            res_["lon_f"] = pd.to_numeric(res_[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            st.session_state["sm_result"] = {"name": name, "n": len(df), "df": res_, "summ": summ_, "rps": list(summ_["rps"]), "ds": ds}
            st.session_state.pop("sm_dmg", None)
            if ds == "eshm20" and auto_sm and _auto_resume("sm", summ_.get("en_attente", 0), summ_["ok"]):
                return
            st.session_state["sm_job"] = False
        except sm.SeismicError as e:
            st.session_state["sm_job"] = False
            st.error(str(e))
    if c2.button("Estimer les dommages", type="primary", key="sm_dmg_go"):
        st.session_state["sm_dmg_open"] = True
    res = st.session_state.get("sm_result")
    if st.session_state.get("sm_dmg_open") and (not res or res["name"] != name or res["n"] != len(df)):
        st.caption("Estimez d'abord l'aléa sismique (bouton de gauche) : les dommages utilisent la PGA calculée.")
    if not res or res["name"] != name or res["n"] != len(df):
        return
    out = res["df"]
    summ = res["summ"]
    m = st.columns(4)
    m[0].metric("Points", summ["points"])
    m[1].metric("PGA lue", summ["ok"])
    m[2].metric("Sans donnée", summ["sans_donnee"])
    m[3].metric("Coordonnées invalides", summ["invalides"])
    if summ.get("en_attente"):
        st.warning(f"{summ['en_attente']} point(s) en attente : le temps imparti est écoulé. Relancez « Estimer l'aléa sismique » : les mailles déjà "
                   "obtenues sont conservées.")
    for k_, m_ in list((summ.get("erreurs") or {}).items())[:3]:
        st.warning(f"Maille {k_} : {m_}")
    if summ["sans_donnee"]:
        st.warning(f"{summ['sans_donnee']} point(s) sans valeur (mer, zone non couverte ou hors de la carte) : voir la colonne seisme_statut.")
    if st.session_state.get("sm_dmg_open"):
        _seisme_damage(res)
    merged = out.copy()
    dm = st.session_state.get("sm_dmg")
    col_opts = [sm.column_name(r) for r in res["rps"] if sm.column_name(r) in out.columns]
    if dm and dm["name"] == name and dm["n"] == len(df):
        o, sm_s = dm["out"], dm["summ"]
        merged = pd.concat([merged, o[[c for c in o.columns if c not in merged.columns]]], axis=1)
        col_opts += [c for c in o.columns if c.startswith("dommage_seisme_RP") or c == "dommage_seisme_PAM_eur"]
        st.markdown("**Dommages : séisme**")
        cols = st.columns(4)
        for i, (c, v) in enumerate(list(sm_s["pertes_totales"].items())[:3]):
            _kpi(cols[i], f"Perte totale RP{__import__('re').search(r'RP(' + chr(92) + 'd+)', c).group(1)}", _eur(v))
        if sm_s["pam_totale"] is not None:
            _kpi(cols[3], "Perte annuelle moyenne (approx.)", _eur(sm_s["pam_totale"]))
        if sm_s.get("valeur_nulle"):
            st.warning("La valeur exposée est nulle ou absente : les pertes en € sont nulles (les taux restent valables).")
        if len(res["rps"]) < 2:
            st.caption("Une seule période de retour : la perte annuelle moyenne n'est pas calculable (il en faut au moins deux).")
    _seisme_curve_viewer(res)
    color_col = st.selectbox("Colorer la carte selon", col_opts, key="sm_color")
    _wind_map(merged, color_col, cls_fn=sm.pga_class, palette=sm.PGA_COLORS, labels=sm.PGA_LABELS, popup_suffix="_pga", key="sm_map")
    show = merged.drop(columns=["lat_f", "lon_f"])
    _stretch(st.dataframe, show, hide_index=True)
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(show), "catmap_seisme.csv", "text/csv", key="sm_csv")
    st.caption("GEM Global Seismic Hazard Map v2023.1 (PGA à 475 ans, sol rocheux, CC BY-NC-SA 4.0, usage non commercial) ou ESHM20 (EFEHR, CC BY 4.0). "
               "Résolution ≈ 5 km : l'effet de sol (amplification) n'est pas pris en compte. Fonction de vulnérabilité à calibrer.")


def _seisme_curve_viewer(res):
    """Contrôle visuel des courbes d'aléa ESHM20 (une courbe par maille de 0,05°) ; GEM n'est pas ajusté (carte déjà calculée)."""
    if res.get("ds", "gem") != "eshm20":
        st.caption("GEM : aucune loi n'est ajustée par CatMap, la valeur est lue dans la carte d'aléa déjà calculée par GEM (475 ans). "
                   "Pas de courbe à contrôler ; pour voir des courbes d'aléa, choisissez ESHM20.")
        return
    if st.button("Visualiser les courbes d'aléa", key="sm_curve_go", type="primary"):
        st.session_state["sm_curve_open"] = not st.session_state.get("sm_curve_open", False)
    if not st.session_state.get("sm_curve_open"):
        return
    curves = sm.load_eshm_curves()
    if not curves:
        st.caption("Aucune courbe en cache : lancez d'abord l'estimation ESHM20.")
        return
    key = st.selectbox("Maille (lat, lon, pas de 0,05°)", sorted(curves), key="sm_curve_sel")
    cur = sm.curve_vs_rp(*curves[key])
    if cur.empty:
        st.warning("Courbe inexploitable pour cette maille.")
        return
    import altair as alt
    pts = pd.DataFrame({"rp": res["rps"], "pga": [sm.pga_at_rp(*curves[key], r) for r in res["rps"]]}).dropna()
    xs = alt.X("rp:Q", scale=alt.Scale(type="log"), title="Période de retour (années)")
    ys = alt.Y("pga:Q", scale=alt.Scale(type="log"), title="PGA (g)")
    try:
        st.altair_chart(alt.layer(alt.Chart(cur).mark_line().encode(x=xs, y=ys),
                                  alt.Chart(pts).mark_point(filled=True, size=70, color="#d62728").encode(x=xs, y=ys, tooltip=["rp", "pga"]))
                        .properties(height=320, title=f"Courbe d'aléa ESHM20, maille {key} (points rouges : périodes de retour retenues)"),
                        use_container_width=True)
    except Exception as e:                                       # noqa: BLE001
        st.warning(f"Tracé avancé indisponible ({type(e).__name__}) : courbe simplifiée.")
        st.line_chart(cur.set_index("rp")["pga"])
    st.caption("Courbe issue du modèle probabiliste ESHM20 (calcul d'aléa sismique par arbre logique), interpolée en log-log ; "
               "aucune extrapolation hors de la courbe fournie. Ce n'est pas un ajustement CatMap : il n'y a pas de loi à valider, mais vous pouvez "
               "contrôler la forme de la courbe et les points retenus.")


def _seisme_damage(res):
    out_df = res["df"]
    base_cols = [c for c in out_df.columns if not c.startswith(("seisme_", "dommage_")) and c not in ("lat_f", "lon_f")]
    with st.container(border=True):
        _help_line("Fonction de vulnérabilité : méthode", "**Version simplifiée de la méthode Hazus (FEMA).** Hazus définit, pour chaque état de dommage (léger, "
                   "modéré, étendu, complet), une fragilité lognormale (médiane, dispersion β), puis calcule le taux moyen de dommage comme la somme des "
                   "probabilités de chaque état pondérées par un taux de réparation, par type de bâtiment et niveau de code parasismique. Ici, on "
                   "**simplifie** : une seule courbe lognormale Φ(ln(PGA/θ)/β), **utilisée directement comme taux moyen de dommage**, sans type de "
                   "bâtiment ni niveau de code, avec la PGA comme grandeur d'entrée.\n\n"
                   "**Hypothèse forte** : une fragilité est la *probabilité d'atteindre ou de dépasser un état de dommage*, pas un taux de dommage moyen : "
                   "les confondre est une approximation de notre part, sans publication qui l'utilise telle quelle. θ (PGA de mi-dommage) et β par défaut "
                   "sont **illustratifs, sans source** : à calibrer selon le type de construction.\n\n"
                   "**Pour comparaison** : Hazus exprime nativement ses courbes en *déplacement spectral* ; sa version en *PGA équivalente* "
                   "(§ 5.4.3, tableaux 5-37 à 5-40, β = 0,64) est calculée pour un spectre de référence (séisme de magnitude 7,0 de l'ouest des États-Unis, "
                   "sol de classe D, distance ≥ 15 km) auquel les médianes sont « très sensibles », et le manuel précise qu'elle n'est pas utilisée "
                   "actuellement dans Hazus pour les bâtiments. Les cartes GEM et ESHM20 donnent la PGA sur **roche** (pas d'effet de site).\n\n"
                   + refs_md("hazus_eq", "gem", "eshm20"), wide=True, label_style="font-size:.875rem")
        st.markdown("**Estimation des dommages sismiques** : perte = taux de dommage(PGA) × valeur exposée, avec une fonction lognormale "
                    "`Φ(ln(PGA/θ)/β)`. Les paramètres par défaut sont **illustratifs** (aucune source) : calibrez-les selon le type de construction.")
        c1, c2, c3 = st.columns(3)
        theta = _num_help(c1, "PGA de mi-dommage θ (g)", "Accélération du sol pour laquelle 50 % de la valeur exposée est détruite. Plus elle est "
                          "faible, plus le bâti est vulnérable (maçonnerie non armée, ancien) ; plus elle est élevée, plus il est résistant "
                          "(parasismique récent).", min_value=0.05, max_value=3.0, value=0.6, step=0.05, key="sm_theta")
        beta = _num_help(c2, "Dispersion β", "Incertitude de la fonction : plus β est grand, plus la courbe s'étale (dommages possibles à faible PGA, "
                         "et partiels à forte PGA).", min_value=0.1, max_value=1.5, value=0.6, step=0.05, key="sm_beta")
        default_cls = c3.selectbox("Type d'actif par défaut", list(dmg.CLASSES), key="sm_cls", index=list(dmg.CLASSES).index("commercial"),
                                   format_func=lambda k: dmg.CLASS_LABELS[k])
        d1, d2 = st.columns(2)
        type_col = d1.selectbox("Colonne du type d'actif (facultatif)", ["(aucune)"] + base_cols, key="sm_typecol",
                                index=_guess_col(base_cols, ("type", "categ", "activ", "naf")))
        country_col = d2.selectbox("Colonne pays (ISO ou nom anglais)", ["(aucune)"] + base_cols, key="sm_ctry",
                                   index=_guess_col(base_cols, ("pays_iso2", "pays_iso", "iso2", "iso3", "country", "pays")))
        type_col = None if type_col == "(aucune)" else type_col
        country_col = None if country_col == "(aucune)" else country_col
        jrc, _err = _dmg_jrc_local()
        jrc_ok = bool(jrc and jrc["maxdmg"] and country_col)
        modes = (["jrc_area"] if jrc_ok else []) + ["area", "col", "const"]
        mode = st.radio("Valeur exposée", modes, key="sm_mode", format_func=lambda m: _DMG_VALUE_MODES[m])
        if not jrc_ok:
            st.caption("Valeurs de remplacement JRC par pays : placez le classeur JRC dans data/flood/ et choisissez une colonne pays.")
        value_col = area_col = None
        unit_value = const = 0.0
        if mode == "col":
            value_col = st.selectbox("Colonne de valeur (€)", base_cols, key="sm_valcol")
        elif mode == "const":
            const = st.number_input("Valeur exposée par point (€)", min_value=0.0, value=0.0, step=100000.0, key="sm_const")
        else:
            area_col = st.selectbox("Colonne de surface (m²)", base_cols, key="sm_area",
                                    index=max(0, _guess_col(base_cols, ("surface", "area", "m2", "m²"), False) - 1))
            if mode == "area":
                unit_value = st.number_input("Valeur de remplacement (€/m²)", min_value=0.0, value=0.0, step=50.0, key="sm_unit")
        if st.button("Calculer les dommages", type="primary", key="sm_dmg_calc"):
            try:
                curve = sm.vulnerability_curve(theta, beta)
                base = out_df.reset_index(drop=True)
                classes = dmg.asset_class_series(base, type_col, default_cls)
                values = dmg.exposed_value(base, mode, value_col, area_col, unit_value, const, country_col, classes, jrc, "total")
                cols = {r: sm.column_name(r) for r in res["rps"] if sm.column_name(r) in base.columns}
                o, summ = dmg.estimate_damages(base, cols, {c: curve for c in classes.dropna().unique()}, classes, values, "Lognormale PGA", None, "seisme")
                summ["valeur_nulle"] = bool((values.fillna(0) <= 0).all())
                st.session_state["sm_dmg"] = {"out": o, "summ": summ, "name": res["name"], "n": res["n"], "curve": curve}
            except (dmg.DamageError, ValueError) as e:
                st.error(str(e))


_GR_UNITS = {"dec": "par décennie", "an": "par an"}


def _grele_section():
    st.subheader("Grêle")
    st.markdown("Fréquence annuelle de grêle très grosse (≥ 5 cm) à la maille ERA5 de 0,25° (≈ 28 km) de chaque point, puis dommages. Même logique que les "
                "autres onglets : colonnes `grele_RP50_freq` (ou `grele_moy_freq`), `dommage_grele_RP50_eur`.")
    df, lat_c, lon_c, name = _points_input("gr_")
    if df is None:
        return
    _help_line("Jeu de données", "**Battaglioli et al. (2025)**, Nature Geoscience (ESSL). *Périmètre* : monde entier, maille ERA5 de 0,25° (≈ 28 km), 1950–2023. "
               "*Contenu* : fréquence des événements de grêle d'au moins 5 cm, estimée par régression logistique (AR-CHaMo) sur la réanalyse ERA5 : "
               "probabilité d'éclair × probabilité conditionnelle de grêle, toutes les 3 h. Calibrée sur l'Europe, les États-Unis et l'Australie.\n\n"
               "*Limites* : c'est une **fréquence modélisée** (pas une observation), à l'échelle d'une maille et non d'un site ; elle ne donne pas la taille "
               "des grêlons au-delà de 5 cm ni l'intensité des dégâts ; l'indépendance entre pas de temps est supposée. "
               "*Licence* CC BY-NC-ND 4.0 : usage non commercial, les fichiers sont lus sans modification.\n\n" + refs_md("grele", "era5"), wide=True,
               label_style="font-size:.875rem")
    _help_line("Approche", "**Climatologie** : nombre moyen annuel d'événements ≥ 5 cm par maille (moyenne 1950–2023). *Hypothèse* : le climat est stationnaire. "
               "*Avantage* : rapide (un seul fichier de 6,5 Mo), robuste. *Inconvénient* : pas de période de retour ; la perte obtenue est une perte "
               "annuelle moyenne uniquement.\n\n**Séries annuelles** : lecture des 74 fichiers annuels à la maille du point, puis niveau annuel dépassé "
               "une année sur RP (quantile empirique, positions de Weibull, sans loi ajustée ni extrapolation : la grêle ≥ 5 cm est souvent nulle "
               "plusieurs années de suite, ce qui exclut les lois d'extrêmes usuelles). *Avantage* : périodes de retour et perte annuelle moyenne par "
               "intégration comme pour les autres aléas. *Inconvénients* : plus long (74 fichiers de 6,5 Mo, lus un à un puis supprimés), "
               "période de retour limitée à 75 ans, valeurs élevées incertaines avec 74 années.\n\n" + refs_md("grele", "weibull", "aal"), wide=True,
               label_style="font-size:.875rem")
    mode = st.selectbox("Approche", list(hl.MODES), key="gr_mode", label_visibility="collapsed", format_func=lambda k: hl.MODES[k])
    rps = []
    if mode == "annual":
        all_rps = _checkbox_with_help("Toutes les périodes de retour disponibles", "Coche 2, 5, 10, 20 et 50 ans. Avec 74 années de données, "
                                      "on ne va pas au-delà de 50 ans (maximum théorique : 75 ans, sans extrapolation).", key="gr_all", value=True)
        rps = list(hl.RPS) if all_rps else st.multiselect("Périodes de retour (années)", list(hl.RPS), default=list(hl.RPS), key="gr_rps", format_func=_rp_fmt)
        if not rps:
            st.caption("Choisissez au moins une période de retour.")
            return
    raw = _checkbox_with_help("Données brutes", "Décoché (par défaut) : lecture du fichier précalculé data/grele/hail_compact.npz, instantanée. "
                              "Coché : ignore ce fichier et interroge directement la source (Zenodo, enregistrement 17379738) : chaque année est "
                              "téléchargée (74 fichiers de 6,5 Mo en séries annuelles), donc beaucoup plus lent ; utile pour contrôler le fichier "
                              "précalculé, visualiser les séries annuelles ou si le fichier est absent.", key="gr_raw", value=False, wide=True)
    clim_total = True
    if raw or not hl.compact_path():
        clim_total = _checkbox_with_help("La climatologie du fichier est un cumul 1950–2023 (à diviser par 74)", "Le diagnostic montre un rapport d'environ 64 "
                                         "entre la climatologie du fichier et la valeur de l'année 2000 (ex. Mendoza : 18,65 contre 0,29), alors que sa "
                                         "description annonce une moyenne annuelle : c'est donc un cumul sur 74 ans. Décochez seulement si votre diagnostic "
                                         "indique un rapport proche de 1. Sans effet si le fichier compact data/grele/hail_compact.npz est présent "
                                         "(la moyenne annuelle y est calculée à partir des 74 fichiers annuels).", key="gr_climtotal", value=True, wide=True)
        st.caption(("Données brutes demandées : " if hl.compact_path() else "Aucun fichier précalculé (data/grele/hail_compact.npz) : ") + "les fichiers sont téléchargés depuis Zenodo (lent en séries annuelles : "
                   "74 fichiers de 6,5 Mo). Voir tools/build_hail_compact.py pour le générer une fois.")
    if not raw and hl.compact_path():
        st.caption("Fichier précalculé data/grele/hail_compact.npz détecté : aucun téléchargement (climatologie, tendance et périodes de retour déjà calculées).")
    use_trend = _checkbox_with_help("Tenir compte de la tendance 1950–2023", "Option : la fréquence est **majorée** si la tendance locale est significative "
                                    "à la hausse, **minorée** si elle l'est à la baisse. Facteur = (fréquence + pente × années) ÷ fréquence, borné. "
                                    "Seules les mailles dont la p-value et la tendance relative dépassent vos seuils sont ajustées. Les seuils sont à "
                                    "étudier (tableau d'aide sous les résultats). Source : fichier de tendances de Battaglioli et al. (2025).\n\n"
                                    + refs_md("grele"), key="gr_trend", value=False, wide=True)
    trend_opts = None
    if use_trend:
        t1, t2, t3 = st.columns(3)
        unit = t1.selectbox("Unité de la tendance", list(_GR_UNITS), key="gr_unit", format_func=lambda k: _GR_UNITS[k])
        p_max = _num_help(t2, "p-value maximale", "Seuil de significativité : seules les mailles dont la p-value de la tendance est inférieure ou égale "
                          "à ce seuil sont ajustées (0,05 = 5 %). Plus il est bas, moins de mailles sont ajustées.", min_value=0.001, max_value=0.5,
                          value=0.05, step=0.01, key="gr_pmax", format="%.3f")
        rel_min = _num_help(t3, "Tendance relative minimale (%/décennie)", "Une tendance faible par rapport à la fréquence moyenne n'est pas retenue. "
                            "0 = aucun seuil ; 10 = seules les mailles dont la fréquence varie d'au moins 10 % par décennie sont ajustées.",
                            min_value=0.0, max_value=200.0, value=0.0, step=5.0, key="gr_rel")
        t4, t5, t6 = st.columns(3)
        years = _num_help(t4, "Années de projection", "Écart entre le milieu de la période (1986,5) et l'année visée : la climatologie est une moyenne "
                          "1950–2023 ; la pente est appliquée sur ce nombre d'années (38 ≈ 2025). 0 = pas de projection (facteur 1).",
                          min_value=0.0, max_value=100.0, value=38.0, step=1.0, key="gr_years")
        fmin = _num_help(t5, "Facteur minimal", "Borne basse du facteur appliqué (minoration maximale). 0,5 = la fréquence ne peut pas être divisée par plus de 2.",
                         min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="gr_fmin")
        fmax = _num_help(t6, "Facteur maximal", "Borne haute du facteur appliqué (majoration maximale). 2 = la fréquence ne peut pas être multipliée par plus de 2.",
                         min_value=1.0, max_value=10.0, value=2.0, step=0.1, key="gr_fmax")
        trend_opts = {"per_decade": unit == "dec", "years": float(years), "p_max": float(p_max), "rel_min_pct": float(rel_min),
                      "fmin": float(fmin), "fmax": float(fmax)}
    with st.expander("Diagnostic de la source (à lancer en cas de problème)"):
        if st.button("Tester les fichiers (climatologie, tendances, une année)", key="gr_diag_go", type="primary"):
            with st.spinner("Téléchargement (≈ 26 Mo) et lecture…"):
                st.session_state["gr_diag"] = hl.diagnose()
        if st.session_state.get("gr_diag"):
            _stretch(st.dataframe, pd.DataFrame(st.session_state["gr_diag"]), hide_index=True)
        st.caption("Le diagnostic donne les noms de variables, la grille, les unités et des valeurs à Mendoza, Dallas, Milan, Paris, Brisbane et "
                   "Johannesbourg. En cas d'ÉCHEC, copiez le tableau. Vous pouvez aussi déposer les fichiers dans data/grele/ (Zenodo 17379738).")
    c1, c2, _sp = st.columns([1, 1, 2])
    auto = _checkbox_with_help("Poursuivre automatiquement jusqu'à la fin (séries annuelles)", "Enchaîne les lots de téléchargement (74 fichiers) sans "
                               "cliquer à nouveau. La page doit rester ouverte ; les années déjà lues sont conservées.", key="gr_auto", value=True)
    job = bool(st.session_state.get("gr_job"))
    if job and st.button("Arrêter la reprise automatique", key="gr_stop"):
        st.session_state["gr_job"] = False
        job = False
    go = c1.button("Estimer l'aléa grêle", type="primary", key="gr_go")
    if go:
        st.session_state["gr_stall"], st.session_state["gr_prev"] = 0, -1
    if go or (job and auto and mode == "annual"):
        try:
            with st.spinner("Lecture de l'aléa grêle (téléchargement Zenodo la première fois, reprise au clic suivant)…"):
                res_, summ_ = hl.estimate(df, lat_c, lon_c, mode, rps, use_trend, trend_opts, clim_total=clim_total, use_compact=not raw)
            if summ_.get("en_attente"):
                st.warning(f"{summ_['en_attente']} année(s) restant à lire sur {len(hl.YEARS)} : relancez le calcul (les années déjà lues sont conservées).")
            for k_, m_ in list((summ_.get("erreurs") or {}).items())[:3]:
                st.warning(f"Année {k_} : {m_}")
            res_["lat_f"] = pd.to_numeric(res_[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            res_["lon_f"] = pd.to_numeric(res_[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            st.session_state["gr_result"] = {"name": name, "n": len(df), "df": res_, "summ": summ_, "mode": mode}
            st.session_state["gr_trend_opts"] = trend_opts
            st.session_state.pop("gr_dmg", None)
            if mode == "annual" and auto and _auto_resume("gr", summ_.get("en_attente", 0), summ_.get("annees_ok", 0)):
                return
            st.session_state["gr_job"] = False
        except hl.HailError as e:
            st.session_state["gr_job"] = False
            st.error(str(e))
    if c2.button("Estimer les dommages", type="primary", key="gr_dmg_go"):
        st.session_state["gr_dmg_open"] = True
    res = st.session_state.get("gr_result")
    if st.session_state.get("gr_dmg_open") and (not res or res["name"] != name or res["n"] != len(df)):
        st.caption("Estimez d'abord l'aléa grêle (bouton de gauche) : les dommages utilisent la fréquence calculée.")
    if not res or res["name"] != name or res["n"] != len(df):
        return
    out, summ = res["df"], res["summ"]
    m = st.columns(4)
    st.caption(f"Source des données : {summ.get('source', 'n/d')}.")
    m[0].metric("Points", summ["points"])
    m[1].metric("Fréquence lue", summ["ok"])
    m[2].metric("Sans donnée", summ["sans_donnee"])
    m[3].metric("Coordonnées invalides", summ["invalides"])
    if summ.get("en_attente"):
        st.warning(f"{summ['en_attente']} année(s) restant à lire : relancez « Estimer l'aléa grêle » (les années déjà lues sont conservées).")
        return
    if summ["sans_donnee"]:
        st.warning(f"{summ['sans_donnee']} point(s) sans valeur (mer, zone non couverte ou hors de la grille) : voir la colonne grele_statut.")
    if summ.get("tendance"):
        st.caption(f"Tendance : {summ.get('hausse', 0)} point(s) majoré(s), {summ.get('baisse', 0)} minoré(s) ; facteur moyen "
                   f"{'n/d' if summ.get('facteur_moyen') is None else round(summ['facteur_moyen'], 3)}. Colonnes d'origine : suffixe « _brut ».")
        _grele_threshold_help(out, summ)
    if st.session_state.get("gr_dmg_open"):
        _grele_damage(res)
    merged = out.copy()
    dm = st.session_state.get("gr_dmg")
    col_opts = list(summ["colonnes"])
    if dm and dm["name"] == name and dm["n"] == len(df):
        o, ds_ = dm["out"], dm["summ"]
        merged = pd.concat([merged, o[[c for c in o.columns if c not in merged.columns]]], axis=1)
        col_opts += [c for c in o.columns if c.startswith("dommage_grele_RP") and c.endswith("_eur") or c == "dommage_grele_PAM_eur"]
        st.markdown("**Dommages : grêle**")
        cols = st.columns(4)
        for i, (c, v) in enumerate(list(ds_["pertes_totales"].items())[:3]):
            _kpi(cols[i], f"Perte totale RP{__import__('re').search(r'RP(' + chr(92) + 'd+)', c).group(1)}", _eur(v))
        if ds_["pam_totale"] is not None:
            _kpi(cols[3], "Perte annuelle moyenne (approx.)", _eur(ds_["pam_totale"]))
        if dm.get("valeur_nulle"):
            st.warning("La valeur exposée est nulle ou absente : les pertes en € sont nulles (les taux restent valables).")
        if res["mode"] == "annual" and len(summ["rps"]) < 2:
            st.caption("Une seule période de retour : la perte annuelle moyenne n'est pas calculable (il en faut au moins deux).")
    if res["mode"] == "annual":
        _grele_series_viewer(res)
    color_col = st.selectbox("Colorer la carte selon", col_opts, key="gr_color")
    _wind_map(merged, color_col, cls_fn=hl.freq_class, palette=hl.FREQ_COLORS, labels=hl.FREQ_LABELS, popup_suffix="_freq", key="gr_map")
    show = merged.drop(columns=["lat_f", "lon_f"])
    _stretch(st.dataframe, show, hide_index=True)
    st.download_button("Télécharger le résultat (CSV)", _csv_bytes(show), "catmap_grele.csv", "text/csv", key="gr_csv")
    st.caption("Grêle ≥ 5 cm : Battaglioli et al. (2025), Nature Geoscience, données ESSL (Zenodo 17379738, CC BY-NC-ND 4.0, usage non commercial). "
               "Fréquence modélisée par maille de 0,25° : pas une mesure au site. Classes de la carte indicatives. Taux de dommage à calibrer.")


def _grele_threshold_help(out, summ):
    ref = hl.CLIM_COL + "_brut" if (hl.CLIM_COL + "_brut") in out.columns else hl.CLIM_COL
    need = (ref, "grele_tendance", "grele_tendance_p")
    if not all(c in out.columns for c in need):
        return
    with st.expander("Aide au choix du seuil de tendance"):
        o = st.session_state.get("gr_trend_opts") or {}
        tab = hl.threshold_table(out[ref], out["grele_tendance"], out["grele_tendance_p"], o.get("per_decade", True), o.get("years", 38.0),
                                 o.get("fmin", 0.5), o.get("fmax", 2.0))
        _stretch(st.dataframe, tab, hide_index=True)
        st.caption("Nombre de points ajustés selon la p-value maximale et la tendance relative minimale : choisissez un seuil qui n'ajuste que des mailles "
                   "où la tendance est nette. Calcul fait avec l'unité, les années de projection et les bornes saisies plus haut au dernier calcul.")


def _grele_series_viewer(res):
    """Contrôle visuel des séries annuelles lues à une maille, avec les niveaux de période de retour retenus."""
    if st.button("Visualiser les séries annuelles", key="gr_curve_go", type="primary"):
        st.session_state["gr_curve_open"] = not st.session_state.get("gr_curve_open", False)
    if not st.session_state.get("gr_curve_open"):
        return
    if res["summ"].get("source") == "fichier précalculé":
        st.caption("Séries annuelles indisponibles avec le fichier précalculé (il ne contient que les niveaux de période de retour, la climatologie et la "
                   "tendance). Pour les visualiser, retirez temporairement data/grele/hail_compact.npz.")
        return
    df = res["df"].dropna(subset=["lat_f", "lon_f"])
    cells = {}
    for la, lo in zip(df["lat_f"], df["lon_f"]):
        cells.setdefault(hl._cell_key(hl.cell_of(la, lo)), 0)
        cells[hl._cell_key(hl.cell_of(la, lo))] += 1
    have = set(hl.series_cache_cells())
    keys = sorted(k for k in cells if k in have)
    if not keys:
        st.caption("Aucune série en cache : lancez d'abord l'estimation (séries annuelles).")
        return
    key = st.selectbox("Maille (lat, lon, pas de 0,25°)", keys, key="gr_curve_sel", format_func=lambda k: f"{k} ({cells[k]} point(s))")
    la, lo = (float(x) for x in key.split(","))
    ser = hl.load_series([(la, lo)])[key]
    yrs = np.array(hl.YEARS)
    data = pd.DataFrame({"année": yrs, "valeur": ser})
    levels = pd.DataFrame([{"rp": r, "valeur": v} for r, v in hl.rp_levels(ser, res["summ"]["rps"]).items() if np.isfinite(v)])
    emp = hl.empirical_curve(ser)
    try:
        import altair as alt
        bars = alt.Chart(data).mark_bar().encode(x=alt.X("année:O", title="Année"), y=alt.Y("valeur:Q", title="Événements ≥ 5 cm par an"),
                                                 tooltip=["année", "valeur"])
        layers = [bars]
        if len(levels):
            layers.append(alt.Chart(levels).mark_rule(color="#d62728").encode(y="valeur:Q", tooltip=["rp", "valeur"]))
        st.altair_chart(alt.layer(*layers).properties(height=280, title=f"Valeurs annuelles, maille {key} (traits rouges : niveaux des périodes de retour retenues)"),
                        use_container_width=True)
        pts = alt.Chart(emp).mark_point(filled=True).encode(x=alt.X("rp:Q", scale=alt.Scale(type="log"), title="Période de retour empirique (années)"),
                                                           y=alt.Y("valeur:Q", title="Événements ≥ 5 cm par an"), tooltip=["rp", "valeur"])
        st.altair_chart(pts.properties(height=280, title="Valeurs classées (positions de Weibull)"), use_container_width=True)
    except Exception as e:                                       # noqa: BLE001
        st.warning(f"Tracé avancé indisponible ({type(e).__name__}) : graphique simplifié.")
        st.bar_chart(data.set_index("année")["valeur"])
    st.caption("Valeurs annuelles issues du modèle de Battaglioli et al. (2025) à la maille choisie. Les niveaux de période de retour sont des quantiles "
               "empiriques : aucune loi n'est ajustée, donc aucun ajustement à valider ; jugez surtout si le nombre d'années non nulles suffit pour "
               "les périodes de retour retenues.")


def _grele_damage(res):
    out_df = res["df"]
    base_cols = [c for c in out_df.columns if not c.startswith(("grele_", "dommage_")) and c not in ("lat_f", "lon_f")]
    with st.container(border=True):
        _help_line("Fonction de vulnérabilité : méthode", "**Linéaire plafonnée** : taux de dommage = min(1, taux par événement × fréquence annuelle) : une "
                   "fréquence de 1 événement par an applique 100 % du taux par événement, 0,5 en applique 50 %, 2 en applique 200 % (plafonné à 100 % de la "
                   "valeur).\n\n**Méthode 1 (saisie)** : le taux par événement est **à calibrer** ; la valeur par défaut est illustrative, elle ne vient "
                   "d'aucune publication.\n\n**Méthode 2 (simplifiée d'après Schmid et al., 2024)** : le taux par événement est celui de la fonction "
                   "d'impact publiée (taux de dommage moyen en fonction de la taille de grêle MESHS, forme sigmoïdale d'Emanuel 2011, paramètres de la "
                   "Table 2), évalué à 50 mm, soit la taille minimale des événements ≥ 5 cm. **Simplifications** : une seule taille (50 mm) alors que les "
                   "événements ≥ 5 cm peuvent être plus gros (le taux est donc minoré) ; calibration sur des sinistres assurés suisses (parc de bâtiments, "
                   "franchises) et sur une grêle estimée par radar, alors que la fréquence vient d'une modélisation ERA5 (grandeurs différentes) ; "
                   "fréquence de la maille de 0,25° appliquée au site, alors qu'un orage ne touche qu'une partie de la maille (surestimation). À lire "
                   "comme un ordre de grandeur.\n\n" + refs_md("schmid_hail") + "\n\nUne maille "
                   "de 0,25° (≈ 28 km) est bien plus grande que l'emprise d'un orage de grêle : le taux moyen par événement et par site est donc "
                   "faible. En mode climatologie, la perte obtenue est une perte annuelle moyenne ; en séries annuelles, la perte à chaque période de "
                   "retour est celle d'une année où la fréquence atteint ce niveau.", wide=True, label_style="font-size:.875rem")
        st.markdown("**Estimation des dommages de grêle** : perte = taux de dommage × valeur exposée, avec un taux par événement ≥ 5 cm saisi par vous "
                    "(méthode 1 : saisie, valeur par défaut **illustrative** sans source ; méthode 2 : fonction d'impact simplifiée d'après Schmid et al., 2024).")
        rmode = st.radio("Taux de dommage par événement", ["manual", "schmid"], horizontal=True, key="gr_ratio_mode",
                         format_func=lambda m: {"manual": "Méthode 1 : saisie (illustratif)",
                                                "schmid": "Méthode 2 : simplifiée d'après Schmid et al. (2024), à 50 mm"}[m])
        dmg_rps = None
        if res["summ"].get("mode") == "annual":
            avail = list(res["summ"]["rps"])
            dmg_rps = st.multiselect("Périodes de retour utilisées pour les dommages", avail, default=avail, key="gr_dmg_rps",
                                     format_func=lambda r: f"RP{r}")
            st.caption("La fréquence annuelle, donc le dommage, dépend de la période de retour : chaque RP choisie donne une colonne de dommage. "
                       "La perte annuelle moyenne est intégrée sur les RP retenues seulement (méthode des trapèzes, sans extrapolation) : "
                       "avec une seule RP, elle n'est pas calculable.")
        c1, c2 = st.columns(2)
        ratio_src = "illustratif"
        if rmode == "schmid":
            asset_k = c1.selectbox("Actif de la fonction d'impact", list(hl.SCHMID_PARAMS), key="gr_schmid_asset",
                                   format_func=lambda k: hl.SCHMID_PARAMS[k]["label"])
            ratio = hl.schmid_mdr(hl.HAIL_SIZE_MM, asset_k)
            ratio_src = f"Schmid et al. 2024, {hl.SCHMID_PARAMS[asset_k]['label'].lower()}, 50 mm"
            c1.caption(f"Taux par événement : {ratio:.4%} de la valeur (fonction d'impact à 50 mm). Calibration suisse, valeur indicative hors Suisse.")
        else:
            ratio = _num_help(c1, "Taux de dommage par événement (%)", "Part de la valeur exposée détruite lors d'un événement de grêle ≥ 5 cm sur la maille. "
                              "Dépend du bâti (toitures, vitrages, panneaux solaires, véhicules, cultures) : à calibrer sur vos sinistres.",
                              min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="gr_ratio", format="%.2f") / 100.0
        default_cls = c2.selectbox("Type d'actif par défaut", list(dmg.CLASSES), key="gr_cls", index=list(dmg.CLASSES).index("commercial"),
                                   format_func=lambda k: dmg.CLASS_LABELS[k])
        d1, d2 = st.columns(2)
        type_col = d1.selectbox("Colonne du type d'actif (facultatif)", ["(aucune)"] + base_cols, key="gr_typecol",
                                index=_guess_col(base_cols, ("type", "categ", "activ", "naf")))
        country_col = d2.selectbox("Colonne pays (ISO ou nom anglais)", ["(aucune)"] + base_cols, key="gr_ctry",
                                   index=_guess_col(base_cols, ("pays_iso2", "pays_iso", "iso2", "iso3", "country", "pays")))
        type_col = None if type_col == "(aucune)" else type_col
        country_col = None if country_col == "(aucune)" else country_col
        jrc, _err = _dmg_jrc_local()
        jrc_ok = bool(jrc and jrc["maxdmg"] and country_col)
        modes = (["jrc_area"] if jrc_ok else []) + ["area", "col", "const"]
        mode = st.radio("Valeur exposée", modes, key="gr_vmode", format_func=lambda m: _DMG_VALUE_MODES[m])
        if not jrc_ok:
            st.caption("Valeurs de remplacement JRC par pays : placez le classeur JRC dans data/flood/ et choisissez une colonne pays.")
        value_col = area_col = None
        unit_value = const = 0.0
        if mode == "col":
            value_col = st.selectbox("Colonne de valeur (€)", base_cols, key="gr_valcol")
        elif mode == "const":
            const = st.number_input("Valeur exposée par point (€)", min_value=0.0, value=0.0, step=100000.0, key="gr_const")
        else:
            area_col = st.selectbox("Colonne de surface (m²)", base_cols, key="gr_area",
                                    index=max(0, _guess_col(base_cols, ("surface", "area", "m2", "m²"), False) - 1))
            if mode == "area":
                unit_value = st.number_input("Valeur de remplacement (€/m²)", min_value=0.0, value=0.0, step=50.0, key="gr_unit_value")
        if st.button("Calculer les dommages", type="primary", key="gr_dmg_calc"):
            try:
                base = out_df.reset_index(drop=True)
                classes = dmg.asset_class_series(base, type_col, default_cls)
                values = dmg.exposed_value(base, mode, value_col, area_col, unit_value, const, country_col, classes, jrc, "total")
                o, summ = hl.estimate_damages(base, res["summ"], ratio, classes, values, ratio_src, dmg_rps)
                summ["valeur_nulle"] = bool((values.fillna(0) <= 0).all())
                st.session_state["gr_dmg"] = {"out": o, "summ": summ, "name": res["name"], "n": res["n"], "valeur_nulle": summ["valeur_nulle"]}
            except (dmg.DamageError, ValueError) as e:
                st.error(str(e))


def page_risques_climatiques():
    st.title("Risque climatiques")
    tabs = st.tabs(["Débordement et submersion", "Ruissellement", "Vent & tempête", "Séisme", "Grêle", "Géorisques"])
    with tabs[0]:
        _safe_tab(_exposition_section, "Débordement et submersion")
    with tabs[1]:
        _safe_tab(_ruissellement_section, "Ruissellement")
    with tabs[2]:
        _safe_tab(_wind_section, "Vent & tempête")
    with tabs[3]:
        _safe_tab(_seisme_section, "Séisme")
    with tabs[4]:
        _safe_tab(_grele_section, "Grêle")
    with tabs[5]:
        _safe_tab(_georisques_section, "Géorisques")


PAGES = {"Géolocalisation": page_geolocalisation, "Temps réel": page_temps_reel,
         "Risque climatiques": page_risques_climatiques}


def main():
    st.set_page_config(page_title="CatMap Risk", page_icon=str(ASSETS / "Logo_CatMapV2.png"), layout="wide")
    css = ASSETS / "styles.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        logo = ASSETS / "Logo_CatMapV2.png"
        if logo.exists():
            st.image(str(logo))
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        sm = ASSETS / "SquareManagement.png"
        if sm.exists():
            st.image(str(sm))
    PAGES[page]()


if __name__ == "__main__":
    main()
