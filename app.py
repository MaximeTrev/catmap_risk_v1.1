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
    if area:
        c[1].metric("Rayon du périmètre", f"{area['radius_m']:,.0f} m".replace(",", " "))
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


def page_geolocalisation():
    st.title("Géolocalisation")
    t_ent, t_sc = st.tabs(["Géolocalisation des entreprises", "Géolocalisation supply chain"])
    with t_ent:
        _tab_entreprises()
    with t_sc:
        _tab_supply_chain()


def _tab_entreprises():
    c_mode, c_area = st.columns([3, 2])
    with c_area:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        area_mode = _checkbox_with_help(
            "Entreprises sur un périmètre géographique",
            "**Principe** : au lieu de chercher une entreprise par son nom, on délimite un cercle sur la carte et on "
            "récupère tous les établissements situés dedans.\n\n"
            "**Coché** : cliquez pour fixer le centre, déplacez la souris pour régler le rayon, cliquez de nouveau pour "
            "valider, puis lancez la recherche. Les objets portant un tag *shop*, *amenity* ou *brand* sont retenus.\n\n"
            "**Décoché** : recherche par nom d'entreprise (saisie, liste ou fichier).",
            key="area_mode", left=True, wide=True)
    with c_mode:
        mode = st.radio("Mode de saisie", ["Un nom d'entreprise", "Liste (une entreprise par ligne)", "Fichier CSV / Excel"],
                        horizontal=True, disabled=area_mode)
    if area_mode:
        _page_area()
        return
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
    _stretch(st.dataframe, res, hide_index=True)
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
        _events_section()
    with t_cy:
        _cyclone_section()
    with t_vg:
        _vigilance_section()
    with t_eu:
        _europe_section()


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


@st.cache_data(ttl=600, show_spinner=False)
def _ev_info(url: str, layer: str, lat: float, lon: float, zoom: float, time_value):
    return evt.get_feature_info(url, layer, lat, lon, zoom, time_value=time_value)


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


def _events_section():
    st.subheader("Suivi évènements climatiques")
    st.caption("Inondations et incendies en temps quasi réel à partir des services publics Copernicus (EMS, EFFIS, EFAS, GloFAS). "
               "Aucune clé n'est nécessaire. Les adresses des services n'ont pas pu être testées depuis le serveur de développement : "
               "chaque onglet fournit un diagnostic des appels.")
    t_act, t_fire, t_flood, t_area = st.tabs(["Activations Copernicus EMS", "Incendies (EFFIS)", "Crues (EFAS / GloFAS)",
                                              "Entreprises dans une zone"])
    with t_act:
        _ev_activations_tab()
    with t_fire:
        _ev_fire_tab()
    with t_flood:
        _ev_flood_tab()
    with t_area:
        _ev_area_tab()


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
    _ev_diag(diag)


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
        _ev_click_values(url, chosen, click, out.get("zoom") or 6, tv, timed)
    if url and chosen:
        with st.expander("Légendes des couches affichées"):
            for n in chosen:
                st.markdown(f"**{dict(layers).get(n, n)}** · `{n}`")
                st.image(evt.legend_url(url, n))
            st.caption("Légendes fournies par le service EFFIS. Classes de l'indice Forêt-Météo utilisées pour le croisement : "
                       + " · ".join(f"{lbl} (< {hi:g})" if hi != float("inf") else f"{lbl} (≥ 50)" for hi, lbl in evt.FWI_CLASSES) + ".")
    if not tbl.empty:
        _stretch(st.dataframe, tbl.head(1000), hide_index=True)
        st.download_button("Télécharger les zones brûlées affichées (CSV)", _csv_bytes(tbl), "effis_zones_brulees.csv", "text/csv",
                           key="ev_fire_dl")
    st.caption("Le service ne filtre pas par date : le filtrage est fait ici sur les colonnes de date, parmi les 5 000 zones les plus "
               "récentes lues dans la zone choisie.")
    _ev_fire_cross(url, layers, gj_all, start_col, end_col, mode, d1, d2, active_days, tv, timed)
    _ev_diag(d_cap + d_ba)


def _ev_click_values(url, chosen, click, zoom, tv, timed):
    st.markdown(f"**Valeurs au point cliqué** ({click['lat']:.4f}, {click['lng']:.4f})")
    rows, raw = [], {}
    for n in chosen:
        txt, _ = _ev_info(url, n, float(click["lat"]), float(click["lng"]), float(zoom), tv if n in timed else None)
        val = evt.first_number(txt)
        rank, label = evt.fwi_class(val) if "fwi" in n.lower() else (0, "")
        rows.append({"Couche": n, "Valeur": val if val is not None else "—", "Niveau": label or "—"})
        raw[n] = txt
    _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)
    with st.expander("Réponse brute du service"):
        for n, t in raw.items():
            st.markdown(f"`{n}`")
            st.code(t or "(réponse vide : couche non interrogeable ou aucune donnée à cet endroit)")


def _ev_fire_cross(url, layers, gj_all, start_col, end_col, mode, d1, d2, active_days, tv, timed):
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
    c1, c2 = st.columns(2)
    snap = c1.slider("Distance maximale à une zone brûlée (km)", 0, 50, 5, key="ev_fire_snap")
    fwi_layers = [n for n, _ in layers if "fwi" in n.lower()]
    use_fwi = False
    fwi_layer = None
    if fwi_layers:
        use_fwi = c2.checkbox("Ajouter le danger météo d'incendie (FWI)", True, key="ev_fire_usefwi")
        if use_fwi:
            fwi_layer = c2.selectbox("Couche FWI", fwi_layers, index=fwi_layers.index("mf010.fwi") if "mf010.fwi" in fwi_layers else 0,
                                     key="ev_fire_fwilayer")
    else:
        c2.caption("Aucune couche FWI disponible : seul le croisement avec les zones brûlées est possible.")
    sig = (up.name, len(df), snap, mode, str(d1), str(d2), active_days, fwi_layer)
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
            if use_fwi and fwi_layer and url:
                vals, info = evt.sample_layer(url, fwi_layer, lat.to_numpy(), lon.to_numpy(), time_value=tv if fwi_layer in timed else None)
                res["fwi_valeur"] = [None if v != v else round(float(v), 1) for v in vals]
                cls = [evt.fwi_class(v) for v in vals]
                res["fwi_rang"] = [c[0] for c in cls]
                res["fwi_niveau"] = [c[1] or "n/d" for c in cls]
                if info["valeurs_lues"] == 0:
                    notes.append("Aucune valeur FWI lue : la couche n'est pas interrogeable ou le service n'a pas répondu.")
                if info["tronque"]:
                    notes.append(f"Seules les {info['mailles']} premières mailles de 5 km ont été interrogées (limite) : les autres points n'ont pas de FWI.")
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
    show = [c for c in res.columns if c.startswith(("incendie_", "fwi_"))]
    first = [c for c in res.columns if c not in show][:6]
    _stretch(st.dataframe, res[first + show], hide_index=True)
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
    if simple:
        import folium
        folium.LayerControl(collapsed=True).add_to(m)
        components.html(m.get_root().render(), height=580, scrolling=False)
        out = {}
    else:
        out = _ev_show_map(m, f"ev_flood_map_{kind}", track_click=True)
    click = out.get("last_clicked") if out else None
    if click and chosen:
        _ev_click_values(url, chosen, click, out.get("zoom") or 4, tv, timed)
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
                st.image(evt.legend_url(url, n))
            st.caption("Légendes fournies par le service. Si une image est vide ou cassée, le service ne publie pas de légende pour cette couche.")
        st.markdown(f"[Tester la première couche dans le navigateur (image brute du service)]({evt.getmap_url(url, chosen[0], time_value=tv if chosen[0] in timed else None)})")
        st.caption("Si ce lien affiche une erreur XML (jeton exigé, date hors plage, couche inconnue), elle explique pourquoi la carte reste vide.")
    st.caption(f"Service interrogé : {url}")
    _ev_flood_cross(url, chosen, tv, timed, kind)
    _ev_diag(d_cap)


def _ev_flood_cross(url, chosen, tv, timed, kind):
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
        _help_line("Seuil de la valeur", "Un point est « touché » si la valeur lue est **strictement supérieure** à ce seuil. Pour une couche "
                   "d'aléa d'inondation, 0 convient (toute valeur positive = dans la zone). Pour un cumul de pluie, choisissez par exemple 50 (mm).",
                   label_style="font-size:.875rem")
        thr = st.slider("Seuil de la valeur", 0.0, 500.0, 0.0, key=f"ev_flood_thr_{kind}", label_visibility="collapsed")
    prec = {"1 km environ (précis, plus d'appels)": 0.01, "5 km environ": 0.05}
    pk = c2.selectbox("Regroupement des points", list(prec), key=f"ev_flood_prec_{kind}")
    sig = (up.name, len(df), tuple(chosen), thr, prec[pk], tv)
    if st.button("Croiser", type="primary", key=f"ev_flood_go_{kind}"):
        with st.spinner("Lecture des valeurs aux points…"):
            lat = pd.to_numeric(df[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
            lon = pd.to_numeric(df[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
            res, notes, hit = df.copy(), [], np.zeros(len(df), dtype=bool)
            answered = False
            for n in chosen:
                vals, info = evt.sample_layer(url, n, lat, lon, time_value=tv if n in timed else None, step=prec[pk], max_cells=500)
                res[f"crue_{n}"] = [None if v != v else round(float(v), 3) for v in vals]
                hit |= np.nan_to_num(vals, nan=-np.inf) > thr
                answered |= info["valeurs_lues"] > 0
                if info["valeurs_lues"] == 0:
                    notes.append(f"Couche « {n} » : aucune valeur lue (couche non interrogeable, ou aucun point dans une zone couverte).")
                if info["tronque"]:
                    notes.append(f"Couche « {n} » : seules les {info['mailles']} premières mailles ont été interrogées (limite) ; "
                                 "les autres points n'ont pas de valeur.")
            res["crue_touche"] = ["Oui" if h else ("Non" if answered else "n/d") for h in hit]
            st.session_state[f"ev_flood_cross_{kind}"] = (sig, res, notes)
    saved = st.session_state.get(f"ev_flood_cross_{kind}")
    if not saved or saved[0] != sig:
        return
    _, res, notes = saved
    for n in notes:
        st.warning(n)
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
                    "<span style='color:#9e9e9e'>■</span> n/d (aucune valeur lue)", unsafe_allow_html=True)
    show = [c for c in res.columns if c.startswith("crue_")]
    first = [c for c in res.columns if c not in show][:6]
    _stretch(st.dataframe, res[first + show], hide_index=True)
    st.download_button("Télécharger le croisement (CSV)", _csv_bytes(res), f"croisement_crues_{kind}.csv", "text/csv",
                       key=f"ev_flood_dl_{kind}")


def _ev_area_tab():
    st.markdown("**Extraire toutes les entreprises situées dans une zone touchée** (même outil que l'onglet Géolocalisation).")
    center, zoom = (46.6, 2.5), 5
    df, _ = _ev_activations()
    if not df.empty:
        geo = df.dropna(subset=["lat", "long"])
        geo = geo[geo["genre"].isin(["flood", "fire"])].head(40)
        opts = {f"{r.code} — {r.nom}": (r.lat, r.long) for r in geo.itertuples()}
        pick = st.selectbox("Centrer la carte sur une activation récente", ["—"] + list(opts), key="ev_area_center")
        if pick in opts:
            center, zoom = opts[pick], 8
    _page_area(kp="ev_area_", result_key="ev_area_result", center=center, zoom=zoom)


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
    else:
        _stretch(st.dataframe, alert[cols].sort_values("vigilance_couleur"), hide_index=True)
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
    else:
        _stretch(st.dataframe, orange[cols].sort_values("europe_niveau_id", ascending=False), hide_index=True)
        _site_map(orange, kp="eu_")
    st.download_button("Télécharger tous les sites avec leurs alertes (CSV)", _csv_bytes(exp[cols]),
                       "catmap_sites_vigilance_europe.csv", "text/csv", key="eu_dl_sites")


# --------------------------------------------------------------------------
def page_risques_climatiques():
    st.title("Risque climatiques")
    st.caption("Section à venir.")


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
