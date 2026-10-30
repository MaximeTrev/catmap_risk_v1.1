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


def _popup_html(row: pd.Series) -> str:
    def esc(v):
        return "" if pd.isna(v) else html.escape(str(v))
    lat, lon = float(row["lat"]), float(row["long"])
    gmaps = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
    streetview = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"
    surf = "" if pd.isna(row.get("surface_m2")) else f"<br>Surface : {row['surface_m2']:,.0f} m²".replace(",", " ")
    return (f"<div style='font:12px sans-serif;min-width:180px'><b>{esc(row['name'])}</b><br>"
            f"{esc(row['nom_demande'])}<br>{esc(row.get('pays'))}<br>{esc(row.get('naf_section'))}{surf}<br>"
            f"Score : {esc(row.get('match_score'))}<br>"
            f"<a href='{esc(row['osm_url'])}' target='_blank' rel='noopener'>Voir sur OpenStreetMap</a><br>"
            f"<a href='{gmaps}' target='_blank' rel='noopener'>Voir sur Google Maps</a><br>"
            f"<a href='{streetview}' target='_blank' rel='noopener'>Voir sur Street View</a></div>")


_COLOR_CHOICES = {"Entreprise": "nom_demande", "Pays": "pays", "Secteur NAF": "naf_section", "Aucune (une seule couleur)": None}


def _site_map(df: pd.DataFrame, area: dict | None = None):
    """Carte interactive des sites du tableau : fonds de carte au choix, regroupement, popups, plein écran."""
    import folium
    from folium.plugins import Fullscreen, MarkerCluster
    from streamlit_folium import st_folium

    d = df.dropna(subset=["lat", "long"])
    if d.empty:
        st.info("Aucun site à afficher.")
        return
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Colorer les points par", list(_COLOR_CHOICES), key="map_color",
                          index=list(_COLOR_CHOICES).index("Secteur NAF") if area else 0)
    cluster_on = c2.checkbox("Regrouper les points proches", True, key="map_cluster")
    color_col = _COLOR_CHOICES[choice]

    if len(d) > MAX_MARKERS:
        st.warning(f"Carte limitée à {MAX_MARKERS} sites sur {len(d)} (les mieux appariés). "
                   "Filtrez par entreprise ou par pays pour tout voir.")
        d = d.sort_values("match_score", ascending=False).head(MAX_MARKERS)

    m = folium.Map(location=[float(d["lat"].mean()), float(d["long"].mean())], zoom_start=5,
                   control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Plan (OpenStreetMap)").add_to(m)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(tiles=esri.format("World_Light_Gray_Base"), attr="Esri", name="Fond clair", max_native_zoom=16).add_to(m)
    folium.TileLayer(tiles=esri.format("World_Dark_Gray_Base"), attr="Esri", name="Fond sombre", max_native_zoom=16).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    Fullscreen().add_to(m)
    m.fit_bounds([[float(d["lat"].min()), float(d["long"].min())], [float(d["lat"].max()), float(d["long"].max())]])

    if color_col:
        keys = sorted(d[color_col].fillna("n/d").astype(str).unique())
        colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
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
    key = f"map_{color_col}_{cluster_on}_{len(d)}_{round(float(d['lat'].sum()), 4)}_{bool(area)}"
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


def _area_new():
    """Bouton « Nouvelle prise » : nouvelle instance de la carte (cercle effacé) et résultats du périmètre retirés."""
    st.session_state["area_pick_n"] = st.session_state.get("area_pick_n", 0) + 1
    res = st.session_state.get("result")
    if res is not None and res.meta.get("area"):
        st.session_state.pop("result", None)


def _page_area():
    s = get_settings()
    with st.expander("Options du périmètre"):
        rel = st.checkbox(
            "Inclure les grands sites (relations)", False, key="area_rel",
            help="**Principe** : un grand site (campus, usine, centre commercial) est parfois décrit par une *relation* "
                 "multipolygone plutôt que par un point ou un contour simple.\n\n"
                 "**Cochée** : les relations sont interrogées aussi (requête plus lourde).\n\n"
                 "**Décochée** : seuls les points (nodes) et les contours (ways) sont interrogés.")
        named = st.checkbox(
            "Uniquement les lieux nommés (name ou brand)", True, key="area_named",
            help="**Principe** : les objets *shop*, *amenity* ou *brand* incluent aussi du mobilier sans nom (bancs, "
                 "poubelles, parkings à vélos…).\n\n"
                 "**Cochée** : on garde les objets qui portent un name ou un brand.\n\n"
                 "**Décochée** : tout est gardé (beaucoup plus de bruit).")
        dedup = st.slider(
            "Fusion des doublons node/way (mètres)", 0, 100, 25, key="area_dedup",
            help="Un même site peut exister en point et en contour dans OpenStreetMap. Valeur > 0 : on fusionne ceux de même "
                 "nom situés à moins de cette distance (ou un point dans un contour). 0 : aucune fusion.")
    st.caption(f"Rayon maximal : **{s.area_max_radius_m:,} m**".replace(",", " ") + f" (au-delà, la recherche serait trop lourde) ; "
               f"au plus {s.area_max_sites:,} sites sont conservés (les plus proches du centre).".replace(",", " "))
    n = st.session_state.get("area_pick_n", 0)
    picked = circle_picker(center=(46.6, 2.5), zoom=6, max_radius_m=s.area_max_radius_m, key=f"area_picker_{n}")
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
        b1.button("Nouvelle prise", key="area_new", on_click=_area_new)
        go = b2.button("Recherche", type="primary", key="area_go")
        if go:
            _run_area(area, AreaOptions(include_relations=rel, named_only=named, dedup_radius_m=float(dedup)))
    res = st.session_state.get("result")
    if res is not None and res.meta.get("area"):
        st.divider()
        _render_result(res)


def _run_area(area: AreaQuery, options: AreaOptions):
    status = st.empty()
    with st.spinner("Interrogation d'Overpass sur le périmètre…"):
        res = locate_area(get_client(), get_settings(), area, options, get_countries(), get_naf(),
                          on_event=lambda m: status.caption(m))
    status.empty()
    st.session_state["result"] = res
    st.session_state["result_options"] = options


def page_geolocalisation():
    st.title("Géolocalisation d'entreprises")
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
    if st.button("Actualiser"):
        _fetch_vigilance_df.clear()
    try:
        return _fetch_vigilance_df()
    except vg.VigilanceError as e:
        st.error(str(e))
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_cyclones(with_rain: bool = True):
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
    st.markdown(f'<span class="{cls}"><span class="cy-tip">{_help_html(text)}</span></span>', unsafe_allow_html=True)


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
    t_cy, t_vg = st.tabs(["Cyclones tropicaux", "Vigilance Météo-France"])
    with t_cy:
        _cyclone_section()
    with t_vg:
        _vigilance_section()


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_om_json(path: str):
    s = get_settings()
    return vg.fetch_path(vg.MeteoFranceAuth(s), s, path)


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_om_zip(path: str):
    s = get_settings()
    return vg.fetch_path(vg.MeteoFranceAuth(s), s, path, binary=True)


def _overseas_section():
    """Vigilance Polynésie, Nouvelle-Calédonie et autres DROM (flux distincts de la métropole)."""
    import json
    import folium
    from streamlit_folium import st_folium

    st.subheader("Outre-mer")
    st.caption("Polynésie française et Nouvelle-Calédonie : couleur par zone. Autres DROM : flux ZIP (voir le diagnostic ci-dessous). "
               "Les contours des zones d'outre-mer ne sont pas encore intégrés : la carte montre un repère par territoire.")
    results = {}
    for name, info in vg.OM_TERRITORIES.items():
        try:
            payload = _fetch_om_json(info["path"])
        except vg.VigilanceError as e:
            st.warning(f"{name} : {e}")
            continue
        try:
            table = vg.parse_vigilance(payload, phenomena={})
        except vg.VigilanceFormatError as e:
            st.warning(f"{name} : {e}")
            with st.expander(f"Réponse brute {name} (début)"):
                st.code(json.dumps(payload, ensure_ascii=False)[:3000])
            continue
        results[name] = (table, vg.worst_by_domain(table, None), payload)

    c = st.columns(max(1, len(results)))
    for col, (name, (table, worst, _)) in zip(c, results.items()):
        mx = int(worst["couleur_id"].max()) if len(worst) and worst["couleur_id"].notna().any() else 0
        col.metric(name, vg.COLORS.get(mx, "n/d"))
    view = st.selectbox("Zoom de la carte", ["Polynésie française", "Nouvelle-Calédonie"] + list(vg.OM_VIEWS), key="om_view")
    center, zoom = (vg.OM_TERRITORIES[view]["center"], vg.OM_TERRITORIES[view]["zoom"]) if view in vg.OM_TERRITORIES else vg.OM_VIEWS[view]
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(tiles=esri.format("Canvas/World_Light_Gray_Base"), attr="Esri", name="Fond clair", max_native_zoom=16).add_to(m)
    folium.TileLayer(tiles=esri.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    for name, (table, worst, _) in results.items():
        mx = int(worst["couleur_id"].max()) if len(worst) and worst["couleur_id"].notna().any() else 0
        zones = "<br>".join(f"{html.escape(str(r.domaine))} : {html.escape(str(r.couleur))}"
                            + (f" — {html.escape(r.phenomenes)}" if r.phenomenes else "") for r in worst.itertuples())
        c0 = vg.OM_TERRITORIES[name]["center"]
        folium.CircleMarker(list(c0), radius=16, color="#222", weight=2, fill=True, fill_opacity=0.85,
                            fill_color=vg.COLOR_HEX.get(mx, "#999999"),
                            popup=folium.Popup(f"<b>{html.escape(name)}</b><br>{zones}", max_width=340),
                            tooltip=f"{name} : {vg.COLORS.get(mx, 'n/d')}").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    try:
        st_folium(m, height=420, returned_objects=[], key=f"om_map_{view}", use_container_width=True)
    except TypeError:
        st_folium(m, height=420, width=1400, returned_objects=[], key=f"om_map_{view}")
    for name, (table, worst, payload) in results.items():
        with st.expander(f"{name} : détail par zone"):
            _stretch(st.dataframe, worst.sort_values("couleur_id", ascending=False), hide_index=True)
            if st.checkbox("Afficher la réponse brute (début)", key=f"raw_{name}"):
                st.code(json.dumps(payload, ensure_ascii=False)[:3000])

    with st.expander("Autres DROM (Antilles, Guyane, La Réunion, Mayotte) : flux ZIP"):
        try:
            data = _fetch_om_zip(vg.OM_DROM_PATH)
        except vg.VigilanceError as e:
            st.warning(str(e))
            return
        frames = vg.parse_zip_json(data)
        if frames:
            tab = pd.concat(frames, ignore_index=True)
            _stretch(st.dataframe, vg.worst_by_domain(tab, None), hide_index=True)
        else:
            st.info("Ce flux ne contient pas de données de couleur exploitables automatiquement (bulletins PDF). "
                    "Contenu du flux :")
            for f in vg.inspect_zip(data):
                st.markdown(f"**{f['fichier']}** ({f['taille']} octets)" + (" : document PDF/binaire" if f["binaire"] else ""))
                if f["debut"]:
                    st.code(f["debut"])
        st.download_button("Télécharger le flux (ZIP)", data, "vigilance_outre_mer.zip", "application/zip", key="om_zip_dl")


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
        esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
        folium.TileLayer(tiles=esri.format("Canvas/World_Light_Gray_Base"), attr="Esri", name="Fond clair", max_native_zoom=16).add_to(m)
        folium.TileLayer("OpenStreetMap", name="Plan (OpenStreetMap)").add_to(m)
        folium.TileLayer(tiles=esri.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
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


def _vigilance_input():
    """Source des sites : fichier chargé (export de la page Géolocalisation) sinon dernière géolocalisation de la session.
    Retourne (df de travail, colonnes d'origine, est_un_fichier) ou None."""
    up = st.file_uploader("Fichier de sites (CSV ou Excel téléchargé depuis la page Géolocalisation, recherche classique ou par périmètre)",
                          type=["csv", "txt", "xlsx"], key="vg_upload")
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
        _site_map(alert)
        st.download_button("Télécharger les sites en vigilance orange ou rouge (CSV)", _csv_bytes(alert[cols]),
                           "catmap_sites_vigilance.csv", "text/csv")
    st.download_button("Télécharger tous les sites avec leur vigilance (CSV)", _csv_bytes(exp[full_cols]),
                       "catmap_sites_vigilance_complet.csv", "text/csv", key="vg_dl_all")


# --------------------------------------------------------------------------
PAGES = {"Géolocalisation": page_geolocalisation, "Temps réel": page_temps_reel}


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
