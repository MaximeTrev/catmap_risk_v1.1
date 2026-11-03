"""Cyclones tropicaux actifs : récupération, normalisation, classification.

Source par défaut : service public « Active Hurricanes, Cyclones and Typhoons »
(ArcGIS Living Atlas / NOAA), qui republie les bulletins du National Hurricane Center
(Atlantique, Pacifique Est et Centre) et du Joint Typhoon Warning Center (Pacifique
Ouest, océan Indien, hémisphère Sud). Couches utilisées :

    0  Forecast Position   (points prévus ; TAU = 0 correspond à la position actuelle)
    1  Observed Position   (trace passée)
    7/8/9  rayons de vent 34 / 50 / 64 kt (polygones)

Aucune clé n'est nécessaire. Le module ne dépend pas de Streamlit : tout est testable
avec une fausse session HTTP. Toute indisponibilité d'un service annexe (précipitations,
rayons de vent) est tolérée : la carte s'affiche sans cette couche.

Hypothèse de classification : la catégorie est calculée à partir du vent soutenu maximal
(échelle de Saffir-Simpson, seuils en nœuds) pour TOUS les bassins, afin d'avoir une
échelle homogène. Les bassins ne l'utilisent pas tous officiellement (ex. Océan Indien).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import requests

SERVICE_URLS = (
    "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/Active_Hurricanes_v1/FeatureServer",
    "https://services9.arcgis.com/RHVPKKiFTONKtxq3/ArcGIS/rest/services/Active_Hurricanes_Sampler/FeatureServer",
)
LAYER_FORECAST = 0
LAYER_OBSERVED = 1
LAYER_RADII = {34: 7, 50: 8, 64: 9}
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

KT_TO_KMH = 1.852
KT_TO_MPH = 1.15078
STALE_OBSERVED_HOURS = 36  # une trace sans prévision plus vieille que ça n'est plus « en cours »

ICON_KEYS = ("TS", "1", "2", "3", "4", "5")
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


class CycloneError(RuntimeError):
    pass


# --------------------------------------------------------------------------- modèle
@dataclass
class TrackPoint:
    tau: int                       # heures après la position actuelle (0 = actuelle)
    lat: float
    lon: float
    valid: Optional[datetime] = None
    wind_kt: Optional[float] = None
    gust_kt: Optional[float] = None
    mslp_hpa: Optional[float] = None
    direction_deg: Optional[float] = None
    speed_kt: Optional[float] = None
    stage: str = ""                # libellé source (ex. « Tropical Storm »)
    precip_mm: Optional[float] = None   # cumul prévu ~24 h autour de l'échéance (Open-Meteo)

    @property
    def icon_key(self) -> str:
        return icon_key_for_wind(self.wind_kt)

    @property
    def category(self) -> int:
        return category_from_wind(self.wind_kt)


@dataclass
class Storm:
    key: str
    name: str
    basin: str
    advisory: str = ""
    advisory_time: Optional[datetime] = None
    current: Optional[TrackPoint] = None
    forecast: list[TrackPoint] = field(default_factory=list)   # y compris la position actuelle en tête
    past: list[tuple[float, float]] = field(default_factory=list)
    source: str = ""
    number: int = 0                      # numéro de la tempête dans le bassin (STORMNUM)
    tau_source: str = "TAU"              # d'où viennent les échéances (TAU, FCSTPRD, horodatage, indisponible)
    tau_ok: bool = True                  # False : échéances non distinctes (toutes à +0 h) et non reconstituables
    tau_raw: list = field(default_factory=list)   # valeurs TAU lues (diagnostic)

    @property
    def kind(self) -> str:
        return storm_kind(self.basin)


# --------------------------------------------------------------------------- classification
def category_from_wind(wind_kt: Optional[float]) -> int:
    """0 = dépression/tempête tropicale, 1..5 = catégories Saffir-Simpson (vent 1 min, nœuds)."""
    if wind_kt is None or (isinstance(wind_kt, float) and math.isnan(wind_kt)):
        return 0
    w = float(wind_kt)
    for cat, floor in ((5, 137), (4, 113), (3, 96), (2, 83), (1, 64)):
        if w >= floor:
            return cat
    return 0


def icon_key_for_wind(wind_kt: Optional[float]) -> str:
    cat = category_from_wind(wind_kt)
    return str(cat) if cat else "TS"


def stage_label(wind_kt: Optional[float], basin: str = "") -> str:
    """Libellé français : dépression / tempête tropicale / ouragan|typhon|cyclone cat. N."""
    cat = category_from_wind(wind_kt)
    if cat:
        return f"{storm_kind(basin).capitalize()} catégorie {cat}"
    if wind_kt is not None and wind_kt < 34:
        return "Dépression tropicale"
    return "Tempête tropicale"


def storm_kind(basin: str) -> str:
    b = (basin or "").strip().upper()
    if b.startswith(("AL", "AT", "EP", "CP", "NA", "NE", "NC")) or b in {"E", "L", "C"}:
        return "ouragan"
    if b.startswith(("WP", "NW", "W")):
        return "typhon"
    return "cyclone"


def compass(deg: Optional[float]) -> str:
    if deg is None or math.isnan(deg):
        return ""
    return COMPASS[int(((deg % 360) + 11.25) // 22.5) % 16]


def movement_text(deg: Optional[float], speed_kt: Optional[float]) -> str:
    if speed_kt is None or math.isnan(speed_kt):
        return "n/d"
    if speed_kt < 1:
        return "Quasi stationnaire"
    d = compass(deg)
    kmh = speed_kt * KT_TO_KMH
    head = f"{d} ({deg:.0f}°) à " if d and deg is not None else "à "
    return f"{head}{speed_kt:.0f} kt ({kmh:.0f} km/h)"


def wind_text(kt: Optional[float]) -> str:
    if kt is None or (isinstance(kt, float) and math.isnan(kt)) or kt <= 0:
        return "n/d"
    return f"{kt:.0f} kt · {kt * KT_TO_KMH:.0f} km/h · {kt * KT_TO_MPH:.0f} mph"


def pressure_text(hpa: Optional[float]) -> str:
    if hpa is None or (isinstance(hpa, float) and math.isnan(hpa)) or hpa <= 0:
        return "n/d"
    return f"{hpa:.0f} hPa"


# --------------------------------------------------------------------------- utilitaires
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f in (-999.0, -9999.0, 9999.0):
        return None
    return f


def _clean_wind(v: Any) -> Optional[float]:
    f = _num(v)
    return f if f is not None and f > 0 else None


def _dt(ms: Any) -> Optional[datetime]:
    f = _num(ms)
    if f is None:
        return None
    try:
        return datetime.fromtimestamp(f / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _title(name: Any) -> str:
    s = str(name or "").strip()
    return s.title() if s else "Sans nom"


def unwrap_longitudes(lons: Iterable[float], ref: float) -> list[float]:
    """Ramène chaque longitude à moins de 180° de la référence (traversée de l'antiméridien)."""
    out = []
    for x in lons:
        while x - ref > 180:
            x -= 360
        while x - ref < -180:
            x += 360
        out.append(x)
    return out


def bearing_speed(p1: tuple[float, float], p2: tuple[float, float], hours: float) -> tuple[Optional[float], Optional[float]]:
    """Cap (°) et vitesse (kt) entre deux positions (lat, lon) ; utilisé si la source ne fournit pas le déplacement."""
    if hours <= 0:
        return None, None
    la1, lo1, la2, lo2 = map(math.radians, (p1[0], p1[1], p2[0], p2[1]))
    dlo = lo2 - lo1
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    nm = 2 * 3440.065 * math.asin(min(1.0, math.sqrt(a)))
    return brg, nm / hours


# --------------------------------------------------------------------------- accès HTTP
def _get_json(url: str, params: dict, *, session=None, timeout: float = 20.0) -> dict:
    s = session or requests
    try:
        r = s.get(url, params=params, timeout=timeout, headers={"User-Agent": "CatMap/3.3"})
    except requests.RequestException as e:
        raise CycloneError(f"Service injoignable : {e}") from e
    if r.status_code != 200:
        raise CycloneError(f"HTTP {r.status_code} sur {url}")
    try:
        data = r.json()
    except ValueError as e:
        raise CycloneError("Réponse non JSON") from e
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        raise CycloneError(f"Erreur service : {err.get('message') if isinstance(err, dict) else err}")
    return data


def _query_layer(base: str, layer: int, *, session=None, geojson: bool = False) -> dict:
    params = {"where": "1=1", "outFields": "*", "f": "geojson" if geojson else "json",
              "returnGeometry": "true" if geojson else "false", "resultRecordCount": 2000}
    if geojson:
        params["outSR"] = 4326
    return _get_json(f"{base}/{layer}/query", params, session=session)


def _attrs(feature: dict) -> dict:
    a = feature.get("attributes")
    if a is None:
        a = feature.get("properties") or {}
    return {str(k).upper(): v for k, v in a.items()}


# --------------------------------------------------------------------------- parsing
def parse_forecast(payload: dict) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for f in payload.get("features", []):
        a = _attrs(f)
        lat, lon = _num(a.get("LAT")), _num(a.get("LON"))
        if lat is None or lon is None:
            geom = f.get("geometry") or {}
            lon = lon if lon is not None else _num(geom.get("x"))
            lat = lat if lat is not None else _num(geom.get("y"))
        tau = _num(a.get("TAU"))
        if lat is None or lon is None or (tau is None and not a.get("STORMNAME")):
            continue
        a["_lat"], a["_lon"], a["_tau"] = lat, lon, (int(tau) if tau is not None else None)
        key = (str(a.get("BASIN") or "").strip().upper(), int(_num(a.get("STORMNUM")) or 0), _title(a.get("STORMNAME")))
        groups.setdefault(key, []).append(a)
    return groups


def parse_observed(payload: dict) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for f in payload.get("features", []):
        a = _attrs(f)
        lat, lon = _num(a.get("LAT")), _num(a.get("LON"))
        if lat is None or lon is None:
            continue
        a["_lat"], a["_lon"], a["_dtg"] = lat, lon, _dt(a.get("DTG"))
        key = (str(a.get("BASIN") or "").strip().upper(), int(_num(a.get("STORMNUM")) or 0))
        groups.setdefault(key, []).append(a)
    for rows in groups.values():
        rows.sort(key=lambda r: r["_dtg"] or datetime.min.replace(tzinfo=timezone.utc))
    return groups



VERSION = 2   # incrémenté à chaque évolution du format (échéances résolues, liens vers les bulletins)
TAU_ALT_KEYS = ("FCSTPRD", "FCST_PRD", "FHOUR", "FORECASTHOUR", "FHR", "HOUR", "FCSTHOUR")


def resolve_taus(rows: list[dict]) -> tuple[list[Optional[int]], str]:
    """Échéances (heures après le bulletin) de chaque point d'une prévision.

    Certaines tempêtes (selon le centre) arrivent avec TAU vide ou égal à 0 sur tous les points : on cherche alors une
    autre colonne d'échéance (FCSTPRD…), puis un horodatage propre à chaque point (heure de validité). Retourne
    (échéances, source) ; source = « indisponible » si rien n'est exploitable (les échéances ne sont alors pas affichées).
    """
    n = len(rows)
    taus = [r.get("_tau") for r in rows]
    if n < 2:
        return [t if t is not None else 0 for t in taus], "TAU"
    if all(t is not None for t in taus) and len(set(taus)) == n:
        return list(taus), "TAU"
    for key in TAU_ALT_KEYS:
        vals = [_num(r.get(key)) for r in rows]
        if all(v is not None for v in vals) and len(set(vals)) == n:
            lo = min(vals)
            return [int(round(v - lo)) for v in vals], key
    keys = [k for k in rows[0] if not str(k).startswith("_")]
    prefer = sorted(keys, key=lambda k: (0 if any(w in str(k) for w in ("VALID", "FCST", "DTG")) else 1))
    for key in prefer:
        vals = [_num(r.get(key)) for r in rows]
        if all(v is not None and v > 1e11 for v in vals) and len(set(vals)) == n:      # millisecondes depuis 1970
            lo = min(vals)
            return [int(round((v - lo) / 3_600_000)) for v in vals], f"horodatage {key}"
    return [t if t is not None else 0 for t in taus], "indisponible"


def build_storms(forecast: dict, observed: dict, *, now: Optional[datetime] = None) -> list[Storm]:
    now = now or datetime.now(timezone.utc)
    storms: list[Storm] = []
    used_obs: set[tuple] = set()

    for (basin, num, name), rows in forecast.items():
        tau_raw = [r.get("_tau") for r in rows]
        taus, tau_source = resolve_taus(rows)
        for r, t in zip(rows, taus):
            r["_tau"] = t
        tau_ok = tau_source != "indisponible"
        rows = sorted(rows, key=lambda r: r["_tau"])          # tri stable : l'ordre d'origine est conservé si tout est à 0
        adv_time = _dt(rows[0].get("ADVDATE"))
        pts: list[TrackPoint] = []
        for r in rows:
            valid = (adv_time + timedelta(hours=r["_tau"])) if (adv_time and tau_ok) else None
            wind = _clean_wind(r.get("MAXWIND"))
            pts.append(TrackPoint(
                tau=r["_tau"], lat=r["_lat"], lon=r["_lon"], valid=valid, wind_kt=wind,
                gust_kt=_clean_wind(r.get("GUST")), mslp_hpa=_clean_wind(r.get("MSLP")),
                direction_deg=_num(r.get("TCDIR")), speed_kt=_num(r.get("TCSPD")),
                stage=str(r.get("TCDVLP") or r.get("DVLBL") or "").strip()))
        ref = pts[0].lon
        for p, lo in zip(pts, unwrap_longitudes([p.lon for p in pts], ref)):
            p.lon = lo
        obs_rows = observed.get((basin, num), [])
        used_obs.add((basin, num))
        past = [(r["_lat"], r["_lon"]) for r in obs_rows]
        if past:
            past = list(zip([p[0] for p in past], unwrap_longitudes([p[1] for p in past], ref)))
        cur = pts[0]
        if cur.wind_kt is None and obs_rows:          # repli : intensité observée la plus récente
            cur.wind_kt = _clean_wind(obs_rows[-1].get("INTENSITY"))
            cur.mslp_hpa = cur.mslp_hpa or _clean_wind(obs_rows[-1].get("MSLP"))
        if cur.speed_kt is None and len(pts) > 1 and pts[1].tau > cur.tau:
            cur.direction_deg, cur.speed_kt = bearing_speed((cur.lat, cur.lon), (pts[1].lat, pts[1].lon), pts[1].tau - cur.tau)
        storms.append(Storm(key=f"{basin}{num:02d}", name=name, basin=basin,
                            advisory=str(rows[0].get("ADVISNUM") or ""), advisory_time=adv_time,
                            current=cur, forecast=pts, past=past, source="NHC/JTWC", number=num,
                            tau_source=tau_source, tau_ok=tau_ok, tau_raw=tau_raw))

    # Tempêtes présentes seulement dans la trace observée (pas de prévision publiée) et récentes.
    for (basin, num), rows in observed.items():
        if (basin, num) in used_obs or not rows:
            continue
        last = rows[-1]
        if not last["_dtg"] or now - last["_dtg"] > timedelta(hours=STALE_OBSERVED_HOURS):
            continue
        direction = speed = None
        if len(rows) > 1 and rows[-2]["_dtg"]:
            hrs = (last["_dtg"] - rows[-2]["_dtg"]).total_seconds() / 3600
            direction, speed = bearing_speed((rows[-2]["_lat"], rows[-2]["_lon"]), (last["_lat"], last["_lon"]), hrs)
        cur = TrackPoint(tau=0, lat=last["_lat"], lon=last["_lon"], valid=last["_dtg"],
                         wind_kt=_clean_wind(last.get("INTENSITY")), mslp_hpa=_clean_wind(last.get("MSLP")),
                         direction_deg=direction, speed_kt=speed)
        past = list(zip([r["_lat"] for r in rows], unwrap_longitudes([r["_lon"] for r in rows], cur.lon)))
        storms.append(Storm(key=f"{basin}{num:02d}", name=_title(last.get("STORMNAME")), basin=basin,
                            advisory_time=last["_dtg"], current=cur, forecast=[cur], past=past, source="observé seul", number=num))

    storms.sort(key=lambda s: -(s.current.wind_kt or 0) if s.current else 0)
    return storms


def fetch_storms(*, session=None, urls: Iterable[str] = SERVICE_URLS, now: Optional[datetime] = None) -> list[Storm]:
    """Essaie chaque URL de service jusqu'à obtenir une réponse valide."""
    last_err: Optional[Exception] = None
    for base in urls:
        try:
            fc = _query_layer(base, LAYER_FORECAST, session=session)
            try:
                ob = _query_layer(base, LAYER_OBSERVED, session=session)
            except CycloneError:
                ob = {"features": []}
            return build_storms(parse_forecast(fc), parse_observed(ob), now=now)
        except CycloneError as e:
            last_err = e
    raise CycloneError(f"Aucune source cyclones disponible ({last_err})")


def fetch_wind_radii(*, session=None, urls: Iterable[str] = SERVICE_URLS) -> dict:
    """GeoJSON des zones de vent ≥ 34/50/64 kt. Retourne {} en cas d'échec (couche facultative)."""
    for base in urls:
        feats: list[dict] = []
        try:
            for kt, layer in LAYER_RADII.items():
                data = _query_layer(base, layer, session=session, geojson=True)
                for f in data.get("features", []):
                    if f.get("geometry"):
                        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": {"kt": kt}})
        except CycloneError:
            continue
        if feats:
            return {"type": "FeatureCollection", "features": feats}
    return {}


# --------------------------------------------------------------------------- précipitations prévues
def attach_precipitation(storms: list[Storm], *, session=None, window_h: int = 24) -> bool:
    """Cumul de pluie prévu (Open-Meteo, modèle global) autour de chaque échéance de la trajectoire.

    Valeur ponctuelle au centre de la tempête : elle sous-estime les maxima dans les bandes
    spiralées ; à lire comme un ordre de grandeur. Retourne False si le service est indisponible.
    """
    ok = False
    for st in storms:
        pts = [p for p in st.forecast if p.valid is not None]
        if not pts:
            continue
        params = {"latitude": ",".join(f"{p.lat:.2f}" for p in pts),
                  "longitude": ",".join(f"{((p.lon + 180) % 360) - 180:.2f}" for p in pts),
                  "hourly": "precipitation", "forecast_days": 10, "timezone": "UTC"}
        try:
            data = _get_json(OPEN_METEO_URL, params, session=session)
        except CycloneError:
            continue
        results = data if isinstance(data, list) else [data]
        for p, res in zip(pts, results):
            hourly = (res or {}).get("hourly") or {}
            times, vals = hourly.get("time") or [], hourly.get("precipitation") or []
            lo, hi = p.valid - timedelta(hours=window_h / 2), p.valid + timedelta(hours=window_h / 2)
            total, n = 0.0, 0
            for t, v in zip(times, vals):
                try:
                    ts = datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if lo <= ts < hi and v is not None:
                    total += float(v)
                    n += 1
            if n >= window_h // 2:     # couverture suffisante de la fenêtre
                p.precip_mm = round(total, 1)
                ok = True
    return ok


# --------------------------------------------------------------------------- lien vers le bulletin source
NHC_HUB = "https://www.nhc.noaa.gov/"
JTWC_HUB = "https://www.metoc.navy.mil/jtwc/jtwc.html"


def _basin_family(basin: str) -> tuple[str, str]:
    """(centre, code court) : ('NHC', 'al'|'ep'|'cp') ou ('JTWC', 'wp'|'io'|'sh') ; ('', '') si inconnu."""
    b = (basin or "").strip().upper()
    direct = {"AL": ("NHC", "al"), "EP": ("NHC", "ep"), "CP": ("NHC", "cp"), "WP": ("JTWC", "wp"), "IO": ("JTWC", "io"),
              "NI": ("JTWC", "io"), "SI": ("JTWC", "sh"), "SP": ("JTWC", "sh"), "SH": ("JTWC", "sh")}
    if b in direct:
        return direct[b]
    for word, val in (("ATLANT", ("NHC", "al")), ("CENTRAL", ("NHC", "cp")), ("EAST", ("NHC", "ep")), ("WEST", ("JTWC", "wp")),
                      ("INDIAN", ("JTWC", "io")), ("SOUTH", ("JTWC", "sh"))):
        if word in b:
            return val
    return "", ""


def source_link(storm: Storm, p: TrackPoint, step_ok: bool = True) -> tuple[str, str]:
    """Adresse du bulletin officiel de la tempête, avec un « fragment de texte » qui fait défiler la page jusqu'à l'heure
    de validité du point (navigateurs récents) : (url, libellé). Le format des adresses n'a pas pu être vérifié en ligne."""
    center, code = _basin_family(storm.basin)
    t = p.valid or storm.advisory_time
    if not center or not storm.number:
        return (NHC_HUB, "Site du National Hurricane Center") if center != "JTWC" else (JTWC_HUB, "Site du JTWC")
    year = (storm.advisory_time or t or datetime.now(timezone.utc)).year
    nn = f"{storm.number:02d}"
    if center == "NHC":
        m = re.match(r"\d+", str(storm.advisory or ""))
        if not m:
            return NHC_HUB, "Site du National Hurricane Center"
        url = f"https://www.nhc.noaa.gov/archive/{year}/{code}{nn}/{code}{nn}{year}.fstadv.{int(m.group()):03d}.shtml"
        frag = t.strftime("%d/%H%MZ") if (t and step_ok) else ""
        label = "Bulletin NHC"
    else:
        url = f"https://www.metoc.navy.mil/jtwc/products/{code}{nn}{str(year)[2:]}web.txt"
        frag = t.strftime("%d%H%MZ") if (t and step_ok) else ""
        label = "Bulletin JTWC"
    if frag:
        from urllib.parse import quote
        url += "#:~:text=" + quote(frag, safe="")
    return url, label


def map_link(storm: Storm) -> Optional[tuple[str, str]]:
    """Adresse de la carte officielle de la tempête (cône de prévision NHC, image de trajectoire JTWC) : (url, libellé).
    None si le centre ou le numéro de la tempête est inconnu. Format des adresses non vérifié en ligne."""
    center, code = _basin_family(storm.basin)
    if not center or not storm.number:
        return None
    if center == "NHC":
        short = "at" if code == "al" else code
        return f"https://www.nhc.noaa.gov/graphics_{short}{storm.number}.shtml", "Carte NHC"
    year = (storm.advisory_time or datetime.now(timezone.utc)).year
    return f"https://www.metoc.navy.mil/jtwc/products/{code}{storm.number:02d}{str(year)[2:]}.gif", "Carte JTWC"
