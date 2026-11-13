"""Hauteur d'eau attendue par point : cartes de débordement fluvial JRC / GloFAS (v2.1.2, 90 m, mondial).

Source : JRC Data Store, ``https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/`` (licence ODbL v1.0 d'après le
README du jeu de données ; citation : Baugh et al. 2024, European Commission JRC). Un dossier par période de retour ; par tuile :
``ID<n>_<N|S><lat>_<E|W><lon>_RP<rp>_depth.tif`` (profondeur brute, m) et ``…_depth_reclass.tif`` (classe 1 : < 1 m ; 2 : 1 à < 3 m ;
3 : 3 à < 10 m ; 4 : > 10 m). ``tile_extents.geojson`` (à la racine) décrit les 271 tuiles.

Principe « à la demande » : le catalogue des tuiles est lu une fois, chaque point est rattaché à sa tuile, et seules les tuiles utiles
sont téléchargées (cache disque) puis lues segment par segment (``tifffile``, Python pur : ni GDAL ni rasterio). La mémoire reste
bornée, même pour une tuile de 12 000 × 12 000 pixels.

Hypothèses NON vérifiées depuis l'environnement de développement (réseau bloqué) et traitées de façon tolérante, avec diagnostic :
nom exact des dossiers de période de retour (``RP50`` / ``RP050``), listing HTTP du dossier, propriétés de ``tile_extents.geojson``,
codec de compression des tuiles. Voir ``diagnose``.

Convention de sortie : pour un point situé dans une tuile, « aucune valeur » (sans-donnée) vaut 0 = aucune inondation modélisée à
cet endroit (≠ absence de risque : les petits cours d'eau, bassins < 500 km², ne sont pas modélisés). Un point hors de toute tuile
reste vide avec le statut « hors tuile ».
"""
from __future__ import annotations

import math
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/"
TILE_EXTENTS_URL = BASE_URL + "tile_extents.geojson"
RPS = (10, 20, 50, 75, 100, 200, 500)
KINDS = ("raw", "reclass")
RISK = "debordement"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "catmap_jrc_tiles"

CLASS_LABELS = {0: "Aucune inondation modélisée", 1: "< 1 m", 2: "1 à < 3 m", 3: "3 à < 10 m", 4: "> 10 m"}
CLASS_COLORS = {None: "#9e9e9e", 0: "#d9ead3", 1: "#9ecae1", 2: "#4292c6", 3: "#08519c", 4: "#54278f"}


class TileError(Exception):
    """Erreur sur une tuile (catalogue, nom de fichier, téléchargement, lecture)."""


def column_name(rp: int, kind: str) -> str:
    """``AAA_BBB_CCC`` : risque, scénario, type de donnée (ex. ``debordement_RP100_raw``)."""
    return f"{RISK}_RP{int(rp)}_{kind}"


def flood_class(value, kind: str) -> Optional[int]:
    """Classe 0 à 4 pour la couleur et la légende (``None`` = pas de donnée)."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    v = float(value)
    if kind == "reclass":
        return int(min(4, max(0, round(v))))
    if v <= 0:
        return 0
    return 1 if v < 1 else 2 if v < 3 else 3 if v < 10 else 4


# --------------------------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class Tile:
    id: int
    west: float
    south: float
    east: float
    north: float
    props: dict = field(default_factory=dict, compare=False, hash=False)

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


def _tile_id(props: dict) -> Optional[int]:
    for k, v in props.items():
        if isinstance(v, str):
            m = re.search(r"ID\s*0*(\d+)", v, re.I)
            if m:
                return int(m.group(1))
    for k, v in props.items():
        if str(k).lower() in ("id", "tile_id", "tileid", "tile", "fid", "objectid") and str(v).strip().lstrip("-").isdigit():
            return int(v)
    for k, v in props.items():
        if "id" in str(k).lower() and str(v).strip().isdigit():
            return int(v)
    return None


def _bbox(geom: dict) -> Optional[tuple[float, float, float, float]]:
    xs: list[float] = []
    ys: list[float] = []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(float(c[0]))
            ys.append(float(c[1]))
        else:
            for x in c or []:
                walk(x)

    walk((geom or {}).get("coordinates"))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def parse_tile_extents(gj: dict) -> list[Tile]:
    tiles: list[Tile] = []
    for i, f in enumerate((gj or {}).get("features") or []):
        props = f.get("properties") or {}
        tid = _tile_id(props)
        bb = _bbox(f.get("geometry") or {})
        if bb is None:
            continue
        tiles.append(Tile(tid if tid is not None else -(i + 1), bb[0], bb[1], bb[2], bb[3], props))
    return tiles


@dataclass
class Catalog:
    tiles: list[Tile]

    def __post_init__(self):
        self._w = np.array([t.west for t in self.tiles])
        self._s = np.array([t.south for t in self.tiles])
        self._e = np.array([t.east for t in self.tiles])
        self._n = np.array([t.north for t in self.tiles])

    def locate_many(self, lats: np.ndarray, lons: np.ndarray) -> list[Optional[Tile]]:
        out: list[Optional[Tile]] = []
        for la, lo in zip(lats, lons):
            if not (np.isfinite(la) and np.isfinite(lo)) or not len(self.tiles):
                out.append(None)
                continue
            hit = np.nonzero((self._s <= la) & (la <= self._n) & (self._w <= lo) & (lo <= self._e))[0]
            out.append(self.tiles[int(hit[0])] if len(hit) else None)
        return out


def _http_get(session, url: str, **kw):
    import requests
    return (session or requests).get(url, timeout=kw.pop("timeout", 60), **kw)


def load_catalog(session=None, url: str = TILE_EXTENTS_URL) -> Catalog:
    try:
        r = _http_get(session, url, headers={"User-Agent": "CatMap"})
    except Exception as e:  # noqa: BLE001
        raise TileError(f"catalogue des tuiles inaccessible ({type(e).__name__}: {str(e)[:150]})") from e
    if r.status_code != 200:
        raise TileError(f"catalogue des tuiles : HTTP {r.status_code}")
    try:
        tiles = parse_tile_extents(r.json())
    except ValueError as e:
        raise TileError("catalogue des tuiles : réponse non JSON") from e
    if not tiles:
        raise TileError("catalogue des tuiles vide ou de structure inattendue")
    return Catalog(tiles)


# --------------------------------------------------------------------------------- noms de fichiers
_HREF = re.compile(r"""href\s*=\s*["']([^"'?#]+?\.tiff?)["']""", re.I)


def _kind_of(name: str) -> Optional[str]:
    n = name.lower()
    if re.search(r"_depth_reclass\.tiff?$", n):
        return "reclass"
    if re.search(r"_depth\.tiff?$", n):
        return "raw"
    return None


def parse_listing(html: str) -> dict[int, dict[str, str]]:
    """Index HTML d'un dossier -> {id de tuile: {'raw': nom, 'reclass': nom}}."""
    out: dict[int, dict[str, str]] = {}
    for href in _HREF.findall(html or ""):
        name = href.rsplit("/", 1)[-1]
        m = re.match(r"ID0*(\d+)_", name, re.I)
        k = _kind_of(name)
        if m and k:
            out.setdefault(int(m.group(1)), {})[k] = name
    return out


def candidate_names(tile: Tile, rp: int, kind: str) -> list[str]:
    """Noms plausibles quand le dossier n'est pas listable : ID<n>_<N|S><lat>_<E|W><lon>_RP<rp>_depth[_reclass].tif(f)."""
    suffix = "_depth_reclass" if kind == "reclass" else "_depth"
    out = []
    for lat in (tile.north, tile.south):
        for lon in (tile.west, tile.east):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            stem = f"ID{tile.id}_{ns}{abs(int(round(lat)))}_{ew}{abs(int(round(lon)))}_RP{rp}{suffix}"
            out += [stem + ".tif", stem + ".tiff"]
    return list(dict.fromkeys(out))


class Resolver:
    """Trouve l'adresse d'un fichier de tuile : dossier de la période de retour (RP50 ou RP050), listing HTTP, sinon noms candidats."""

    def __init__(self, session=None, base_url: str = BASE_URL):
        self.session, self.base_url = session, base_url
        self.folders: dict[int, Optional[str]] = {}
        self.listings: dict[int, dict[int, dict[str, str]]] = {}
        self.notes: list[str] = []
        self._lock = threading.Lock()

    def _exists(self, url: str) -> bool:
        try:
            r = _http_get(self.session, url, stream=True, headers={"Range": "bytes=0-0", "User-Agent": "CatMap"}, timeout=30)
        except Exception:  # noqa: BLE001
            return False
        ok = r.status_code in (200, 206)
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
        return ok

    def folder(self, rp: int) -> str:
        with self._lock:
            if rp in self.folders:
                if self.folders[rp] is None:
                    raise TileError(f"dossier RP{rp} introuvable ({'; '.join(self.notes[-2:])})")
                return self.folders[rp]
            for cand in (f"RP{rp}", f"RP{rp:03d}", f"RP{rp:04d}"):
                url = f"{self.base_url}{cand}/"
                try:
                    r = _http_get(self.session, url, headers={"User-Agent": "CatMap"})
                except Exception as e:  # noqa: BLE001
                    self.notes.append(f"{cand}: {type(e).__name__}")
                    continue
                if r.status_code == 200:
                    lst = parse_listing(r.text)
                    if lst:
                        self.folders[rp], self.listings[rp] = url, lst
                        return url
                    self.notes.append(f"{cand}: dossier lisible mais sans fichier .tif reconnu")
                    self.folders[rp], self.listings[rp] = url, {}     # dossier existe, listing inexploitable : noms candidats
                    return url
                self.notes.append(f"{cand}: HTTP {r.status_code}")
            self.folders[rp] = None
            raise TileError(f"dossier RP{rp} introuvable ({'; '.join(self.notes[-3:])})")

    def url(self, tile: Tile, rp: int, kind: str) -> str:
        folder = self.folder(rp)
        name = self.listings.get(rp, {}).get(tile.id, {}).get(kind)
        if name:
            return folder + name
        for cand in candidate_names(tile, rp, kind):
            if self._exists(folder + cand):
                return folder + cand
        raise TileError(f"fichier introuvable pour la tuile ID{tile.id} (RP{rp}, {kind})")


# --------------------------------------------------------------------------------- téléchargement et lecture
def tiff_complete(path: Path) -> tuple[bool, str]:
    """Contrôle d'intégrité : tous les blocs annoncés par le fichier doivent tenir dans sa taille (détecte un téléchargement tronqué)."""
    path = Path(path)
    try:
        import tifffile
        size = path.stat().st_size
        with tifffile.TiffFile(str(path)) as tf:
            pg = tf.pages[0]
            ends = [o + c for o, c in zip(pg.dataoffsets, pg.databytecounts)]
            if not ends:
                return False, "aucun bloc de données"
            if max(ends) > size:
                return False, f"fichier tronqué : {size / 1e6:.1f} Mo reçus pour {max(ends) / 1e6:.1f} Mo attendus"
        return True, ""
    except ImportError:
        return True, "contrôle non effectué (tifffile absent)"
    except Exception as e:  # noqa: BLE001
        return False, f"en-tête illisible ({type(e).__name__}: {str(e)[:120]})"


def _cache_path(url: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / re.sub(r"[^\w.\-]+", "_", url.split("flood_hazard/")[-1])


def download_tile(url: str, cache_dir: Path = DEFAULT_CACHE, session=None, max_mb: float = 400.0, attempts: int = 3) -> Path:
    """Télécharge une tuile dans le cache disque. Un fichier en cache n'est réutilisé que s'il est complet. Un flux interrompu est repris
    (requête partielle si le serveur l'accepte, sinon nouveau téléchargement), puis le fichier est contrôlé. ``TileError`` en cas d'échec."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(url, cache_dir)
    if dest.exists() and dest.stat().st_size > 0:
        ok, _ = tiff_complete(dest)
        if ok:
            return dest
        dest.unlink(missing_ok=True)                       # fichier en cache incomplet ou corrompu : on le refait
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = "échec"
    try:
        tmp.unlink(missing_ok=True)
        expected = None
        for attempt in range(1, attempts + 1):
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "CatMap", "Accept-Encoding": "identity"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                r = _http_get(session, url, stream=True, headers=headers, timeout=120)
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            if have and r.status_code == 200:              # le serveur ignore Range : on repart de zéro
                tmp.unlink(missing_ok=True)
                have = 0
            elif r.status_code not in (200, 206):
                raise TileError(f"téléchargement refusé (HTTP {r.status_code})")
            if expected is None:
                cl = r.headers.get("Content-Length")
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr and cr.rsplit("/", 1)[-1].isdigit():
                    expected = int(cr.rsplit("/", 1)[-1])
                elif cl and cl.isdigit():
                    expected = have + int(cl)
            try:
                with open(tmp, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            if f.tell() > max_mb * 1e6:
                                raise TileError(f"fichier trop volumineux (> {max_mb:.0f} Mo)")
            except TileError:
                raise
            except Exception as e:  # noqa: BLE001 - flux coupé : on reprendra là où il s'est arrêté
                last = f"flux interrompu ({type(e).__name__})"
            size = tmp.stat().st_size if tmp.exists() else 0
            if size and (expected is None or size >= expected):
                ok, why = tiff_complete(tmp)
                if ok:
                    tmp.replace(dest)
                    return dest
                last = why
                tmp.unlink(missing_ok=True)
                expected = None
            else:
                last = f"{size / 1e6:.1f} Mo reçus sur {(expected or 0) / 1e6:.1f} Mo"
        raise TileError(f"téléchargement incomplet après {attempts} essais ({last})")
    except TileError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise TileError(f"téléchargement impossible ({type(e).__name__}: {str(e)[:150]})") from e


def cache_info(cache_dir: Path = DEFAULT_CACHE) -> tuple[int, float]:
    """(nombre de fichiers, taille en Mo) du cache de tuiles."""
    d = Path(cache_dir)
    files = [f for f in d.glob("*") if f.is_file()] if d.exists() else []
    return len(files), round(sum(f.stat().st_size for f in files) / 1e6, 1)


def clear_cache(cache_dir: Path = DEFAULT_CACHE) -> tuple[int, float]:
    """Supprime les tuiles du cache ; retourne (fichiers supprimés, Mo libérés)."""
    n, mb = cache_info(cache_dir)
    for f in (Path(cache_dir).glob("*") if Path(cache_dir).exists() else []):
        if f.is_file():
            f.unlink(missing_ok=True)
    return n, mb


def _georef(page) -> tuple[float, float, float, float]:
    """(x0, y0, sx, sy) : coin haut-gauche et taille de pixel (sy > 0 vers le bas) lus dans les tags GeoTIFF."""
    tag = page.tags.get
    if tag(34264) is not None:                               # ModelTransformation
        m = tag(34264).value
        return float(m[3]), float(m[7]), float(m[0]), -float(m[5])
    if tag(33550) is not None and tag(33922) is not None:
        scale, tie = tag(33550).value, tag(33922).value
        return (float(tie[3]) - float(tie[0]) * float(scale[0]), float(tie[4]) + float(tie[1]) * float(scale[1]),
                float(scale[0]), float(scale[1]))
    raise TileError("géoréférencement absent du fichier (tags GeoTIFF)")


def _nodata(page) -> Optional[float]:
    nd = page.tags.get(42113).value if page.tags.get(42113) is not None else None
    try:
        return float(str(nd).strip("\x00 ")) if nd is not None else None
    except ValueError:
        return None


def to_classes(arr: np.ndarray, kind: str, nodata: Optional[float]) -> np.ndarray:
    """Classe de légende 0 à 4 par pixel (0 = aucune inondation / sans-donnée). Mêmes seuils que ``flood_class``."""
    a = np.asarray(arr, dtype="float64")
    ok = np.isfinite(a) & (a > 0)
    if nodata is not None:
        ok &= a != nodata
    if kind == "reclass":
        return np.where(ok, np.clip(a, 0, 4), 0).astype("uint8")
    cls = np.digitize(a, [1.0, 3.0, 10.0]) + 1             # <1 -> 1 ; 1..<3 -> 2 ; 3..<10 -> 3 ; >=10 -> 4
    return np.where(ok, cls, 0).astype("uint8")


def render_preview(path: Path, kind: str, max_px: int = 1000) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Image RGBA de la tuile aux couleurs de la légende, pour la carte. Retourne (rgba, (sud, ouest, nord, est)).
    Lecture bloc par bloc ; la réduction garde la classe MAXIMALE de chaque groupe de pixels (les cellules inondées ne disparaissent pas).
    Les pixels de classe 0 (aucune inondation, sans-donnée) sont transparents."""
    try:
        import tifffile
    except ImportError as e:
        raise TileError("bibliothèque « tifffile » absente : ajoutez-la à requirements.txt") from e
    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            H, W = int(page.imagelength), int(page.imagewidth)
            x0, y0, sx, sy = _georef(page)
            nodata = _nodata(page)
            f = max(1, int(math.ceil(max(H, W) / float(max_px))))
            ph, pw = int(math.ceil(H / f)), int(math.ceil(W / f))
            prev = np.zeros((ph, pw), dtype="uint8")
            tiled = bool(page.is_tiled)
            th, tw = (int(page.tilelength), int(page.tilewidth)) if tiled else (int(page.rowsperstrip), W)
            across = int(math.ceil(W / tw))
            fh = tf.filehandle
            for seg in range(len(page.dataoffsets)):
                fh.seek(page.dataoffsets[seg])
                data = fh.read(page.databytecounts[seg])
                arr, _, _ = page.decode(data, seg)
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    raise TileError(f"organisation de fichier non gérée (dimensions {arr.shape})")
                r0, c0 = ((seg // across) * th, (seg % across) * tw) if tiled else (seg * th, 0)
                h, w = min(arr.shape[0], H - r0), min(arr.shape[1], W - c0)
                if h <= 0 or w <= 0:
                    continue
                c = to_classes(arr[:h, :w], kind, nodata)
                if not c.any():
                    continue
                cr, cc = (r0 + np.arange(h)) // f, (c0 + np.arange(w)) // f
                rs = np.flatnonzero(np.r_[True, np.diff(cr) != 0])
                cs = np.flatnonzero(np.r_[True, np.diff(cc) != 0])
                red = np.maximum.reduceat(np.maximum.reduceat(c, rs, axis=0), cs, axis=1)
                ix = np.ix_(cr[rs], cc[cs])
                prev[ix] = np.maximum(prev[ix], red)
    except TileError:
        raise
    except Exception as e:  # noqa: BLE001
        raise TileError(f"image impossible à produire ({type(e).__name__}: {str(e)[:160]})") from e
    pal = np.zeros((5, 4), dtype="uint8")
    for k in (1, 2, 3, 4):
        hx = CLASS_COLORS[k].lstrip("#")
        pal[k] = [int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16), 255]
    return pal[prev], (y0 - H * sy, x0, y0, x0 + W * sx)


def sample_geotiff(path: Path, lons, lats) -> list[Optional[float]]:
    """Valeur du pixel pour chaque point (``None`` = sans-donnée ou hors image). Lecture par segments : mémoire bornée."""
    try:
        import tifffile
    except ImportError as e:
        raise TileError("bibliothèque « tifffile » absente : ajoutez-la à requirements.txt") from e
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    out: list[Optional[float]] = [None] * len(lons)
    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            H, W = int(page.imagelength), int(page.imagewidth)
            x0, y0, sx, sy = _georef(page)
            nodata = _nodata(page)
            cols = np.floor((lons - x0) / sx).astype(np.int64)
            rows = np.floor((y0 - lats) / sy).astype(np.int64)
            inside = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
            tiled = bool(page.is_tiled)
            th, tw = (int(page.tilelength), int(page.tilewidth)) if tiled else (int(page.rowsperstrip), W)
            across = int(math.ceil(W / tw))
            seg_of = (rows // th) * across + (cols // tw) if tiled else rows // th
            fh = tf.filehandle
            for seg in np.unique(seg_of[inside]):
                fh.seek(page.dataoffsets[int(seg)])
                data = fh.read(page.databytecounts[int(seg)])
                arr, idx, _ = page.decode(data, int(seg))
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    raise TileError(f"organisation de fichier non gérée (dimensions {arr.shape})")
                for i in np.nonzero(inside & (seg_of == seg))[0]:
                    ry = int(rows[i] - (seg // across) * th) if tiled else int(rows[i] % th)
                    rx = int(cols[i] % tw) if tiled else int(cols[i])
                    if ry >= arr.shape[0] or rx >= arr.shape[1]:
                        continue
                    v = float(arr[ry, rx])
                    if not math.isfinite(v) or (nodata is not None and v == nodata):
                        continue
                    out[i] = v
    except TileError:
        raise
    except Exception as e:  # noqa: BLE001 - codec manquant, fichier tronqué…
        raise TileError(f"lecture impossible ({type(e).__name__}: {str(e)[:200]})") from e
    return out


# --------------------------------------------------------------------------------- calcul
def plan(df: pd.DataFrame, lat_col: str, lon_col: str, catalog: Catalog) -> dict:
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(float)
    tiles = catalog.locate_many(lat, lon)
    ids = sorted({t.id for t in tiles if t is not None})
    return {"lat": lat, "lon": lon, "tiles": tiles, "tile_ids": ids}


def estimate_depths(df: pd.DataFrame, lat_col: str, lon_col: str, rps: list[int], kinds: list[str], catalog: Optional[Catalog] = None,
                    session=None, cache_dir: Path = DEFAULT_CACHE,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> tuple[pd.DataFrame, dict]:
    """Ajoute ``debordement_RP<rp>_<raw|reclass>`` + ``debordement_statut`` + ``debordement_tuile``. Ne lève pas pour une tuile en échec."""
    rps = [r for r in rps if r in RPS]
    kinds = [k for k in kinds if k in KINDS]
    catalog = catalog or load_catalog(session)
    out = df.copy().reset_index(drop=True)
    pl = plan(out, lat_col, lon_col, catalog)
    tiles = pl["tiles"]
    by_tile: dict[int, list[int]] = {}
    for i, t in enumerate(tiles):
        if t is not None:
            by_tile.setdefault(t.id, []).append(i)
    tile_by_id = {t.id: t for t in tiles if t is not None}
    resolver = Resolver(session)
    cols = {column_name(rp, k): [None] * len(out) for rp in rps for k in kinds}
    problems: dict[str, list[str]] = {}
    err_by_row: dict[int, list[str]] = {}
    jobs = [(rp, k, tid) for rp in rps for k in kinds for tid in by_tile]
    n_dl = 0
    files_read: dict[str, str] = {}
    for done, (rp, k, tid) in enumerate(jobs, start=1):
        name = column_name(rp, k)
        rows = by_tile[tid]
        try:
            url = resolver.url(tile_by_id[tid], rp, k)
            path = download_tile(url, cache_dir, session)
            n_dl += 1
            files_read[f"{name}|{tid}"] = str(path)
            try:
                vals = sample_geotiff(path, pl["lon"][rows], pl["lat"][rows])
            except TileError as first:                     # fichier en cache abîmé : on le supprime et on retélécharge une fois
                Path(path).unlink(missing_ok=True)
                path = download_tile(url, cache_dir, session)
                try:
                    vals = sample_geotiff(path, pl["lon"][rows], pl["lat"][rows])
                except TileError as again:
                    raise TileError(f"{again} (déjà observé au premier essai : {first}) ; fichier peut-être incomplet côté serveur") from again
            for i, v in zip(rows, vals):
                cols[name][i] = 0.0 if v is None else (int(v) if k == "reclass" else round(float(v), 3))
        except TileError as e:
            problems.setdefault(f"RP{rp} {k} tuile ID{tid}", []).append(str(e))
            for i in rows:
                err_by_row.setdefault(i, []).append(f"RP{rp} {k}: {e}")
        if on_progress:
            on_progress(done, len(jobs))
    for name, vals in cols.items():
        out[name] = vals
    out[f"{RISK}_tuile"] = [f"ID{t.id}" if t is not None else None for t in tiles]
    status = []
    for i, t in enumerate(tiles):
        if t is None:
            status.append("hors tuile (mer ou zone non couverte)")
        elif i in err_by_row:
            status.append("partiel : " + " ; ".join(dict.fromkeys(err_by_row[i]))[:300])
        else:
            status.append("ok")
    out[f"{RISK}_statut"] = status
    detail = []
    for tid, rows in by_tile.items():
        t = tile_by_id[tid]
        errs = [f"{key.split(' tuile')[0]}: {v[0]}" for key, v in problems.items() if key.endswith(f"tuile ID{tid}")]
        detail.append({"id": tid, "west": t.west, "south": t.south, "east": t.east, "north": t.north, "points": len(rows),
                       "statut": "erreur" if errs else "ok", "erreurs": errs})
    summary = {"fichiers_lus": files_read, "tuiles_detail": detail, "points": len(out), "hors_tuile": int(sum(t is None for t in tiles)), "tuiles": len(by_tile),
               "fichiers": len(jobs), "telecharges_ou_en_cache": n_dl, "problemes": problems, "notes_dossiers": resolver.notes}
    return out, summary


def new_columns(rps: list[int], kinds: list[str]) -> list[str]:
    return [column_name(r, k) for r in rps for k in kinds] + [f"{RISK}_tuile", f"{RISK}_statut"]


def diagnose(rp: int = 100, session=None, catalog: Optional[Catalog] = None) -> list[dict]:
    """Contrôle pas à pas des hypothèses (catalogue, dossier, noms, requêtes partielles) : une ligne par étape."""
    rows: list[dict] = []

    def add(step, ok, detail):
        rows.append({"étape": step, "résultat": "OK" if ok else "ÉCHEC", "détail": str(detail)[:300]})

    try:
        cat = catalog or load_catalog(session)
        t = cat.tiles[0]
        add("Catalogue tile_extents.geojson", True, f"{len(cat.tiles)} tuile(s) ; exemple : ID{t.id} emprise ({t.west}, {t.south}, {t.east}, {t.north}) ; propriétés : {list(t.props)[:6]}")
    except TileError as e:
        add("Catalogue tile_extents.geojson", False, e)
        return rows
    res = Resolver(session)
    try:
        folder = res.folder(rp)
        n = len(res.listings.get(rp, {}))
        add(f"Dossier RP{rp}", True, f"{folder} ; fichiers listés : {n}" + ("" if n else " (listing HTTP inexploitable : noms déduits des coordonnées)"))
    except TileError as e:
        add(f"Dossier RP{rp}", False, e)
        return rows
    try:
        url = res.url(cat.tiles[0], rp, "raw")
        add("Nom de fichier (raw)", True, url)
    except TileError as e:
        add("Nom de fichier (raw)", False, e)
        return rows
    try:
        r = _http_get(session, url, stream=True, headers={"Range": "bytes=0-99", "User-Agent": "CatMap"}, timeout=40)
        size = r.headers.get("Content-Range") or r.headers.get("Content-Length")
        add("Requêtes partielles (Range)", r.status_code == 206, f"HTTP {r.status_code} ; taille annoncée : {size}" +
            ("" if r.status_code == 206 else " (le serveur ignore Range : téléchargement complet des tuiles utiles, ce qui reste possible)"))
        r.close()
    except Exception as e:  # noqa: BLE001
        add("Requêtes partielles (Range)", False, f"{type(e).__name__}: {e}")
    try:
        path = download_tile(url, DEFAULT_CACHE, session)
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            pg = tf.pages[0]
            add("Téléchargement et lecture de l'en-tête", True,
                f"{path.stat().st_size / 1e6:.1f} Mo ; {pg.imagewidth}×{pg.imagelength} px ; {'tuilé' if pg.is_tiled else 'par bandes'} ; "
                f"compression {pg.compression.name if hasattr(pg.compression, 'name') else pg.compression} ; type {pg.dtype}")
        c = cat.tiles[0]
        v = sample_geotiff(path, [(c.west + c.east) / 2], [(c.south + c.north) / 2])
        add("Lecture d'un pixel au centre de la tuile", True, f"valeur : {v[0]} (vide = sans-donnée)")
    except TileError as e:
        add("Téléchargement / lecture", False, e)
    except Exception as e:  # noqa: BLE001
        add("Téléchargement / lecture", False, f"{type(e).__name__}: {e}")
    return rows
