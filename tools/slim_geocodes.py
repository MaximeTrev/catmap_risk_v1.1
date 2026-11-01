"""Allège le fichier « Geocodes » de MeteoAlarm (31 Mo) pour le déposer sur GitHub (limite 25 Mo).

Usage (à lancer depuis la racine du dépôt) :
    python tools/slim_geocodes.py chemin/vers/geocodes.json data/meteoalarm_geocodes_light.json.gz
Produit un fichier compressé de quelques Mo (contours simplifiés, champs utiles seulement).
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catmap import meteoalarm as ma  # noqa: E402


def main(src: str, dst: str) -> None:
    geo = ma.load_geocodes_bytes(Path(src).read_bytes())
    out = {}
    for code, (ctry, name, rings) in geo["by_code"].items():
        out[code] = {"country": ctry, "name": name, "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[[[float(x), float(y)] for x, y in r]] for r in rings]}}
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(out)} zones -> {dst} ({Path(dst).stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
