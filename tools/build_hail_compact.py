"""Précalcule, pour chaque maille de 0,25°, les grandeurs utilisées par l'onglet Grêle et les range dans data/grele/hail_compact.npz.

Contenu : moyenne annuelle d'événements ≥ 5 cm (climatologie), tendance par décennie, p-value, niveaux annuels de périodes de retour 2, 5, 10, 20 et 50 ans.
Une seule exécution suffit ; ensuite l'application lit ce fichier (quelques Mo) au lieu de télécharger les 74 fichiers annuels.

Usage :
    python tools/build_hail_compact.py "C:\\Users\\vous\\Desktop\\Hail_data" data/grele/hail_compact.npz

Dossier source : prob_hail_ge5cm_1950.nc … prob_hail_ge5cm_2023.nc et hail_ge5cm_trends_1950_2023.nc (hail_ge5cm_climatology_1950_2023.nc facultatif : contrôle).
Dépendances : numpy et une bibliothèque NetCDF (netCDF4 ou h5netcdf, déjà présente si vous utilisez xarray). Mémoire nécessaire : environ 1 Go.
Licence des données (CC BY-NC-ND 4.0) : ce fichier est dérivé ; ne le publiez pas si votre dépôt est public.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catmap import hail  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(hail.build_compact(sys.argv[1], out))
