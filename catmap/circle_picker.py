"""Composant Streamlit « sélection d'un cercle sur une carte » (sans étape de build).

Clic 1 : centre. Déplacement de la souris : le rayon grandit, affiché en mètres au-dessus de la carte en
temps réel. Clic 2 : valide le rayon. Le composant renvoie ``{"lat", "lon", "radius_m"}`` une fois le cercle validé
(``None`` avant). Pour recommencer, changer la ``key`` (nouvelle instance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

_FRONTEND = Path(__file__).parent / "components" / "circle_picker"
_component = None


def circle_picker(center: Sequence[float] = (46.6, 2.5), zoom: int = 6, max_radius_m: float = 2000,
                  key: str = "area_picker", height: int = 640) -> Optional[dict]:
    global _component
    if _component is None:
        import streamlit.components.v1 as components  # import tardif : les tests n'ont pas besoin de Streamlit
        _component = components.declare_component("catmap_circle_picker", path=str(_FRONTEND))
    return _component(center=[float(center[0]), float(center[1])], zoom=int(zoom),
                      max_radius_m=float(max_radius_m), height=int(height), key=key, default=None)
