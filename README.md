# CatMap Risk v3

Géolocalisation d'établissements d'entreprises à partir d'OpenStreetMap (Overpass), rattachement au **pays**
et à un **secteur NAF approché**, cartographie, et croisement avec la **vigilance Météo-France**.

Cette version reprend l'application `CatMap_risk` et corrige les 20 anomalies relevées dans la documentation
technique (annexe A). La logique métier est dans le paquet `catmap/` (testé sans Streamlit) ; `app.py` ne contient que l'interface.

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # renseigner au minimum CATMAP_CONTACT (e-mail ou URL) puis exporter les variables
streamlit run app.py            # peut être lancé depuis n'importe quel dossier
python -m unittest discover -s tests -t .      # 83 tests (ou : pytest)
```

Aucune bibliothèque géospatiale lourde n'est requise (ni GDAL, geopandas, shapely, pyproj, overpy, selenium).

## Structure

| Chemin | Rôle |
|---|---|
| `app.py` | Interface Streamlit : Géolocalisation, Cartographies avancées, Temps réel |
| `catmap/names.py` | Nettoyage des formes juridiques, normalisation, regex serveur, appariement strict et score |
| `catmap/overpass.py` | Construction de requête, client (retries, bascule, résultats partiels, cache disque) |
| `catmap/parsing.py` | JSON Overpass → sites : géométrie, surface, multipolygones, fusion des doublons |
| `catmap/geo.py` | Géométrie en Python pur : surface exacte sur l'ellipsoïde, centroïde, anneaux |
| `catmap/shapefile.py`, `polygon_index.py`, `countries.py` | Lecture du shapefile pays et jointure spatiale vectorisée |
| `catmap/naf.py`, `data/naf_mapping.csv` | Nomenclature NAF complète et classification des tags OSM |
| `catmap/pipeline.py` | Orchestration, rapport par entreprise, parallélisme borné |
| `catmap/inputs.py` | Lecture robuste des listes (CSV, XLSX, texte) |
| `catmap/vigilance.py` | Authentification, analyse de la réponse, départements, exposition des sites |
| `catmap/qa.py`, `scripts/qa_*.py` | Contrôle qualité reproductible des coordonnées |
| `scripts/build_naf_mapping.py` | Régénère `data/naf_mapping.csv` depuis des règles lisibles |
| `docker-compose.overpass.yml` | Instance Overpass locale (optionnel) |

## Configuration

Variables d'environnement ou `.streamlit/secrets.toml` (voir `.env.example`). Rien de secret dans le code.

| Variable | Défaut | Effet |
|---|---|---|
| `CATMAP_CONTACT` | vide | Ajouté au User-Agent (bonne pratique Overpass) |
| `OVERPASS_URLS` | 2 serveurs publics | Liste d'endpoints séparés par des virgules ; instance locale possible |
| `OVERPASS_TIMEOUT_S` | 180 | `[timeout:]` de la requête |
| `OVERPASS_MAX_ATTEMPTS` | 3 | Essais par endpoint (attente exponentielle + jitter, `Retry-After` respecté) |
| `OVERPASS_MIN_INTERVAL_S` | 1.0 (0 en local) | Délai minimal entre requêtes |
| `CATMAP_MAX_WORKERS` | 1 (4 en local) | Parallélisme ; à n'augmenter qu'avec une instance locale |
| `CATMAP_CACHE_TTL_HOURS` | 168 | `never` = instantané figé |
| `CATMAP_CACHE_NAMESPACE` | vide | Isole les résultats d'un instantané (ex. `fr-2026-09`) |
| `METEOFRANCE_API_KEY` ou `..._CLIENT_ID` / `..._CLIENT_SECRET` | vide | Page « Temps réel » |

## Comment la recherche fonctionne

1. Le nom est nettoyé (formes juridiques finales uniquement, en mots entiers) puis normalisé.
2. Le **serveur** reçoit une regex volontairement simple (sous-chaîne tolérante : accents, casse, séparateurs, sans classe de caractères ni échappement) sur `name` et `brand` (et `operator` en option), plus `brand:wikidata` si fourni.
3. Côté **Python**, chaque élément est vérifié en **mots entiers** et reçoit un score : 1,00 (Wikidata), 0,90 (nom identique), 0,75 (commence par), 0,60 (contenu), moins 0,20 pour `operator`.
4. Géométrie : nodes, ways fermés et relations multipolygones ; surface exacte sur WGS84 ; contours ouverts sans surface (valeur vide, pas 0) ; point représentatif garanti dans le polygone.
5. Fusion des doublons node/way d'un même site (rayon réglable, 25 m par défaut).
6. Pays par jointure spatiale (méthode `within`, sinon `nearest` à moins de 50 km, indiquée dans `pays_methode`).
7. Secteur NAF : priorité `amenity` > `shop` > `office` > `craft` > … avec `naf_source` et `naf_confidence`.

**Pour les noms courts ou ambigus** (Total, Apple, Orange) : fournir l'identifiant Wikidata de la marque (colonne `wikidata` d'un fichier, ou `Nom;Q123` en saisie texte), limiter à un pays, ou relever le score minimal.

## Overpass local pour un grand périmètre

1. Démarrer l'instance : `docker compose -f docker-compose.overpass.yml up -d` (extrait de test Monaco ; adapter l'URL d'extrait).
2. `export OVERPASS_URLS=http://localhost:12345/api/interpreter` et `export CATMAP_CACHE_NAMESPACE=<extrait-date>`.
3. Les réglages par défaut s'adaptent seuls (délai minimal nul, 4 travailleurs, retries courts). Le cache disque est isolé par espace de noms : changer d'instantané ne mélange pas les résultats.
4. Le filtre par pays utilise les « areas » Overpass : elles doivent être générées (variable `OVERPASS_RULES_LOAD` dans le fichier Docker).

Une instance locale supprime quotas et files d'attente, mais **pas le coût d'une regex sur tous les éléments nommés** : pour de très gros lots, regrouper ou extraire une fois les éléments nommés (voir la documentation technique, section 3).

## Contrôle qualité des coordonnées

```bash
python scripts/qa_sample.py sites.csv --n 500 --seed 42 --strata pays --out echantillon.csv
# remplir la colonne « precision » (1 = bonne, 0 = mauvaise, 2 = incertaine) à partir des liens fournis
python scripts/qa_report.py echantillon_annote.csv --by pays
```

Échantillon à graine fixe et stratifié ; précision avec intervalle de Wilson à 95 % et bornes tenant compte des cas incertains.

## Traçabilité : anomalies corrigées

| ID | Problème de la v1 | Correction v3 | Où / test |
|---|---|---|---|
| A1 | Dictionnaire NAF vide ; graphique sur la mauvaise colonne | Nomenclature complète (88 divisions, 21 sections), mapping de 441 règles, graphiques sur `naf_section` | `naf.py`, `naf_mapping.csv` / `test_inputs_naf_countries` |
| A2 | Secret OAuth2 en clair | Clé API ou OAuth2 via environnement / `st.secrets` ; test anti-secret dans le code | `config.py`, `vigilance.py` / `test_no_secret_in_source` |
| A3 | `__suppr__` supprimait des sous-chaînes, oubliait « SCA » | Mots entiers en fin de nom, multi-suffixes, formes étrangères, casse conservée | `names.clean_company_name` / `TestClean` |
| A4 | Résultat partiel (`remark`) accepté | Traité comme un échec, jamais mis en cache, statut « partiel » | `overpass.fetch` / `test_partial_*` |
| A5 | Échappement de regex fragile | Regex serveur sans échappement ; précision côté Python | `names.server_regex`, `NameMatcher` / `TestRegex` |
| A6 | Plantages sur résultat vide ; « sans résultat » faux ; CSV fragile | Rapport par entreprise demandée ; DataFrame vide typé ; lecture CSV/XLSX robuste | `pipeline.py`, `inputs.py` / `test_empty_result_is_safe` |
| A7 | `Pays=[]` mutable ; doublons de jointure ; pays vide près des côtes | Index de polygones sans état partagé, une entité par point, repli `nearest` | `polygon_index.py`, `countries.py` / `TestCountries` |
| A8 | Surface fictive des ways ouverts, UTM, pas de multipolygones | Surface géodésique exacte ; contours fermés seulement ; relations assemblées avec trous | `geo.py`, `parsing.py` / `TestArea`, `TestMultipolygon` |
| A9 | `hasattr(way, "center_lat")` toujours vrai | Géométrie lue explicitement, jamais de `float(None)` | `parsing._element_geometry` / `test_missing_geometry` |
| A10 | Chronométrage cassé, progression > 100 % | Durées dans le rapport ; progression exacte, sans dépendance à Streamlit | `pipeline.py` / `test_threads` |
| A11 | Script Selenium interactif, aléatoire, sans sauvegarde | Échantillon reproductible, métriques avec intervalle, aucun effet de bord à l'import | `qa.py`, `scripts/qa_*.py` / `test_qa` |
| A12 | Code mort (`loadDatas`) | Supprimé | – |
| A13 | Bloc NAF dupliqué, carte lente, chemins relatifs | Logique unique, `MarkerCluster`, plafond d'affichage, chemins via `__file__` | `app.py` / `test_app_smoke` |
| A14 | Dépendances non épinglées / incomplètes | Bornes minimales, dépendances inutiles retirées | `requirements.txt` |
| A15 | Tables NAF incohérentes (cafe → Construction…) | Mapping curaté avec confiance et commentaire, valeurs inconnues listées | `build_naf_mapping.py` / `test_known_v1_errors_fixed` |
| A16 | Appariement sur `name` seul, pas d'identifiant OSM | `brand`, `operator`, `brand:wikidata`, adresse, id et URL OSM, score, fusion node/way | `parsing.py` / `TestParsing` |
| A17 | Format de vigilance non confronté au réel | Analyse du JSON documenté (`product.periods…`), repli GeoJSON, erreur explicite avec clés reçues | `vigilance.py` / `test_vigilance` |
| A18 | `print` en production | `logging` | partout |
| A19 | README obsolète | Ce document | – |
| A20 | Aucun cache : tout relancé à chaque interaction | Bouton « Lancer » + résultat en `session_state` + cache disque des réponses | `app.py`, `overpass.py` / `test_second_run_uses_cache` |

## Limites connues (à lire)

- **Non testé en conditions réelles** : les serveurs Overpass, l'API Météo-France, Docker et un vrai Streamlit n'étaient pas accessibles pendant le développement. Les 83 tests couvrent la logique avec un Overpass simulé, le vrai shapefile pays, et l'interface via un Streamlit simulé ; un premier essai réel est nécessaire.
- **Dialecte de regex d'Overpass** : volontairement évité (sous-chaîne simple), mais la requête complète n'a pas été envoyée à un serveur ; une erreur de syntaxe serait remontée explicitement (statut « erreur », message du serveur).
- **Format de vigilance** : basé sur la structure documentée ; à confirmer sur votre abonnement. L'identifiant de domaine (code département, Corse `2A`/`2B`) est géré avec tolérance mais non vérifié sur données réelles. La carte nécessite un GeoJSON de départements (téléchargé et mis en cache, ou fourni via `CATMAP_DEPARTEMENTS_GEOJSON`).
- **Mapping NAF** : proposition argumentée (441 règles, niveaux de confiance) à **faire valider par le métier** ; un tag OSM décrit l'établissement, pas l'activité du groupe. Surcharges possibles via un CSV supplémentaire (`NafClassifier(extra_csv=…)`).
- **Cas non couverts** : ligatures `œ`/`æ` dans la regex serveur (traitées côté Python), polygones OSM invalides (auto-intersections), surface = emprise au sol (pas la surface plancher), fichier pays généralisé (213 entités avec géométrie sur 216).
