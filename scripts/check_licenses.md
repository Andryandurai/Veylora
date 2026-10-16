# Licence checklist (do this before publishing the repository or the demo bundle)

Nothing here is legal advice. "Evidence" says where the statement comes from; **VERIFY** means a person must confirm
the current terms at the source. The repository never sends data anywhere; the question is what you may *publish*.

## Data sources

| Source | Used for | Evidence found | Obligation / risk | Status |
|---|---|---|---|---|
| **IMD gridded rainfall and temperature** (via `imdlib`) | district rain, Tmax, Tmin 1991-2024 (`weather_daily`) | No licence text found. The IRI/LDEO Data Library states it has *not* been granted permission by IMD to distribute IMD datasets to third parties ([IRI](https://iridl.ldeo.columbia.edu/SOURCES/.IMD/index.html)). A news article says IMD has historically sold data ([India Water Portal](https://www.indiawaterportal.org/articles/should-imd-give-away-meteorological-data-free)). | Redistribution of IMD-derived series (the demo bundle's `weather_daily.parquet`, `data/sample/`) may need IMD's permission. Always cite IMD. | **VERIFY with IMD Pune before pushing `data/sample/` to a public repository** |
| **NASA POWER** | humidity, wind, solar (and a weather fallback) | The AWS Open Data Registry states there are no restrictions on use, access or download, and asks that the POWER project be acknowledged ([registry](https://registry.opendata.aws/nasa-power/)); the `nasapower` documentation gives the acknowledgement wording ([rOpenSci](https://docs.ropensci.org/nasapower/)). | Acknowledge: "These data were obtained from the NASA Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science/Applied Science Program." | Acknowledgement is in `README.md`; **VERIFY** wording at power.larc.nasa.gov |
| **ICRISAT District Level Database** (apportioned APY) | district yields 2000-2017 (`crops_annual`) | Described as free and open ([TCI Cornell](https://tci.cornell.edu/?p=796), [ICRISAT press room](https://pressroom.icrisat.org/tci-and-icrisat-continue-partnership-to-improve-access-to-indian-food-systems-data)); no formal licence text found. | Cite ICRISAT/TCI; confirm the reuse terms on the DLD download page. | **VERIFY** terms at the DLD site |
| **geoBoundaries ADM2 India (gbOpen)** | district polygons (`districts.parquet`, maps) | The stored metadata (`data/raw/boundaries/ADM2_meta.json`) says: source "Pathways Data Pvt. Ltd., lgdirectory.gov.in", licence **Open Data Commons Open Database License 1.0** (ODbL). ADM1 (state) metadata says CC BY 2.5 India. geoBoundaries asks for attribution and notes that per-file licences govern ([geoBoundaries terms as mirrored](https://ftp.fau.de/cran/web/packages/geobounds/COPYRIGHTS)). | ODbL is share-alike: a database derived from these polygons (the unit polygons in `districts.parquet`, the demo bundle) is likely a *Derived Database* that must be offered under ODbL, with attribution and a notice. Maps/figures are "Produced Works" and need attribution only. | **VERIFY**; if `data/sample/` is published, add an ODbL notice for the geometry and attribute geoBoundaries and the source |
| **Terrain tiles (elevation)** | real relief of a 10 km window per simulator district (`simulator/terrain.py`, cached in `data/raw/elevation`, shown in `sim_data.json` and the simulator) | The tile provider's attribution page requires: "United States 3DEP (formerly NED) and global GMTED2010 and SRTM terrain data courtesy of the U.S. Geological Survey" ([attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)); the AWS registry lists the dataset as open ([registry](https://registry.opendata.aws/terrain-tiles)). | The attribution is stored in `meta.attribution` and shown in the simulator footer; SRTM is a US government product, the registry does not state further conditions | **VERIFY** the current attribution wording before publishing |
| **Satellite NDVI / soil moisture** | real: MODIS MOD13Q1 NDVI and ERA5-Land layer-1 soil moisture, averaged per district unit through Google Earth Engine (`satellite_16d`) | From the dataset catalogue entries as I know them, not checked against the current licence texts: MODIS products (NASA LP DAAC) are openly available and ask for citation; ERA5-Land is under the Copernicus licence (free, with attribution). Earth Engine access is under its non-commercial terms (project `veylora-ee`, registered until 2028-04-16). | Only district means are redistributed, not imagery. Cite MODIS MOD13Q1 v061 and ERA5-Land and add the Copernicus attribution wherever the data is shown. A commercial use of Veylora would need Earth Engine's commercial terms. | Add attributions; confirm the licence texts before publishing |

## Software and assets

| Item | Licence | Evidence | Status |
|---|---|---|---|
| Three.js r160.1 (vendored `simulator/vendor/three.module.min.js`) | MIT | `simulator/vendor/THREE_NOTICE.txt` (also records the SHA-256) | Keep the notice with the file |
| Python dependencies (157 packages) | see `docs/DEPENDENCY_LICENSES.md` (generated from package metadata by `scripts/list_licenses.py`) | metadata | Review the "needs a human look" section: `psycopg` / `psycopg-binary` are LGPL-3.0 (used unmodified as a library), `certifi` and `tqdm` carry MPL-2.0, `llvmlite` has no licence field in its metadata |
| Fonts, icons, photographs | none bundled; Streamlit and the browser supply fonts, the simulator draws procedural geometry | repository contents | - |
| Advisory translations | written by an AI assistant, **not reviewed** (decision D-026) | `advisory/templates.yaml` | Native-speaker and agronomist review before any real use |
| Crop parameters and prices (`config/crop_params.yaml`, `config/economics.yaml`) | typed from memory of public documents, **unverified** (D-021, D-022) | files carry the warnings | Verify each value against its source before relying on it |

## Before you `git push`

1. Decide whether `data/sample/` (derived IMD series, ODbL-derived polygons) may be public. If unsure, delete it from the commit
   (`git rm -r --cached data/sample`), keep it local, and deploy privately or after the permissions arrive.
2. Keep the attributions in `README.md` (IMD, NASA POWER, ICRISAT/TCI, geoBoundaries and its source, Three.js).
3. Run `python scripts/check_secrets.py $(git ls-files)` (also done by the pre-commit hook) and confirm `.env` is not tracked.
4. Re-run `python scripts/list_licenses.py` after changing dependencies.
