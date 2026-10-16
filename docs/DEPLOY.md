# Deployment

Three ways to run the dashboards (and optionally the advisory bot). All of them start from the **demo bundle**
(`data/sample/`: derived district-level tables, three serving models, the simulator page), so no pipeline run or
data download is needed. **Before making any deployment or repository public, read `scripts/check_licenses.md`**:
the bundle contains IMD-derived series and ODbL-derived polygons whose redistribution terms were not confirmed.

## 1. Docker (local or any container host)

```bash
docker compose up --build app          # http://localhost:8501
```

The image (`Dockerfile`) is `python:3.11-slim`, runs as the non-root user `app`, installs CPU-only PyTorch and the exact
versions in `requirements.txt`, and has a health check on `/_stcore/health`. On first start the app copies the bundle
into place; mount your own `data/processed` and `models` over `/app/data/processed` and `/app/models` to use real
pipeline outputs. The bot: `docker compose --profile bot up bot` (needs `TWILIO_AUTH_TOKEN` in `.env`; refuses unsigned
requests; replies only; outbound sending stays a dry run unless `DRY_RUN=false`).

Verified in this repository (`python scripts/check_container.py`; durations, test counts, memory and image size are in
`reports/container_checks.json` and in the final report): the serving image builds from `requirements.txt`, starts as a non-root
user, answers its health check, restores the bundle and runs the models; an image built from `requirements.lock` runs the whole
test suite inside the container against the compose database (browser and node tests skip there: the image has neither Chromium nor node).

## 2. Streamlit Community Cloud (manual, needs your GitHub account)

1. Push the repository to GitHub (after the licence check above).
2. On <https://share.streamlit.io> choose **New app**, pick the repository and branch, set the main file to
   `dashboards/app.py`.
3. **Advanced settings**: Python version **3.11**. Dependencies come from `requirements.txt` at the repository root (it
   already points pip at the CPU-only PyTorch wheels).
4. Deploy. The first load restores the bundle and loads the models (the cold and warm load times of every page are in `reports/dashboard_checks.json`).
5. Do not add Twilio secrets to a public app: the dashboards never send messages.

Limits to expect: community instances have limited memory, and the image is large because of PyTorch, XGBoost, SHAP and GeoPandas
(sizes and the memory used in the container check are in `reports/container_checks.json`; every additional concurrent session adds to
the memory). If the build runs out of memory or disk, deploy with Docker (option 1 or 3) instead.

## 3. Hugging Face Spaces (Docker Space, manual, needs your Hugging Face account)

1. Create a new Space with SDK **Docker**.
2. Push the repository to the Space. The Space reads its settings from the YAML block at the top of the Space's
   `README.md`; add this block to the **Space copy** of the README (not to the GitHub README):

   ```yaml
   ---
   title: Veylora
   sdk: docker
   app_port: 8501
   pinned: false
   ---
   ```

3. Spaces build the repository `Dockerfile` and start its default command (`streamlit run dashboards/app.py`). The binary
   files in `data/sample/` are larger than 10 MB: enable Git LFS for `*.parquet`, `*.html` and `*.joblib` in the Space
   (`git lfs track`) if the push is refused.

## After deploying

* Open every page once and check that the yellow synthetic-data banner and the Methodology page are present.
* Put the public URL into `docs/DEMO_SCRIPT.md` / the CV (`make_report` prints it as `MISSING` until you do; it cannot know it).
* Never commit `.env`. The pre-commit hook `no-secrets` scans staged files (`scripts/check_secrets.py`).

## Reference: environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `DB_URL` | analytics, loaders | SQLAlchemy URL of the warehouse; the app falls back to the parquet files when it is unreachable |
| `DRY_RUN` | `advisory/twilio_client.py` | `true` (default): nothing is sent; only an explicit `false` allows real sending |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_SMS_FROM`, `TWILIO_WHATSAPP_FROM` | sender and bot | Twilio credentials; the bot refuses all requests without the token |
| `PUBLIC_BASE_URL` | bot | Public URL Twilio calls (tunnel or proxy), used to validate signatures |
| `ALLOW_UNSIGNED_WEBHOOK` | bot | `true` only for local tests |
| `ADVISORY_AS_OF` | bot | Date used when a message names none (default: last observed day) |
| `GEE_PROJECT` | ETL | Earth Engine project, after `earthengine authenticate` |

## 4. Vercel (the static Veylora site) plus a Streamlit host for the dashboards

Vercel cannot run the Streamlit dashboards: they need a long-lived websocket connection, which Vercel functions do not
support, and the serving stack (PyTorch, XGBoost, SHAP, GeoPandas) is far larger than a function may be. Vercel hosts a
**static site** instead (`web/`: landing page, the self-contained 3D simulator, the headline results and the limits), and the
dashboards run on Streamlit Community Cloud (section 2), Hugging Face Spaces (section 3) or Docker (section 1).

1. Build the site. Every number on the page is read from `reports/`; a missing result shows `MISSING`.
   ```bash
   python -m scripts.make_site --dashboards-url https://<your-app>.streamlit.app   # omit the flag until the dashboards are hosted
   ```
   The page uses the dashboards' colour tokens. Commit `web/` (the simulator page `web/simulator.html` is not committed:
   `vercel.json` copies it from `data/sample/simulator/` at build time).
2. Create the Vercel project named **veylora**, either
   - in the dashboard: **Add New, Project**, import the GitHub repository, leave the framework preset on **Other**
     (`vercel.json` already sets the build command and the output directory `web`), project name `veylora`; or
   - with the CLI from the repository root: `npx vercel login`, `npx vercel link --project veylora`, `npx vercel deploy --prod`
     (`.vercelignore` uploads only `web/`, `vercel.json` and the bundled simulator page).
3. After the dashboards have a public address, rebuild the page with `--dashboards-url`, commit and push; Vercel redeploys.

The simulator page contains weather series derived from IMD data; read `scripts/check_licenses.md` before making the site public.

