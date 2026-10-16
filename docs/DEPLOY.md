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

Verified in this repository: image build from `requirements.txt`, the test suite inside a container built from
`requirements.lock`, and the dashboards' health check (see the verification notes at the end of the final report).

## 2. Streamlit Community Cloud (manual, needs your GitHub account)

1. Push the repository to GitHub (after the licence check above).
2. On <https://share.streamlit.io> choose **New app**, pick the repository and branch, set the main file to
   `dashboards/app.py`.
3. **Advanced settings**: Python version **3.11**. Dependencies come from `requirements.txt` at the repository root (it
   already points pip at the CPU-only PyTorch wheels).
4. Deploy. The first load restores the bundle and loads the models (about 10-20 s); later loads take a few seconds.
5. Do not add Twilio secrets to a public app: the dashboards never send messages.

Limits to expect: community instances have limited memory; the app needs roughly 1.5-2 GB while PyTorch, XGBoost and
GeoPandas are loaded. If the build runs out of memory, deploy with Docker (option 1 or 3) instead.

## 3. Hugging Face Spaces (Docker Space, manual, needs your Hugging Face account)

1. Create a new Space with SDK **Docker**.
2. Push the repository to the Space. The Space reads its settings from the YAML block at the top of the Space's
   `README.md`; add this block to the **Space copy** of the README (not to the GitHub README):

   ```yaml
   ---
   title: ClimateResilience India
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
