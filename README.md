# Weed scout

**Live demo: https://weed-scout-939814041857.us-central1.run.app** (the sample walks replay instantly; a fresh walk takes about 8 to 12 minutes, with every model call at max effort)

Upload the photos from one walk through one field. A Muse Spark agent drives SAM 3.1 to separate weeds from crop at every stop, code measures weed intensity and applies threshold rules, and the report lists herbicide options whose EPA label covers this crop, this weed, and this weed size, each with a page citation.

While a walk runs, you watch the agent work photo by photo: each marking picture, each close-up, the final overlay, and the agent's own reasoning. The sample walks replay a saved run, so they cost nothing to open.

**Built during the Meta Global AI Developer Hackathon (Oct 30 – Nov 4, 2026).** Everything in this repository was written during the event. Earlier projects by the author ([Kronia](https://github.com/AjjayK/Kronia-Pesticide-LLM) and [sf-epa-data-ingestion](https://github.com/AjjayK/sf-epa-data-ingestion)) inspired the label idea; no code was reused.

## How it works
1. **Survey.** The agent says which crop the photo shows and about how many weeds it sees. If the crop doesn't match the walk, the report says so at the top.
2. **Mark.** Muse Spark boxes the crop rows and every weed it can tell apart, up to 120 boxes in one call; SAM 3.1 segments them 16 per request, in parallel. Where weeds are too dense or too small to box one by one, the agent draws a weed area instead, and code measures it from green pixels minus the crop.
3. **Check.** After every mark the agent gets back a picture of what is marked so far, compares it with the photo, and fixes what it missed. It zooms into doubtful plants. If its marks fall short of its own estimate, the stop is flagged.
4. **Identify.** For each kind of weed, the agent picks up to three of its clearest plants. Each goes to a panel of seven separate Muse Spark calls as a full-resolution leaf close-up plus a wider view of the plant. Each call describes the leaf shape, margin, veins, hairs, stem and habit before naming the species and ruling out look-alikes, and the panel votes. It stops waiting once the votes still out could not change the winner. In the agent's own busy context, a burcucumber vine came back as morningglory or velvetleaf; the panel names it. The species list is not a closed menu: a weed that isn't on it is named in its own right (common and scientific name) instead of being forced onto the nearest listed weed, and a panel that can't tell says so.
5. **Measure.** Code computes weed cover, counts, density (straight-down shots), and in-row vs between-row share. The model never writes a number.
6. **Decide.** Rules compare species against sourced thresholds (zero tolerance for resistant amaranths), and choose no action, scout again, or treat (spot or broadcast).
7. **Ground in labels.** A table built from EPA label PDFs (each record's quote verified on its page) lists products labeled for this crop, weed, and size. It holds every weed the labels name, not only the catalog's 36, so a weed named off the list still finds its products.

## Meta models used
Every Muse Spark call runs at `max` reasoning effort (the API's default is `high`).
- `muse-spark-1.3`: agent tool loop (survey, mark, zoom, identify, classify, measure), reasoning summaries shown live, label extraction to structured JSON, field explanation.
- `muse-spark-1.3` as the identification panel: up to seven structured-output calls per kind of weed. At `max`, three panels of five on burcucumber close-ups all named it (13 votes of 15).
- `sam-3.1`: multi-object box prompts, concept prompts ("sky", "crop row"), confidence scores.

## Run locally
    python -m venv .venv && source .venv/Scripts/activate
    pip install -r requirements.txt
    cp .env.example .env   # add MODEL_API_KEY
    python -m pytest -q
    python -m uvicorn scout.app.main:create_app --factory --port 8000

Scout photos from the command line, without the web app:

    PYTHONPATH=. python scripts/run_stop.py --crop corn photo1.jpg photo2.jpg

## Deploy to Google Cloud Run
In [Cloud Shell](https://shell.cloud.google.com), with a project that has billing turned on:

    gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
    PROJECT=$(gcloud config get-value project)
    NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
    gcloud projects add-iam-policy-binding "$PROJECT" \
      --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" --role=roles/run.builder
    git clone https://github.com/AjjayK/weed-or-wheat.git && cd weed-or-wheat
    read -rsp "MODEL_API_KEY: " KEY && echo
    gcloud run deploy weed-scout --source . --region us-central1 --port 7860 --memory 2Gi --cpu 1 \
      --timeout 3600 --max-instances 1 --no-cpu-throttling --allow-unauthenticated \
      --set-env-vars "MODEL_API_KEY=$KEY"

`--max-instances 1` is required: a running walk lives in the server's memory, so every request must reach the same instance. `--no-cpu-throttling` keeps a walk going if the viewer closes the page. Cloud Run caps an upload at 32 MiB, so the page limits a walk's photos to 30 MB in all.

How hard the model works is set by environment variables too: `AGENT_EFFORT` (default `max`) for the agent loop, label extraction and field explanation, `ID_SAMPLES` (default 7), `ID_EFFORT` (default `max`) and `ID_TIMEOUT_S` (default 150) for the identification panel, and `STOP_TIMEOUT_S` (default 600) for one photo. `AGENT_EFFORT=high` roughly halves a walk's time; in our tests it named the weeds as well, but `max` sometimes marked more of them.

New walks are limited per hour by two environment variables: `WALKS_PER_IP_PER_HOUR` (default 10) and `WALKS_PER_HOUR` (default 30, site-wide). Sample walks and repeat uploads of the same photos don't count. Change a limit without rebuilding:

    gcloud run services update weed-scout --region us-central1 --update-env-vars WALKS_PER_HOUR=50

That starts a new revision, which ends any walk in progress and forgets fresh walks' reports (the container's disk is its memory, and a 10-photo walk saves about 30 MB). The limits don't set the Cloud Run bill; `--max-instances 1` does. One instance with 1 vCPU and 2 GiB costs about $0.08 an hour while it's up and scales to zero when idle, and roughly the first 60 hours each month are free.

## Limits
Decision support, not a prescription. Read the label before use; the label is the law. Density is only reported for straight-down photos. Weed counts in weed areas are approximate. Thresholds marked "tool default" are not from an agronomic source; every weed named off the catalog is graded this way.
