# Weed scout

**Live demo: https://weed-scout-939814041857.us-central1.run.app** (the sample walks replay instantly; a fresh walk takes about 4 to 6 minutes)

Upload the photos from one walk through one field. A Muse Spark agent drives SAM 3.1 to separate weeds from crop at every stop, code measures weed intensity and applies threshold rules, and the report lists herbicide options whose EPA label covers this crop, this weed, and this weed size, each with a page citation.

While a walk runs, you watch the agent work photo by photo: each marking picture, each close-up, the final overlay, and the agent's own reasoning. The sample walks replay a saved run, so they cost nothing to open.

**Built during the Meta Global AI Developer Hackathon (Oct 30 – Nov 4, 2026).** Everything in this repository was written during the event. Earlier projects by the author ([Kronia](https://github.com/AjjayK/Kronia-Pesticide-LLM) and [sf-epa-data-ingestion](https://github.com/AjjayK/sf-epa-data-ingestion)) inspired the label idea; no code was reused.

## How it works
1. **Survey.** The agent says which crop the photo shows and about how many weeds it sees. If the crop doesn't match the walk, the report says so at the top.
2. **Mark.** Muse Spark boxes the crop rows and every weed it can tell apart, up to 120 boxes in one call; SAM 3.1 segments them 16 per request, in parallel. Where weeds are too dense or too small to box one by one, the agent draws a weed area instead, and code measures it from green pixels minus the crop.
3. **Check.** After every mark the agent gets back a picture of what is marked so far, compares it with the photo, and fixes what it missed. It zooms into doubtful plants and names each weed and its size. If its marks fall short of its own estimate, the stop is flagged.
4. **Measure.** Code computes weed cover, counts, density (straight-down shots), and in-row vs between-row share. The model never writes a number.
5. **Decide.** Rules compare species against sourced thresholds (zero tolerance for resistant amaranths), and choose no action, scout again, or treat (spot or broadcast).
6. **Ground in labels.** A table built from EPA label PDFs (each record's quote verified on its page) lists products labeled for this crop, weed, and size.

## Meta models used
- `muse-spark-1.3`: agent tool loop (survey, mark, zoom, classify, measure), reasoning summaries shown live, label extraction to structured JSON, field explanation.
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

## Limits
Decision support, not a prescription. Read the label before use; the label is the law. Density is only reported for straight-down photos. Weed counts in weed areas are approximate. Thresholds marked "tool default" are not from an agronomic source.
