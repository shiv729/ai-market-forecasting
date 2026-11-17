# Can AI Forecast Better Than Prediction Markets?

A small, reproducible Python project for a prospective comparison of an LLM's binary-event probabilities with Polymarket's contemporaneous YES-token midpoint. It collects public information and evaluates forecasts; it never places trades, connects to a wallet, or needs trading credentials.

## Research question and score

The primary question is whether one fixed LLM forecasting setup has a lower Brier score than the market probability captured immediately before each forecast. For a binary outcome \(y\in\{0,1\}\) and probability \(p\), the Brier score is \((p-y)^2\); lower is better. If the event resolves YES and a forecaster assigned 0.65, its score is \((0.65-1)^2=0.1225\). A confident wrong forecast is penalized more than a cautious one.

The reported difference is **AI Brier − market Brier**. Negative values favor AI. The primary estimator averages daily scores within a market, then averages market means so that each enrolled market receives equal weight. Confidence intervals resample whole markets (or, optionally, event groups), never individual days as independent samples. A 0.5 forecast is included as a descriptive baseline. About 50 markets cannot establish universal superiority, related markets can remain dependent, and a lower Brier score does not establish trading profitability.

## Setup

Python 3.9 or newer is supported. Install the project and its dashboard/test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
cp .env.example .env
```

Edit `.env` and set `STUDY_DEADLINE` to the preregistered ISO-8601 UTC timestamp (for example `2027-06-30T23:59:59Z`). There is deliberately no semester date or deadline embedded in the code. For real model forecasts set `OPENAI_API_KEY`; for current web search also set `TAVILY_API_KEY`. The public Polymarket data endpoints used here do not require authentication. Do not put keys in the study config, database, or manifest.

The primary default identifier is `gpt-4o-mini`, the prompt version is `primary-v1`, the primary quote is the midpoint of the YES token's best bid/ask, and the default research provider is Tavily. Decide the study deadline, thresholds, model, research setup, run limits, and cost controls before enrollment. Freeze the resulting configuration in the manifest; changes to model, prompt, evidence setup, or forecast cutoff should be treated as a separate study/configuration.

## Synthetic end-to-end demo

The demo requires no credentials and uses only explicitly marked fabricated data:

```bash
forecast-study demo --data-dir data/demo
```

It exercises daily snapshot/forecast collection, duplicate protection, resolution, Brier evaluation, bootstrap summaries, CSV exports, and a Markdown report. It stores a separate synthetic study ID and database; synthetic observations cannot be combined with a real study. Inspect `data/demo/exports/report.md`, then view it in the dashboard with:

```bash
STUDY_DATABASE=data/demo/research.sqlite STUDY_ID=synthetic-demo streamlit run dashboard.py
```

## Start a real study

First configure the deadline and selection thresholds in `.env` and `config/study.json`. Discovery writes a JSON report for human review and a CSV summary. It uses public Gamma market metadata and checks each YES order book against the configured liquidity, spread, and staleness criteria:

```bash
forecast-study discover --config config/study.json --max-pages 5 --output data/candidate-report.json
```

Review the candidate and exclusion reports. Enrollment defaults to at most one market per event group, never relaxes criteria to reach 50, records a seed if random selection is requested, and writes a frozen content-hashed manifest:

```bash
forecast-study enroll --config config/study.json --candidates data/candidate-report.json --count 50 --output data/study-manifest.json
```

The manifest is the preregistration artifact: inspect it and preserve it unchanged. It contains market/event identifiers, question and rules, YES/NO token mapping, URL, category, expected end date, selection metadata, enrollment time, deadline, criteria, exclusions, and a SHA-256 hash. Save/commit the manifest before scheduling collection; it contains no credentials.

Check wiring without paid research or model calls, then run one collection:

```bash
forecast-study collect --config config/study.json --manifest data/study-manifest.json --dry-run
forecast-study collect --config config/study.json --manifest data/study-manifest.json
```

Repeat `collect` once per day. A market/date has one primary snapshot; retries retain that original quote and are logged as separate attempts. A successful rerun on the same UTC date is a no-op. Failed research/model calls never become a 0.5 forecast. If the market quote is unusable, its issue is saved and it is not included in paired scoring.

Refresh resolutions and generate analysis/exports whenever needed:

```bash
forecast-study resolve --config config/study.json --manifest data/study-manifest.json
forecast-study evaluate --config config/study.json
forecast-study export --config config/study.json --output-dir data/exports
streamlit run dashboard.py
```

`forecast-study --help` and `forecast-study <command> --help` describe all options. You can override database and manifest paths per command. Set maximum markets, research calls, model calls, output tokens, retries, quote age/spread, snapshot/forecast time tolerance, and bootstrap count/seed in `config/study.json`. `--dry-run` checks current quotes but does not persist them as primary snapshots and makes no research or model calls; the next real run captures a fresh quote. A spend cap is enforced using a pre-call token estimate only when both input/output token price estimates are configured; estimates must be supplied by the researcher and are not provider pricing.

## Data and reproducibility

SQLite is the durable source of truth. The versioned schema stores immutable study/manifests, enrollment records, one primary market snapshot per UTC market/date, every forecast attempt (including failures and raw responses), evidence excerpts/URLs/retrieval timestamps, successful forecasts, and append-only resolution revisions. SQLite uses foreign keys and WAL mode. The database and raw responses are private local research data; `.gitignore` excludes local SQLite files.

The workflow and database record study mode, model/provider, prompt version/hash and exact model input, research config, timestamps, software/runtime versions, response usage where returned, estimated cost if researcher-configured, quote method, exact bid/ask/spread, time gap, validation state, and run ID. Prompt template `src/forecaststudy/prompts/primary-v1.txt` is versioned with the package. Daily market midpoint capture uses the documented public CLOB order book for the YES token, not a later substituted price. It validates two-sided quotes, timestamp freshness, price bounds, and spread. Outcome token IDs are selected from protocol-version-specific fields and mapped by the outcome labels in Gamma metadata.

Resolution tracking uses the public Data API v2 `GET /v2/resolutions?condition=...` lifecycle and its payouts. Only a verified resolved binary payout vector maps to YES=1 or NO=0. Proposed/disputed outcomes, canceled or split payouts, missing payout vectors, and closed-but-unverified markets are retained for review and are not forced into 0/1. Scheduled end dates do not guarantee resolution. Every check is stored as a revision.

### Research evidence and leakage

The model prompt contains the question, complete available resolution rules, current UTC time, cutoff, and timestamped evidence, but no current/historical market prices. Search queries use only the event/rules. Tavily results on Polymarket domains and likely price-reporting pages are filtered; evidence published after the forecast-time cutoff is rejected. This reduces leakage but cannot guarantee independence: other publishers may repeat market-derived probabilities, source dates may be uncertain, and public model pretraining can contain prior market information. Evidence excerpts and metadata are retained for audit. Webpage text is treated as untrusted data.

Tavily search is an optional paid/current-evidence integration. If using manual evidence, configure `research_provider` as `manual` and replace the illustrative `examples/evidence.json` entries with real, timestamped excerpts. The sample `.invalid` URL is intentionally not a usable source. Without a retrieval provider the system does not claim the model knows current news. `OPENAI_API_KEY` is required for real model calls; the model endpoint is the OpenAI Responses API with JSON Schema structured output. Credentials and integration calls were not exercised in this repository because no credentials are supplied.

### Cost control and scheduling

Approximate model workload is **markets × collection days × model calls per forecast** (including malformed-output retries). Search calls are separately capped. Provider prices change; configure researcher-owned token price estimates if you need an estimated daily spend cap. The cap reserves estimated input/output cost before each model request and records actual token usage/cost estimates when available; these are not guaranteed charges.

An optional GitHub Actions workflow runs daily or manually. It serializes jobs with a concurrency group, restores the prior successful run's `study-state` artifact, and uploads the updated SQLite database/raw records as the next artifact. Artifact retention is 90 days; this is cross-run persistence, not archival storage. Keep the frozen manifest committed at `data/study-manifest.json`, configure repository secrets `STUDY_DEADLINE`, `OPENAI_API_KEY`, and `TAVILY_API_KEY`, and enable Actions. A failed artifact upload can lose that run's new state; copy/export data to institution-managed durable storage for long-term preservation. Local collection remains fully functional without Actions.

## Results and limitations

`evaluate` and the dashboard report coverage, forecast failures, quote/time-gap exclusions, unresolved markets, paired scores, market-level win/tie/loss counts, bootstrap interval, first/last-pair sensitivity results, horizon buckets (only when at least five markets are represented), and calibration counts. Empty or insufficient data is shown as “not enough resolved data”; charts are not populated with invented values. Dashboard study mode is explicit, and synthetic records are separate and labeled.

No retrospective replacement of poorly performing markets is supported by the study procedure. Enrollment exclusions and reasons remain in the candidate report/manifest. Probability forecasts on one event can share evidence and shocks, so even a market-cluster bootstrap can be optimistic when event dependence remains. Report the exact enrollment, missingness, configuration, and limitations alongside any conclusions.

## Verification and official API references

Polymarket integration paths and response conventions were checked against the official docs:

- [Discover markets / Gamma API](https://docs.polymarket.com/market-data/discover-markets)
- [Market metadata and outcome-ID mapping](https://docs.polymarket.com/market-data/market-details)
- [CLOB prices and order books](https://docs.polymarket.com/market-data/prices-order-books)
- [Data API v2 resolution state](https://docs.polymarket.com/api-reference/markets/get-resolution-state)
- [Resolution lifecycle](https://docs.polymarket.com/concepts/resolution)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

Polymarket read APIs are public and unauthenticated. The Polymarket resolution endpoint returns lifecycle status and payouts; the implementation requires final binary payouts rather than inferring settlement from `closed`. OpenAI and Tavily request shapes follow their published docs, but live credentialed behavior, account access, rate limits, and current provider pricing have not been verified.

## Tests

```bash
python -m pytest
```

Tests use local fixtures/mocked HTTP responses and include the synthetic end-to-end pipeline.
