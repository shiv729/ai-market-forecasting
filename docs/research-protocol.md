# Research protocol: Can AI Forecast Better Than Prediction Markets?

## Primary estimand

For each preregistered market and scheduled collection date before its outcome cutoff, capture a YES-token midpoint from a valid, fresh two-sided Polymarket order book, then obtain one structured probability from the fixed model/research/prompt configuration. For resolved binary outcomes, calculate squared probability error for both sources. Average daily valid paired errors within each market and then equally average market means. The primary difference is AI minus market Brier score; a negative value favors AI.

Only observations with a successful validated AI forecast, a valid YES midpoint, a verified binary outcome, and a snapshot-to-completion gap within the preregistered tolerance contribute to paired scoring. Failures, missing quotes, too-wide/stale quotes, long time gaps, and unresolved/ambiguous outcomes are retained and reported rather than imputed.

## Enrollment and stopping

Review candidates before enrollment. Criteria in the candidate report are frozen with the final manifest: binary YES/NO outcome labels, active/unresolved status, expected end date before study deadline, liquidity/activity floors, readable rules/source, and valid quote quality. Deduplicate by event group to reduce near-identical questions. Record selection seed when random selection is used, all exclusions and reasons, actual enrolled count, and manifest hash. If fewer than 50 qualify, retain that sample size; do not lower thresholds or replace markets after observing forecasts/outcomes.

The study deadline is researcher-configured. Stop forecasts at the earlier of the expected event end minus the configured buffer and the study deadline. Check official resolution lifecycle before each forecast; stop when resolved or when a proposal/dispute means the outcome may already be known. An uncertain status requires review. A market being closed is not itself a verified outcome.

## Information and leakage control

The AI receives the exact question, available complete resolution rules, UTC as-of time, cutoff, and timestamped public evidence excerpts. It does not receive market prices. The configured evidence provider searches using only question/rules. Exclude Polymarket pages and obvious reporting of market odds/probabilities; reject evidence published after the forecast's as-of time. Record included evidence and its retrieval metadata. Source filtering and public-data exclusion reduce leakage but cannot guarantee independence from market-derived reporting, model pretraining, or undetected timestamps.

The primary analysis must use one fixed model identifier, prompt version/hash, evidence provider/configuration, and cutoff. Any material changes form a separate study/configuration and should not be pooled into the primary estimate.

## Quote and resolution rules

Use the midpoint of the best valid YES-token bid and ask, computed from the official CLOB order book. Determine the YES token by protocol version and matching outcome label. Record the book timestamp, snapshot time, bid, ask, spread, midpoint, and status. Missing/one-sided/crossed/invalid/stale/wide quotes do not enter paired scoring; do not substitute a later quote or silently use a last-trade value.

Use Polymarket Data API v2 resolution state. Score only a resolved binary payout vector mapping to one winning outcome. Proposed, challenged/disputed, canceled, ambiguous, split, non-binary, closed-but-unverified, and missing-payout cases remain unscored and visible. Append corrections as resolution revisions.

## Statistical analysis

Report the equal-market-weighted AI and market Brier scores, their difference, paired observation count, a 0.5 baseline, market-level win/tie/loss counts, and a percentile 95% bootstrap interval with reproducible seed. Bootstrap markets as clusters, retaining all daily observations for a sampled market. Optionally cluster by event; related events/evidence may still be dependent. Report first/last valid pair sensitivities and predefined forecast-horizon buckets only when at least five markets are present. Show calibration bins with counts and a small-sample warning.

Do not treat daily observations as independent samples. Do not interpret score differences as trading returns or universal superiority. Report enrollment, resolved/scored counts, failures, exclusions, missingness, unresolved markets, study mode, and configuration alongside results.

## Cost and data handling

Bound market, research, model, output-token, retry, and daily spend estimates before collection. Provider price estimates are researcher-entered and may not equal actual charges. Preserve raw responses, evidence excerpts, timestamps, config, and run IDs in versioned SQLite. Do not commit credentials or local database files. Synthetic data is a pipeline test only and must not enter real-study analysis.
