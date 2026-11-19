# Validation results

All numbers are mean pinball loss on the logit scale (the contest's scoring: per trophy and level,
`2 * max(tau * d, (tau - 1) * d)` with `d = logit(y) - logit(q)`, rates clamped to 0.01–99.99 first),
produced by `python3 validate.py` unless noted. Lower is better.

Baselines are the house's own two entries, implemented as RULES.txt describes them:

- **persistence** — median = the last read's rate, spread `0.5 * sqrt(horizon days)` on the logit scale
  (the kit's starter, number for number).
- **curve** — per-trophy least-squares line of logit(rate) on `ln(1 + hours since release)` over the last
  24 h of reads, read at the target; spread = the random walk's and the fit's residual combined.

## 1. Practice holdout (the brief's protocol)

For each practice case the last history read is dropped, the forecaster runs as of the second-to-last read
toward the last read's timestamp (a 30-minute horizon in both cases), and the result is scored against that
read.

| case | model | persistence | curve |
| --- | --- | --- | --- |
| p00 | **0.010482** | 0.014354 | 0.014354 |
| p01 | **0.010482** | 0.014354 | 0.014354 |

Caveat, stated plainly: both practice histories are a single frozen page (638 players tracked, identical
rates in all 7–8 reads — the page Exophase had not yet refreshed since before launch). The held-out read is
therefore an exact copy of every earlier read, and this holdout can only measure spread discipline around a
known median, not forecasting skill. The model wins it by keeping a tight interval when nothing has ever
been seen to move (σ ≈ 0.053 at a 30-minute horizon, vs the baselines' 0.072). The real practice answers are
the *refreshed* page (3,868 players, nearly all rates lower), which no case-internal signal predicts; the
practice pays nothing and RULES.txt itself calls it "little guide to the final's week".

## 2. Synthetic benchmark (where the dynamics actually move)

Worlds with known ground truth: the tracked pool grows along a saturating curve revealed in refresh steps
every 2.5–6.5 h; each cohort of players earns trophy *i* with probability
`A_i * (1 - exp(-(age / tau_i)^0.8))`, with `A_i` and `tau_i` drawn per trophy (platinum slowest and rarest).
Reads every 30 minutes print the state as of the last refresh, rates rounded to 2 decimals, like the real
page. Cases are f00-shaped: checkpoint 59.5 h after release, target at 168 h (the final's shape: deadline
2026-11-21T11:30Z, target 2026-11-26T00:00Z).

| world | model | persistence | curve | model 90% coverage |
| --- | --- | --- | --- | --- |
| seed 1 | **0.1306** | 0.3497 | 0.2393 | 1.00 |
| seed 2 | **0.1439** | 0.3330 | 0.2527 | 0.98 |
| seed 3 | **0.1568** | 0.3455 | 0.2634 | 0.98 |
| seed 4 | **0.1544** | 0.3090 | 0.2656 | 0.98 |
| seed 5 | **0.1649** | 0.3618 | 0.2501 | 0.98 |
| **mean** | **0.1501** | 0.3398 | 0.2542 | 0.99 |

Held-out worlds (seeds 6–10, not used while tuning): **model 0.1474**, persistence 0.3219, curve 0.2596.

Coverage runs slightly conservative (0.98–1.00 against the nominal 0.90): the synthetic worlds are smooth,
and the spread keeps a margin for the real page's lumpier arrivals and refresh timing inside the target
hour. On the tuning worlds the emitted σ averaged 0.60 logit against a realized error std of 0.21 around a
−0.17 mean bias (a small systematic overforecast, mostly fast-saturating trophies); the fixed spread
coefficients were trimmed once against this (walk 0.22 → 0.18, model-form 0.35 → 0.28) and then left alone.

## 3. Edge cases

`validate.py` also runs ten malformed/boundary variants — empty history, one read, a trophy null in every
read, a trophy null in the last read, rates of exactly 0 and 100, `players_tracked` zero or missing, missing
levels/target/horizon, no `trophies_earned` anywhere, a single-trophy case, a zero horizon. Every one
produces a contract-valid forecast (right shape, ids covered once each, five finite quantiles in 0–100,
non-decreasing, ≤ 65,536 bytes).

## 4. Runtime contract

Checked the way the grader runs it — `python3 -P -s main.py`, clean environment, one case file in the input
directory, both via `MAKEFUN_INPUT_DIR`/`MAKEFUN_OUTPUT_DIR` and via the default `/input` → `/output`:

- output parses and satisfies every format rule in RULES.txt;
- two runs (including under a different `PYTHONHASHSEED`) write byte-identical output;
- a 360-read, 55-trophy case (f00's scale: 10-minute reads for 2.5 days) forecasts in ~0.01 s against the
  8 s cap; a practice output is 2,666 bytes against the 65,536-byte cap.
