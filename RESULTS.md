# Validation results

All numbers are mean pinball loss on the logit scale (the contest's scoring: per trophy and level,
`2 * max(tau * d, (tau - 1) * d)` with `d = logit(y) - logit(q)`, rates clamped to 0.01–99.99 first),
produced by `python3 validate.py` unless noted. Lower is better.

Baselines are the house's own two entries, implemented as RULES.txt describes them:

- **persistence** — median = the last read's rate, spread `0.5 * sqrt(horizon days)` on the logit scale
  (the kit's starter, number for number).
- **curve** — per-trophy least-squares line of logit(rate) on `ln(1 + hours since release)` over the last
  24 h of reads, read at the target; spread = the random walk's and the fit's residual combined.

## Recalibration (2026-11-20)

The model was re-fit before the final. Diagnostics on 24 fresh synthetic worlds (seeds 101–124) showed
the first calibration was far too wide and systematically high: emitted σ averaged 0.49 logit against a
realized error std of 0.19, 90% coverage ran at 0.99, and the median overforecast by 0.16 logit on
average — worst (−0.70) for high-share trophies whose earner curves had already saturated, where the
log-time line keeps extrapolating growth that has already flattened. The constants were re-fit against
an extended suite of 13 world regimes (section 2b), accepting a change only when it improved both the
tuning worlds and the held-out worlds of every regime:

- fit window 36 h → 24 h (the launch rush poisons the longer window; this alone cut the early-checkpoint
  regime's loss from 0.69 to ~0.15);
- slope damping exponent 0.5 → 1.1 and a 0.95 shrink on the extrapolated logit move;
- a bias correction of 0.3 × share subtracted from the median's logit (the saturation overforecast
  grows with the share);
- spread rescaled (walk 0.18 → 0.10, model-form 0.28 → 0.22, overall × 0.65), landing 90% coverage on
  its nominal value instead of 0.99;
- the frozen-history spread trimmed (walk 0.30 → 0.22, floor 0.03 → 0.02).

After recalibration, on the same 24 fresh worlds: bias +0.06 logit, realized error std 0.19 against a
mean emitted σ of 0.21, coverage 0.93. The model class is unchanged — epoch decomposition, shrunk
per-trophy earner slopes, logit-line median — only its calibration moved.

## 1. Practice holdout (the brief's protocol)

For each practice case the last history read is dropped, the forecaster runs as of the second-to-last read
toward the last read's timestamp (a 30-minute horizon in both cases), and the result is scored against that
read.

| case | model | persistence | curve |
| --- | --- | --- | --- |
| p00 | **0.007463** | 0.014354 | 0.014354 |
| p01 | **0.007463** | 0.014354 | 0.014354 |

(Before recalibration the model scored 0.010482 here.)

Caveat, stated plainly: both practice histories are a single frozen page (638 players tracked, identical
rates in all 7–8 reads — the page Exophase had not yet refreshed since before launch). The held-out read is
therefore an exact copy of every earlier read, and this holdout can only measure spread discipline around a
known median, not forecasting skill. The model wins it by keeping a tight interval when nothing has ever
been seen to move (σ ≈ 0.037 at a 30-minute horizon, vs the baselines' 0.072). The real practice answers are
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
| seed 1 | **0.1058** | 0.3497 | 0.2393 | 0.95 |
| seed 2 | **0.1087** | 0.3330 | 0.2527 | 0.89 |
| seed 3 | **0.1075** | 0.3455 | 0.2634 | 0.89 |
| seed 4 | **0.1047** | 0.3090 | 0.2656 | 0.91 |
| seed 5 | **0.1208** | 0.3618 | 0.2501 | 0.89 |
| **mean** | **0.1095** | 0.3398 | 0.2542 | 0.91 |

Held-out worlds (seeds 6–10, not used while tuning): **model 0.0955**, persistence 0.3219, curve 0.2596,
coverage 0.94. Before recalibration the model scored 0.1501 on seeds 1–5 and 0.1474 on seeds 6–10, with
coverage 0.98–1.00.

## 2b. Extended regimes (the recalibration suite)

One knob of the world moved at a time — pool growth fast/slow/small, checkpoint early (36 h) or late
(100 h), refresh cadence fast or slow, earning curves steep (exponent 1.25) or flat (0.55), trophies rare
or common, and a second pool wave at 72 h. Five tuning worlds (seeds 201–205) and five held-out worlds
(seeds 301–305) per regime; `validate.py`'s `run_extended` reproduces the table.

| regime | tuning | held-out | | regime | tuning | held-out |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.0981 | 0.0905 | | fast_refresh | 0.0770 | 0.0721 |
| fast_pool | 0.1064 | 0.0958 | | slow_refresh | 0.1168 | 0.1212 |
| slow_pool | 0.0931 | 0.0867 | | steep | 0.1871 | 0.1798 |
| small_pool | 0.1018 | 0.0948 | | flat | 0.0707 | 0.0668 |
| early | 0.1456 | 0.1665 | | rare | 0.0956 | 0.0994 |
| late | 0.0713 | 0.0643 | | common | 0.1426 | 0.1338 |
| surge | 0.0736 | 0.0700 | | **mean** | **0.1061** | **0.1032** |

The incumbent calibration scored 0.1959 (tuning) / 0.1929 (held-out) averaged over the same regimes, with
the early-checkpoint regime at 0.69/0.71 — its single worst failure, now 0.15/0.17. The steep and common
regimes remain the hardest (coverage 0.67–0.79 there): worlds whose earning curves bend the other way, or
where most trophies sit at high shares, still fool the log-time extrapolation more than the spread allows
for. No regime got worse under the recalibration, on either split.

## 3. Edge cases

`validate.py` also runs ten malformed/boundary variants — empty history, one read, a trophy null in every
read, a trophy null in the last read, rates of exactly 0 and 100, `players_tracked` zero or missing, missing
levels/target/horizon, no `trophies_earned` anywhere, a single-trophy case, a zero horizon. Every one
produces a contract-valid forecast (right shape, ids covered once each, five finite quantiles in 0–100,
non-decreasing, ≤ 65,536 bytes).

## 4. Runtime contract

Checked the way the grader runs it — `python3 -P -s main.py`, clean environment, both via
`MAKEFUN_INPUT_DIR`/`MAKEFUN_OUTPUT_DIR` and via the default `/input` → `/output`:

- output parses and satisfies every format rule in RULES.txt;
- two runs (including under a different `PYTHONHASHSEED`) write byte-identical output;
- a 360-read, 55-trophy case forecasts in ~0.1 s against the 8 s cap; a practice output is 2,643 bytes
  against the 65,536-byte cap.
