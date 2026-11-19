# GTA VI Trophy Rarity Forecaster

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sidiq66/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Sidiq66/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for every PlayStation 5 trophy of Grand Theft Auto VI, forecast five quantiles (5%, 25%, 50%, 75%, 95%) of
its earn rate on Exophase at the case's target — for the final, 2026-11-26T00:00:00Z. Scoring is mean pinball
loss on the logit scale, so rare trophies count as much as common ones and an honest spread matters as much
as the median.

## Run it

The grader runs the program once per case, offline, as `python3 -P -s /bot/main.py` (Python 3.13, standard
library only) with only that case's file in `/input`; it writes `/output/<id>.out`. Locally:

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`validate.py` reproduces every number in [RESULTS.md](RESULTS.md): the practice holdout, a synthetic
benchmark with known ground truth, and malformed-input edge cases. The practice cases ship in `practice/`.

## The model

The page a case's history records only changes when Exophase refreshes it, so the history first collapses
into **refresh epochs** — runs of identical reads. Between epochs two things happen at once, and the model
keeps them apart:

- **Dilution.** The tracked-player pool grows in steps at each refresh. Newly tracked players have earned
  almost nothing, so pool growth drags every earn rate down mechanically.
- **Progression.** The players keep playing, so each trophy's earner count (`rate / 100 * players_tracked`)
  grows as cohorts reach it — fast for early story trophies, slow for endgame and online ones.

Both sides are fitted separately over the recent window (the last 36 h of epochs; launch-hour growth is
explosive and unrepresentative): a line for `ln(players)` and a line per trophy for `ln(earners)`, each
against `ln(1 + hours since release)`, each slope damped for the read past the last epoch because growth
decelerates as the launch rush fades. Per-trophy earner slopes are shrunk toward the whole game's earner
slope (from `trophies_earned`), so a noisy trophy borrows the game's trend.

At the last observed rate the two slopes combine into the share's **logit slope**,
`(earner slope − pool slope) / (1 − share)`, and the median forecast reads that slope along the logit line.
Working on the logit scale is what stops a high-share trophy from extrapolating past 100%, and it is the
scale the contest scores on. The platinum's median is capped at the rarest other trophy's, since it is
earned only by players who earned everything else.

The spread is a budget on the logit scale: a random-walk term in `sqrt(horizon)`, the trophy's own residual
wiggle around its fitted trend, the slope fits' extrapolation uncertainty (standard error × distance), a
model-form term for the damped extrapolation, and a floor for the page's 2-decimal rounding. A history that
has never been seen to move (one frozen epoch, like the practice cases) gets a tight interval around the
last rate; a trophy no read has ever listed gets a tier prior with a wide interval. The program never
crashes on malformed input: a defensive persistence fallback sits behind the model, and every output row is
kept finite, inside 0–100, and non-decreasing.

## Validation summary

- Practice holdout (last read of each practice case held out; both histories are one frozen page, so this
  measures spread discipline only): **model 0.0105** vs house persistence 0.0144 and house curve 0.0144, on
  both cases.
- Synthetic benchmark (cohort worlds with known ground truth, f00-shaped cases: checkpoint 59.5 h after
  release, target at 168 h): **model 0.150** vs persistence 0.340 and curve 0.254 (mean over 5 worlds;
  0.147 vs 0.322 / 0.260 on 5 further worlds held out from tuning).
- Runtime contract: deterministic (byte-identical across runs and hash seeds), ~0.01 s on a 360-read case
  against the 8 s cap, output ≈2.7 KB against the 65,536-byte cap.

Full numbers and method notes: [RESULTS.md](RESULTS.md).

## License

MIT — see `LICENSE`. The practice inputs in `practice/` are the competition's published practice cases.
