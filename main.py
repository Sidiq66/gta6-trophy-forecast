"""gta6-trophy-rarity-forecast-01: earn-rate quantile forecaster (Python 3.13, standard library only).

Model in one paragraph: the page a case's history records only changes when Exophase refreshes it, so the
history collapses into refresh epochs (runs of identical reads). Inside an epoch nothing moves; between epochs
two things happen at once: the tracked-player pool grows, which dilutes every earn rate, and the players keep
playing, which grows each trophy's earner count. The two effects are separated by fitting them apart over the
recent window (the last 24 h of epochs): a line for ln(players) against ln(1 + hours since release), and a
line per trophy for ln(earners), earners = rate / 100 * players, each slope damped for the read past the last
epoch and the per-trophy slopes shrunk toward the whole game's earner slope. At the last observed rate the
two slopes combine into the share's logit slope, (earner slope - pool slope) / (1 - share), and the median
forecast reads that slope along the logit line -- which is what stops a high-share trophy extrapolating past
100%. The spread combines a random-walk term, the trophy's own observed step volatility between epochs, and
the slope fits' extrapolation uncertainty, all on the logit scale the contest scores on.

Contract (RULES.txt): read every <id>.in in $MAKEFUN_INPUT_DIR (else /input), write <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}, one row per
trophy of the case, q in percent, non-decreasing, finite, 0..100. Deterministic: no randomness, no clock, no
network; the same case always writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory off sys.path; put it back first (harmless when run by hand).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
RATE_LO = 0.01    # the scorer clamps rates into 0.01..99.99 before the logit; forecasts stay inside it too
RATE_HI = 99.99
EARNER_EPS = 0.5  # half an earner: keeps log(earners) finite for trophies nobody has earned yet

# Median rates (percent) used only for a trophy no read has ever listed, by tier: a last-resort prior.
TIER_PRIOR = {"platinum": 0.05, "gold": 0.4, "silver": 1.0, "bronze": 2.5}

# Trend fits use only the last FIT_WINDOW_HOURS of epochs: growth right after launch is explosive and
# unrepresentative, and a line fitted through it extrapolates absurdly. Slopes are also damped when read
# past the last epoch (growth decelerates as the launch rush fades): the extrapolated change is the slope
# times the log-time distance times (x_last / x_target) ** SLOPE_DAMP_EXP.
FIT_WINDOW_HOURS = 24.0
SLOPE_DAMP_EXP = 1.1
SHRINK_WEIGHT = 3.0  # pseudo-count of the game's slope in each trophy's shrunk slope

# Spread components on the logit scale (see forecast_case): the random-walk coefficient per sqrt(day)
# of horizon, the model-form coefficient per unit of log-time distance extrapolated, and the rounding
# floor. SIGMA_SCALE multiplies the combined spread; MOVE_SCALE multiplies the extrapolated logit move
# of the median; BIAS_C0 + BIAS_C1 * share is subtracted from the median's logit (a recalibration for
# the systematic overforecast of fast-saturating trophies). FROZEN_* set the spread when the history
# holds a single epoch and nothing has ever been seen to move.
#
# Recalibration (2026-11-20, see RESULTS.md section 2): the first calibration ran far too wide and too
# high -- on synthetic worlds the emitted sigma averaged ~0.49 logit against a realized error std of
# ~0.19, coverage ran 0.99 against the 0.90 nominal, and the median overforecast by ~0.16 logit,
# worst for high-share trophies whose earner curves had already saturated. The constants below were
# re-fit against an extended synthetic suite (13 regimes: pool growth, refresh cadence, checkpoint and
# target placement, earning-curve shape, rarity mix, a mid-window pool surge), accepting a change only
# when it improved both the tuning and the held-out worlds of every regime: a shorter fit window (the
# launch rush poisons a 36 h window), stronger slope damping with a small move shrink, a share-proportional
# bias correction, and a spread rescale that lands coverage on its nominal 0.90.
WALK_COEF = 0.10
FORM_COEF = 0.22
SIGMA_FLOOR = 0.04
SIGMA_SCALE = 0.65
MOVE_SCALE = 0.95
BIAS_C0 = 0.0
BIAS_C1 = 0.3
FROZEN_WALK_COEF = 0.22
FROZEN_FLOOR = 0.02


def clamp_rate(pct: float) -> float:
    return min(RATE_HI, max(RATE_LO, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def round4(x: float) -> float:
    """Round to 4 decimals, half up (all values here are positive)."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_time(text):
    """ISO 8601 UTC -> aware datetime, or None when unparseable."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


class Fit:
    """Weighted least squares y ~= a + b*x with residual scale and slope standard error."""

    __slots__ = ("a", "b", "resid", "se_b", "n", "x_mean")

    def __init__(self, a, b, resid, se_b, n, x_mean):
        self.a = a
        self.b = b
        self.resid = resid
        self.se_b = se_b
        self.n = n
        self.x_mean = x_mean

    def at(self, x):
        return self.a + self.b * x


def fit_wls(xs, ys, ws):
    """Fit a line; None when fewer than 2 points or all x equal."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ym - b * xm
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / (n - 2)
    else:
        var = 0.0
    resid = math.sqrt(max(var, 0.0))
    se_b = resid / math.sqrt(sxx) if sxx > 0 else 0.0
    return Fit(a, b, resid, se_b, n, xm)


def build_epochs(history):
    """Collapse consecutive identical reads (same players, same earners total, same rates) into epochs.

    An epoch's state is the page as Exophase last refreshed it; its time is the first read that showed it.
    Returns a list of {"at": datetime, "players": float, "earned": float|None, "rates": list}."""
    epochs = []
    for read in history:
        if not isinstance(read, dict):
            continue
        at = parse_time(read.get("slot")) or parse_time(read.get("at"))
        if at is None:
            continue
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players <= 0:
            players = epochs[-1]["players"] if epochs else 1.0
        rates = read.get("rates") if isinstance(read.get("rates"), list) else []
        earned = read.get("trophies_earned")
        if not isinstance(earned, (int, float)):
            earned = None
        key = (float(players), earned, tuple(None if r is None else float(r) for r in rates))
        if epochs and epochs[-1]["key"] == key:
            continue
        epochs.append({"key": key, "at": at, "players": float(players), "earned": earned, "rates": key[2]})
    return epochs


def release_origin(case, epochs):
    """The instant progression is measured from: the game's release day at 00:00 UTC."""
    game = case.get("game") if isinstance(case.get("game"), dict) else {}
    dt = parse_time(str(game.get("release_date", "")) + "T00:00:00Z")
    if dt is not None:
        return dt
    if epochs:
        first = epochs[0]["at"]
        return first.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def forecast_case(case):
    """The forecast object {"trophies": [...]} for one parsed case."""
    trophies = case.get("trophies") or []
    levels = case.get("levels")
    if not (isinstance(levels, list) and len(levels) == 5 and all(isinstance(t, (int, float)) for t in levels)):
        levels = DEFAULT_LEVELS
    nd = NormalDist()
    zs = [nd.inv_cdf(min(0.999, max(0.001, float(t)))) for t in levels]

    history = case.get("history") or []
    epochs = build_epochs(history)
    at = parse_time(case.get("at")) or (epochs[-1]["at"] if epochs else None)
    horizon_min = case.get("horizon_minutes")
    if not isinstance(horizon_min, (int, float)) or horizon_min < 0:
        horizon_min = 0.0
    target = parse_time(case.get("target"))
    if target is None and at is not None:
        from datetime import timedelta

        target = at + timedelta(minutes=float(horizon_min))
    if at is None or target is None:  # nothing to anchor on: flat priors, wide spread
        return {"trophies": [prior_row(t, zs, 1.6, k + 1) for k, t in enumerate(trophies)]}
    if horizon_min == 0.0:
        horizon_min = max(0.0, hours_between(at, target) * 60.0)
    h_days = max(horizon_min, 0.0) / 1440.0

    origin = release_origin(case, epochs)

    def x_of(dt):
        return math.log1p(max(0.0, hours_between(origin, dt)))

    x_target = x_of(target)
    x_last = x_of(epochs[-1]["at"]) if epochs else x_of(at)
    # Slopes are read from the last epoch onward; growth decelerates, so the read is damped.
    damp = (x_last / x_target) ** SLOPE_DAMP_EXP if x_target > x_last > 0.0 else 1.0
    dx = max(0.0, x_target - x_last)

    n_last = epochs[-1]["players"] if epochs else 0.0

    # The epochs the trend fits see: the recent window (always at least the last three epochs).
    if epochs:
        fit_epochs = [e for e in epochs if hours_between(e["at"], epochs[-1]["at"]) <= FIT_WINDOW_HOURS]
        if len(fit_epochs) < min(3, len(epochs)):
            fit_epochs = epochs[-min(3, len(epochs)):]
    else:
        fit_epochs = []

    # --- pool size at the target: slope from the window, anchored at the last observed size ------------
    pool_fit = None
    if len(fit_epochs) >= 2:
        pool_fit = fit_wls([x_of(e["at"]) for e in fit_epochs], [math.log(e["players"]) for e in fit_epochs],
                           [1.0] * len(fit_epochs))
    if pool_fit is not None and n_last > 0:
        n_target = n_last * math.exp(pool_fit.b * damp * dx)
        n_target = max(n_target, n_last * math.exp(-0.1 * math.sqrt(h_days)))  # the pool barely ever shrinks
        pool_se = pool_fit.se_b
    elif n_last > 0:
        n_target = n_last  # a single frozen epoch: no growth is observable, assume none
        pool_se = 0.0
    else:
        n_target = 1.0
        pool_se = 0.0

    # --- the whole game's earner slope over the window, the shrinkage target ----------------------------
    game_slope = 0.0
    game_resid = 0.15  # the game's own wiggle around its trend; the fallback noise scale
    if len(fit_epochs) >= 2 and all(e["earned"] is not None for e in fit_epochs):
        gfit = fit_wls([x_of(e["at"]) for e in fit_epochs], [math.log(e["earned"] + 1.0) for e in fit_epochs],
                       [1.0] * len(fit_epochs))
        if gfit is not None:
            game_slope = gfit.b
            game_resid = gfit.resid
    if game_slope == 0.0 and len(fit_epochs) >= 2:
        # No usable trophies_earned series: fall back to the median trophy's own earner slope, so the
        # shrinkage target is still the game's typical progression rather than a flat line.
        slopes = []
        for i in range(len(trophies)):
            pts = [(x_of(e["at"]), math.log(clamp_rate(e["rates"][i]) / 100.0 * e["players"] + EARNER_EPS))
                   for e in fit_epochs if i < len(e["rates"]) and e["rates"][i] is not None]
            f = fit_wls([p[0] for p in pts], [p[1] for p in pts], [1.0] * len(pts)) if len(pts) >= 2 else None
            if f is not None:
                slopes.append(f.b)
        if slopes:
            slopes.sort()
            game_slope = slopes[len(slopes) // 2]

    fit_ids = {id(e) for e in fit_epochs}
    pool_slope = pool_fit.b if pool_fit is not None else 0.0
    rows = []
    medians = []  # (index, median rate) for the platinum cap afterwards
    for i, trophy in enumerate(trophies):
        obs = []       # (x, log earners, weight, rate) over every epoch that lists the trophy
        obs_fit = []   # the same, restricted to the fit window
        for e in epochs:
            r = e["rates"][i] if i < len(e["rates"]) else None
            if r is None:
                continue
            earners = clamp_rate(float(r)) / 100.0 * e["players"]
            item = (x_of(e["at"]), math.log(earners + EARNER_EPS),
                    e["players"] / (e["players"] + 500.0), float(r))  # small pools print coarse rates
            obs.append(item)
            if id(e) in fit_ids:
                obs_fit.append(item)
        sigma_extra = 0.0
        if obs and len(epochs) >= 2:
            anchor_x, _, _, last_rate = obs[-1]
            fit = fit_wls([o[0] for o in obs_fit], [o[1] for o in obs_fit],
                          [o[2] for o in obs_fit]) if len(obs_fit) >= 2 else None
            if fit is not None:
                slope_e = (fit.n * fit.b + SHRINK_WEIGHT * game_slope) / (fit.n + SHRINK_WEIGHT)
            else:  # too little in the window: borrow the game's slope whole
                slope_e = game_slope
                sigma_extra = 0.35
            slope_e = min(4.0, max(-1.5, slope_e))
            # The dilution/progression split, expressed on the scale the contest scores: the share's logit
            # moves with (earner growth - pool growth), steepened by 1 / (1 - share); reading that slope
            # along the logit line is what keeps a high-share trophy from extrapolating past 100%.
            share = clamp_rate(last_rate) / 100.0
            resid_logit = (fit.resid if fit is not None else game_resid) / max(0.02, 1.0 - share)
            slope_logit = (slope_e - pool_slope) / max(0.02, 1.0 - share)
            slope_logit = min(2.5, max(-2.5, slope_logit))
            dist = max(0.0, x_target - anchor_x)
            if fit is not None:
                sigma_extra = min(0.9, abs(fit.se_b) / max(0.02, 1.0 - share) * dist)
            mu_med = logit(last_rate) + MOVE_SCALE * slope_logit * damp * dist \
                - (BIAS_C0 + BIAS_C1 * share)
            median_rate = clamp_rate(inv_logit(mu_med))
        elif obs:  # a single frozen epoch: nothing has been seen to move
            median_rate = clamp_rate(obs[-1][3])
            resid_logit = game_resid
            dist = dx
        else:  # never listed in any read: tier prior, honestly wide
            median_rate = TIER_PRIOR.get(trophy.get("tier"), 1.0)
            sigma_extra = 0.8
            resid_logit = game_resid
            dist = dx
        if len(epochs) >= 2:
            # walk + the series' own wiggle around its trend + slope uncertainty + pool-slope uncertainty
            # + a model-form term for the damped extrapolation + a floor for the page's rounding
            var = (WALK_COEF * math.sqrt(h_days)) ** 2 + resid_logit ** 2 + sigma_extra ** 2 \
                + (pool_se * dx) ** 2 + (FORM_COEF * dist) ** 2 + SIGMA_FLOOR ** 2
            sigma = min(2.4, max(0.05, SIGMA_SCALE * math.sqrt(var)))
        else:  # frozen or empty history: nothing has been seen to move, so the interval stays tight
            sigma = min(2.4, math.sqrt((FROZEN_WALK_COEF * math.sqrt(h_days)) ** 2 + FROZEN_FLOOR ** 2))
        medians.append((i, median_rate))
        rows.append((trophy.get("id") if isinstance(trophy.get("id"), int) else i + 1, median_rate, sigma))

    # The platinum is earned only by players who earned everything else: its rate cannot pass the rarest
    # other trophy's. Pull its median down to that cap when the fits disagree.
    plat = [k for k, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat and len(rows) > 1:
        p = plat[0]
        others = [m[1] for m in medians if m[0] != p]
        if others and medians[p][1] > min(others):
            rows[p] = (rows[p][0], min(others), rows[p][2])

    out = []
    for tid, median_rate, sigma in rows:
        mu = logit(median_rate)
        q = [round4(inv_logit(mu + z * sigma)) for z in zs]
        for j in range(1, len(q)):  # rounding at the extremes can tie neighbours; keep the row increasing
            if q[j] <= q[j - 1]:
                q[j] = round4(min(RATE_HI, q[j - 1] + 0.0001))
        out.append({"id": tid, "q": q})
    return {"trophies": out}


def prior_row(trophy, zs, sigma, fallback_id):
    mu = logit(TIER_PRIOR.get(trophy.get("tier"), 1.0))
    q = [round4(inv_logit(mu + z * sigma)) for z in zs]
    for j in range(1, len(q)):
        if q[j] <= q[j - 1]:
            q[j] = round4(min(RATE_HI, q[j - 1] + 0.0001))
    tid = trophy.get("id")
    return {"id": tid if isinstance(tid, int) else fallback_id, "q": q}


def fallback_case(case):
    """Persistence with the house spread, built defensively; used only if the model itself fails."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    h_days = max(0.0, float(case.get("horizon_minutes") or 0.0)) / 1440.0
    spread = 0.5 * math.sqrt(h_days)
    nd = NormalDist()
    zs = [nd.inv_cdf(t) for t in DEFAULT_LEVELS]
    last = history[-1] if history else {}
    rates = last.get("rates") if isinstance(last.get("rates"), list) else []
    rows = []
    for i, trophy in enumerate(trophies):
        r = rates[i] if i < len(rates) else None
        mu = logit(float(r)) if isinstance(r, (int, float)) else logit(TIER_PRIOR.get(trophy.get("tier"), 1.0))
        q = [round4(inv_logit(mu + z * spread)) for z in zs]
        for j in range(1, len(q)):
            if q[j] <= q[j - 1]:
                q[j] = round4(min(RATE_HI, q[j - 1] + 0.0001))
        tid = trophy.get("id")
        rows.append({"id": tid if isinstance(tid, int) else i + 1, "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(case.get("trophies"), list):  # a malformed trophy entry must not sink the whole case
            case["trophies"] = [t if isinstance(t, dict) else {} for t in case["trophies"]]
        try:
            forecast = forecast_case(case)
            if len(forecast["trophies"]) != len(case.get("trophies") or []):
                raise ValueError("row count mismatch")
        except Exception:
            forecast = fallback_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
