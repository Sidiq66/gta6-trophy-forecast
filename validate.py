"""Holdout validation for main.py's forecaster.

Three things are checked:

1. The brief's holdout on the practice cases: drop each case's last history read, forecast from the
   second-to-last read toward the last read's timestamp, and score with mean pinball loss on the logit
   scale (the contest's own scoring). NOTE: both practice histories are one frozen page (638 players,
   identical rates in every read), so the last read is an exact copy of every earlier read and this
   holdout can only measure spread discipline, not forecasting skill. It is reported as-is.

2. A synthetic benchmark: worlds with known ground truth (a tracked pool that grows in refresh steps,
   cohorts of players who earn trophies as they age), cut into f00-shaped cases (checkpoint ~60 h after
   release, target at 168 h). Here the dilution/progression dynamics actually move, so this is where the
   model has to beat the baselines.

3. Format: every forecast produced here is checked against the output contract in RULES.txt.

Baselines, implemented as RULES.txt describes the house's own two entries:
  A "persistence": median = last read's rate, spread 0.5 * sqrt(horizon days) on the logit scale.
  B "curve": per-trophy least-squares line of logit(rate) on ln(1 + hours since release) over the last
     24 h of reads, read at the target; spread = the random walk's and the fit residual combined.
"""

import copy
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ND = NormalDist()
ZS = [ND.inv_cdf(t) for t in LEVELS]
HERE = Path(__file__).resolve().parent


def pinball_loss(forecast, case, actual_rates):
    """Mean pinball loss on the logit scale of a forecast against one read's rates (percent)."""
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    total, count = 0.0, 0
    for i, trophy in enumerate(case["trophies"]):
        y = actual_rates[i] if i < len(actual_rates) else None
        if y is None:
            continue
        ly = model.logit(float(y))
        for tau, q in zip(LEVELS, rows[trophy["id"]]):
            d = ly - model.logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count if count else float("nan")


def check_format(forecast, case):
    """Assert a forecast object satisfies the RULES.txt output contract for its case."""
    assert set(forecast.keys()) == {"trophies"}, forecast.keys()
    rows = forecast["trophies"]
    assert len(rows) == len(case["trophies"]), (len(rows), len(case["trophies"]))
    ids = [r["id"] for r in rows]
    assert sorted(ids) == sorted(t["id"] for t in case["trophies"]), "ids must cover the case's trophies once each"
    for r in rows:
        assert set(r.keys()) == {"id", "q"}, r.keys()
        q = r["q"]
        assert len(q) == 5, q
        assert all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 100.0 for v in q), q
        assert all(q[j] >= q[j - 1] for j in range(1, 5)), q
    blob = json.dumps(forecast, separators=(",", ":")).encode()
    assert len(blob) <= 65536, len(blob)
    return True


def baseline_persistence(case):
    h_days = max(0.0, float(case.get("horizon_minutes") or 0.0)) / 1440.0
    spread = 0.5 * math.sqrt(h_days)
    last = case["history"][-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i] if i < len(last["rates"]) else None
        mu = model.logit(float(r)) if isinstance(r, (int, float)) else model.logit(1.0)
        rows.append({"id": trophy["id"], "q": [model.round4(model.inv_logit(mu + z * spread)) for z in ZS]})
    return {"trophies": rows}


def baseline_curve(case):
    trophies = case["trophies"]
    history = case["history"]
    if len(history) < 3:
        return baseline_persistence(case)
    origin = model.release_origin(case, model.build_epochs(history))
    at = model.parse_time(case["at"])
    target = model.parse_time(case["target"])
    h_days = max(0.0, float(case.get("horizon_minutes") or 0.0)) / 1440.0
    walk = 0.5 * math.sqrt(h_days)
    recent = [r for r in history if model.hours_between(model.parse_time(r["slot"]), at) <= 24.0] or history

    def x_of(dt):
        return math.log1p(max(0.0, model.hours_between(origin, dt)))

    rows = []
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for read in recent:
            r = read["rates"][i] if i < len(read["rates"]) else None
            if r is not None:
                xs.append(x_of(model.parse_time(read["slot"])))
                ys.append(model.logit(float(r)))
        fit = model.fit_wls(xs, ys, [1.0] * len(xs)) if len(xs) >= 3 else None
        if fit is None:
            last = history[-1]["rates"][i]
            mu = model.logit(float(last)) if isinstance(last, (int, float)) else model.logit(1.0)
            sigma = walk
        else:
            mu = fit.at(x_of(target))
            sigma = math.sqrt(walk ** 2 + fit.resid ** 2)
        q = [model.round4(model.inv_logit(mu + z * sigma)) for z in ZS]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def holdout(case):
    """The brief's holdout: forecast the last history read from the reads before it."""
    truncated = copy.deepcopy(case)
    last = truncated["history"].pop()
    prev = truncated["history"][-1]
    t_prev = model.parse_time(prev["slot"])
    t_last = model.parse_time(last["slot"])
    truncated["at"] = prev["slot"]
    truncated["target"] = last["slot"]
    truncated["horizon_minutes"] = (t_last - t_prev).total_seconds() / 60.0
    return truncated, last["rates"]


def run_practice():
    print("== practice holdout (last read held out; histories are frozen, see docstring) ==")
    results = {}
    for name in ("p00", "p01"):
        case = json.loads((HERE / "practice" / f"{name}.in").read_text())
        trunc, actual = holdout(case)
        scores = {}
        for label, fn in (("model", model.forecast_case), ("persistence", baseline_persistence),
                          ("curve", baseline_curve)):
            fc = fn(trunc)
            check_format(fc, trunc)
            scores[label] = pinball_loss(fc, trunc, actual)
        results[name] = scores
        print(f"{name}: model {scores['model']:.6f} | persistence {scores['persistence']:.6f} | "
              f"curve {scores['curve']:.6f}")
        # the full case must also produce a valid forecast
        check_format(model.forecast_case(case), case)
    return results


# --- synthetic worlds -----------------------------------------------------------
# Ground truth: the pool grows along a saturating curve and is revealed in refresh steps; each cohort of
# players earns trophy i with probability F_i(age) = A_i * (1 - exp(-(age/tau_i)^0.8)). Reads every
# 30 minutes print the state as of the last refresh, rates rounded to 2 decimals, like the real page.

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def synth_world(seed):
    rng = random.Random(seed)
    trophies = []
    for k in range(55):
        if k == 0:
            trophies.append({"id": 1, "name": "Platinum", "description": "all", "tier": "platinum",
                             "points": 300, "secret": False, "_A": 0.004, "_tau": 260.0})
            continue
        tier = "gold" if k <= 2 else "silver" if k <= 8 else "bronze"
        a = math.exp(rng.uniform(math.log(0.01), math.log(0.97)))
        tau = math.exp(rng.uniform(math.log(2.0), math.log(320.0)))
        trophies.append({"id": k + 1, "name": f"T{k + 1}", "description": "synthetic", "tier": tier,
                         "points": {"gold": 90, "silver": 30, "bronze": 15}[tier],
                         "secret": bool(rng.getrandbits(1)), "_A": a, "_tau": tau})

    def pool(hours):  # the pool's true size at a moment
        return 600.0 + 80000.0 * (1.0 - math.exp(-((hours / 72.0) ** 0.85)))

    refreshes = []  # (time, pool size) steps
    t = 0.0
    while t < 200.0:
        refreshes.append((t, pool(t)))
        t += rng.uniform(2.5, 6.5)

    def state_at(hours):
        """(players, rates) as the page shows them: the last refresh at or before the moment."""
        step = max((s for s in refreshes if s[0] <= hours), key=lambda s: s[0])
        n = step[1]
        earners = []
        for tr in trophies:
            e = 0.0
            for j, (s0, n0) in enumerate(refreshes):
                if s0 > step[0]:
                    break
                cohort = n0 - (refreshes[j - 1][1] if j else 0.0)
                age = step[0] - s0
                e += cohort * tr["_A"] * (1.0 - math.exp(-((age / tr["_tau"]) ** 0.8))) if age > 0 else 0.0
            earners.append(e)
        rates = [round(100.0 * e / n, 2) if n else 0.0 for e in earners]
        return n, rates, sum(earners)

    history = []
    read_h = -0.17  # first read 2026-11-18T23:50Z, as in the real recording
    checkpoint_h, target_h = 59.5, 168.0
    while read_h < checkpoint_h:
        n, rates, earned = state_at(max(read_h, 0.0))
        history.append({"slot": iso(RELEASE + timedelta(hours=read_h)), "at": iso(RELEASE + timedelta(hours=read_h)),
                        "players_tracked": int(round(n)), "trophies_earned": int(round(earned)), "rates": rates})
        read_h += 0.5
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": f"s{seed}", "at": iso(RELEASE + timedelta(hours=checkpoint_h)),
            "horizon_minutes": (target_h - checkpoint_h) * 60.0, "target": iso(RELEASE + timedelta(hours=target_h)),
            "levels": LEVELS, "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                                       "release_date": "2026-11-19", "source": "synthetic"},
            "trophies": [{k: v for k, v in tr.items() if not k.startswith("_")} for tr in trophies],
            "history": history, "notes": "synthetic"}
    _, truth, _ = state_at(target_h)
    return case, truth


def run_synthetic(seeds=(1, 2, 3, 4, 5)):
    print("== synthetic benchmark (f00-shaped: checkpoint 59.5 h, target 168 h) ==")
    agg = {"model": [], "persistence": [], "curve": []}
    cover = []
    for seed in seeds:
        case, truth = synth_world(seed)
        scores = {}
        for label, fn in (("model", model.forecast_case), ("persistence", baseline_persistence),
                          ("curve", baseline_curve)):
            fc = fn(case)
            check_format(fc, case)
            scores[label] = pinball_loss(fc, case, truth)
            agg[label].append(scores[label])
        rows = {r["id"]: r["q"] for r in model.forecast_case(case)["trophies"]}
        inside = sum(1 for i, tr in enumerate(case["trophies"])
                     if rows[tr["id"]][0] <= truth[i] <= rows[tr["id"]][4])
        cover.append(inside / len(case["trophies"]))
        print(f"seed {seed}: model {scores['model']:.4f} | persistence {scores['persistence']:.4f} | "
              f"curve {scores['curve']:.4f} | model 90% coverage {cover[-1]:.2f}")
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"mean:    model {mean['model']:.4f} | persistence {mean['persistence']:.4f} | "
          f"curve {mean['curve']:.4f} | mean coverage {sum(cover) / len(cover):.2f}")
    return mean


def synth_world_x(seed, pool_start=600.0, pool_add=80000.0, pool_tau=72.0, pool_exp=0.85,
                  refresh_lo=2.5, refresh_hi=6.5, checkpoint_h=59.5, target_h=168.0,
                  a_lo=0.01, a_hi=0.97, tau_lo=2.0, tau_hi=320.0, earn_exp=0.8,
                  surge_h=None, surge_add=0.0):
    """synth_world with the world's knobs exposed; the defaults reproduce synth_world exactly.

    The knobs move one feature of the world at a time: how fast and how far the tracked pool grows,
    how often the page refreshes, where the checkpoint and target sit, how common the trophies are,
    how their earning curves bend, and a second pool wave (surge) partway through the window."""
    rng = random.Random(seed)
    trophies = []
    for k in range(55):
        if k == 0:
            trophies.append({"id": 1, "name": "Platinum", "description": "all", "tier": "platinum",
                             "points": 300, "secret": False, "_A": 0.004, "_tau": 260.0})
            continue
        tier = "gold" if k <= 2 else "silver" if k <= 8 else "bronze"
        a = math.exp(rng.uniform(math.log(a_lo), math.log(a_hi)))
        tau = math.exp(rng.uniform(math.log(tau_lo), math.log(tau_hi)))
        trophies.append({"id": k + 1, "name": f"T{k + 1}", "description": "synthetic", "tier": tier,
                         "points": {"gold": 90, "silver": 30, "bronze": 15}[tier],
                         "secret": bool(rng.getrandbits(1)), "_A": a, "_tau": tau})

    def pool(hours):
        base = pool_start + pool_add * (1.0 - math.exp(-((hours / pool_tau) ** pool_exp)))
        if surge_h is not None and hours > surge_h:
            base += surge_add * (1.0 - math.exp(-((hours - surge_h) / 30.0)))
        return base

    refreshes = []
    t = 0.0
    while t < target_h + 10.0:
        refreshes.append((t, pool(t)))
        t += rng.uniform(refresh_lo, refresh_hi)

    def state_at(hours):
        step = max((s for s in refreshes if s[0] <= hours), key=lambda s: s[0])
        n = step[1]
        earners = []
        for tr in trophies:
            e = 0.0
            for j, (s0, n0) in enumerate(refreshes):
                if s0 > step[0]:
                    break
                cohort = n0 - (refreshes[j - 1][1] if j else 0.0)
                age = step[0] - s0
                e += cohort * tr["_A"] * (1.0 - math.exp(-((age / tr["_tau"]) ** earn_exp))) if age > 0 else 0.0
            earners.append(e)
        rates = [round(100.0 * e / n, 2) if n else 0.0 for e in earners]
        return n, rates, sum(earners)

    history = []
    read_h = -0.17
    while read_h < checkpoint_h:
        n, rates, earned = state_at(max(read_h, 0.0))
        history.append({"slot": iso(RELEASE + timedelta(hours=read_h)), "at": iso(RELEASE + timedelta(hours=read_h)),
                        "players_tracked": int(round(n)), "trophies_earned": int(round(earned)), "rates": rates})
        read_h += 0.5
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": f"s{seed}", "at": iso(RELEASE + timedelta(hours=checkpoint_h)),
            "horizon_minutes": (target_h - checkpoint_h) * 60.0, "target": iso(RELEASE + timedelta(hours=target_h)),
            "levels": LEVELS, "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                                       "release_date": "2026-11-19", "source": "synthetic"},
            "trophies": [{k: v for k, v in tr.items() if not k.startswith("_")} for tr in trophies],
            "history": history, "notes": "synthetic"}
    _, truth, _ = state_at(target_h)
    return case, truth


REGIMES = {
    "base": {},
    "fast_pool": {"pool_tau": 30.0, "pool_add": 150000.0},
    "slow_pool": {"pool_tau": 150.0, "pool_add": 45000.0},
    "small_pool": {"pool_start": 250.0, "pool_add": 8000.0, "pool_tau": 60.0},
    "early": {"checkpoint_h": 36.0, "target_h": 120.0},
    "late": {"checkpoint_h": 100.0, "target_h": 220.0},
    "fast_refresh": {"refresh_lo": 0.75, "refresh_hi": 2.0},
    "slow_refresh": {"refresh_lo": 6.0, "refresh_hi": 12.0},
    "steep": {"earn_exp": 1.25, "tau_lo": 6.0, "tau_hi": 420.0},
    "flat": {"earn_exp": 0.55, "tau_lo": 1.0, "tau_hi": 200.0},
    "rare": {"a_lo": 0.001, "a_hi": 0.45},
    "common": {"a_lo": 0.25, "a_hi": 0.995},
    "surge": {"surge_h": 72.0, "surge_add": 50000.0},
}


def run_extended():
    """The recalibration suite: every regime on tuning seeds (201-205) and held-out seeds (301-305)."""
    print("== extended synthetic regimes (model only; tuning seeds 201-205, held-out 301-305) ==")
    agg = {"tune": [], "hold": []}
    for name, kw in REGIMES.items():
        line = f"{name:12s}"
        for split, seeds in (("tune", range(201, 206)), ("hold", range(301, 306))):
            scores, cover = [], []
            for seed in seeds:
                case, truth = synth_world_x(seed, **kw)
                fc = model.forecast_case(case)
                check_format(fc, case)
                scores.append(pinball_loss(fc, case, truth))
                rows = {r["id"]: r["q"] for r in fc["trophies"]}
                cover.append(sum(1 for i, tr in enumerate(case["trophies"])
                                 if rows[tr["id"]][0] <= truth[i] <= rows[tr["id"]][4]) / len(case["trophies"]))
            mean = sum(scores) / len(scores)
            agg[split].append(mean)
            line += f" | {split} {mean:.4f} (cov {sum(cover) / len(cover):.2f})"
        print(line)
    print(f"mean over regimes: tune {sum(agg['tune']) / len(agg['tune']):.4f} | "
          f"hold {sum(agg['hold']) / len(agg['hold']):.4f}")


def edge_tests():
    """Malformed and boundary cases: the forecaster must always return a contract-valid forecast."""
    print("== edge cases ==")
    base_trophies = [
        {"id": 1, "name": "P", "description": "", "tier": "platinum", "points": 300, "secret": False},
        {"id": 2, "name": "A", "description": "", "tier": "bronze", "points": 15, "secret": True},
        {"id": 3, "name": "B", "description": "", "tier": "silver", "points": 30, "secret": False},
    ]

    def read(slot, players, rates, earned=100):
        return {"slot": slot, "at": slot, "players_tracked": players, "trophies_earned": earned, "rates": rates}

    def case(**kw):
        c = {"schema": "makefun/gta6-trophies-case/v1", "id": "edge", "at": "2026-11-20T00:00:00Z",
             "horizon_minutes": 1440, "target": "2026-11-21T00:00:00Z", "levels": LEVELS,
             "game": {"name": "Grand Theft Auto VI", "platform": "PS5", "release_date": "2026-11-19",
                      "source": "test"},
             "trophies": copy.deepcopy(base_trophies), "history": [], "notes": ""}
        c.update(kw)
        return c

    variants = {
        "empty history": case(),
        "one read": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.1, 50.0, 3.0])]),
        "all-null trophy": case(history=[read("2026-11-19T01:00:00Z", 1000, [None, 50.0, 3.0]),
                                         read("2026-11-19T05:00:00Z", 2000, [None, 55.0, 4.0], 300)]),
        "null in last read": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.1, 50.0, 3.0]),
                                           read("2026-11-19T05:00:00Z", 2000, [0.2, None, 4.0], 300)]),
        "rates 0 and 100": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.0, 100.0, 0.0]),
                                         read("2026-11-19T05:00:00Z", 2000, [0.0, 100.0, 0.01], 300)]),
        "players zero/missing": case(history=[read("2026-11-19T01:00:00Z", 0, [0.1, 50.0, 3.0]),
                                              {"slot": "2026-11-19T05:00:00Z", "at": "2026-11-19T05:00:00Z",
                                               "rates": [0.2, 55.0, 4.0]}]),
        "no levels/target/horizon": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.1, 50.0, 3.0])],
                                         levels=None, target=None, horizon_minutes=None),
        "no trophies_earned": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.1, 50.0, 3.0], None),
                                            read("2026-11-19T05:00:00Z", 2000, [0.3, 55.0, 4.0], None)]),
        "single trophy": case(trophies=[base_trophies[1]],
                              history=[read("2026-11-19T01:00:00Z", 1000, [50.0]),
                                       read("2026-11-19T05:00:00Z", 2000, [55.0], 300)]),
        "zero horizon": case(history=[read("2026-11-19T01:00:00Z", 1000, [0.1, 50.0, 3.0])],
                             horizon_minutes=0, target="2026-11-20T00:00:00Z"),
    }
    for name, c in variants.items():
        fc = model.forecast_case(c)
        check_format(fc, c)
        print(f"  ok: {name}")
    print("  all edge cases produced valid forecasts")


if __name__ == "__main__":
    run_practice()
    run_synthetic()
    run_extended()
    edge_tests()
