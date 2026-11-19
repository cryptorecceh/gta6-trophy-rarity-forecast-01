"""Forecast GTA VI trophy earn rates a week after launch.

Reads one case per file from the input directory (JSON: the house's recording of
Exophase's GTA VI trophy page up to the case checkpoint) and writes, for every
trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the case
target, as /output/<id>.out.

Method (pure Python standard library, deterministic, no network):
  - Work on the logit scale, against x = ln(1 + hours since 2026-11-19T00:00Z).
    Trophy rates follow a saturating progression curve, so in logit-vs-log-time
    space the recent trajectory is close to a straight line that flattens.
  - Collapse the history to distinct reads (Exophase refreshes its page every
    few hours, so consecutive reads are usually identical), then per trophy fit
    a least-squares line of logit(rate) on x over the last 24 hours of reads.
  - Shrink each slope toward the cross-trophy median slope (empirical Bayes:
    noisy trophies borrow strength from the field), then damp the slope by 0.7
    because the curve flattens as it approaches its ceiling. The intercept is
    refit at the damped slope, and the line is read at the target.
  - Spread: the fit's prediction standard error combined with a random-walk
    term for unmodelled movement, scaled by 0.7, floored at 0.35 and capped
    at 2.0 logit units. Quantiles are the median shifted by the standard
    normal quantiles times the spread.
  - Guards from the trophy list itself: the platinum (earn every other trophy)
    can never exceed the rarest other trophy, and the chapter trophies form a
    chain (Chapter 1 >= Chapter 2 >= ... >= final mission). Medians that
    violate either are pooled back into order.
  - Holders never decrease: a median is never forecast below the rate implied
    by the holders already counted, even if the tracked-player count grew
    twenty-fold by the target.

With fewer than three usable reads for a trophy, fall back to persistence
(the last read's rate) with the random-walk spread. Nothing here reads the
network, the clock, or the environment beyond the input/output directories.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Time origin for the log-time axis: the launch day, 2026-11-19T00:00:00Z.
ORIGIN = datetime(2026, 11, 19, tzinfo=timezone.utc)
# How much of the history the line is fit on (hours back from the last read).
WINDOW_HOURS = 24.0
# Damping on the fitted slope: the growth curve flattens toward its ceiling.
DAMP = 0.7
# Prior spread of slopes for the empirical-Bayes shrinkage, in logit per log-hour.
SHRINK_TAU = 0.6
# Spread scale, floor and cap, in logit units.
SPREAD_SCALE = 0.7
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.0
# Random-walk spread of a rate's move, in logit units per square-root day.
RW_SIGMA = 0.5
# Holders cannot decrease: the median never goes below the rate implied by
# the holders already counted if the tracked count grew this many-fold.
MAX_TRACKED_GROWTH = 20.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    """ISO-8601 timestamp (with Z or an offset) -> hours since ORIGIN."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (moment - ORIGIN).total_seconds() / 3600


def read_time(read):
    """A read's time in hours since ORIGIN, from `at`, else from `slot`."""
    hours = parse_time(read.get("at"))
    if hours is None:
        hours = parse_time(read.get("slot"))
    return hours


def distinct_reads(history, times):
    """Collapse consecutive reads with identical data to one, at its last time.

    Exophase refreshes every few hours, so most reads repeat the previous one
    exactly; keeping each copy would overweight the flat stretches.
    """
    out = []
    for read, hours in zip(history, times):
        if hours is None:
            continue
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if out and out[-1][0] == key:
            out[-1] = (key, hours, read)
        else:
            out.append((key, hours, read))
    return out


def fit_line(points):
    """Weighted least squares of y on x. Returns (a, b, s2, xbar, sxx, n)."""
    total_w = sum(w for _, _, w in points)
    xbar = sum(w * x for x, _, w in points) / total_w
    ybar = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if sxx <= 0:
        return ybar, 0.0, 0.0, xbar, 0.0, len(points)
    slope = sum(w * (x - xbar) * (y - ybar) for x, y, w in points) / sxx
    intercept = ybar - slope * xbar
    dof = max(len(points) - 2, 1)
    resid2 = sum((y - (intercept + slope * x)) ** 2 for x, y, _ in points)
    return intercept, slope, resid2 / dof, xbar, sxx, len(points)


def chapter_chain(trophies):
    """Chapter trophies in story order, from their descriptions."""
    chapters = []
    final = None
    for trophy in trophies:
        desc = trophy.get("description") or ""
        match = re.match(r"Complete Chapter (\d+)\.", desc)
        if match:
            chapters.append((int(match.group(1)), trophy["id"]))
        elif desc.startswith("Complete the final mission"):
            final = trophy["id"]
    chapters.sort()
    chain = [tid for _, tid in chapters]
    if final is not None:
        chain.append(final)
    return chain


def pool_adjacent(values):
    """Make a list non-increasing by pooling adjacent violators (averaging)."""
    pooled = list(values)
    i = 0
    while i < len(pooled) - 1:
        if pooled[i] < pooled[i + 1]:
            mean = (pooled[i] + pooled[i + 1]) / 2
            pooled[i] = pooled[i + 1] = mean
            i = max(i - 1, 0)
        else:
            i += 1
    return pooled


def forecast_case(case: dict, name: str) -> dict:
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    target_hours = parse_time(case.get("target"))
    times = [read_time(read) for read in history]
    # Fill in any unreadable timestamps so a sloppy clock never drops a read:
    # step 30 minutes past the previous known time (the recorder's cadence).
    for i, hours in enumerate(times):
        if hours is None:
            times[i] = (times[i - 1] + 0.5) if i > 0 and times[i - 1] is not None else 0.0
    known = [t for t in times if t is not None]
    t_last = max(known) if known else 0.0
    if target_hours is None:
        target_hours = t_last + horizon_minutes / 60
    x_target = math.log1p(max(target_hours, -0.999))
    days = horizon_minutes / 1440
    rw = RW_SIGMA * math.sqrt(days) if days > 0 else 0.0

    obs = distinct_reads(history, times)
    last_read = history[-1] if history else {}
    last_rates = last_read.get("rates") or []
    last_n = last_read.get("players_tracked") or 0

    # Per-trophy line fits on the last day of distinct reads.
    fits = []
    for idx, trophy in enumerate(trophies):
        points = []
        for _, hours, read in obs:
            if hours < t_last - WINDOW_HOURS:
                continue
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            points.append((math.log1p(max(hours, -0.999)), logit(rates[idx]), 1.0))
        if len(points) >= 3:
            a, b, s2, xbar, sxx, n = fit_line(points)
            se_b = math.sqrt(max(s2, 1e-9) / sxx) if sxx > 0 else 9.9
            fits.append({"points": points, "a": a, "b": b, "s2": s2,
                         "xbar": xbar, "sxx": sxx, "n": n, "se_b": se_b})
        else:
            rate = last_rates[idx] if idx < len(last_rates) else None
            fits.append({"points": points, "a": logit(rate if rate is not None else 0.01),
                         "b": 0.0, "s2": 0.0, "xbar": 0.0, "sxx": 0.0, "n": len(points),
                         "se_b": 9.9})

    # Shrink the slopes toward the field's median slope.
    slopes = sorted(f["b"] for f in fits) if fits else [0.0]
    median_slope = slopes[len(slopes) // 2]
    rows = {}
    for idx, trophy in enumerate(trophies):
        fit = fits[idx]
        se2 = fit["se_b"] ** 2
        shrunk = (fit["b"] / se2 + median_slope / SHRINK_TAU ** 2) / (1 / se2 + 1 / SHRINK_TAU ** 2)
        slope = DAMP * shrunk
        points = fit["points"]
        if points:
            total_w = sum(w for _, _, w in points)
            intercept = sum(w * (y - slope * x) for x, y, w in points) / total_w
        else:
            intercept = fit["a"]
        median = clamp(inv_logit(intercept + slope * x_target))
        # Holders never decrease: floor at the counted holders over a
        # generously grown tracked-player count.
        rate = last_rates[idx] if idx < len(last_rates) else None
        if rate is not None and last_n:
            holders = rate / 100 * last_n
            floor = clamp(100 * holders / (MAX_TRACKED_GROWTH * last_n))
            median = max(median, floor)
        # Spread: prediction error of the fit plus the random walk.
        if fit["sxx"] > 0 and fit["n"] >= 2:
            se_pred = math.sqrt(max(fit["s2"], 0.0) *
                                (1 + 1 / fit["n"] + (x_target - fit["xbar"]) ** 2 / fit["sxx"]))
        else:
            se_pred = math.sqrt(max(fit["s2"], 0.0))
        spread = SPREAD_SCALE * math.sqrt(se_pred ** 2 + rw ** 2)
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        centre = logit(median)
        rows[trophy["id"]] = [round4(inv_logit(centre + z * spread)) for z in Z]

    # Guards from the trophy list: platinum <= every other trophy, and the
    # chapter chain non-increasing. Rebuild the rows of any median that moves.
    def rebuild(tid, median):
        centre = logit(median)
        old = rows[tid]
        spread = (logit(old[4]) - logit(old[0])) / (Z[4] - Z[0])
        rows[tid] = [round4(inv_logit(centre + z * spread)) for z in Z]

    platinum = [t["id"] for t in trophies if (t.get("tier") or "") == "platinum"]
    if platinum and len(rows) > 1:
        pid = platinum[0]
        cap = min(v[2] for k, v in rows.items() if k != pid)
        if rows[pid][2] > cap:
            rebuild(pid, cap)
    chain = chapter_chain(trophies)
    if len(chain) >= 2 and all(tid in rows for tid in chain):
        pooled = pool_adjacent([rows[tid][2] for tid in chain])
        for tid, value in zip(chain, pooled):
            if value != rows[tid][2]:
                rebuild(tid, value)

    ordered = [{"id": t["id"], "q": rows[t["id"]]} for t in trophies if t["id"] in rows]
    return {"trophies": ordered}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case without a forecast: persistence on the last
            # read's rates, or the smallest non-zero rate when there is none.
            trophies = case.get("trophies") or []
            history = case.get("history") or []
            last = history[-1] if history else {}
            rates = last.get("rates") or []
            rows = []
            for idx, trophy in enumerate(trophies):
                rate = rates[idx] if idx < len(rates) and rates[idx] is not None else 0.01
                centre = logit(rate)
                rows.append({"id": trophy["id"],
                             "q": [round4(inv_logit(centre + z * SPREAD_FLOOR)) for z in Z]})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
