"""Roblox Up-and-Coming top-10 forecaster v4 (make.fun
roblox-up-and-coming-forecast-01).

Reads the chart history in /input (or $MAKEFUN_INPUT_DIR) and writes one
forecast to /output/forecast.json (or $MAKEFUN_OUTPUT_DIR): for each of
the top 10 places, a probability for each game it names being at that
place, plus its guess at the top 10 in order. The single forecast is
scored against every hourly read of the 72 hours after entries close, so
it is a pooled occupancy distribution, not a point prediction.

Method: a three-way ensemble, blended 0.6 / 0.2 / 0.2.
1. Monte Carlo rank diffusion (weight 0.6). Each game in the last chart
   has a rank and a shrunk recent velocity (24h window, decay 0.5;
   velocity weight 0.4: stronger persistence than the measured 0.25
   correlation, validated on walk-forward backtests in Nov 2026).
   Each hour for 72 hours the rank moves by velocity plus sigma(rank)
   times a standard normal draw, with sigma scaled by 0.5: the archive
   shows the raw diffusion over-disperses the pooled occupancy. Sigma grows with rank
   (about 0.5 ranks/sqrt-hour at the top, about 3.7 deep in the chart),
   estimated from the input history and shrunk toward priors measured
   on the public archive. No systematic drift is applied: in an
   independent-diffusion model the measured downward drift of mid-ranked
   games is an artifact of new games entering at the bottom. Games older
   No hard aging-out: `created` age does not predict a 28-day exit (see
   forecast_mc note; verified against the games API, Nov 2026).
2. Empirical transition occupancy (weight 0.2). Future ranks are drawn
   from the empirical transition distribution measured on the input
   history: every pair of reads contributes (start-rank bucket, lag
   bucket) -> end rank or exit. Draw-and-sort over horizons 3..72h
   gives occupancy that captures true diffusion, exits, jumps and mean
   reversion without assuming Gaussian walks. A velocity shift and a
   temperature of 0.6 are applied.
3. Direct occupancy kernel regression (weight 0.2). Every historical
   (read, game) moment is a training instance with features rank,
   velocity and lifecycle L (drawdown from the game's best rank and
   time since its peak); its target is the game's place-occupancy
   vector over the following 72h. Each game in the last chart averages
   the targets of similar moments under a Gaussian kernel, with a
   temperature of 0.6.

The three normalized forecasts are averaged per place and renormalized
to leave 0.35 of places 1-3 unnamed and 0.15 of places 4-10 unnamed:
the top places need more insurance for games that enter the chart
during the window and rocket into the top 10, which cannot be named in
advance, while the lower places want sharper named mass.

Player counts do not enter the model: log playing growth is uncorrelated
with rank moves (correlation -0.02 on the archive), because the chart is
computed from growth, not live counts, and counts are dominated by
time-of-day noise. Cross-chart membership (Top Trending / Top Playing
Now) and name update-tags were tested and carry no usable signal.

Validated on the practice window (10 archive reads 2026-11-08..2026-11-10)
and seven walk-forward backtests inside the 34-read history. Practice
log score about -32.5 nats (v3: -32.7), walk-forward average about -35.0.
Deterministic: fixed seeds, sorted output.
"""
import json
import math
import os
import random
from datetime import datetime

# Tuned parameters (MC component).
MC_PATHS = 3000
MC_HOURS = 72
MC_GAMES = 100
MC_VEL_W = 0.4
MC_VEL_HALF_LIFE_H = 12.0
MC_SIGMA_SCALE = 0.5
MC_UNNAMED = 0.22
MC_P_CAP = 0.8
MC_SEED = 12345
MAX_RANK = 130.0

# Tuned parameters (empirical transition component).
OCC_DRAWS = 1500
OCC_HORIZONS = [3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
OCC_VEL_SHIFT = 0.5
OCC_VEL_HALF_LIFE_H = 12.0
OCC_TEMPERATURE = 0.6
OCC_SEED = 12345
OCC_RANK_BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 18), (19, 30), (31, 50),
                    (51, 75), (76, 120)]
OCC_LAG_BUCKETS = [(0, 8), (8, 16), (16, 30), (30, 50), (50, 80), (80, 200)]
OCC_EXIT_RANK = 200.0
OCC_REENTRY_MAX_GAP_H = 72.0

# Tuned parameters (kernel regression component).
KNN_BW_RANK = 8.0
KNN_BW_VEL = 0.5
KNN_BW_L = 0.35
KNN_TEMPERATURE = 0.6
KNN_MIN_FUTURE = 2

# Blend.
W_MC = 0.6
W_OCC = 0.2
W_KNN = 0.2
BLEND_UNNAMED_TOP3 = 0.35
BLEND_UNNAMED_REST = 0.15
BLEND_P_CAP = 0.8
BLEND_P_FLOOR = 0.001

# Prior drift (ranks/hour) and sigma (ranks/sqrt-hour) at anchor ranks,
# measured on the archive's 34 reads 2026-10-30..2026-11-07. Re-estimated
# from the input history at run time and shrunk toward these priors.
# Drift is estimated but not applied inside the simulation (see docstring).
DRIFT_ANCHORS = [(2, 0.058), (7, 0.119), (15, 0.149), (30, 0.148),
                 (55, 0.038), (85, -0.378)]
SIGMA_ANCHORS = [(2, 0.471), (7, 0.957), (15, 1.680), (30, 2.696),
                 (55, 3.369), (85, 3.685)]


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _interp(anchors, r):
    if r <= anchors[0][0]:
        return anchors[0][1]
    if r >= anchors[-1][0]:
        return anchors[-1][1]
    for (r0, v0), (r1, v1) in zip(anchors, anchors[1:]):
        if r0 <= r <= r1:
            f = (r - r0) / (r1 - r0)
            return v0 + f * (v1 - v0)
    return anchors[-1][1]


def load_history(input_dir):
    try:
        with open(os.path.join(input_dir, "history.ndjson"),
                  encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        raw = []
    reads = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        try:
            at = _parse_time(d["at"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        rows = d.get("up_and_coming") or []
        ranked = []
        seen = set()
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            try:
                uid = int(r["universe_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if uid in seen:
                continue
            seen.add(uid)
            rank = r.get("rank")
            if not isinstance(rank, (int, float)):
                rank = i + 1
            ranked.append((rank, uid))
        chart = [uid for _, uid in sorted(ranked, key=lambda p: p[0])]
        created = {}
        details = d.get("details") or {}
        if isinstance(details, dict):
            for uid, det in details.items():
                if isinstance(det, dict) and det.get("created"):
                    try:
                        created[int(uid)] = _parse_time(det["created"])
                    except (ValueError, TypeError):
                        pass
        reads.append({"at": at, "chart": chart, "created": created})
    # Time order is assumed downstream (reads[-1] is latest, _build_pools
    # breaks on h > 200). Stable: no-op on already-sorted inputs.
    reads.sort(key=lambda r: r["at"])
    try:
        with open(os.path.join(input_dir, "task.json"),
                  encoding="utf-8") as f:
            task = json.load(f)
    except (OSError, ValueError):
        task = {}
    if not isinstance(task, dict):
        task = {}
    return reads, task


def estimate_dynamics(reads):
    """Per-hour drift and diffusion sigma by rank, estimated from
    consecutive reads and shrunk toward the measured priors."""
    buckets = {a[0]: [] for a in SIGMA_ANCHORS}
    for a, b in zip(reads, reads[1:]):
        h = (b["at"] - a["at"]).total_seconds() / 3600.0
        if h <= 0 or h > 24:
            continue
        pos_b = {u: i + 1 for i, u in enumerate(b["chart"])}
        for i, u in enumerate(a["chart"]):
            if u not in pos_b:
                continue
            r = i + 1
            anchor = min(buckets, key=lambda x: abs(x - r))
            buckets[anchor].append((pos_b[u] - r, h))
    drift_pts, sigma_pts = [], []
    for (ar, pd), (_, ps) in zip(DRIFT_ANCHORS, SIGMA_ANCHORS):
        obs = buckets[ar]
        if len(obs) >= 20:
            d = sum(x for x, _ in obs) / sum(h for _, h in obs)
            var = sum((x - d * h) ** 2 / h for x, h in obs) / len(obs)
            w = len(obs) / (len(obs) + 50.0)
            drift_pts.append((ar, w * d + (1 - w) * pd))
            sigma_pts.append((ar, w * math.sqrt(max(var, 0.01)) + (1 - w) * ps))
        else:
            drift_pts.append((ar, pd))
            sigma_pts.append((ar, ps))
    return (lambda r: _interp(drift_pts, r),
            lambda r: _interp(sigma_pts, r))


def estimate_velocity(reads, game, decay=0.5):
    """Shrunk recent rank velocity (ranks/hour) from the pairs spanning
    roughly the last 24 hours (up to 8 pairs). Negative means rising."""
    vel = 0.0
    wsum = 0.0
    w = 1.0
    used = 0
    span = 0.0
    for a, b in reversed(list(zip(reads, reads[1:]))):
        h = (b["at"] - a["at"]).total_seconds() / 3600.0
        if h <= 0 or h > 24:
            continue
        if used >= 8 or span >= 24.0:
            break
        pa = {u: i + 1 for i, u in enumerate(a["chart"])}
        pb = {u: i + 1 for i, u in enumerate(b["chart"])}
        if game in pa and game in pb:
            vel += w * (pb[game] - pa[game]) / h
            wsum += w
        w *= decay
        used += 1
        span += h
    if wsum == 0:
        return 0.0
    return vel / wsum


def _normalize(probs, unnamed_top, unnamed_rest, p_cap, p_floor):
    """Renormalize a {(game, place): p} map into forecast rows."""
    rows = []
    for k in range(1, 11):
        cand = [(u, p) for (u, pl), p in probs.items()
                if pl == k and p >= p_floor]
        cand = [(u, min(p, p_cap)) for u, p in cand]
        s = sum(p for _, p in cand)
        if s <= 0:
            continue
        scale = (1.0 - (unnamed_top if k <= 3 else unnamed_rest)) / s
        # The format allows at most 100 rows per place: keep the 100
        # largest (ties broken by universe id, deterministically) and
        # rescale so the place keeps its exact unnamed share. Never
        # binds on ordinary inputs (observed maxima are ~30 rows).
        if len(cand) > 100:
            cand = sorted(cand, key=lambda t: (-t[1], t[0]))[:100]
            s = sum(p for _, p in cand)
            scale = (1.0 - (unnamed_top if k <= 3 else unnamed_rest)) / s
        for u, p in sorted(cand):
            rows.append({"universe_id": u, "place": k,
                         "p": round(p * scale, 6)})
    return rows


def forecast_mc(reads):
    """Component 1: Monte Carlo rank diffusion."""
    last = reads[-1]
    chart = last["chart"]
    created = {}
    for r in reads:
        created.update(r["created"])
    t0 = last["at"]
    _, sigma_fn = estimate_dynamics(reads)
    vel_decay = 0.5 ** (1.0 / MC_VEL_HALF_LIFE_H)

    games = chart[:MC_GAMES]
    init = []
    for u in games:
        r0 = chart.index(u) + 1
        v0 = MC_VEL_W * estimate_velocity(reads, u)
        # NO hard aging-out: verified 2026-11-13 against the games API that
        # games with universe `created` age up to 369 days still appear on the
        # chart, and exit hazard rises smoothly with age (no 28-day cliff).
        # The old created+28d rule zeroed old games only when `details` were
        # present (i.e. only on final input), costing v1 -4.3 nats and v2
        # -1.7 nats in a created-date fixture test. Exits are left to the
        # empirical diffusion/transition behavior learned from history.
        age_out_h = None
        init.append((u, float(r0), v0, age_out_h))

    rng = random.Random(MC_SEED)
    occ = {}
    rank_sum = {u: 0.0 for u in games}
    rank_cnt = {u: 0 for u in games}
    for _ in range(MC_PATHS):
        states = [[r, v, ao] for (_, r, v, ao) in init]
        for h in range(1, MC_HOURS + 1):
            order = []
            for i, (u, r0, v0, ao) in enumerate(init):
                r, v, _ = states[i]
                if ao is not None and h >= ao:
                    r = MAX_RANK + 10.0
                else:
                    r = r + v
                    r += MC_SIGMA_SCALE * sigma_fn(r) * rng.gauss(0.0, 1.0)
                    v = v * vel_decay
                    if r < 1.0:
                        r = 1.0
                    if r > MAX_RANK:
                        r = MAX_RANK
                states[i] = [r, v, ao]
                order.append((r, i))
            order.sort()
            for place, (_, i) in enumerate(order[:10], 1):
                u = games[i]
                occ[(u, place)] = occ.get((u, place), 0) + 1
            for (r, i) in order:
                u = games[i]
                rank_sum[u] += r
                rank_cnt[u] += 1

    total = MC_PATHS * MC_HOURS
    probs = {}
    for (u, place), c in occ.items():
        probs[(u, place)] = c / total
    rows = _normalize(probs, MC_UNNAMED, MC_UNNAMED, MC_P_CAP, 0.001)
    mean_rank = {u: rank_sum[u] / max(rank_cnt[u], 1) for u in games}
    held = sorted({u for (u, _) in occ}, key=lambda u: mean_rank[u])
    top10 = held[:10]
    if len(top10) < 10:
        rest = [u for u in sorted(games, key=lambda u: mean_rank[u])
                if u not in top10]
        top10 = (top10 + rest)[:10]
    return {"probabilities": rows, "top10": top10}


def _bucket_of(buckets, x):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= x <= hi:
            return i
    return len(buckets) - 1 if x > buckets[-1][1] else 0


def _build_pools(reads):
    """Empirical transition cells + re-entry pool (see module docstring)."""
    cells = {}
    reentry = []
    pos = [{u: i + 1 for i, u in enumerate(r["chart"])} for r in reads]
    for i in range(len(reads)):
        for j in range(i + 1, len(reads)):
            h = (reads[j]["at"] - reads[i]["at"]).total_seconds() / 3600.0
            if h <= 0 or h > 200:
                break
            lb = _bucket_of(OCC_LAG_BUCKETS, h)
            for u, r in pos[i].items():
                rb = _bucket_of(OCC_RANK_BUCKETS, r)
                end = float(pos[j][u]) if u in pos[j] else OCC_EXIT_RANK
                cells.setdefault((rb, lb), []).append(end)
            seen_before = set()
            for k in range(i):
                seen_before.update(pos[k])
            for u, r in pos[j].items():
                if u not in pos[i] and u in seen_before:
                    last = max(k for k in range(i) if u in pos[k])
                    gap = (reads[i]["at"] - reads[last]["at"]).total_seconds() / 3600.0
                    reentry.append((gap, float(r), h))
    for k in cells:
        cells[k].sort()
    return cells, reentry


def _cell_sample(rng, cells, rb, lb, lag_ids):
    arr = cells.get((rb, lb))
    if not arr:
        for dlb in sorted(lag_ids, key=lambda x: abs(x - lb)):
            arr = cells.get((rb, dlb))
            if arr:
                break
    if not arr:
        merged = []
        for (r2, l2), a in cells.items():
            if r2 == rb:
                merged.extend(a)
        arr = sorted(merged) if merged else [float(OCC_RANK_BUCKETS[rb][0])]
    return arr[rng.randrange(len(arr))]


def _apply_temperature(probs, temperature):
    if temperature == 1.0:
        return probs
    for k in range(1, 11):
        items = [(u, p) for (u, pl), p in probs.items() if pl == k]
        if not items:
            continue
        pw = [(u, p ** (1.0 / temperature)) for u, p in items]
        s = sum(p for _, p in pw)
        s0 = sum(p for _, p in items)
        if s > 0:
            for u, p in pw:
                probs[(u, k)] = p / s * s0
    return probs


def forecast_occ(reads):
    """Component 2: empirical transition occupancy (draw-and-sort)."""
    last = reads[-1]
    chart = last["chart"]
    t0 = last["at"]
    cells, reentry_pool = _build_pools(reads)
    lag_ids = sorted(set(lb for (_, lb) in cells))

    games = chart[:100]
    vel = {u: estimate_velocity(reads, u) for u in games}
    pos_last = set(chart)
    last_seen = {}
    for i, r in enumerate(reads):
        for u in r["chart"]:
            last_seen[u] = i
    cands = []
    for u, i in last_seen.items():
        if u in pos_last:
            continue
        gap = (t0 - reads[i]["at"]).total_seconds() / 3600.0
        if gap <= OCC_REENTRY_MAX_GAP_H:
            cands.append(u)
    n_pairs = sum(1 for i in range(len(reads)) for j in range(i + 1, len(reads))
                  if 0 < (reads[j]["at"] - reads[i]["at"]).total_seconds() / 3600.0 <= 200)
    reentry_rate = min(0.9, len(reentry_pool) / max(n_pairs, 1) / max(len(cands), 1) * 8.0) \
        if cands else 0.0
    reentry_ranks = sorted(r for (_, r, _) in reentry_pool) or [50.0]

    rng = random.Random(OCC_SEED)
    occ = {}
    rank_sum = {u: 0.0 for u in games}
    rank_cnt = {u: 0 for u in games}
    vel_decay = 0.5 ** (1.0 / OCC_VEL_HALF_LIFE_H)
    for h in OCC_HORIZONS:
        lb = _bucket_of(OCC_LAG_BUCKETS, float(h))
        eff_h = OCC_VEL_HALF_LIFE_H / math.log(2) * (1 - vel_decay ** h)
        for _ in range(OCC_DRAWS):
            draws = []
            for u in games:
                r0 = chart.index(u) + 1
                rb = _bucket_of(OCC_RANK_BUCKETS, r0)
                r = _cell_sample(rng, cells, rb, lb, lag_ids)
                r = r + OCC_VEL_SHIFT * vel[u] * eff_h
                if r < 1.0:
                    r = 1.0
                if r > MAX_RANK and r < OCC_EXIT_RANK:
                    r = MAX_RANK
                draws.append((r, u))
            for u in cands:
                if rng.random() < reentry_rate:
                    r = reentry_ranks[rng.randrange(len(reentry_ranks))]
                    draws.append((r, u))
            draws.sort()
            for place, (_, u) in enumerate(draws[:10], 1):
                occ[(u, place)] = occ.get((u, place), 0) + 1
            for (r, u) in draws:
                if u in rank_sum:
                    rank_sum[u] += min(r, MAX_RANK)
                    rank_cnt[u] += 1

    total = OCC_DRAWS * len(OCC_HORIZONS)
    probs = {}
    for (u, place), c in occ.items():
        probs[(u, place)] = c / total
    probs = _apply_temperature(probs, OCC_TEMPERATURE)
    rows = _normalize(probs, BLEND_UNNAMED_TOP3, BLEND_UNNAMED_REST, 0.8, 0.001)
    mean_rank = {u: rank_sum[u] / max(rank_cnt[u], 1) for u in games}
    held = sorted({u for (u, _) in occ if u in mean_rank},
                  key=lambda u: mean_rank[u])
    top10 = held[:10]
    if len(top10) < 10:
        rest = [u for u in sorted(games, key=lambda u: mean_rank[u])
                if u not in top10]
        top10 = (top10 + rest)[:10]
    return {"probabilities": rows, "top10": top10}


def _knn_features(reads, i, u, pos_list):
    """Features of game u at read i using reads 0..i only: rank, 24h
    velocity, lifecycle L (drawdown from best rank + time since peak)."""
    r = pos_list[i][u]
    vel = 0.0
    wsum = 0.0
    w = 1.0
    used = 0
    span = 0.0
    for a in range(i, 0, -1):
        h = (reads[a]["at"] - reads[a - 1]["at"]).total_seconds() / 3600.0
        if h <= 0 or h > 24:
            continue
        if used >= 8 or span >= 24.0:
            break
        if u in pos_list[a - 1] and u in pos_list[a]:
            vel += w * (pos_list[a][u] - pos_list[a - 1][u]) / h
            wsum += w
        w *= 0.5
        used += 1
        span += h
    vel = vel / wsum if wsum else 0.0
    past = [(k, pos_list[k][u]) for k in range(i + 1) if u in pos_list[k]]
    peak = min(x[1] for x in past)
    kpeak = max(k for k, x in past if x == peak)
    tsp = (reads[i]["at"] - reads[kpeak]["at"]).total_seconds() / 3600.0
    dd = r - peak
    L = math.exp(-dd / 8.0) * math.exp(-tsp / 48.0)
    return (float(r), vel, L)


def forecast_knn(reads):
    """Component 3: direct occupancy kernel regression."""
    last = reads[-1]
    chart = last["chart"]
    pos_list = [{u: i + 1 for i, u in enumerate(r["chart"])} for r in reads]
    n = len(reads)
    instances = []
    for i in range(n):
        fut = [j for j in range(i + 1, n)
               if (reads[j]["at"] - reads[i]["at"]).total_seconds() / 3600.0 <= 72.0]
        if len(fut) < KNN_MIN_FUTURE:
            continue
        for u in reads[i]["chart"]:
            occ = [0.0] * 10
            for j in fut:
                p = pos_list[j].get(u)
                if p is not None and p <= 10:
                    occ[p - 1] += 1.0
            occ = [x / len(fut) for x in occ]
            feat = _knn_features(reads, i, u, pos_list)
            instances.append((feat, occ, min(len(fut), 10)))

    games = chart[:100]
    i_last = len(reads) - 1
    probs = {}
    mean_place = {}
    for u in games:
        f = _knn_features(reads, i_last, u, pos_list)
        num = [0.0] * 10
        den = 0.0
        for (fi, occ, wt) in instances:
            d2 = ((f[0] - fi[0]) / KNN_BW_RANK) ** 2
            d2 += ((f[1] - fi[1]) / KNN_BW_VEL) ** 2
            d2 += ((f[2] - fi[2]) / KNN_BW_L) ** 2
            if d2 > 25.0:
                continue
            w = wt * math.exp(-0.5 * d2)
            den += w
            for k in range(10):
                num[k] += w * occ[k]
        if den <= 0:
            continue
        occ = [x / den for x in num]
        for k in range(10):
            if occ[k] > 0:
                probs[(u, k + 1)] = occ[k]
        s = sum(occ)
        mean_place[u] = sum((k + 1) * occ[k] for k in range(10)) / s if s > 0 else 99.0

    probs = _apply_temperature(probs, KNN_TEMPERATURE)
    rows = _normalize(probs, 0.22, 0.22, 0.8, 0.001)
    top10 = sorted(mean_place, key=lambda u: mean_place[u])[:10]
    if len(top10) < 10:
        rest = [u for u in games if u not in top10]
        top10 = (top10 + rest)[:10]
    return {"probabilities": rows, "top10": top10}


def blend(fcs, weights):
    """Arithmetic blend of component forecasts, renormalized per place."""
    probs = {}
    for fc, w in zip(fcs, weights):
        for r in fc["probabilities"]:
            probs[(r["universe_id"], r["place"])] = \
                probs.get((r["universe_id"], r["place"]), 0.0) + w * r["p"]
    rows = _normalize(probs, BLEND_UNNAMED_TOP3, BLEND_UNNAMED_REST,
                      BLEND_P_CAP, BLEND_P_FLOOR)
    return {"probabilities": rows, "top10": fcs[0]["top10"]}


def _complete_top10(top10, reads):
    """Ensure exactly 10 distinct universe ids: pad with games from the
    history (most recent reads first), then synthetic ids that cannot
    collide with real universe ids. Only binds on degenerate inputs."""
    out = list(dict.fromkeys(top10))
    if len(out) < 10:
        seen = set(out)
        for r in reversed(reads):
            for u in r["chart"]:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
                    if len(out) == 10:
                        break
            if len(out) == 10:
                break
    i = 0
    while len(out) < 10:
        out.append(10 ** 15 + i)
        i += 1
    return out[:10]


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    reads, task = load_history(input_dir)
    out_path = os.path.join(output_dir,
                            task.get("output") or "forecast.json")
    os.makedirs(output_dir, exist_ok=True)
    if not reads:
        # No usable history: emit a valid-shape forecast (the unnamed
        # spread degrades to uniform over each read's chart) rather
        # than crashing and scoring the floor on every place.
        result = {"probabilities": [],
                  "top10": _complete_top10([], reads)}
    else:
        mc = forecast_mc(reads)
        occ = forecast_occ(reads)
        knn = forecast_knn(reads)
        result = blend([mc, occ, knn], [W_MC, W_OCC, W_KNN])
        result["top10"] = _complete_top10(result["top10"], reads)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
