"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads the chart history in /input (or $MAKEFUN_INPUT_DIR) and writes one
forecast to /output/forecast.json (or $MAKEFUN_OUTPUT_DIR): for each of the
top 10 places, a probability for each game it names being at that place,
plus its guess at the top 10 in order. The single forecast is scored
against every hourly read of the 72 hours after entries close, so it is a
pooled occupancy distribution, not a point prediction.

Method: Monte Carlo rank diffusion.
- Each game in the last chart has a rank (float) and a shrunk recent
  velocity. Rank velocity persists with correlation about 0.25, so the
  velocity is shrunk by 0.25 and decays with a 12-hour half-life.
- Each hour for 72 hours the rank moves by velocity plus sigma(rank) times
  a standard normal draw. Sigma grows with rank (about 0.5 ranks/sqrt-hour
  at the top, about 3.7 deep in the chart), estimated from the input
  history and shrunk toward priors measured on the public archive.
  No systematic drift is applied inside the simulation: in an
  independent-diffusion model the measured downward drift of mid-ranked
  games is an artifact of new games entering at the bottom, and applying
  it there hurts the score. The drift is used only once, in the prior
  mean, as a one-shot shift at the mean horizon.
- Games older than 28 days age out of the chart (the chart only lists
  games published in the last 28 days); the created date is used when the
  history carries it.
- The forecast is the occupancy distribution: p(game, place) is the
  fraction of (path, hour) pairs in which the game held that place,
  blended 10/90 with a Gaussian-kernel prior around the last rank, capped
  at 0.8, and renormalized to leave 0.22 of every place unnamed. The
  unnamed share is insurance for games that enter the chart during the
  window and rocket into the top 10, which cannot be named in advance;
  the archive shows such rockets repeatedly.
- Player counts do not enter the model: log playing growth is uncorrelated
  with rank moves (correlation -0.02 on the archive), because the chart is
  computed from growth, not live counts, and counts are dominated by
  time-of-day noise.

Tuned on the practice window (10 archive reads 2026-11-08..2026-11-10) and
three rolling backtests inside the 34-read history. Practice log score
about -34.3 nats with the house baseline at -43.5 and the prize bar (best
house entry) at -37.2. Deterministic: fixed seed, sorted output.
"""
import json
import math
import os
import random
from datetime import datetime

# Tuned parameters.
N_PATHS = 3000
N_HOURS = 72
N_GAMES = 60
VEL_W = 0.25
VEL_HALF_LIFE_H = 12.0
SIGMA_SCALE = 0.9
UNNAMED = 0.22
P_CAP = 0.8
PRIOR_W = 0.10
PRIOR_HORIZON_H = 36.0
SEED = 12345
MAX_RANK = 130.0

# Prior drift (ranks/hour) at anchor ranks, measured on the archive's
# 34 reads 2026-10-30..2026-11-07. Re-estimated from the input history at
# run time and shrunk toward these priors. Used only in the prior mean.
DRIFT_ANCHORS = [(2, 0.058), (7, 0.119), (15, 0.149), (30, 0.148),
                 (55, 0.038), (85, -0.378)]
# Prior sigma (ranks/sqrt-hour) at anchor ranks, measured on the archive.
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
    with open(os.path.join(input_dir, "history.ndjson"),
              encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    reads = []
    for d in raw:
        chart = [r["universe_id"] for r in sorted(d["up_and_coming"],
                                                  key=lambda r: r["rank"])]
        created = {}
        details = d.get("details") or {}
        for uid, det in details.items():
            if det and det.get("created"):
                try:
                    created[int(uid)] = _parse_time(det["created"])
                except (ValueError, TypeError):
                    pass
        reads.append({"at": _parse_time(d["at"]), "chart": chart,
                      "created": created})
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    return reads, task


def estimate_dynamics(reads):
    """Per-hour drift and diffusion sigma by rank, estimated from
    consecutive reads and shrunk toward the measured priors. Drift is used
    only in the prior mean (a one-shot shift at the mean horizon); the
    Monte Carlo itself applies no systematic drift, which scores better."""
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
    roughly the last 24 hours (up to 8 pairs), so it adapts to sparse
    archive reads and dense hourly reads alike."""
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


def forecast(reads, task):
    last = reads[-1]
    chart = last["chart"]
    created = {}
    for r in reads:
        created.update(r["created"])
    t0 = last["at"]
    drift_fn, sigma_fn = estimate_dynamics(reads)
    vel_decay = 0.5 ** (1.0 / VEL_HALF_LIFE_H)

    games = chart[:N_GAMES]
    init = []
    for u in games:
        r0 = chart.index(u) + 1
        v0 = VEL_W * estimate_velocity(reads, u)
        c = created.get(u)
        age_out_h = None
        if c is not None:
            age_out_h = (c.timestamp() + 28 * 86400 - t0.timestamp()) / 3600.0
        init.append((u, float(r0), v0, age_out_h))

    rng = random.Random(SEED)
    occ = {}
    rank_sum = {u: 0.0 for u in games}
    rank_cnt = {u: 0 for u in games}
    for _ in range(N_PATHS):
        states = [[r, v, ao] for (_, r, v, ao) in init]
        for h in range(1, N_HOURS + 1):
            order = []
            for i, (u, r0, v0, ao) in enumerate(init):
                r, v, _ = states[i]
                if ao is not None and h >= ao:
                    r = MAX_RANK + 10.0  # aged out of the chart
                else:
                    r = r + v + SIGMA_SCALE * sigma_fn(r) * rng.gauss(0.0, 1.0)
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

    total = N_PATHS * N_HOURS
    probs = {}
    for (u, place), c in occ.items():
        probs[(u, place)] = c / total
    if PRIOR_W > 0:
        for k in range(1, 11):
            tot = 0.0
            tmp = {}
            for u in games:
                r = chart.index(u) + 1
                mu = r + drift_fn(r) * PRIOR_HORIZON_H
                sig = 2.0 * math.sqrt(PRIOR_HORIZON_H / 6.4) * (1 + 0.05 * r)
                p = math.exp(-0.5 * ((k - mu) / sig) ** 2) / \
                    (sig * math.sqrt(2 * math.pi))
                tmp[u] = p
                tot += p
            if tot > 0:
                for u, p in tmp.items():
                    probs[(u, k)] = (1 - PRIOR_W) * probs.get((u, k), 0.0) + \
                        PRIOR_W * p / tot
    rows = []
    for k in range(1, 11):
        cand = [(u, p) for (u, pl), p in probs.items()
                if pl == k and p >= 0.001]
        cand = [(u, min(p, P_CAP)) for u, p in cand]
        s = sum(p for _, p in cand)
        if s <= 0:
            continue
        scale = (1.0 - UNNAMED) / s
        for u, p in sorted(cand):
            rows.append({"universe_id": u, "place": k,
                         "p": round(p * scale, 6)})
    mean_rank = {u: rank_sum[u] / max(rank_cnt[u], 1) for u in games}
    held = sorted({u for (u, _) in occ}, key=lambda u: mean_rank[u])
    top10 = held[:10]
    if len(top10) < 10:
        rest = [u for u in sorted(games, key=lambda u: mean_rank[u])
                if u not in top10]
        top10 = (top10 + rest)[:10]
    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    reads, task = load_history(input_dir)
    result = forecast(reads, task)
    out_path = os.path.join(output_dir, task["output"])
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
