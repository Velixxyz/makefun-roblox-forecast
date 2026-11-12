# Roblox Up-and-Coming Top-10 Forecaster v3

Entry for make.fun `roblox-up-and-coming-forecast-01`: a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, with
a probability for each game at each place, for the 72 hours after entries
close.

v3 = v2 (three-way ensemble, same parameters) with ONE fix: the hard
created+28d aging-out rule in the Monte Carlo component is REMOVED.

## Why the aging-out rule was wrong

The rule assumed a game leaves the chart 28 days after its universe `created`
date. Verified 2026-11-13 against the public games API (246 of 253 games in
the history + practice reads): games with `created` age up to 369 days still
appear on the chart (median age 22.8 days, p90 33.3 days), and exit hazard
rises smoothly with age (0.019 at 7-14d to 0.105 at 28d+) with no cliff at
28 days. The chart's "last 28 days" eligibility does not run on the universe
`created` clock.

The bug was invisible in all practice/walk-forward testing because the
archive history carries no `details` (no created dates), so the rule never
triggered. The FINAL input does carry `details` (house recorder reads call
games/v1), so the rule would have triggered only on the final grading run.

Measured damage in a created-date fixture test (same practice reads,
`details` injected from the games API):

- v1 (submitted): -34.263 without created dates, -38.597 with them (-4.33 nats, hits 7.3 -> 4.5)
- v2: -32.705 without, -34.411 with them (-1.71 nats, hits 7.3 -> 4.5)
- v3 (fix): -32.705 with created dates, byte-identical output to v2 on the
  original no-details input (regression check), deterministic across runs.

Exits are left to the empirical diffusion/transition behavior learned from
history (the OCC component models exits from observed transitions).

## How it works

A three-way ensemble, blended 0.6 / 0.2 / 0.2:

1. **Monte Carlo rank diffusion (0.6).** Each game in the last chart has a
   rank and a shrunk recent velocity (24h window, decay 0.5; velocity
   persists with correlation about 0.25). Each hour for 72 hours the rank
   moves by velocity plus sigma(rank) times a standard normal draw, with
   sigma scaled by 0.6: the archive shows raw diffusion over-disperses the
   pooled occupancy. Sigma grows with rank and is estimated from the input
   history, shrunk toward archive priors. No systematic drift is applied.
2. **Empirical transition occupancy (0.2).** Future ranks are drawn from
   the empirical transition distribution measured on the input history
   (start-rank bucket x lag bucket -> end rank or exit). Draw-and-sort
   over horizons 3..72h captures true diffusion, exits, jumps and mean
   reversion without assuming Gaussian walks. Velocity shift + temperature
   0.6.
3. **Direct occupancy kernel regression (0.2).** Every historical (read,
   game) moment is a training instance with features rank, velocity and
   lifecycle L (drawdown from the game's best rank + time since its peak);
   its target is the game's place-occupancy vector over the following 72h.
   Each game in the last chart averages similar moments under a Gaussian
   kernel. Temperature 0.6.

The three normalized forecasts are averaged per place and renormalized to
leave 0.35 of places 1-3 unnamed and 0.15 of places 4-10 unnamed: the top
places need more insurance for games that enter during the window and
rocket into the top 10, while lower places want sharper named mass.

Player counts are deliberately ignored: on the archive, log playing growth
is uncorrelated with rank moves (correlation -0.02), because the chart is
computed from growth, not live counts. Cross-chart membership and name
update-tags were tested and carry no usable signal.

## Validation

Practice window (10 archive reads 2026-11-08..2026-11-10): log score about
-32.7 nats (v1 Monte Carlo: -34.3). Seven walk-forward backtests inside
the 34-read history average about -35.1. A 70-config sweep on 2026-11-13
(blend weights, unnamed splits, per-place temperature, floors/caps,
persistence blend) found the v2 configuration at a flat optimum: no variant
beat it by more than 0.03 nats on the practice+WF blend metric, and every
sharpening variant gained on practice but lost more on walk-forward.
Ideas tested and rejected on walk-forward evidence: mean reversion to a
historical anchor, last-pair velocity, lifecycle peak drift, per-game sigma,
temperature below 0.5, persistence blending, per-place temperature, and a
standalone kernel-regression forecaster.

Deterministic: fixed seeds, sorted output. Standard library only, no
network, one run takes well under the 300s limit.
