# Roblox Up-and-Coming Top-10 Forecaster v6

Entry for make.fun `roblox-up-and-coming-forecast-01`: a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, with
a probability for each game at each place, for the 72 hours after entries
close. Deterministic, offline, standard library only.

Model: 3-way blend (0.6 Monte Carlo rank diffusion / 0.2 empirical
transition occupancy / 0.2 KNN direct occupancy), as in v4, plus the
round-3 growth signal in TWO components plus an adaptive tail trim:

1. MC growth velocity: the Monte Carlo initial velocity is augmented by
   the game's deseasonalized 24h log playing-count growth
   (MC_GROWTH_W = 0.3), estimated from the input's `playing` counts after
   removing the pooled time-of-day curve. Growth predicts 72h rank change
   (slope -12.3 ranks per log-unit) independently of rank velocity.
2. OCC growth shift: empirical-transition draws are shifted by the same
   growth signal (OCC_GROWTH_W = 0.5 times the effective horizon).
3. Adaptive tail trim: after blending, named rows below the place's
   implied unnamed rate (unnamed share / unnamed chart games) are dropped
   and survivors rescaled (up to 5 passes).

No aging-out rule; `details.created` is ignored (output invariant to it).

Verification (production config, 3000 paths):
- Seed 12345: practice -32.563, WF avg -34.220, blend -33.392 (v4: -33.773, delta +0.381).
- Seed 777: practice -32.570, WF avg -34.204, blend -33.387 (v4: -33.790, delta +0.403).
No walk-forward regression on either seed. Stress-tested on final-shaped
fixtures (determinism + runtime + output validity + created/order
invariance).

Frozen by round-3 subagent session e0bfafde-5ffe-4a5b-ba91-4fe85b75d9de.
