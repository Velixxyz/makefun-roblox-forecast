# Roblox Up-and-Coming Top-10 Forecaster

Entry for make.fun `roblox-up-and-coming-forecast-01`: a program that reads
the history of Roblox's Up-and-Coming chart and forecasts its top 10, with
a probability for each game at each place, for the 72 hours after entries
close.

## How it works

Monte Carlo rank diffusion. Each game in the last chart has a rank and a
shrunk recent velocity (velocity persists with correlation about 0.25, so it
is shrunk by 0.25 and decays with a 12-hour half-life). Each hour for 72
hours the rank moves by velocity plus sigma(rank) times a standard normal
draw, where sigma grows with rank and is estimated from the input history.
Games older than 28 days age out of the chart when a created date is known.
The forecast is the occupancy distribution over 3,000 simulated paths,
blended 10/90 with a Gaussian-kernel prior, capped at 0.8, leaving 0.22 of
every place unnamed as insurance for new games that rocket into the top 10.

Player counts are deliberately ignored: on the archive, log playing growth
is uncorrelated with rank moves (correlation -0.02).

## Tuning

Tuned on the practice window (10 archive reads 2026-11-08..2026-11-10) and
three rolling backtests inside the 34-read history. Practice log score
about -34.3 nats; the house baseline scores -43.5 and the prize bar (best
house entry) is -37.2.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, offline, deterministic (fixed seed, sorted output).
