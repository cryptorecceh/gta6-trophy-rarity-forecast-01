# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cryptorecceh/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cryptorecceh/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one: for every trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the target, from the house's recording of the trophy page up to the checkpoint. Pure Python standard library, deterministic, no network.

## Method

Per trophy, a least-squares line of the logit's earn rate on log-time over the last 24 hours of distinct reads, with the slope shrunk toward the field's median slope and damped by 0.7 (the growth curve flattens toward its ceiling), read at the target. The spread combines the fit's prediction error with a random-walk term. Guards from the trophy list itself: the platinum never exceeds the rarest other trophy, the chapter trophies stay in story order, and holders never decrease.

Validated on synthetic launch worlds (lognormal and Weibull earn-time distributions): mean pinball loss 0.18 / 0.24 versus 0.32 / 0.40 for the house curve baseline and 0.59 / 0.61 for persistence, beating the house baseline in 40/40 worlds in each family.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s main.py` with one case file in `/input`; the forecast lands in `/output/<id>.out` as `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zdkc9vmx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
