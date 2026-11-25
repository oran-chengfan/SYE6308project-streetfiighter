# SYE-6308 Street fighter

Group 10’s Ryu won the 2026 Street Fighter championship in **SYE6308 – Statistical Learning in Manufacturing Quality Control**. Watch the quarterfinal, semifinal and final in the [championship recordings (ZIP)](https://github.com/oran-chengfan/SYE6308project-streetfiighter/releases/download/championship-2026/championship_records.zip).

The TA’s Kaggle Arena used Python 3.13.15, PyTorch 2.11.0+cpu, NumPy 2.1.3, Gymnasium 1.2.0, Stable-Baselines3 2.7.0, Stable-Retro 1.0.1 and Gym 0.26.2.

## What changed from the classroom PPO

- **Memory.** The classroom policy sees four stacked frames. Our 128-unit GRU retains context across decisions, combining visual features, fighter coordinates and previous inputs.
- **Counterfactual practice.** We restored a fight state and compared retreating, waiting and blocking for the same duration. The same policy finished each branch so we could measure the effect on damage and the round outcome.
- **Matchup sampling.** Ryu faced 12 CPU characters across eight difficulties. PvP practice varied opponent style, opening delay and controller port.
- **External opponents.** League, PSRO, FSP and SFighterAI exposed Ryu to independently trained opponents. [FightLadder paper](https://proceedings.mlr.press/v235/li24q.html) · [SFighterAI](https://github.com/linyiLYi/street-fighter-ai)
- **Targeted teaching.** We replayed successful sequences from teachers specializing in spacing, counterattacks and finishing rounds. G’s final updates protected the ranking of proven actions while learning responses to P1 pressure.
- **Basic controls.** Each decision selects a six-frame input. Neutral, eight directions and six attack buttons provide 15 basic actions. Another 49 templates cover directional attacks and special-move inputs. [Action list](counter_ryu/actions.json)

## Models and code

| Ryu | Use | Four-bout qualification replay |
| --- | --- | ---: |
| [F](models/F/policy.zip) | Baseline and practice opponent | 1136 |
| [G](models/G/Group_10.zip) | Final championship submission and practice opponent | 1243 |

With the course game installed, run `python -m counter_ryu.play --p1 G --p2 F`. The notebook replays qualification and swaps both controller ports. [Match notebook](championship.ipynb) · [Policy](counter_ryu/model.py) · [Training code](training/) · [G’s Kaggle Arena result](models/G/arena_result.json)

## Limitations and next steps

Repeated special moves were common across teams in the finals. Ryu often chained Shoryukens and Hadokens. Immediate damage rewards may favor these attacks over footwork with a delayed payoff. Test this by comparing moving, waiting and blocking against opponents who punish missed attacks.

Teachers can disagree in similar positions. Select demonstrations by the opponent’s response and the outcome of the next exchange. Both of G’s losses on the 20-game held-out panel came at P1, and its broader CPU coverage fell from F’s 96/96 KOs to 79/96. Future evaluations should reserve entire opponent policies, swap ports and track character-by-difficulty KO coverage alongside qualification score.
