# SYE-6308 Street fighter

This Ryu won the 2026 Street Fighter championship in **SYE6308 – Statistical Learning in Manufacturing Quality Control** at CityUHK. Watch the quarterfinal, semifinal and final in the [championship recordings (ZIP)](https://github.com/oran-chengfan/SYE6308project-streetfiighter/releases/download/championship-2026/championship_records.zip).

The final Kaggle Arena used Python 3.13.15, PyTorch 2.11.0+cpu, NumPy 2.1.3, Gymnasium 1.2.0, Stable-Baselines3 2.7.0, Stable-Retro 1.0.1 and Gym 0.26.2. But we trained the model locally. If you do so, please remember trying your model in Kaggle!

## What changed from the classroom PPO

- **Memory.** The classroom policy sees 4 stacked and historical frames. Our 128-unit GRU retains context across decisions, combining visual features, fighter coordinates and previous inputs.
- **Counterfactual practice.** We restored a fight state and compared retreating, waiting and blocking for the same duration. The same policy finished each branch so we could measure the effect on damage and the round outcome.
- **Matchup sampling.** Ryu faced 12 CPU characters across eight-level difficulties. PvP practice varied opponent style, opening delay and controller port.
- **External opponents.** League, PSRO, FSP and SFighterAI exposed Ryu to independently trained opponents. [FightLadder paper](https://proceedings.mlr.press/v235/li24q.html) · [SFighterAI](https://github.com/linyiLYi/street-fighter-ai)
- **Targeted teaching.** We replayed successful sequences from teachers specializing in spacing, counterattacks and finishing rounds. Final model G’s final updates protected the ranking of proven actions while learning responses to P1 pressure.
- **Basic controls.** Each decision selects a six-frame input. Neutral, eight directions and six attack buttons provide 15 basic actions. Another 49 templates cover directional attacks and special-move inputs. [Action list](counter_ryu/actions.json)

## Models and code

| Ryu | Use | Four-bout qualification replay |
| --- | --- | ---: |
| [F](models/F/policy.zip) | Baseline and practice opponent | 1136 |
| [G](models/G/Group_10.zip) | Final championship submission and practice opponent | 1243 |

It should be noted that the locally tranining results does not represent the score you get in the Kaggle Arena, at least these two scores are different from what we saw in the class championship (PvP). Because opearting system, version as well as hardware affect the performance, and there is randomess.

With the course game installed, run `python -m counter_ryu.play --p1 G --p2 F`. The notebook replays qualification and swaps both controller ports. [Match notebook](championship.ipynb) · [Policy](counter_ryu/model.py) · [Training code](training/) · [G’s Kaggle Arena result](models/G/arena_result.json)

## Limitations and future directions

Repeated special moves were common across teams in the finals. The role Ryu often chained Shoryukens (升龙) and Hadokens (波动拳). Immediate damage rewards may favor these attacks over footwork (走位) with a delayed payoff. Test this by comparing moving, waiting and blocking against opponents who punish missed attacks.

Teacher models can disagree in similar positions. Select demonstrations by the opponent’s response and the outcome of the next exchange. Both of G’s losses on the 20-game held-out panel came at P1, and its broader CPU coverage fell down. Future evaluations should reserve entire opponent policies, swap ports and track character-by-difficulty KO coverage alongside qualification score.

Different RL algorithms beyond PPO should be encouraged for trials.
