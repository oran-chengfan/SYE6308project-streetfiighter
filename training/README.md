# Training code

`ppo.py` contains the recurrent sequence evaluation and clipped PPO objective. The final stage used supervised replay. `fit.py` reconstructs each action history, `teacher.py` computes action and retention losses, and `ranking.py` constrains the output layer so verified actions keep their rank on the protected histories.

G’s last 32 updates trained the final output layer. Every eight updates, the ranking constraint used a margin of 0.05 on selected successful histories.

To rerun that stage with the archived training inputs, restore its trusted parent checkpoint, replay banks and fit configuration at the paths recorded in the configuration. From the repository root run:

```sh
python -m training.fit --initial "$PARENT" --config "$FIT_CONFIG" --out artifacts/fit --seed 201131 --trainable-scope output --max-updates 32
```

The F and G releases are playable without those training archives. Their ZIPs contain deployment weights. The historical optimizer state and training examples are held in the local project archive.
