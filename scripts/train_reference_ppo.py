"""Small ordinary-PPO continuation entry; never overwrites/promotes the reference.

This preserves the existing policy architecture and uses the existing Ryu
environment. It is a transport/smoke entry, not a demonstrated adaptive method.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.league_adaptive.runtime import ROOT, setup, write
from experiments.ryu_direct_league.policy import save
from experiments.ryu_direct_league.neural_mixture import NeuralMixturePolicy, mixture_kwargs
from experiments.ryu_counterq.environment import Worker


class Checkpoints(BaseCallback):
    def __init__(self, out, definition, checkpoint_every, max_seconds):
        super().__init__()
        self.out, self.definition = out, definition
        self.checkpoint_every, self.max_seconds = checkpoint_every, max_seconds
        self.last_save, self.started = 0, time.monotonic()
        self.rows = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if info.get("result") not in (None, "ongoing"):
                self.rows.append({k: info[k] for k in
                                  ("opponent", "player", "offset", "result", "self_hp", "opponent_hp", "native_frames")})
        if self.num_timesteps - self.last_save >= self.checkpoint_every:
            save(self.out / f"checkpoint_{self.num_timesteps}.pt", self.model.policy, self.definition,
                 dict(algorithm="ordinary_ppo_reference_continuation", steps=self.num_timesteps), self.model.policy.optimizer)
            write(self.out / "episodes.json", self.rows)
            self.last_save = self.num_timesteps
        return not (self.out / "STOP").exists() and time.monotonic() - self.started < self.max_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="models/counter_ryu_final/agent.pt")
    parser.add_argument("--pool", default="configs/train_pool.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1022)
    parser.add_argument("--checkpoint-every", type=int, default=12288)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.steps <= 0 or args.max_seconds <= 0:
        parser.error("Provide a positive, explicit training budget.")
    setup()
    out = Path(args.out).resolve()
    if not out.is_relative_to((ROOT / "artifacts").resolve()):
        parser.error("Training output must be a new directory under artifacts/.")
    out.mkdir(parents=True, exist_ok=False)
    source = torch.load(args.initial, map_location="cpu", weights_only=False)
    definition = copy.deepcopy(source["definition"])
    if definition["kind"] != "neural_mixture":
        raise ValueError("This small reference entry only continues the retained neural-mixture PPO.")
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    if any(r.get("character", "Ryu") != "Ryu" or r["kind"] == "cpu" for r in pool):
        raise ValueError("Supply a Ryu-only PvP pool; CPU levels are added separately.")
    cpu = [dict(id=f"CPU_Ryu_{s}", kind="cpu", character="Ryu", stars=s) for s in range(1, 9)]
    workers = [Worker(definition, pool, 0, args.seed, "win"),
               Worker(definition, pool, 1, args.seed + 317, "win"),
               Worker(definition, cpu, 2, args.seed + 634, "win")]
    env = SubprocVecEnv(workers, start_method="spawn")
    callback = Checkpoints(out, definition, args.checkpoint_every, args.max_seconds)
    try:
        model = PPO(NeuralMixturePolicy, env, policy_kwargs=mixture_kwargs(definition),
                    device="cpu", seed=args.seed, learning_rate=1e-4, n_steps=256,
                    batch_size=128, n_epochs=4, gamma=1., gae_lambda=.95,
                    clip_range=.1, ent_coef=.005, target_kl=.03, verbose=1)
        model.policy.load_state_dict(source["state_dict"], strict=True)
        write(out / "config.json", dict(arguments=vars(args), cpu=cpu, pool=pool,
              architecture="Unchanged retained PPO mixture; frozen expert constituents, trainable router/critic.",
              reward="Existing win objective: victory+1, every other completed outcome-1, telescoping 0.25 HP potential; gamma1.",
              scope="Small ordinary PPO continuation; not a validated adaptive improvement or exact recreation of the old training run."))
        save(out / "initial.pt", model.policy, definition, dict(source=args.initial))
        model.learn(total_timesteps=args.steps, callback=callback)
        save(out / "last.pt", model.policy, definition,
             dict(algorithm="ordinary_ppo_reference_continuation", steps=model.num_timesteps), model.policy.optimizer)
        write(out / "summary.json", dict(steps=model.num_timesteps, episodes=callback.rows,
              native_frames=sum(env.get_attr("native_frames")),
              wall_seconds=time.monotonic() - callback.started,
              promoted=False, requires_frozen_complete_match_evaluation=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
