"""Play a local round with the frozen competition policies."""

import argparse
import io
import json
from collections import deque
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .game import CHARACTERS, NativeGame
from .model import compiled, construct_sequence_q
from .observation import preprocess

SOURCE = Path(__file__).resolve().parent
MODELS = SOURCE.parent / "models"


def load_policy(name):
    """Read the trained tensors directly, preserving the original release files."""
    if name == "G":
        with ZipFile(MODELS / "G/Group_10.zip") as outer:
            with ZipFile(io.BytesIO(outer.read("agent.zip"))) as agent:
                archive = io.BytesIO(agent.read("policy.zip"))
    elif name == "F":
        archive = MODELS / "F/policy.zip"
    else:
        raise ValueError(f"Unknown model {name!r}")
    with ZipFile(archive) as policy_zip:
        weights = torch.load(
            io.BytesIO(policy_zip.read("policy.pth")),
            map_location="cpu",
            weights_only=True,
        )
    config = json.loads((SOURCE / "policy_config.json").read_text())
    config["actions"] = json.loads((SOURCE / "actions.json").read_text())
    policy = construct_sequence_q(config)
    policy.load_state_dict(weights, strict=True)
    policy.set_training_mode(False)
    policy.reset_inference_state()
    return policy


def observation(frames, raw, player):
    opponent = 3 - player
    x, y = raw[f"p{player}_x"], raw[f"p{player}_y"]
    ox, oy = raw[f"p{opponent}_x"], raw[f"p{opponent}_y"]
    return dict(
        image=np.concatenate(frames),
        position=np.array([x, y, ox, oy, ox - x, oy - y], np.float32) / 1024,
    )


def play(p1="G", p2="F", cpu=None, stars=2, offset=0):
    torch.set_num_threads(1)
    game = NativeGame()
    actors = [load_policy(p1)]
    if cpu is None:
        actors.append(load_policy(p2))
    commands = compiled(json.loads((SOURCE / "actions.json").read_text()))
    try:
        game.open("pve" if cpu else "pvp", stars if cpu else 1, cpu or "Ryu")
        for _ in range(offset):
            game.step_keys()
        frames = deque([preprocess(game.rgb)] * 4, maxlen=4)
        reason = None
        while reason is None:
            inputs = []
            for port, actor in enumerate(actors, 1):
                action, _ = actor.predict(
                    observation(frames, game.raw, port), deterministic=True
                )
                inputs.append(commands[int(action)])
            if cpu:
                inputs.append([()] * 6)
            for left, right in zip(*inputs):
                raw, winner, reason = game.step_keys(left, right)
                if reason:
                    break
            frames.append(preprocess(game.rgb))
        return dict(
            p1=p1,
            p2=cpu or p2,
            winner=winner,
            reason=reason,
            hp=[max(0, raw["p1_hp"]), max(0, raw["p2_hp"])],
            frames=game.elapsed,
        )
    finally:
        game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=("F", "G"), default="G")
    opponent = parser.add_mutually_exclusive_group()
    opponent.add_argument("--p2", choices=("F", "G"), default="F")
    opponent.add_argument("--cpu", choices=CHARACTERS)
    parser.add_argument("--stars", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--offset", type=int, choices=(0, 12), default=0)
    print(json.dumps(play(**vars(parser.parse_args()))))
