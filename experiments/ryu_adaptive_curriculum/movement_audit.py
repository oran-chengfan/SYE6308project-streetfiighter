"""Paired movement interventions followed by the same frozen match policy.

An intervention is a whole prefix, never a value label for its first input.
The opponent continues to act during every native frame. Roots include both
controller histories and the learner history when present.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from experiments.league_adaptive.runtime import load_actor, setup, write
from experiments.ryu_direct_league.environment import RyuEnv


def outcome(info):
    return {k: info[k] for k in ("result", "end_reason", "self_hp", "opponent_hp", "native_frames")}


def score(info):
    return (int(info["result"] == "win") - int(info["result"] == "loss"),
            max(0, info["self_hp"]) - max(0, info["opponent_hp"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="models/counter_ryu_final/agent.pt")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--roots", default="0,16,48")
    parser.add_argument("--lengths", default="1,4,8,16")
    args = parser.parse_args()
    setup()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    actor = load_actor(args.agent)
    pool = json.loads(Path(args.pool).read_text())
    env = RyuEnv(actor.definition, pool, 0, gamma=1.)
    names = {a["name"]: i for i, a in enumerate(actor.actions)}
    requested = [int(x) for x in args.roots.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    cache = {}
    stateful = actor.definition["kind"] in ("memory_ppo", "recurrent_direct_ppo")
    if actor.definition.get("inference_hold_steps", 0):
        raise ValueError("A route-commitment intervention needs an explicit actual-input state update")
    rows = []
    started = time.monotonic()

    def choose(obs):
        if stateful:
            return int(actor.predict(obs, True)[0])
        key = obs["image"].tobytes() + obs["position"].tobytes()
        if key not in cache:
            if len(cache) > 12000:
                cache.clear()
            cache[key] = int(actor.predict(obs, True)[0])
        return cache[key]

    def finish(obs):
        while True:
            obs, _, terminated, truncated, info = env.step(choose(obs))
            if terminated or truncated:
                return info

    try:
        for index, rival in enumerate(pool):
            for player in ((1,) if rival["kind"] == "cpu" else (1, 2)):
                for offset in ((0,) if rival["kind"] == "cpu" else (0, 12)):
                    actor.reset()
                    obs, _ = env.reset(options=dict(opponent=index, player=player, offset=offset))
                    roots = []
                    for step in range(max(requested) + 1):
                        if step in requested:
                            roots.append(dict(saved=env.save(), actor_state=copy.deepcopy(actor.get_state()),
                                              decision=step, raw=dict(env.game.raw)))
                        if step == max(requested):
                            break
                        obs, _, terminated, truncated, _ = env.step(choose(obs))
                        if terminated or truncated:
                            break
                    for root in roots:
                        plans = [("baseline", 0)] + [(mode, length) for mode in ("toward", "away", "uppercut") for length in lengths]
                        root_rows = []
                        for mode, length in plans:
                            obs = env.restore(root["saved"])
                            actor.set_state(copy.deepcopy(root["actor_state"]))
                            prefix = []
                            ended = False
                            for _ in range(length):
                                raw = env.game.raw
                                dx = raw[f"p{3-player}_x"] - raw[f"p{player}_x"]
                                forward = "RIGHT" if dx >= 0 else "LEFT"
                                name = forward if mode == "toward" else ("LEFT" if forward == "RIGHT" else "RIGHT") if mode == "away" else f"SHORYUKEN_{'R' if dx >= 0 else 'L'}_X"
                                action = names[name]
                                if stateful:
                                    if not hasattr(actor.policy, "observe_and_force"):
                                        raise ValueError("The actor cannot record a forced actual input")
                                    actor.policy.observe_and_force(obs, action)
                                obs, _, terminated, truncated, info = env.step(action)
                                prefix.append(dict(input=name, self_x=env.game.raw[f"p{player}_x"],
                                                   self_y=env.game.raw[f"p{player}_y"],
                                                   distance=abs(env.game.raw[f"p{3-player}_x"]-env.game.raw[f"p{player}_x"]),
                                                   self_hp=info["self_hp"], opponent_hp=info["opponent_hp"]))
                                if terminated or truncated:
                                    ended = True
                                    break
                            if not ended:
                                info = finish(obs)
                            row = dict(opponent=rival["id"], player=player, offset=offset,
                                       root_decision=root["decision"], root_readings=root["raw"],
                                       intervention=mode, prefix_decisions=length, prefix=prefix, **outcome(info))
                            root_rows.append(row)
                        obs = env.restore(root["saved"])
                        actor.set_state(copy.deepcopy(root["actor_state"]))
                        replay = finish(obs)
                        if outcome(replay) != outcome(root_rows[0]):
                            raise RuntimeError("Baseline full-game replay changed")
                        base = root_rows[0]
                        for row in root_rows:
                            row["outcome_improved"] = score(row)[0] > score(base)[0]
                            row["outcome_worsened"] = score(row)[0] < score(base)[0]
                            row["hp_margin_change"] = score(row)[1] - score(base)[1]
                        rows.extend(root_rows)
                        write(out / "progress.json", dict(completed_roots=len(rows)//len(plans), seconds=time.monotonic()-started,
                                                          native_frames=env.native_frames, rows=rows))
                        print("MOVEMENT_AUDIT", rival["id"], player, offset, root["decision"],
                              [(r["intervention"], r["prefix_decisions"], r["result"], r["hp_margin_change"]) for r in root_rows], flush=True)
        grouped = {}
        for mode in ("toward", "away", "uppercut"):
            cases = [r for r in rows if r["intervention"] == mode]
            grouped[mode] = dict(branches=len(cases), improved=sum(r["outcome_improved"] for r in cases),
                                 worsened=sum(r["outcome_worsened"] for r in cases),
                                 mean_hp_margin_change=sum(r["hp_margin_change"] for r in cases)/max(1,len(cases)))
        write(out / "summary.json", dict(agent=args.agent, arguments=vars(args), comparisons=grouped,
                                         seconds=time.monotonic()-started, native_frames=env.native_frames, rows=rows,
                                         scope="Whole-prefix conditional effects against the same acting opponent and frozen continuation. Not a first-action oracle, trained policy, or independent generalization estimate."))
    finally:
        env.close()


if __name__ == "__main__":
    main()
