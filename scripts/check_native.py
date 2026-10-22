"""Check nine Ryu starts and exact same-state controller replay."""
import argparse
import json
import numpy as np
from experiments.league_adaptive.runtime import NativeGame, setup, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/native_check.json")
    args = parser.parse_args()
    setup()
    game, rows = NativeGame(), []
    try:
        for mode, stars in [("pvp", 1)] + [("pve", s) for s in range(1, 9)]:
            game.open(mode, stars, "Ryu")
            initial = game.snapshot()
            for _ in range(6):
                game.step_keys(("RIGHT",), ("LEFT",) if mode == "pvp" else ())
            expected = game.snapshot()
            game.restore(initial)
            for _ in range(6):
                game.step_keys(("RIGHT",), ("LEFT",) if mode == "pvp" else ())
            observed = game.snapshot()
            if (observed["readings"] != expected["readings"] or
                    not np.array_equal(observed["ram"], expected["ram"]) or
                    not np.array_equal(observed["rgb"], expected["rgb"])):
                raise RuntimeError("Same-state replay changed.")
            rows.append(dict(mode=mode, stars=stars, replay_equal=True,
                             initial_hp=[initial["readings"]["p1_hp"], initial["readings"]["p2_hp"]],
                             replay_native_frames=observed["elapsed"]))
    finally:
        game.close()
    report = dict(passed=True, cases=rows,
                  scope="Native starts and short replay only; no training or complete-match strength claim.")
    write(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
