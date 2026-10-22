"""Compare reference predictions against the 80 saved legal observations."""
import argparse
import json
import numpy as np
import torch
from experiments.league_adaptive.runtime import ROOT, load_actor, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="models/counter_ryu_final/agent.pt")
    args = parser.parse_args()
    setup()
    folder = ROOT / "models/counter_ryu_final"
    with np.load(folder / "reload_inputs.npz", allow_pickle=False) as data:
        observations = {key: data[key].copy() for key in ("image", "position")}
    with np.load(folder / "reload_predictions.npz", allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    actor = load_actor(args.agent)
    actor.reset()
    with torch.no_grad():
        actions = np.asarray(actor.predict(observations, True)[0])
        tensor, _ = actor.policy.obs_to_tensor(observations)
        probabilities = actor.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
        values = actor.policy.predict_values(tensor).cpu().numpy()
    action_errors = int(np.sum(actions != expected["action"]))
    probability_error = float(np.max(np.abs(probabilities - expected["probability"])))
    value_error = float(np.max(np.abs(values - expected["value"])))
    report = dict(observations=len(actions), action_differences=action_errors,
                  probability_max_absolute_error=probability_error, value_max_absolute_error=value_error,
                  passed=action_errors == 0 and probability_error <= 1e-5 and value_error <= 1e-5,
                  scope="Prediction comparison only; no ROM, training or official Arena validation.")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Reference inference differs. Diagnose versions/observations before training.")


if __name__ == "__main__":
    main()
