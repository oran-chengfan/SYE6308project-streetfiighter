"""Rebuild archived P1 replay banks with the recorded inputs and trainer."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys


PREFIX = Path("artifacts/ryu_p1_pressure")
CONFIG = Path("configs/ryu_p1_pressure.json")
RANK = PREFIX / "runs/201131/plain/p1_rank_guarded_retention"
BANKS = (
    "p1_rehearsal_coverage", "closing_pressure_accumulation",
    "closing_pressure_timing", "neural_pressure_opening",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def original_input(source, relative):
    mapping = source / "artifacts/post_competition_deletion_candidates_20261124.csv"
    if (source / relative).is_file() or not mapping.is_file():
        return relative
    with mapping.open(encoding="utf-8-sig", newline="") as stream:
        match = next((row["retained"] for row in csv.DictReader(stream)
                      if Path(row["path"]) == relative and row["retained"]), None)
    return Path(match) if match else relative


def put(source, target, paths):
    aliases = {}
    for relative in map(Path, paths):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a repository-relative input: {relative}")
        actual = original_input(source, relative)
        if actual.is_absolute() or not (source / actual).resolve().is_relative_to(source):
            raise ValueError(f"Invalid retained input: {actual}")
        if actual != relative:
            aliases[relative.as_posix()] = actual.as_posix()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / actual, destination)
    return aliases


def run(target, arguments):
    result = subprocess.run(
        [sys.executable, "-B", *arguments], cwd=target, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)


def fit(source, target, stage, initial, scope, updates, output):
    cfg = stage / "fit_config.json"
    rule = read(source / cfg)["teacher_fit"]
    aliases = put(source, target, [cfg, rule["paired_initial_source"], *[
        path for path in rule["data_files"] if not (target / path).is_file()
    ]])
    if initial != Path(rule["paired_initial_source"]):
        aliases.update(put(source, target, [initial]))
    run(target, [
        "-m", "experiments.ryu_adaptive_curriculum.train_direct_sequence",
        "--offline-teacher", "--config", str(cfg), "--initial", str(initial),
        "--trainable-scope", scope, "--seed", "201131",
        "--max-updates", str(updates), "--out", output,
    ])
    return read(target / output / "result.json"), aliases


def coverage_inputs(source):
    transition = PREFIX / "p1_continuation"
    margin = PREFIX / "decision_margin"
    audit = read(source / margin / "audit.json")
    selection = read(source / transition / "transition_selection.json")
    development = next(
        row["development"] for row in selection["candidates"]
        if row["checkpoint"] == audit["models"]["C35"]
    )
    return [
        development, PREFIX / "p1_transition_rehearsal.pt",
        transition / "transition_plan.json", transition / "transition_selection.json",
        margin / "audit.json", margin / "matched_complete_followups.json",
    ]


def bank_recipe(source, bank):
    closing = PREFIX / "closing_pressure"
    neural = PREFIX / "neural_practice"
    if bank == "p1_rehearsal_coverage":
        return "prepare_p1_rehearsal_coverage", "p1_continuation/coverage_plan.json", coverage_inputs(source), False
    if bank == "closing_pressure_timing":
        return "prepare_timing_replay", "closing_pressure/timing_plan.json", [
            PREFIX / "closing_pressure_replay.pt", closing / "replay_selection.json",
            closing / "replay_plan.json", closing / "timing_support/registration.json",
            closing / "timing_support/diagnosis.json", closing / "timing_support/parent/histories.pt",
        ], False
    if bank == "neural_pressure_opening":
        return "prepare_neural_openings", "neural_practice/opening_plan.json", [
            PREFIX / "neural_pressure_retention.pt", neural / "readout_selection.json",
            neural / "training_plan.json", neural / "mixed_plan.json",
        ], False
    previous = closing / "balance_context_selection.json"
    followup = closing / "parent_followup/diagnosis.json"
    shared = closing / "accumulation_shared_opening/diagnosis.json"
    settings = read(source / CONFIG)["closing_pressure"]["accumulation_replay"]
    board = next(row["development"] for row in read(source / previous)["candidates"]
                 if (row["seed"], row["arm"]) == (settings["parent_seed"], settings["parent_arm"]))
    demonstrated = [Path(row["path"]) / "conditional_demonstrations.pt"
                    for row in read(source / followup)["results"] if row["continued_wins"]]
    others = [shared, Path(read(source / shared)["demonstrations"]) / "conditional_demonstrations.pt"]
    return "prepare_accumulation", "closing_pressure/accumulation_plan.json", [
        PREFIX / "closing_pressure_context.pt", previous, followup, board,
        closing / "balance_context_plan.json", neural / "training_plan.json",
        closing / "accumulation_development/baselines.json",
        closing / "accumulation_autonomous/histories.pt", *demonstrated, *others,
    ], True


def rebuild(source, target, bank):
    routine, plan_path, inputs, needs_rivals = bank_recipe(source, bank)
    aliases = put(source, target, [CONFIG, *inputs])
    run(target, ["-c",
        "from pathlib import Path; "
        f"from experiments.ryu_adaptive_curriculum.ryu_p1_pressure import read, {routine}; "
        f"{routine}(read({str(CONFIG)!r}), {'{}, ' if needs_rivals else ''}Path({str(PREFIX)!r}))",
    ])
    return plan_path, inputs, aliases


def value_equal(a, b):
    import numpy as np
    import torch

    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(value_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(value_equal(x, y) for x, y in zip(a, b))
    return a == b


def compare_banks(original, rebuilt):
    import torch

    source = torch.load(original, map_location="cpu", weights_only=False)
    candidate = torch.load(rebuilt, map_location="cpu", weights_only=False)
    if not value_equal(source, candidate):
        raise ValueError("Replay contents or ordering changed during reconstruction")
    return len(source["episodes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new directory under workspace/artifacts")
    parser.add_argument("--mode", choices=("verify", "rebuild", "resume"), default="verify")
    parser.add_argument("--bank", choices=BANKS, default=BANKS[0])
    args = parser.parse_args()
    source, target = args.workspace.resolve(), args.out.resolve()
    if not target.is_relative_to(source / "artifacts") or target == source / "artifacts":
        parser.error("Choose a new output directory under workspace/artifacts")
    target.mkdir()
    shutil.copytree(source / "experiments", target / "experiments",
                    ignore=shutil.ignore_patterns("__pycache__"))
    receipt = {}
    if args.mode == "resume":
        result, aliases = fit(source, target, RANK, RANK / "fit/last.pt", "output", 33, "resume")
        if result["updates"] != 33 or len(result["sampling_trace"]) != 1:
            raise ValueError("The restored trainer did not read and finish exactly one update")
        receipt["resume"] = {"from": 32, "to": 33, **result["sampling_trace"][0]}
        receipt["retained_input_aliases"] = aliases
    if args.mode in ("verify", "rebuild"):
        plan_path, inputs, aliases = rebuild(source, target, args.bank)
        old_plan = PREFIX / plan_path
        stage = PREFIX / "runs/201131/plain" / args.bank
        plan = read(target / old_plan)
        rule = read(source / stage / "fit_config.json")["teacher_fit"]
        if plan != read(source / old_plan) or plan["fixed_windows"] != rule["required_success_schedule"]:
            raise ValueError("The complete plan or fixed sampling schedule changed")
        receipt["bank"] = args.bank
        receipt["inputs"] = [Path(path).as_posix() for path in [CONFIG, *inputs, old_plan]]
        if args.mode == "verify":
            path = PREFIX / f"{args.bank}.pt"
            receipt["reconstructed_episodes"] = compare_banks(source / path, target / path)
        old_result = read(source / stage / "fit/result.json")
        result, training_aliases = fit(source, target, stage, Path(rule["paired_initial_source"]),
                                       old_result["trainable_scope"], 1, "batch")
        reference = old_result["sampling_trace"][0]
        if result["sampling_trace"] != [reference]:
            raise ValueError("The real fixed replay batch differed from the archived update")
        receipt["fixed_batch"] = {"plan_equal": True, **reference}
        receipt["retained_input_aliases"] = aliases | training_aliases
    (target / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
