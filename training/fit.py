"""Final supervised update on recorded, verified fight histories."""

import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from counter_ryu.game import ROOT
from .optimizer import restore_optimizer, save


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Windows readers and indexers can briefly deny replacement. Keep the
    # previous complete document available, and bound retries to one second.
    for attempt in range(21):
        try:
            tmp.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 20:
                raise
            time.sleep(0.05)


def fit_teacher_sequences(args):
    """Offline sequential imitation with shared CE and optional additive retention.

    No environment, value objective, refresh or PPO minibatch is constructed.
    A declared paired run can measure and weight KL on its fixed references.
    """
    from counter_ryu.model import construct_sequence_q
    from .optimizer import set_policy_learning_scope, policy_optimizer_step
    from .replay import SequenceReplay, audit_success_history_labels
    from .teacher import (
        recurrent_success_loss,
        teacher_sequence_metrics,
        recurrent_window_logits,
    )

    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rule = spec["teacher_fit"]
    verified_sources = rule.get("verified_mixed_sources", False)
    retain = "behavior_retention_coefficient" in rule
    retention_coefficient = float(rule.get("behavior_retention_coefficient", 0.0))
    if retain and (
        not verified_sources
        or not rule.get("paired_fixed_updates")
        or not 0 <= retention_coefficient <= 1
        or not rule.get("behavior_reference_source")
    ):
        raise ValueError(
            "Paired additive retention needs a declared frozen source and coefficient in [0, 1]"
        )
    paired_initial = rule.get(
        "paired_initial_source", rule.get("behavior_reference_source")
    )
    reference_sources = {
        rule.get("behavior_reference_source"),
        *rule.get("additional_behavior_reference_sources", ()),
    }
    priority_roles = set(rule.get("success_priority_roles", ()))
    priority_fraction = float(rule.get("success_priority_fraction", 0.0))
    if (
        bool(priority_roles) != bool(priority_fraction)
        or not 0 <= priority_fraction <= 1
    ):
        raise ValueError(
            "Success priority needs declared roles and a fraction in [0, 1]"
        )
    source = torch.load(args.initial, map_location="cpu", weights_only=False)
    definition = copy.deepcopy(source["definition"])
    if definition["kind"] != "sequence_ppo" or not definition.get("physical_actions"):
        raise ValueError("Use the existing physical-action recurrent PPO architecture")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = construct_sequence_q(definition)
    saved = source.get("training", {}).get("teacher_state")
    if (
        retain
        and saved is None
        and (
            args.teacher_initialization != "F"
            or Path(args.initial).resolve() != (ROOT / paired_initial).resolve()
        )
    ):
        raise ValueError(
            "Both paired fits must initialize from the registered frozen reference"
        )
    if saved is not None or args.teacher_initialization == "F":
        policy.load_state_dict(source["state_dict"], strict=True)
    scope = args.trainable_scope
    feature_gradients = scope in ("all", "encoder_readout")
    if saved is not None and saved.get("trainable_scope", "all") != scope:
        raise ValueError(
            "Resuming teacher optimization requires the same trainable scope"
        )
    set_policy_learning_scope(policy, scope, critic_active=False)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    policy.optimizer = torch.optim.Adam(parameters, lr=rule["learning_rate"])
    if saved is not None:
        restore_optimizer(policy.optimizer, source["optimizer_state"])
    expected = {id(p) for p in parameters}
    actual = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
    if actual != expected:
        raise RuntimeError("Teacher optimizer omitted intended parameters")
    splits = {"fit": [], "check": []}
    sources = []
    forbidden = (
        "conditional_preferences",
        "conditional_option",
        "retention_allowed_actions",
    )
    for path in rule["data_files"]:
        document = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        for i, original in enumerate(document["episodes"]):
            if not verified_sources and (
                not original.get("legal_controller_recording")
                or original.get("independent_learned_control")
            ):
                raise ValueError(
                    "Teacher-only data must have explicit actual legal controller provenance"
                )
            e = copy.deepcopy(original)
            reference = "policy_probabilities" in e
            if any(k in e for k in forbidden) or reference and not retain:
                raise ValueError(
                    "Offline fit received an undeclared soft or comparison target"
                )
            if reference and (
                e.get("policy_version") not in reference_sources
                or not e.get("independent_learned_control")
                or e.get("retention_target") != "physical_distribution"
            ):
                raise ValueError(
                    "KL is restricted to protected successes run by the frozen behavior policy"
                )
            if e["action_names"] != [a["name"] for a in definition["actions"]]:
                raise ValueError("Teacher physical action meanings differ")
            if not reference:
                e["retention_target"] = "executed_action"
            e["auxiliary_priority"] = e.get("data_role") in priority_roles
            e["persistent_auxiliary_priority"] = False
            group = (
                "check"
                if path in rule.get("check_data_files", ())
                or e["opponent_config"]["character"]
                in rule["check_opponent_characters"]
                else "fit"
            )
            e["teacher_source"] = dict(file=path, episode=i)
            splits[group].append(e)
            sources.append(
                dict(
                    file=path,
                    episode=i,
                    split=group,
                    opponent=e["opponent_config"],
                    player=e["player"],
                    transitions=len(e["actions"]),
                    label_first=e.get("supervision_start", 0),
                )
            )
    if not splits["fit"] or not splits["check"]:
        raise ValueError("Declare nonempty whole-game fit and check splits")
    audit = audit_success_history_labels(splits["fit"], use_soft_references=False)
    if audit["both_hard_target_conflicts"]:
        raise ValueError("Teacher hard labels conflict on identical legal histories")
    length = (
        max(
            len(e["actions"]) - int(e.get("supervision_start", 0))
            for e in splits["fit"]
        )
        if rule.get("complete_sequences", False)
        else rule["sequence_length"]
    )
    bank = SequenceReplay(
        capacity=rule.get("capacity_transitions", 10000),
        byte_capacity=rule.get("byte_capacity", 67108864),
        seed=args.seed,
        burn=0,
        learn=length,
        nstep=1,
    )
    for e in splits["fit"]:
        if rule.get("complete_sequences", False):
            e["sample_complete_supervision_window"] = True
        bank.add_verified_demonstration(e)
    if retain and (
        len(bank.episodes) != len(splits["fit"])
        or not any(
            e.get("retention_target") == "physical_distribution"
            for e in bank.episodes.values()
        )
    ):
        raise ValueError(
            "Paired training needs a full, unevicted bank with frozen protected distributions"
        )
    required_schedule = rule.get("required_success_schedule", [[]])
    if not required_schedule or any(
        not isinstance(conditions, list) for conditions in required_schedule
    ):
        raise ValueError(
            "A fixed supervised schedule must contain lists of verified conditions"
        )
    for conditions in required_schedule:
        bank.required_success_sources(conditions)
    aggregated = False
    if saved is not None:
        if saved["sources"] != sources or saved["rule"] != rule:
            mutable = ("data_files", "complete_sequences")
            old_rule = {k: v for k, v in saved["rule"].items() if k not in mutable}
            new_rule = {k: v for k, v in rule.items() if k not in mutable}
            if (
                not args.teacher_phase_transition
                or old_rule != new_rule
                or not all(item in sources for item in saved["sources"])
                or [s for s in sources if s["split"] == "check"]
                != [s for s in saved["sources"] if s["split"] == "check"]
            ):
                raise ValueError(
                    "A declared teacher phase may only append fit histories or change whole-sequence sampling; check split/objective stay fixed"
                )
            aggregated = True
        bank.rng.bit_generator.state = saved["sampling_rng"]
    updates = 0 if saved is None else saved["updates"]
    history = [] if saved is None else saved["history"]
    disagreement_interval = rule.get("disagreement_refresh_updates", 0)
    if (
        type(disagreement_interval) is not int
        or disagreement_interval < 0
        or disagreement_interval
        and not verified_sources
    ):
        raise ValueError(
            "An explicit disagreement interval requires verified mixed-source supervision"
        )
    assessments = []
    sampling_trace = []
    minimum_phase_end = updates + (rule["check_every"] if aggregated else 0)
    initialized = (
        paired_initial
        if retain
        else args.teacher_initialization if saved is None else saved["initialization"]
    )
    guard = rule.get("verified_output_guard")
    guard_events = [] if saved is None else saved.get("output_guard_events", [])[:]
    if guard is not None:
        from .optimizer import policy_output_parameter_names
        from .ranking import project_verified_output

        if (
            scope != "output"
            or not retain
            or type(guard["every_updates"]) is not int
            or guard["every_updates"] < 1
        ):
            raise ValueError(
                "An output ranking guard needs a declared output-only replay interval"
            )
        names = policy_output_parameter_names(policy)
        (weight,) = [
            k for k in names if k.startswith("action_net.") and k.endswith(".weight")
        ]
        bias = weight.removesuffix("weight") + "bias"
        reference = (
            source
            if saved is None
            else torch.load(
                ROOT / paired_initial, map_location="cpu", weights_only=False
            )
        )["state_dict"]
        columns = policy.deployment_action_indices or list(range(policy.action_space.n))
        guard_start = torch.cat(
            (reference[weight][columns], reference[bias][columns, None]), dim=1
        ).double()

    def metrics():
        return dict(
            fit=teacher_sequence_metrics(policy, splits["fit"]),
            check=teacher_sequence_metrics(policy, splits["check"]),
        )

    def passed(value):
        return all(
            value["fit"].get(k, {}).get("accuracy", 0) >= threshold
            for k, threshold in [
                ("all", 0.95),
                ("direction", 0.90),
                ("P1", 0.95),
                ("P2", 0.95),
                ("P1/direction", 0.90),
                ("P2/direction", 0.90),
            ]
        )

    def save_state(value, status):
        state = dict(
            updates=updates,
            initialization=initialized,
            sources=sources,
            rule=rule,
            trainable_scope=scope,
            sampling_rng=bank.rng.bit_generator.state,
            history=history,
        )
        extra = {} if guard is None else dict(output_guard_events=guard_events)
        state.update(extra)
        save(
            out / "last.pt",
            policy,
            definition,
            dict(
                algorithm=(
                    "verified_history_supervision"
                    if verified_sources
                    else "pure_teacher_supervision"
                ),
                teacher_state=state,
            ),
            policy.optimizer,
        )
        write(
            out / "result.json",
            dict(
                status=status,
                updates=updates,
                initialization=initialized,
                metrics=value,
                PPO_updates=0,
                value_updates=0,
                retention_updates=(
                    len(sampling_trace) if retention_coefficient > 0 else 0
                ),
                native_frames=0,
                trainable_scope=scope,
                total_loss=(
                    "categorical CE + weighted frozen physical-action KL"
                    if retain
                    else "only shared hard-action categorical cross entropy"
                ),
                source_trajectories=sources,
                actual_teacher_use=bank.retention_usage,
                seconds=time.monotonic() - started,
                history=history,
                disagreement_assessments=assessments,
                retention_coefficient=retention_coefficient,
                sampling_trace=sampling_trace,
                **extra,
            ),
        )

    started = time.monotonic()
    value = metrics()
    if not history or aggregated:
        history.append(
            dict(
                updates=updates,
                event="teacher_fit_data_aggregated" if aggregated else "initial",
                **value,
            )
        )
    save_state(value, "FIT_CHECK_RUNNING")
    print("TEACHER_FIT", initialized, updates, json.dumps(value), flush=True)
    while updates < (args.max_updates or rule["segment_updates"]) and (
        rule.get("paired_fixed_updates")
        or not passed(value)
        or updates < minimum_phase_end
    ):
        if disagreement_interval and (
            not assessments or updates % disagreement_interval == 0
        ):
            assessments.append(bank.assess_success_disagreements(policy, updates))
        policy.set_training_mode(True)
        policy.optimizer.zero_grad(set_to_none=True)
        loss, n, parts, selected = recurrent_success_loss(
            policy,
            bank,
            count=min(rule["batch_sequences"], len(bank.episodes)),
            length=length,
            gradient_context=rule["gradient_context"],
            use_soft_references=retain,
            retain_supervised_reference=retain,
            retention_coefficient=retention_coefficient,
            feature_gradients=feature_gradients,
            return_components=True,
            return_sources=True,
            priority_fraction=priority_fraction,
            required_conditions=required_schedule[updates % len(required_schedule)],
            disagreement_scope=(
                "verified_success" if disagreement_interval else "failed"
            ),
            disagreement_fraction=0.75 if disagreement_interval else 0.0,
        )
        if not torch.allclose(
            loss, parts["supervision"] + parts["retention"], atol=1e-7
        ):
            raise RuntimeError(
                "Offline objective differs from its logged CE and KL terms"
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite teaching objective")
        before_parameters = (
            {name: p.detach().clone() for name, p in policy.named_parameters()}
            if updates == 0
            else None
        )
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, rule["gradient_clip"]))
        if not np.isfinite(norm):
            raise RuntimeError("Nonfinite teacher gradient")
        policy_optimizer_step(policy, 1.0)
        updates += 1
        if guard is not None and (
            updates % guard["every_updates"] == 0
            or updates == (args.max_updates or rule["segment_updates"])
        ):
            policy.set_training_mode(False)
            proof = project_verified_output(
                policy,
                splits["fit"],
                len(splits["fit"]),
                guard,
                feasible_start=guard_start,
            )
            if not proof["applied"]:
                raise RuntimeError(
                    "The next mixed block could not preserve the declared legal-history rankings"
                )
            event = {
                k: v for k, v in proof.items() if k not in ("histories", "encoding")
            }
            guard_events.append(dict(update=updates, **event))
            print(
                "OUTPUT_RANK_GUARD",
                updates,
                event["installed_displacement_from_feasible"],
                flush=True,
            )
        if retain:
            sampling_trace.append(
                dict(
                    update=updates,
                    windows=[
                        [
                            r[k]
                            for k in (
                                "episode",
                                "first",
                                "last",
                                "effective_action_labels",
                                "additive_KL_labels",
                            )
                        ]
                        for r in selected
                    ],
                    CE=float(parts["supervision"].detach()),
                    weighted_KL=float(parts["retention"].detach()),
                    unweighted_KL=sum(r["unweighted_additive_KL"] for r in selected)
                    / len(selected),
                )
            )
        if updates == 1:
            before = float(loss.detach())
            after = []
            with torch.no_grad():
                for row in selected:
                    ep = bank.episodes[row["episode"]]
                    score = recurrent_window_logits(
                        policy,
                        ep,
                        row["first"],
                        row["last"],
                        rule["gradient_context"],
                        feature_gradients,
                    )[0]
                    target = torch.as_tensor(ep["actions"][row["first"] : row["last"]])
                    known = torch.tensor(
                        [
                            i not in ep.get("uncertain_action_decisions", ())
                            for i in range(row["first"], row["last"])
                        ]
                    )
                    per_action = torch.nn.functional.cross_entropy(
                        score / policy.action_temperature, target, reduction="none"
                    )
                    after.append((per_action * known).sum() / known.sum().clamp_min(1))
            after_loss = float(torch.stack(after).mean())
            changes = {
                prefix: max(
                    float((p.detach() - before_parameters[name]).abs().max())
                    for name, p in policy.named_parameters()
                    if name.startswith(prefix)
                )
                for prefix in ("features_extractor", "sequence_gru", "action_net")
            }
            write(
                out / "first_update.json",
                dict(
                    total_loss=before,
                    supervision=float(parts["supervision"].detach()),
                    retention=float(parts["retention"].detach()),
                    PPO=0.0,
                    value=0.0,
                    recomputed_same_histories_loss_after=after_loss,
                    mean_teacher_log_probability_before=-before,
                    mean_teacher_log_probability_after=-after_loss,
                    gradient_norm_before_clip=norm,
                    parameter_changes=changes,
                    optimizer_parameters=len(parameters),
                    selected=[
                        {k: v for k, v in row.items() if k != "loss"}
                        for row in selected
                    ],
                    scope="Same legal histories reconstructed under updated parameters; no stored hidden states or policy probability truncation.",
                ),
            )
        if updates % rule["check_every"] == 0 or updates == (
            args.max_updates or rule["segment_updates"]
        ):
            value = metrics()
            history.append(dict(updates=updates, **value))
            save_state(
                value, "FIT_THRESHOLD_REACHED" if passed(value) else "FIT_CHECK_RUNNING"
            )
            if verified_sources:
                save(
                    out / f"checkpoint_{updates}.pt",
                    policy,
                    definition,
                    dict(algorithm="verified_history_supervision", updates=updates),
                    None,
                )
            print("TEACHER_FIT", initialized, updates, json.dumps(value), flush=True)
    save_state(
        value,
        (
            "SUPERVISED_PAIR_ENDPOINT"
            if rule.get("paired_fixed_updates")
            else (
                "FIT_THRESHOLD_REACHED"
                if passed(value)
                else "SEGMENT_CHECKPOINT_NOT_COMPLETION"
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial",
        required=True,
        help="A trusted training checkpoint, with definition and state_dict",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON containing the teacher_fit rule and replay paths",
    )
    parser.add_argument("--out", required=True, help="Directory for this run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "readout", "encoder_readout", "hidden_readout", "output"],
        default="output",
    )
    parser.add_argument(
        "--teacher-initialization",
        choices=["F", "random"],
        default="F",
        help="F restores the supplied parent weights",
    )
    parser.add_argument("--teacher-phase-transition", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fit_teacher_sequences(args)


if __name__ == "__main__":
    main()
