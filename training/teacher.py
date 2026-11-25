import numpy as np
import torch
from torch.nn import functional as F


def recurrent_window_logits(
    policy, episode, first, last, gradient_context=0, feature_gradients=True
):
    """One current-weight legal-prefix reconstruction for auxiliary windows."""
    from .replay import episode_values

    gradient_start = max(0, first - int(gradient_context))
    hidden = None
    with torch.no_grad():
        for start in range(0, gradient_start, 32):
            _, hidden = episode_values(
                policy, episode, start, min(start + 32, gradient_start), hidden
            )
    logits, _ = episode_values(
        policy,
        episode,
        gradient_start,
        last,
        hidden,
        detach_features=not feature_gradients,
    )
    return logits[:, first - gradient_start :]


def teacher_sequence_metrics(policy, episodes):
    """Complete legal-history action fit; never autonomous tactical success.

    Sources are split by whole games before calling this function. Reconstruct
    all real prefix actions, including an unlabelled student-controlled prefix.
    Categories describe requested inputs, not native moves or useful walking.
    """
    from .replay import full_values
    from collections import defaultdict

    groups = defaultdict(lambda: dict(samples=0, correct=0, loss_sum=0.0))

    def add(key, loss, correct):
        item = groups[key]
        item["samples"] += 1
        item["correct"] += int(correct)
        item["loss_sum"] += float(loss)

    was_training = policy.training
    policy.set_training_mode(False)
    try:
        for episode in episodes:
            first = int(episode.get("supervision_start", 0))
            last = int(episode.get("supervision_end", len(episode["actions"])))
            logits = (
                full_values(policy, episode)[first:last].double()
                / policy.action_temperature
            )
            target = torch.as_tensor(episode["actions"][first:last])
            prediction = logits.argmax(1)
            loss = F.cross_entropy(logits, target, reduction="none")
            names = episode["action_names"]
            for i, (action, value, correct) in enumerate(
                zip(target, loss, prediction == target)
            ):
                name = names[int(action)]
                at = first + i
                if at in episode.get("uncertain_action_decisions", ()):
                    continue
                if name in ("LEFT", "RIGHT"):
                    forward = "RIGHT" if episode["positions"][at, 4] >= 0 else "LEFT"
                    category = "approach" if name == forward else "retreat"
                    add("direction", value, correct)
                    add(f"P{episode['player']}/direction", value, correct)
                elif name in ("DOWN_LEFT", "DOWN_RIGHT"):
                    category = "guard"
                elif name == "NO_OP":
                    category = "neutral"
                else:
                    category = "attack"
                add("all", value, correct)
                add(category, value, correct)
                add(f"P{episode['player']}", value, correct)
        return {
            key: dict(
                samples=x["samples"],
                correct=x["correct"],
                accuracy=x["correct"] / x["samples"],
                cross_entropy=x["loss_sum"] / x["samples"],
            )
            for key, x in groups.items()
        }
    finally:
        policy.set_training_mode(was_training)


def conditional_preference_term(log_probabilities, pair):
    """Rank tested choices without requiring a unique winning action."""
    good = pair.get("better_actions", [pair.get("better")])
    bad = pair.get("worse_actions", [pair.get("worse")])
    if (
        not good
        or not bad
        or any(
            type(i) is not int or not 0 <= i < len(log_probabilities)
            for i in good + bad
        )
        or set(good) & set(bad)
        or not 0 < float(pair["weight"]) <= 2
    ):
        raise ValueError(
            "A preference needs distinct tested physical choices and a bounded weight"
        )
    difference = log_probabilities[good].max() - log_probabilities[bad].max()
    objective = pair.get("objective", "relative_change")
    if objective == "relative_change":
        difference = difference - float(pair["reference_log_odds"])
    elif objective != "absolute_rank":
        raise ValueError("Unknown conditional preference objective")
    # Summing valid-action mass could still leave a known bad greedy winner.
    # Max ranks a valid winner while leaving ties within the valid set free.
    return float(pair["weight"]) * F.softplus(
        float(pair.get("log_odds_margin", 1.0)) - difference
    )


def verified_reference_sets(episode, first, last, action_count):
    """One validator for physical, history/version-bound auxiliary sets."""
    result = []
    seen = set()
    for permission in episode.get("retention_allowed_actions", []):
        at = permission["decision"]
        choices = permission["actions"]
        if (
            type(at) is not int
            or at in seen
            or not 0 <= at < len(episode["actions"])
            or permission.get("continuation")
            != episode.get("reference_policy_version", episode.get("policy_version"))
            or len(choices) < 2
            or len(set(choices)) != len(choices)
            or any(type(a) is not int or not 0 <= a < action_count for a in choices)
        ):
            raise ValueError(
                "Pooled retention requires distinct, version-bound physical choices on actual history"
            )
        seen.add(at)
        if int(np.asarray(episode["policy_probabilities"])[at].argmax()) not in choices:
            raise ValueError(
                "Pooled retention must include the frozen reference winner"
            )
        if first <= at < last:
            result.append((at - first, choices))
    return result


def conditional_option_term(own_logs, own_actions, other_logs, other_actions, option):
    """Relative finite-program likelihood, not a value-confidence estimate."""
    chosen = own_logs.gather(1, own_actions[:, None]).mean()
    other = other_logs.gather(1, other_actions[:, None]).mean()
    return float(option["weight"]) * F.softplus(
        float(option.get("log_odds_margin", 1.0))
        - (chosen - other - float(option.get("reference_log_odds", 0.0)))
    )


def recurrent_success_loss(
    policy,
    replay,
    count=4,
    length=16,
    disagreement_fraction=0.0,
    disagreement_scope="failed",
    error_balance=0.0,
    priority_fraction=0.0,
    gradient_context=0,
    use_soft_references=False,
    hard_failed_conditions=False,
    disagreement_selection="random",
    feature_gradients=True,
    required_conditions=(),
    supervise_verified_context=False,
    return_components=False,
    retention_coefficient=1.0,
    return_sources=False,
    retain_supervised_reference=False,
):
    """Finite action supervision on verified complete self-successes.

    Reconstruct the entire actual prefix with current visual/GRU weights.
    The selected suffix supplies labels. An explicitly configured preceding
    context can carry gradients through its actual recurrent inputs. A reserved
    verified hard demonstration can also label that already computed context,
    within its verified bounds; soft old references keep their original scope.
    There are no replay returns,
    critic targets, hidden-state labels or PPO advantage samples here.
    """
    if not getattr(policy, "probability_policy", False) or not policy.physical_actions:
        raise ValueError(
            "Sequence success supervision requires physical categorical actions"
        )
    if not 0 <= error_balance <= 1:
        raise ValueError("Invalid supervised error balance")
    if not np.isfinite(retention_coefficient) or not 0 <= retention_coefficient <= 1:
        raise ValueError("Declare a finite behavior-retention coefficient in [0,1]")
    if int(gradient_context) != gradient_context or gradient_context < 0:
        raise ValueError("Invalid gradient context")
    if not isinstance(feature_gradients, bool):
        raise ValueError("Auxiliary feature gradients require an explicit boolean")
    if not isinstance(supervise_verified_context, bool):
        raise ValueError("Verified context supervision requires an explicit boolean")
    if return_sources and not return_components:
        raise ValueError("Source attribution requires loss components")
    if retain_supervised_reference and (
        not use_soft_references or hard_failed_conditions or error_balance
    ):
        raise ValueError(
            "Additive behavior retention requires fixed soft references beside executed-action supervision"
        )
    from .replay import same_episode_condition

    losses = []
    samples = 0
    components = {"supervision": [], "retention": []}
    sources = []
    for episode, first, last in replay.success_sequences(
        count,
        length,
        disagreement_fraction,
        disagreement_scope,
        priority_fraction,
        disagreement_selection,
        required_conditions,
    ):
        reference_soft = (
            use_soft_references
            and episode.get("retention_target") == "physical_distribution"
        )
        repair = (
            reference_soft
            and hard_failed_conditions
            and episode.get("current_complete_useful") is False
        )
        joint = reference_soft and retain_supervised_reference
        soft = reference_soft and not (repair or joint)
        if episode.get("retention_allowed_actions") and not soft:
            raise ValueError("Verified physical sets require a soft frozen reference")
        train_features = feature_gradients
        gradient_start = max(0, first - int(gradient_context))
        label_first = first
        if (
            supervise_verified_context
            and not soft
            and any(
                rule.get("supervise_context", True)
                and same_episode_condition(episode, rule)
                and episode.get("policy_version") == rule["policy_version"]
                for rule in required_conditions
            )
        ):
            # Reuse the already computed recurrent gradient context. Never
            # label an unverified prefix or sharpen a soft old reference.
            label_first = max(gradient_start, int(episode.get("supervision_start", 0)))
        logits = recurrent_window_logits(
            policy,
            episode,
            label_first,
            last,
            label_first - gradient_start,
            train_features,
        )
        action = torch.as_tensor(
            episode["actions"][label_first:last], device=logits.device
        )
        use = replay.retention_usage[episode["auxiliary_registry_id"]]
        extra = first - label_first
        if extra:
            use["auxiliary_transitions"] += extra
            use["phase_auxiliary_transitions"] = (
                use.get("phase_auxiliary_transitions", 0) + extra
            )
            use["context_supervised_action_labels"] = (
                use.get("context_supervised_action_labels", 0) + extra
            )
            replay.auxiliary_sampling_totals[episode["group"]]["transitions"] += extra
            errors = np.asarray(episode.get("supervised_disagreements", []))
            use["covered_disagreements"] = use.get("covered_disagreements", 0) + int(
                ((errors >= label_first) & (errors < first)).sum()
            )
            windows = use.setdefault("context_supervised_windows", {})
            key = f"{label_first}:{first}"
            windows[key] = windows.get(key, 0) + 1
        context = first - gradient_start
        use["reencoded_context_transitions"] = (
            use.get("reencoded_context_transitions", 0) + context
        )
        use["gradient_context_transitions"] = use.get(
            "gradient_context_transitions", 0
        ) + (context if train_features else 0)
        known = torch.ones(len(action), dtype=torch.bool, device=logits.device)
        if soft or "uncertain_action_decisions" in episode:
            uncertain = episode.get(
                (
                    "retention_excluded_decisions"
                    if soft
                    else "uncertain_action_decisions"
                ),
                (),
            )
            if any(
                type(at) is not int or not 0 <= at < len(episode["actions"])
                for at in uncertain
            ):
                raise ValueError("Uncertain labels must refer to actual decisions")
            for at in uncertain:
                if label_first <= at < last:
                    known[at - label_first] = False
            counter = (
                "retention_excluded_transitions"
                if soft
                else "uncertain_action_transitions"
            )
            use[counter] = use.get(counter, 0) + int((~known).sum())
        use["unmasked_supervision_transitions"] = use.get(
            "unmasked_supervision_transitions", 0
        ) + int(known.sum())
        actual_bad = int(((logits[0].argmax(1) != action) & known).sum())
        assessed_errors = episode.get("supervised_disagreements", [])
        if len(assessed_errors) and label_first <= int(assessed_errors[0]) < last:
            use["phase_assessed_frontier_windows"] = (
                use.get("phase_assessed_frontier_windows", 0) + 1
            )
        use["actual_disagreements_trained"] = (
            use.get("actual_disagreements_trained", 0) + actual_bad
        )
        use["phase_actual_disagreements_trained"] = (
            use.get("phase_actual_disagreements_trained", 0) + actual_bad
        )
        if reference_soft and "policy_probabilities" not in episode:
            raise ValueError("Missing physical reference distribution")
        retention_loss = None
        retention_labels = 0
        if soft or joint:
            if "policy_probabilities" not in episode:
                raise ValueError(
                    "A soft reference must contain actual frozen physical probabilities"
                )
            target = torch.as_tensor(
                episode["policy_probabilities"][label_first:last],
                device=logits.device,
                dtype=logits.dtype,
            )
            if (
                target.shape != logits[0].shape
                or not torch.isfinite(target).all()
                or (target < 0).any()
                or (target.sum(1) <= 0).any()
            ):
                raise ValueError("Invalid physical probability reference")
            target = target / target.sum(1, keepdim=True)
            temperature = float(
                episode.get(
                    "retention_temperature",
                    episode.get("reference_temperature", policy.action_temperature),
                )
            )
            if (
                not np.isfinite(temperature)
                or temperature <= 0
                or (
                    "retention_temperature" in episode
                    and episode.get("reference_temperature") != temperature
                )
            ):
                raise ValueError(
                    "Retention temperature must match its declared probability reference"
                )
            if episode.get("retention_role") == "qualified_function_reference":
                use["reference_argmax_disagreements_trained"] = use.get(
                    "reference_argmax_disagreements_trained", 0
                ) + int(
                    ((logits[0].detach().argmax(1) != target.argmax(1)) & known).sum()
                )
            mode = episode.get("retention_mode", "distribution")
            if joint and (
                mode != "distribution"
                or episode.get("conditional_preferences")
                or episode.get("conditional_option")
                or episode.get("reference_role") == "skill_demonstration"
            ):
                raise ValueError(
                    "Additive KL is defined only on protected executed-action histories"
                )
            if episode.get("retention_allowed_actions") and mode != "distribution":
                raise ValueError(
                    "Verified physical sets require distribution retention"
                )
            if mode == "distribution":
                per_reference = retention_kl(
                    logits[0] / temperature, target, reduction="none"
                )
                for index, choices in verified_reference_sets(
                    episode, label_first, last, target.shape[1]
                ):
                    mask = torch.zeros_like(target[index : index + 1], dtype=torch.bool)
                    mask[:, choices] = True
                    per_reference = per_reference.clone()
                    per_reference[index] = retention_kl(
                        logits[0, index : index + 1] / temperature,
                        target[index : index + 1],
                        mask,
                    )
                    use["verified_pooled_reference_transitions"] = use.get(
                        "verified_pooled_reference_transitions", 0
                    ) + int(known[index])
            elif mode == "greedy_margin":
                # A finite behavior constraint, not a return-confidence score:
                # keep the source winner above alternatives on these verified
                # actual histories without reproducing its entire posterior.
                margins = torch.as_tensor(
                    episode["reference_logit_margin"][label_first:last],
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if (
                    margins.shape != (len(action),)
                    or not torch.isfinite(margins).all()
                    or (margins < 0).any()
                    or (margins > 1).any()
                ):
                    raise ValueError(
                        "A greedy reference needs bounded source logit margins"
                    )
                chosen = target.argmax(1)
                own = logits[0].gather(1, chosen[:, None]).squeeze(1)
                other = (
                    logits[0]
                    .masked_fill(F.one_hot(chosen, logits.shape[-1]).bool(), -torch.inf)
                    .max(1)
                    .values
                )
                per_reference = (
                    F.relu(margins - own + other) / policy.action_temperature
                )
                use["greedy_margin_reference_transitions"] = use.get(
                    "greedy_margin_reference_transitions", 0
                ) + int(known.sum())
            else:
                raise ValueError("Unknown behavior-retention mode")
            reference_known = known
            if joint:
                reference_known = known.clone()
                for at in episode.get("retention_excluded_decisions", ()):
                    if type(at) is not int or not 0 <= at < len(episode["actions"]):
                        raise ValueError("Protection exclusions must be real decisions")
                    if label_first <= at < last:
                        reference_known[at - label_first] = False
            retention_labels = int(reference_known.sum())
            retention_loss = (
                per_reference * reference_known
            ).sum() / reference_known.sum().clamp_min(1)
            loss = retention_loss
        if not soft:
            if bool(known.all()):
                loss = F.cross_entropy(logits[0] / policy.action_temperature, action)
            else:
                per_action = F.cross_entropy(
                    logits[0] / policy.action_temperature, action, reduction="none"
                )
                loss = (per_action * known).sum() / known.sum().clamp_min(1)
        preferences = episode.get("conditional_preferences", [])
        option = episode.get("conditional_option")
        comparison_coefficient = float(episode.get("comparison_coefficient", 1.0))
        if (
            not np.isfinite(comparison_coefficient)
            or not 0 <= comparison_coefficient <= 1
        ):
            raise ValueError("Declare a finite comparison coefficient in [0,1]")
        if option:
            tail = episode.get("conditional_tail_reference", False)
            if tail and (
                not soft
                or episode.get("reference_role") != "skill_demonstration"
                or episode.get("target_composition", {}).get("executed_exchange")
                != [option["first"], option["last"]]
            ):
                raise ValueError(
                    "A comparative tail needs its verified finite exchange and actual-history soft continuation"
                )
            if not tail and (
                soft or label_first != option["first"] or last != option["last"]
            ):
                raise ValueError(
                    "A continuation preference requires its complete actual prefix window"
                )
            # A later sampled tail still learns the verified controller on its
            # actual new history. Rank only a complete sampled option, never
            # a truncated first input or a window on the other branch's state.
            if label_first <= option["first"] and option["last"] <= last:
                first_option, last_option = option["first"], option["last"]
                other = {
                    key: episode["comparison_" + key]
                    for key in ("frames", "positions", "previous", "actions")
                }
                if "comparison_input_durations" in episode:
                    other["input_durations"] = episode["comparison_input_durations"]
                alternative = recurrent_window_logits(
                    policy,
                    other,
                    first_option,
                    last_option,
                    gradient_context,
                    train_features,
                )[0]
                other_actions = torch.as_tensor(
                    other["actions"][first_option:last_option],
                    device=alternative.device,
                )
                own = logits[0, first_option - label_first : last_option - label_first]
                own_actions = action[
                    first_option - label_first : last_option - label_first
                ]
                loss = float(
                    episode.get("continuation_supervision_weight", 0.25)
                ) * loss + comparison_coefficient * conditional_option_term(
                    F.log_softmax(own / policy.action_temperature, 1),
                    own_actions,
                    F.log_softmax(alternative / policy.action_temperature, 1),
                    other_actions,
                    option,
                )
                use["conditional_option_pairs"] = (
                    use.get("conditional_option_pairs", 0) + 1
                )
                use["active_conditional_option_pairs"] = use.get(
                    "active_conditional_option_pairs", 0
                ) + int(comparison_coefficient > 0 and option["weight"] > 0)
        elif preferences:
            tail = episode.get("conditional_tail_reference", False)
            exchange = episode.get("target_composition", {}).get("executed_exchange")
            if soft and (
                not tail
                or episode.get("reference_role") != "skill_demonstration"
                or not isinstance(exchange, list)
                or len(exchange) != 2
                or not 0 <= exchange[0] < exchange[1] <= len(episode["actions"])
                or any(
                    not exchange[0] <= p["decision"] < exchange[1] for p in preferences
                )
            ):
                raise ValueError(
                    "Matched comparisons need a verified finite exchange, not an old policy posterior"
                )
            logs = F.log_softmax(logits[0] / policy.action_temperature, 1)
            terms = []
            for pair in preferences:
                at = int(pair["decision"]) - label_first
                if not 0 <= at < len(action):
                    continue
                terms.append(conditional_preference_term(logs[at], pair))
            if not terms and not (soft and tail):
                raise ValueError(
                    "Comparison was sampled outside its verified root window"
                )
            if terms:
                loss = (
                    float(episode.get("continuation_supervision_weight", 0.25)) * loss
                    + comparison_coefficient * torch.stack(terms).mean()
                )
            use["conditional_preference_pairs"] = use.get(
                "conditional_preference_pairs", 0
            ) + len(terms)
            use["active_conditional_preference_pairs"] = use.get(
                "active_conditional_preference_pairs", 0
            ) + len(terms) * int(comparison_coefficient > 0)
        skill_demo = soft and episode.get("reference_role") == "skill_demonstration"
        kind = (
            "soft_demonstration"
            if skill_demo
            else (
                "soft_reference"
                if soft
                else (
                    "conditional_preference" if preferences or option else "hard_action"
                )
            )
        )
        if repair:
            use["outcome_gated_hard_transitions"] = use.get(
                "outcome_gated_hard_transitions", 0
            ) + len(action)
        use[kind + "_transitions"] = use.get(kind + "_transitions", 0) + len(action)
        use[kind + "_loss_weighted_sum"] = use.get(
            kind + "_loss_weighted_sum", 0.0
        ) + float(loss.detach()) * len(action)
        if error_balance and not soft:
            wrong = (logits[0].detach().argmax(1) != action) & known
            if wrong.any():
                loss = (1 - error_balance) * loss + error_balance * F.cross_entropy(
                    logits[0, wrong] / policy.action_temperature, action[wrong]
                )
            use["error_balanced_samples"] = use.get("error_balanced_samples", 0) + int(
                wrong.sum()
            )
        losses.append(loss + retention_loss if joint else loss)
        component = "retention" if soft and not skill_demo else "supervision"
        components[component].append(loss)
        if joint:
            components["retention"].append(retention_loss)
            use["additive_KL_labels"] = (
                use.get("additive_KL_labels", 0) + retention_labels
            )
            use["additive_KL_loss_sum"] = use.get("additive_KL_loss_sum", 0.0) + float(
                retention_loss.detach()
            )
        if return_sources:
            sources.append(
                dict(
                    episode=episode["auxiliary_registry_id"],
                    group=episode.get("group"),
                    reference=episode.get(
                        "reference_policy_version", episode.get("policy_version")
                    ),
                    role=episode.get("data_role"),
                    component=component,
                    first=label_first,
                    last=last,
                    loss=loss,
                    effective_action_labels=int(known.sum()),
                    additive_KL_labels=retention_labels if joint else 0,
                    unweighted_additive_KL=(
                        float(retention_loss.detach()) if joint else 0.0
                    ),
                )
            )
        samples += len(action)
    total = torch.stack(losses).mean()
    if retention_coefficient != 1:
        # Preserve the identical sampled histories, label exposure and batch
        # denominator for a loss-component diagnostic. Zero is not a claim
        # of a safe deployable learning rule.
        total = (
            sum(components["supervision"], total * 0)
            + retention_coefficient * sum(components["retention"], total * 0)
        ) / len(losses)
    if return_components:
        parts = {
            name: sum(values, total * 0) / len(losses)
            for name, values in components.items()
        }
        if retention_coefficient != 1:
            parts["retention"] = parts["retention"] * retention_coefficient
        if return_sources:
            return (
                total,
                samples,
                parts,
                [
                    dict(
                        row,
                        loss=row["loss"]
                        / len(losses)
                        * (
                            retention_coefficient
                            if row["component"] == "retention"
                            else 1.0
                        ),
                    )
                    for row in sources
                ],
            )
        return total, samples, parts
    return total, samples


def retention_kl(logits, target, allowed=None, *, reduction="mean"):
    """Physical KL, optionally pooling only a verified outcome-equivalent set.

    Pooling removes preference *within* that set, without rewarding its use
    or granting permission to any other action. A singleton is ordinary KL.
    """
    logp = F.log_softmax(logits, 1)
    terms = target * (target.clamp_min(1e-12).log() - logp)
    if reduction not in ("mean", "none"):
        raise ValueError("Unknown physical retention reduction")
    if allowed is None:
        values = terms.sum(1)
        return values if reduction == "none" else values.mean()
    mass = (target * allowed).sum(1)
    pooled = mass * (
        mass.clamp_min(1e-12).log()
        - logp.masked_fill(~allowed, -torch.inf).logsumexp(1)
    )
    values = terms.masked_fill(allowed, 0).sum(1) + pooled
    return values if reduction == "none" else values.mean()
