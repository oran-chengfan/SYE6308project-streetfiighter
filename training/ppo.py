"""Recurrent PPO rollout geometry, sequence evaluation and clipped objective."""

import torch


def recurrent_ppo_configuration(shared):
    """Course controls and research use this same recurrent update geometry."""
    shared = shared or {}
    settings = shared.get("ppo", {})
    defaults = dict(
        n_envs=4,
        n_steps=512,
        sequence_length=32,
        batch_sequences=shared.get("learner", {}).get("ppo_batch_sequences", 8),
        n_epochs=4,
        clip_range=0.1,
        target_kl=0.04,
    )
    result = {key: settings.get(key, value) for key, value in defaults.items()}
    for key in ("n_envs", "n_steps", "sequence_length", "batch_sequences", "n_epochs"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Recurrent PPO dimensions must be positive integers")
    if (
        result["n_steps"] % result["sequence_length"]
        or result["batch_sequences"]
        > result["n_envs"] * (result["n_steps"] // result["sequence_length"])
        or not 0 < result["clip_range"] < 1
        or not 0 < result["target_kl"] < 1
    ):
        raise ValueError("Invalid recurrent PPO rollout, clipping or KL settings")
    return result


def recurrent_step(
    policy,
    features,
    hidden,
    previous,
    starts,
    positions,
    remaining,
    health,
    left,
    elapsed,
    durations,
):
    """One training/inference recurrent API, with explicit legal input history."""
    if getattr(policy, "probability_policy", False):
        return policy.sequence_step(
            features,
            hidden,
            previous,
            starts,
            positions[:, 4] < 0,
            input_durations=durations,
        )
    return policy.sequence_step(
        features,
        hidden,
        previous,
        starts,
        positions[:, 4] < 0,
        remaining,
        health,
        positions,
        left,
        elapsed,
    )


def evaluate_recurrent_batch(
    policy,
    images,
    positions,
    hidden,
    previous,
    starts,
    actions,
    input_durations,
    remaining=None,
    durations=None,
    controller_left=None,
    elapsed=None,
    return_history=False,
):
    """The shared contiguous PPO forward pass, also used for update diagnosis."""
    length, batch = actions.shape
    sequence_ppo = getattr(policy, "probability_policy", False)
    observation = dict(image=images.flatten(0, 1), position=positions.flatten(0, 1))
    features = policy.extract_features(observation).reshape(length, batch, -1)
    health = (
        None if sequence_ppo else policy.read_health(dict(image=observation["image"]))
    )
    if health is not None:
        health = health.reshape(length, batch, 2)
    if (
        sequence_ppo
        and return_history
        and not torch.is_grad_enabled()
        and hasattr(policy, "sequence_values")
    ):
        # Reconstruct current histories with the existing sequence GRU. Split
        # only where a real stream resets; no cached old-parameter state and
        # no per-decision Python replay of an otherwise contiguous sequence.
        boundaries = (
            [0] + (torch.nonzero(starts[1:].any(1)).flatten() + 1).tolist() + [length]
        )
        logs = []
        values = []
        entropies = []
        history = []
        for first, last in zip(boundaries, boundaries[1:]):
            before = hidden
            hidden = hidden * (~starts[first, :, None]).float()
            actual_previous = torch.where(
                starts[first:last],
                torch.zeros_like(previous[first:last]),
                previous[first:last],
            )
            actual_durations = torch.where(
                starts[first:last, :, None],
                torch.zeros_like(input_durations[first:last]),
                input_durations[first:last],
            )
            score, next_hidden, combined = policy.sequence_values(
                features[first:last, :, :-6].transpose(0, 1),
                positions[first:last].transpose(0, 1),
                actual_previous.transpose(0, 1),
                hidden[None],
                input_durations=actual_durations.transpose(0, 1),
                return_features=True,
            )
            score = score.transpose(0, 1)
            flip = (positions[first:last, :, 4] < 0) & (not policy.physical_actions)
            logits = torch.where(
                flip[:, :, None], score[:, :, policy.physical_mirror], score
            )
            critic = (
                score.mean(2)
                if policy.deployment_action_indices is None
                else policy.last_internal_critic.transpose(0, 1)
            )
            distribution = torch.distributions.Categorical(
                logits=logits / policy.action_temperature
            )
            logs.append(distribution.log_prob(actions[first:last]))
            values.append(critic)
            entropies.append(distribution.entropy())
            recurrent = combined[:, :, -policy.memory_hidden :].transpose(0, 1)
            history.append(torch.cat((before[None], recurrent[:-1]), 0))
            hidden = next_hidden[0]
        values = torch.cat(values)
        return (
            torch.cat(logs),
            values,
            torch.cat(entropies),
            values * 0,
            None,
            (torch.cat(history), hidden),
        )
    logs = []
    values = []
    entropies = []
    families = []
    history = []
    for offset in range(length):
        if return_history:
            history.append(hidden)
        flip = positions[offset, :, 4] < 0
        distribution, value, hidden = recurrent_step(
            policy,
            features[offset],
            hidden,
            previous[offset],
            starts[offset],
            positions[offset],
            None if remaining is None else remaining[offset],
            None if health is None else health[offset],
            None if controller_left is None else controller_left[offset],
            None if elapsed is None else elapsed[offset],
            input_durations[offset],
        )
        duration_distribution = (
            None
            if sequence_ppo
            else policy.duration_distribution(hidden, actions[offset], flip)
        )
        logs.append(
            distribution.log_prob(actions[offset])
            + (0 if sequence_ppo else duration_distribution.log_prob(durations[offset]))
        )
        values.append(value)
        entropies.append(
            distribution.entropy()
            if sequence_ppo
            else policy.joint_entropy(distribution, hidden, flip)
        )
        families.append(
            value * 0 if sequence_ppo else policy.family_entropy(distribution)
        )
    result = (
        torch.stack(logs),
        torch.stack(values),
        torch.stack(entropies),
        torch.stack(families),
        health,
    )
    return (*result, (torch.stack(history), hidden)) if return_history else result


def ppo_objective_terms(
    new_log,
    values,
    entropies,
    family_entropies,
    old_log,
    advantage,
    mask,
    returns,
    entropy_coefficient,
    family_entropy_coefficient=0.0,
    health_predictions=None,
    health_labels=None,
    clip_range=0.1,
):
    """One PPO objective for training and fixed-batch update attribution.

    The advantage is already normalized on the current rollout mask. This
    function accepts no replay rewards or demonstration advantages.
    """
    change = (new_log - old_log)[mask]
    ratio = change.exp()
    actor = -torch.minimum(
        advantage * ratio, advantage * ratio.clamp(1 - clip_range, 1 + clip_range)
    ).mean()
    value = (returns - values).square().mean()
    entropy = entropies[mask].mean()
    family = family_entropies[mask].mean()
    health = (
        actor.new_zeros(())
        if health_predictions is None
        else (health_predictions - health_labels).square().mean()
    )
    return dict(
        actor=actor,
        value=value,
        entropy=entropy,
        family_entropy=family,
        health=health,
        joint=actor
        + 0.5 * value
        - entropy_coefficient * entropy
        - family_entropy_coefficient * family
        + 10.0 * health,
        kl=((ratio - 1) - change).mean().detach(),
    )
