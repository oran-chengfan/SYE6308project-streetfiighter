import time
import numpy as np
import torch
from .replay import episode_values


def readout_step_weights(episode, first, last):
    """One trajectory's mass, retaining a declared tactical window's emphasis.

    Additional follow-through labels are the same physically verified game,
    never another independent success or an on-policy advantage sample.
    """
    weights = torch.full((last - first,), 1.0 / (last - first))
    window = episode.get("readout_priority_window")
    if window is not None:
        start, end, fraction = window["first"], window["last"], window["fraction"]
        if not first <= start < end <= last or not 0 < fraction < 1:
            raise ValueError(
                "Priority window must be inside its complete label history"
            )
        weights *= 1 - fraction
        weights[start - first : end - first] += fraction / (end - start)
    return weights


def readout_action_scores(policy, scores):
    """Select acting columns after full internal dueling/mirror arithmetic."""
    indices = policy.deployment_action_indices
    return scores if indices is None else scores[..., indices]


def readout_action_labels(policy, episode, first, last):
    """Canonical labels for recorded physical inputs, never soft-reference modes."""
    actions = torch.as_tensor(episode["actions"][first:last]).long()
    if policy.facing_canonical:
        actions = torch.where(
            torch.as_tensor(episode["positions"][first:last, 4] < 0),
            policy.physical_mirror[actions],
            actions,
        )
    return actions


def readout_data(
    policy,
    episodes,
    verification=None,
    trajectory_masses=None,
    use_soft_references=False,
    encoding_sequence_length=32,
):
    """Current legal raw histories, not cached teacher recurrent states."""
    if (
        type(encoding_sequence_length) is not int
        or not 1 <= encoding_sequence_length <= 32
    ):
        raise ValueError(
            "Readout encoding needs a declared sequence length from one to32"
        )
    features = []
    targets = []
    weights = []
    slices = []
    offset = 0
    reconstruction_error = 0.0
    logit_scale = 1.0
    probability_error = 0.0
    greedy_changes = 0
    full_error = 0.0
    head_probability_error = 0.0
    temperature = (
        float(policy.action_temperature)
        if getattr(policy, "probability_policy", False)
        else 1.0
    )
    with torch.no_grad():
        for index, episode in enumerate(episodes):
            parts = []
            hidden = None
            n = len(episode["actions"])
            for first in range(0, n, encoding_sequence_length):
                last = min(first + encoding_sequence_length, n)
                scores, hidden, combined = episode_values(
                    policy, episode, first, last, hidden, return_features=True
                )
                head = policy.action_net(combined)
                direct = policy.value_net(combined) + head - head.mean(2, keepdim=True)
                if policy.facing_canonical:
                    flip = torch.as_tensor(episode["positions"][first:last, 4] < 0)[
                        None, :, None
                    ]
                    mirror = (
                        policy.physical_mirror
                        if policy.deployment_action_indices is None
                        else policy.internal_action_mirror
                    )
                    direct = torch.where(flip, direct[:, :, mirror], direct)
                    head = torch.where(flip, head[:, :, mirror], head)
                direct = readout_action_scores(policy, direct)
                head = readout_action_scores(policy, head)
                reconstruction_error = max(
                    reconstruction_error,
                    float(
                        (
                            (scores - scores.mean(2, keepdim=True))
                            - (head - head.mean(2, keepdim=True))
                        )
                        .abs()
                        .max()
                    ),
                )
                logit_scale = max(
                    logit_scale, float(scores.abs().max()), float(direct.abs().max())
                )
                full_error = max(full_error, float((scores - direct).abs().max()))
                probability_error = max(
                    probability_error,
                    float(
                        (
                            (scores / temperature).softmax(2)
                            - (direct / temperature).softmax(2)
                        )
                        .abs()
                        .max()
                    ),
                )
                head_probability_error = max(
                    head_probability_error,
                    float(
                        (
                            (scores / temperature).softmax(2)
                            - (head / temperature).softmax(2)
                        )
                        .abs()
                        .max()
                    ),
                )
                greedy_changes += int((scores.argmax(2) != head.argmax(2)).sum())
                parts.append(combined[0])
            minimum = episode.get("supervision_start", 0)
            maximum = episode.get("supervision_end", n)
            if not 0 <= minimum < maximum <= n:
                raise ValueError(
                    "Readout supervision needs a nonempty continuous label window"
                )
            x = torch.cat(parts)[minimum:maximum]
            y = readout_action_labels(policy, episode, minimum, maximum)
            if use_soft_references:
                if episode.get("retention_target") == "physical_distribution":
                    y = torch.as_tensor(
                        episode["policy_probabilities"][minimum:maximum],
                        dtype=torch.float32,
                    )
                    if (
                        y.shape != (len(x), len(policy.physical_mirror))
                        or not torch.isfinite(y).all()
                        or (y < 0).any()
                        or (y.sum(1) <= 0).any()
                    ):
                        raise ValueError("Invalid physical soft readout references")
                    y = y / y.sum(1, keepdim=True)
                    if policy.facing_canonical:
                        y = torch.where(
                            torch.as_tensor(
                                episode["positions"][minimum:maximum, 4] < 0
                            )[:, None],
                            y[:, policy.physical_mirror],
                            y,
                        )
                else:
                    y = torch.nn.functional.one_hot(
                        y, len(policy.physical_mirror)
                    ).float()
            mass = (
                1.0 / len(episodes)
                if trajectory_masses is None
                else trajectory_masses[index]
            )
            features.append(x)
            targets.append(y)
            weights.append(
                torch.full((len(y),), mass / len(y))
                if "readout_priority_window" not in episode
                else mass * readout_step_weights(episode, minimum, maximum)
            )
            slices.append(
                dict(
                    first=offset,
                    last=offset + len(y),
                    opponent=episode["opponent_config"],
                    player=episode["player"],
                    offset=episode["offset"],
                    supervision_start=minimum,
                    supervision_end=maximum,
                    source=episode["policy_version"],
                )
            )
            offset += len(y)
    tolerance = 8 * torch.finfo(torch.float32).eps * logit_scale
    check = dict(
        centered_logit_absolute_error=reconstruction_error,
        logit_scale=logit_scale,
        encoding_sequence_length=encoding_sequence_length,
        scale_aware_roundoff_tolerance=tolerance,
        maximum_probability_error=probability_error,
        full_dueling_reconstruction_error=full_error,
        head_only_probability_error=head_probability_error,
        probability_temperature=temperature,
        greedy_action_changes=greedy_changes,
    )
    if verification is not None:
        verification.update(check)
    if (
        reconstruction_error > tolerance
        or full_error > tolerance
        or probability_error > 1e-6
        or greedy_changes
    ):
        raise RuntimeError(
            f"Readout reconstruction changes the physical policy: {check}"
        )
    return (
        torch.cat(features).double(),
        torch.cat(targets),
        torch.cat(weights).double(),
        slices,
        reconstruction_error,
    )


def fixed_feature_margin_probe(
    features,
    source_weights,
    labels,
    protected,
    *,
    margin=1.0,
    maximum_iterations=3000,
    maximum_seconds=120.0,
    whitening_floor=1e-6,
    objective="feasible",
    feasible_start=None,
):
    """Find a sufficient linear readout on recorded histories, using torch only.

    Columns include the caller's bias feature. Each update projects onto one
    violated action-ranking half-space in whitened coordinates. This is a
    bounded representational diagnostic. The default is not PPO, a minimum-
    change solution, an infeasibility certificate, or unseen-state protection.
    nearest_source requires a supplied feasible matrix and uses a bounded
    primal active-set descent toward the original parameter matrix. Its
    finite-data certificate is separate from game-outcome protection.
    A returned candidate must still pass float32 policy and complete-game checks.
    """
    x = torch.as_tensor(features, dtype=torch.float64).detach().cpu()
    original = torch.as_tensor(source_weights, dtype=torch.float64).detach().cpu()
    y = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    protected = torch.as_tensor(protected, dtype=torch.bool).detach().cpu()
    if (
        x.ndim != 2
        or original.ndim != 2
        or x.shape[1] != original.shape[1]
        or y.shape != (len(x),)
        or protected.shape != y.shape
        or not len(x)
        or original.shape[0] < 2
        or bool((y < 0).any())
        or bool((y >= len(original)).any())
        or not torch.isfinite(x).all()
        or not torch.isfinite(original).all()
    ):
        raise ValueError(
            "Margin probe needs finite aligned features, labels and source action rows"
        )
    if (
        not np.isfinite(margin)
        or margin <= 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
        or not np.isfinite(maximum_seconds)
        or maximum_seconds <= 0
        or not 0 < whitening_floor < 1
    ):
        raise ValueError(
            "Declare positive bounded projection limits and whitening floor"
        )
    if objective not in ("feasible", "nearest_source"):
        raise ValueError("Declare feasible or nearest_source projection")
    if objective == "nearest_source":
        return _nearest_source_readout(
            x,
            original,
            y,
            protected,
            feasible_start,
            margin,
            maximum_iterations,
            maximum_seconds,
        )
    if feasible_start is not None:
        raise ValueError("A feasible start belongs to nearest_source only")
    started = time.monotonic()
    covariance = x.T @ x / len(x)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = max(
        float(eigenvalues[-1]) * whitening_floor, torch.finfo(torch.float64).eps
    )
    transform = (
        eigenvectors * eigenvalues.clamp_min(floor).rsqrt()[None]
    ) @ eigenvectors.T
    rotated = x @ transform
    scales = rotated.norm(dim=1).clamp_min(1e-12)
    unit = rotated / scales[:, None]
    scores = (x @ original.T) / scales[:, None]
    delta = torch.zeros_like(original)
    rows = torch.arange(len(x))
    threshold = margin / scales
    history = []
    reason = "iteration_limit"
    completed = 0
    for iteration in range(maximum_iterations + 1):
        others = scores.clone()
        others[rows, y] = -torch.inf
        competitor, which = others.max(1)
        violation = competitor - scores[rows, y] + threshold
        if iteration % 100 == 0 or iteration == maximum_iterations:
            correct = scores.argmax(1) == y
            history.append(
                dict(
                    iteration=iteration,
                    maximum_normalized_violation=float(violation.max()),
                    protected_correct=int(correct[protected].sum()),
                    protected_total=int(protected.sum()),
                    new_correct=int(correct[~protected].sum()),
                    new_total=int((~protected).sum()),
                )
            )
        if float(violation.max()) <= 1e-9:
            reason = "cached_constraints_satisfied"
            break
        if iteration == maximum_iterations:
            break
        if time.monotonic() - started >= maximum_seconds:
            reason = "time_limit"
            break
        at = int(violation.argmax())
        better = int(y[at])
        worse = int(which[at])
        v = unit[at]
        step = float(violation[at]) / (2.0 * float(v.square().sum()))
        change = step * v
        delta[better] += change
        delta[worse] -= change
        effect = step * (unit @ v)
        scores[:, better] += effect
        scores[:, worse] -= effect
        completed += 1
    candidate = original + delta @ transform.T
    # Recompute from the actual returned matrix, not accumulated score updates.
    final = x @ candidate.T
    other = final.clone()
    other[rows, y] = -torch.inf
    gaps = final[rows, y] - other.max(1).values
    correct = final.argmax(1) == y
    audit = dict(
        reason=reason,
        projection_steps=completed,
        seconds=time.monotonic() - started,
        observations=len(x),
        features=x.shape[1],
        actions=len(original),
        margin=margin,
        all_cached_actions_match=bool(correct.all()),
        minimum_physical_margin=float(gaps.min()),
        cached_constraints_satisfied=bool((gaps >= margin - 1e-6).all()),
        protected_correct=int(correct[protected].sum()),
        protected_total=int(protected.sum()),
        new_correct=int(correct[~protected].sum()),
        new_total=int((~protected).sum()),
        maximum_parameter_change=float((candidate - original).abs().max()),
        history=history,
        scope="Sufficient fixed-history linear feasibility only; failure is inconclusive. No native frames or PPO updates.",
    )
    return candidate, audit


def _nearest_source_readout(
    x, original, y, protected, start, margin, iterations, seconds
):
    """Feasible active-set descent for a finite linear ranking projection.

    The objective is half the squared Euclidean distance to original. Working
    rows are sparse class differences, so no full constraint matrix is built.
    A truncated run remains a feasible approximation, not a minimum claim.
    """
    if start is None:
        raise ValueError("Nearest-source descent requires an explicitly feasible start")
    current = torch.as_tensor(start, dtype=x.dtype).detach().cpu().clone()
    if current.shape != original.shape or not torch.isfinite(current).all():
        raise ValueError("Invalid feasible readout matrix")
    scales = x.norm(dim=1)
    if bool((scales <= 0).any()):
        raise ValueError("Positive ranking margins require nonzero features")
    unit = x / scales[:, None]
    threshold = margin / scales
    rows = torch.arange(len(x))

    def margins(w):
        logits = x @ w.T
        other = logits.clone()
        other[rows, y] = -torch.inf
        return logits, logits[rows, y] - other.max(1).values

    _, initial_gaps = margins(current)
    if bool((initial_gaps < margin - 1e-6).any()):
        raise ValueError("Supplied readout is not feasible")
    started = time.monotonic()
    active = []
    history = []
    reason = "iteration_limit"
    completed = 0
    best = current.clone()
    best_norm = float((current - original).square().sum())

    def working(displacement):
        if not active:
            return None, None, torch.empty(0, dtype=x.dtype), -displacement
        a = torch.tensor([i for i, j in active])
        b = torch.tensor([j for i, j in active])
        good = y[a]
        u = unit[a]
        gram = u @ u.T
        signs = (
            (good[:, None] == good[None]).to(x.dtype)
            - (good[:, None] == b[None]).to(x.dtype)
            - (b[:, None] == good[None]).to(x.dtype)
            + (b[:, None] == b[None]).to(x.dtype)
        )
        rhs = ((displacement[good] - displacement[b]) * u).sum(1)
        multipliers = torch.linalg.lstsq(
            gram * signs, rhs[:, None], rcond=1e-12, driver="gelsd"
        ).solution[:, 0]
        direction = -displacement.clone()
        change = multipliers[:, None] * u
        direction.index_add_(0, good, change)
        direction.index_add_(0, b, -change)
        return a, b, multipliers, direction

    for iteration in range(iterations + 1):
        displacement = current - original
        scores = unit @ current.T
        slack = scores[rows, y, None] - scores - threshold[:, None]
        slack[rows, y] = torch.inf
        physical_minimum = float((slack.min(1).values * scales + margin).min())
        norm = float(displacement.square().sum())
        if physical_minimum >= margin - 1e-6 and norm < best_norm:
            best = current.clone()
            best_norm = norm
        if iteration % 20 == 0 or iteration == iterations:
            history.append(
                dict(
                    iteration=iteration,
                    active_constraints=len(active),
                    displacement_norm=float(displacement.norm()),
                    minimum_physical_margin=physical_minimum,
                )
            )
        if iteration == iterations:
            break
        if time.monotonic() - started >= seconds:
            reason = "time_limit"
            break
        a, b, multipliers, direction = working(displacement)
        norm2 = float(direction.square().sum())
        if norm2 <= 1e-16 * max(1.0, float(displacement.square().sum())):
            if len(multipliers) and float(multipliers.min()) < -1e-8:
                active.pop(int(multipliers.argmin()))
                completed += 1
                continue
            reason = "source_projection_KKT_satisfied"
            break
        derivative = float((displacement * direction).sum())
        if derivative >= -1e-12:
            reason = "numerical_stationarity"
            break
        change = unit @ direction.T
        direction_gap = change[rows, y, None] - change
        direction_gap[rows, y] = 0
        if a is not None:
            direction_gap[a, b] = 0
        ratios = torch.full_like(slack, torch.inf)
        decreasing = direction_gap < -1e-12
        ratios[decreasing] = slack[decreasing].clamp_min(0) / (
            -direction_gap[decreasing]
        )
        flat = int(ratios.argmin())
        blocking = (flat // len(original), flat % len(original))
        limit = float(ratios.flatten()[flat])
        optimum = -derivative / norm2
        length = min(optimum, limit)
        current += length * direction
        if limit < optimum - 1e-10:
            if blocking in active:
                reason = "numerical_duplicate_constraint"
                break
            active.append(blocking)
        completed += 1
    final, gaps = margins(current)
    if bool((gaps < margin - 1e-6).any()):
        current = best
        final, gaps = margins(current)
        reason += ";returned_best_feasible"
    correct = final.argmax(1) == y
    displacement = current - original
    a, b, multipliers, _ = working(displacement)
    dual_vector = torch.zeros_like(displacement)
    linear = 0.0
    if a is not None:
        positive = multipliers.clamp_min(0)
        u = unit[a]
        change = positive[:, None] * u
        dual_vector.index_add_(0, y[a], change)
        dual_vector.index_add_(0, b, -change)
        source_scores = unit @ original.T
        linear = float(
            (
                positive * (threshold[a] - source_scores[a, y[a]] + source_scores[a, b])
            ).sum()
        )
    primal = 0.5 * float(displacement.square().sum())
    lower = linear - 0.5 * float(dual_vector.square().sum())
    feasible = bool((gaps >= margin - 1e-6).all())
    audit = dict(
        reason=reason,
        projection_steps=completed,
        seconds=time.monotonic() - started,
        observations=len(x),
        features=x.shape[1],
        actions=len(original),
        margin=margin,
        all_cached_actions_match=bool(correct.all()),
        minimum_physical_margin=float(gaps.min()),
        cached_constraints_satisfied=feasible,
        protected_correct=int(correct[protected].sum()),
        protected_total=int(protected.sum()),
        new_correct=int(correct[~protected].sum()),
        new_total=int((~protected).sum()),
        maximum_parameter_change=float(displacement.abs().max()),
        history=history,
        objective="nearest_source",
        parameter_displacement_norm=float(displacement.norm()),
        starting_displacement_norm=float((torch.as_tensor(start) - original).norm()),
        dual_active_constraints=len(active),
        primal_objective=primal,
        dual_lower_bound=lower,
        primal_dual_gap=primal - lower,
        nearest_solution_certified=feasible
        and abs(primal - lower) <= 1e-6 * max(1.0, primal),
        scope="Bounded feasible nearest-source projection. The certificate covers the supplied finite linear rankings only, never unseen states or game outcomes.",
    )
    return current, audit


def project_verified_output(
    policy, histories, protected_count, limits, *, feasible_start=None
):
    """Project the existing final output using complete verified legal histories."""
    if not 0 < protected_count <= len(histories):
        raise ValueError("Declare which histories the projection protects")
    verification = {}
    raw, labels, _, spans, _ = readout_data(policy, histories, verification)
    layout = policy.action_net
    output = layout if isinstance(layout, torch.nn.Linear) else layout[-1]
    columns = policy.deployment_action_indices or list(range(policy.action_space.n))
    with torch.no_grad():
        encoded = raw.float() if layout is output else layout[:-1](raw.float())
        features = torch.cat(
            (encoded.double(), encoded.new_ones((len(encoded), 1)).double()), dim=1
        )
        original = torch.cat(
            (output.weight[columns], output.bias[columns, None]), dim=1
        ).double()
        baseline = readout_action_scores(policy, layout(raw.float())).argmax(1)
    if not torch.equal((features @ original.T).argmax(1), baseline):
        raise RuntimeError(
            "Cached final-layer projection changed source greedy actions"
        )
    _, index = torch.unique(features, dim=0, return_inverse=True)
    low = torch.full((int(index.max()) + 1,), len(columns), dtype=torch.long)
    high = torch.full_like(low, -1)
    low.scatter_reduce_(0, index, labels, reduce="amin")
    high.scatter_reduce_(0, index, labels, reduce="amax")
    valid = (low == high)[index]
    protected = torch.arange(len(labels)) < spans[protected_count - 1]["last"]
    audit = dict(
        encoding=verification,
        omitted_conflicting_observations=int((~valid).sum()),
        source_matches=int(((baseline == labels) & valid).sum()),
        holds=len(spans),
    )
    for span in spans:
        window = slice(span["first"], span["last"])
        span["ambiguous_actions"] = int((~valid[window]).sum())
    audit["histories"] = spans
    if not valid.any():
        return dict(audit, applied=False)
    x, y, kept = features[valid], labels[valid], protected[valid]
    feasible = feasible_start
    if feasible is None:
        feasible, probe = fixed_feature_margin_probe(
            x,
            original,
            y,
            kept,
            margin=limits["margin"],
            maximum_iterations=limits["feasible_iterations"],
            maximum_seconds=limits["feasible_seconds"],
        )
        audit["bounded_feasibility"] = {
            k: v for k, v in probe.items() if k != "history"
        }
        if not probe["cached_constraints_satisfied"]:
            return dict(audit, applied=False)
    chosen, nearest = fixed_feature_margin_probe(
        x,
        original,
        y,
        kept,
        margin=limits["margin"],
        objective="nearest_source",
        feasible_start=feasible,
        maximum_iterations=limits["nearest_iterations"],
        maximum_seconds=limits["nearest_seconds"],
    )
    audit["bounded_nearest"] = {k: v for k, v in nearest.items() if k != "history"}
    if not nearest["cached_constraints_satisfied"]:
        return dict(audit, applied=False)
    with torch.no_grad():
        output.weight[columns] = chosen[:, :-1].to(output.weight)
        output.bias[columns] = chosen[:, -1].to(output.bias)
        installed = readout_action_scores(policy, layout(raw.float())).argmax(1)
        mismatched = int(((installed != labels) & valid).sum())
        if mismatched:
            output.weight[columns] = original[:, :-1].to(output.weight)
            output.bias[columns] = original[:, -1].to(output.bias)
    return dict(
        audit,
        applied=not mismatched,
        installed_action_mismatches=mismatched,
        proposed_displacement_from_feasible=float((original - feasible).norm()),
        installed_displacement_from_feasible=(
            float((chosen - feasible).norm()) if not mismatched else 0.0
        ),
        scope="Ranking on recorded legal observations only. Replayed training constraints are not independent wins.",
    )
