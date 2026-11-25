import copy
from collections import Counter
import numpy as np
import torch


class SequenceReplay:
    """Replay full histories for the supervised fit."""

    def __init__(
        self,
        capacity=60000,
        seed=1,
        burn=40,
        learn=40,
        nstep=5,
        byte_capacity=402653184,
        alpha=0.6,
    ):
        self.capacity, self.burn, self.learn, self.nstep = (
            int(capacity),
            int(burn),
            int(learn),
            int(nstep),
        )
        self.rng = np.random.default_rng(seed)
        self.episodes = {}
        self.priorities = {}
        self.count = self.transitions = 0
        self.byte_capacity = int(byte_capacity)
        self.bytes = 0
        self.alpha = float(alpha)
        self.history_ids = set()
        self.history_fraction = 0.0
        self.history_sample_fraction = 0.0
        self.retention_usage = {}
        self.retention_events = []
        self.auxiliary_sampling_totals = {}

    @staticmethod
    def episode_bytes(episode):
        return sum(v.nbytes for v in episode.values() if isinstance(v, np.ndarray))

    def _remove(self, number):
        removed = self.episodes.pop(number)
        self.transitions -= len(removed["actions"])
        self.bytes -= self.episode_bytes(removed)
        self.history_ids.discard(number)
        for key in [k for k in self.priorities if k[0] == number]:
            del self.priorities[key]

    def _trim(self):
        while self.episodes:
            history = [n for n in self.episodes if n in self.history_ids]
            recent = [n for n in self.episodes if n not in self.history_ids]
            hn = sum(len(self.episodes[n]["actions"]) for n in history)
            hb = sum(self.episode_bytes(self.episodes[n]) for n in history)
            if self.history_fraction and (
                hn > int(self.capacity * self.history_fraction)
                or hb > int(self.byte_capacity * self.history_fraction)
            ):
                self._remove(history[0])
                continue
            # The historical cap is a maximum, not empty reserved storage.
            # Recent data can use unused space so both arms share total caps.
            rc = self.capacity - hn
            rb = self.byte_capacity - hb
            if recent and (self.transitions - hn > rc or self.bytes - hb > rb):
                self._remove(recent[0])
                continue
            break

    def add(self, episode):
        n = len(episode["actions"])
        raw = "frames" in episode
        if (
            n < 1
            or (
                len(episode["frames"]) != n + 4
                if raw
                else len(episode["maps"]) != n + 1
            )
            or len(episode["positions"]) != n + 1
        ):
            raise ValueError(
                "Continuous replay requires one more legal observation than actions"
            )
        if (not raw and not episode["dones"][-1]) or np.any(episode["dones"][:-1]):
            raise ValueError("Replay episodes must end once and cannot cross resets")
        number = self.count
        self.count += 1
        self.episodes[number] = episode
        self.transitions += n
        self.bytes += self.episode_bytes(episode)
        initial_priority = max(self.priorities.values(), default=1.0)
        for anchor in range(int(episode.get("learn_start", 0)), n, self.learn):
            self.priorities[(number, anchor)] = initial_priority
        self._trim()
        return number

    def add_verified_demonstration(self, episode, frame=0, *, replace_condition=False):
        """Replayed controlled or independent wins supply auxiliary data only.

        Keep its real source-greedy labels; never invent greedy/on-policy
        status for a search controller. This bank may supervise actions only.
        """
        independent = (
            episode.get("independent_learned_control", False)
            and episode.get("validated_complete")
            and np.array_equal(episode["actions"], episode.get("greedy_actions"))
        )
        legal_rule = legal_rule_demonstration(episode)
        if not successful_episode(episode) or not (
            episode.get("controlled_replay_equal") or independent or legal_rule
        ):
            raise ValueError(
                "A demonstration needs a reproduced complete native win/CPU KO"
            )
        if (
            not 0
            <= episode.get("supervision_start", 0)
            < episode.get("supervision_end", len(episode["actions"]))
            <= len(episode["actions"])
        ):
            raise ValueError(
                "A demonstration needs a nonempty validated learning window"
            )
        fresh = copy.deepcopy(episode)
        fresh["supervision_only"] = True
        if legal_rule:
            fresh.setdefault("retain_on_self_success", True)
            fresh.setdefault(
                "persistent_auxiliary_priority", fresh["retain_on_self_success"]
            )
            fresh.setdefault(
                "auxiliary_priority", fresh["persistent_auxiliary_priority"]
            )
        fresh["source"] = (
            "verified_independent_success"
            if independent
            else (
                "verified_legal_rule_success"
                if legal_rule
                else "verified_controlled_demonstration"
            )
        )
        fresh["verified_frame"] = int(frame)
        if replace_condition:
            for number, old in list(self.episodes.items()):
                if not same_episode_condition(old, fresh):
                    continue
                self.retention_usage[number]["retired_frame"] = int(frame)
                self.retention_usage[number][
                    "retirement_reason"
                ] = "replaced_by_verified_coherent_response"
                self._remove(number)
        number = self.add(fresh)
        if number not in self.episodes:
            raise ValueError("The full demonstration exceeds the bounded bank")
        self.retention_usage[number] = dict(
            first_protected_frame=int(frame),
            retired_frame=None,
            sampled_sequences=0,
            auxiliary_transitions=0,
            TD_transitions=0,
            history_TD_transitions=0,
            policy_version=fresh["policy_version"],
            opponent=fresh["opponent_config"],
            player=fresh["player"],
            offset=fresh["offset"],
            group=fresh["group"],
        )
        self.retention_usage[number]["auxiliary_priority"] = bool(
            fresh.get("auxiliary_priority", False)
        )
        return number

    def required_success_sources(self, required_conditions):
        """Resolve frozen coaches or the unique refreshable ordinary slot.

        A retained coach and a learner's ordinary success may share a fight.
        Refresh permission follows the ordinary slot, never both sources.
        An explicit owner may replace its coach only after a real independent
        complete success. This selects training data, never deployment actions.
        This read-only preflight does not draw RNG or mark any sample as used.
        """
        required = []
        for rule in required_conditions:
            refresh = rule.get("allow_verified_source_refresh", False)
            registry = rule.get("auxiliary_registry_id")
            if registry is not None and (
                type(registry) is not int
                or registry < 0
                or refresh
                or rule.get("prefer_learner_success_owner")
            ):
                raise ValueError(
                    "A fixed auxiliary registry source cannot also refresh its owner"
                )
            matches = [
                n
                for n, e in self.episodes.items()
                if all(
                    e.get(k) == rule[k] for k in ("opponent_config", "player", "offset")
                )
                and (
                    not e.get("retain_on_self_success", False)
                    if refresh
                    else e.get("policy_version") == rule["policy_version"]
                )
                and (
                    registry is None
                    or n == registry
                    and e.get("auxiliary_registry_id") == registry
                )
            ]
            owner = rule.get("prefer_learner_success_owner")
            if owner:
                mastered = [
                    n
                    for n, e in self.episodes.items()
                    if all(
                        e.get(k) == rule[k]
                        for k in ("opponent_config", "player", "offset")
                    )
                    and not e.get("retain_on_self_success", False)
                    and e.get("learner_success_owner") == owner
                    and e.get("independent_learned_control")
                    and e.get("validated_complete")
                    and successful_episode(e)
                    and np.array_equal(e["actions"], e.get("greedy_actions"))
                ]
                if mastered:
                    matches = mastered
            if len(matches) != 1:
                raise ValueError(
                    "A required auxiliary condition must identify one frozen trajectory"
                )
            number = matches[0]
            episode = self.episodes[number]
            if (
                number in required
                or not successful_episode(episode)
                or not episode.get("validated_complete")
                or not (
                    episode.get("controlled_replay_equal")
                    or episode.get("independent_learned_control")
                    or legal_rule_demonstration(episode)
                )
            ):
                raise ValueError(
                    "Required auxiliary trajectories must be distinct verified complete successes"
                )
            required.append(number)
        return required

    def success_sequences(
        self,
        count,
        length,
        disagreement_fraction=0.0,
        disagreement_scope="failed",
        priority_fraction=0.0,
        disagreement_selection="random",
        required_conditions=(),
    ):
        """Group-balanced, distinct actual histories for an auxiliary loss.

        Optional frozen-condition reservations consume an existing group slot.
        Their last assessed disagreement supplies a window, never a game action.
        A declared fraction may revisit a complete verified response even when
        an earlier disagreement remains. It does not add trajectories or slots.
        Explicit outcome-retention reservations may follow a unique refreshed
        verified source; tactical coaches keep their default version binding.
        """
        if disagreement_scope not in ("failed", "verified_success"):
            raise ValueError("Unknown success-disagreement scope")
        if disagreement_selection not in ("random", "first"):
            raise ValueError("Unknown success frontier selection")
        if not 0 <= priority_fraction <= 1:
            raise ValueError("Invalid within-group new-skill sampling fraction")
        if len(self.episodes) < count:
            raise ValueError("Distinct success trajectories are missing")
        groups = ["pve", "p1", "p2"]
        self.rng.shuffle(groups)
        chosen = []
        required = self.required_success_sources(required_conditions)
        quotas = Counter(groups[i % 3] for i in range(count))
        if any(
            sum(self.episodes[n]["group"] == g for n in required) > quotas[g]
            for g in groups
        ):
            raise ValueError(
                "Required trajectories exceed the unchanged auxiliary group budget"
            )
        for i in range(count):
            available = [
                n
                for n, e in self.episodes.items()
                if n not in chosen and e["group"] == groups[i % 3]
            ]
            if not available:
                available = [n for n in self.episodes if n not in chosen]
            pending = [n for n in required if n in available and n not in chosen]
            if pending:
                number = pending[0]
            elif priority_fraction:
                new = [
                    n
                    for n in available
                    if self.episodes[n].get("auxiliary_priority", False)
                    or self.episodes[n].get("auxiliary_disagreement_priority", False)
                ]
                old = [n for n in available if n not in new]
                if new and old:
                    available = new if self.rng.random() < priority_fraction else old
                number = int(self.rng.choice(available))
            else:
                number = int(self.rng.choice(available))
            chosen.append(number)
        result = []
        for number in chosen:
            episode = self.episodes[number]
            minimum = int(episode.get("supervision_start", 0))
            end = int(episode.get("supervision_end", len(episode["actions"])))
            if not 0 <= minimum < end <= len(episode["actions"]):
                raise ValueError("Invalid auxiliary learning window")
            if "tactical_window" in episode:
                raise ValueError(
                    "Unknown tactical_window field; use validated auxiliary_priority_window"
                )
            tactical = episode.get("auxiliary_priority_window")
            if tactical is not None:
                if not isinstance(tactical, dict) or set(tactical) != {
                    "first",
                    "last",
                    "fraction",
                }:
                    raise ValueError(
                        "A tactical sampling window needs first, last and fraction"
                    )
                low, high = tactical["first"], tactical["last"]
                fraction = float(tactical["fraction"])
                verified_start = max(
                    minimum, int(episode.get("original_supervision_start", minimum))
                )
                if (
                    type(low) is not int
                    or type(high) is not int
                    or not verified_start <= low < high <= end
                    or high - low > length
                    or not 0 <= fraction <= 1
                    or episode.get("sample_complete_supervision_window", False)
                    or not episode.get("validated_complete")
                    or not successful_episode(episode)
                    or not (
                        episode.get("controlled_replay_equal")
                        or episode.get("independent_learned_control")
                    )
                ):
                    raise ValueError(
                        "Tactical sampling must stay inside one verified complete response"
                    )
            first = int(self.rng.integers(minimum, end))
            errors = episode.get("supervised_disagreements", np.empty(0, np.int64))
            eligible = (
                episode.get("current_complete_useful") is False
                if disagreement_scope == "failed"
                else successful_episode(episode)
            )
            focused = (
                eligible
                and len(errors) > 0
                and (
                    number in required
                    or disagreement_fraction > 0
                    and self.rng.random() < disagreement_fraction
                )
            )
            if focused:
                center = int(self.rng.choice(errors))
                if disagreement_selection == "first":
                    center = int(errors[0])
                first = max(minimum, min(center - length // 2, end - length))
            last = min(first + length, end)
            if episode.get("sample_complete_supervision_window", False):
                if end - minimum > length:
                    raise ValueError(
                        "The requested sequence cannot hold the complete verified window"
                    )
                # Keep the real preceding history for reconstruction, but teach
                # the whole short verified exchange, not a random tail of it.
                # These remain auxiliary soft/action labels, never PPO returns.
                first, last = minimum, end
            reserved = (
                tactical is not None and fraction > 0 and self.rng.random() < fraction
            )
            if reserved:
                # An earlier entry error must not indefinitely starve the new
                # verified response. Keep the complete raw history for current
                # recurrent reconstruction, and the same one-source batch cap.
                first, last = low, high
                focused = False
            use = self.retention_usage[number]
            use["sampled_sequences"] += 1
            use["auxiliary_transitions"] += last - first
            total = self.auxiliary_sampling_totals.setdefault(
                episode["group"], dict(sequences=0, transitions=0)
            )
            total["sequences"] += 1
            total["transitions"] += last - first
            use["phase_auxiliary_transitions"] = (
                use.get("phase_auxiliary_transitions", 0) + last - first
            )
            key = (
                "tactical_window_sequences"
                if reserved
                else "disagreement_sequences" if focused else "uniform_sequences"
            )
            use[key] = use.get(key, 0) + 1
            if reserved:
                use["tactical_window_labels"] = (
                    use.get("tactical_window_labels", 0) + last - first
                )
            use["covered_disagreements"] = use.get("covered_disagreements", 0) + int(
                ((errors >= first) & (errors < last)).sum()
            )
            if number in required:
                use["required_condition_sequences"] = (
                    use.get("required_condition_sequences", 0) + 1
                )
                use["required_assessed_disagreement_windows"] = use.get(
                    "required_assessed_disagreement_windows", 0
                ) + int(focused)
            if episode.get("auxiliary_disagreement_priority", False):
                use["disagreement_priority_sequences"] = (
                    use.get("disagreement_priority_sequences", 0) + 1
                )
            if episode.get("persistent_auxiliary_priority", False):
                # Record the actual loss windows without changing sampling.
                # Episode counts alone cannot show whether a recently fixed
                # tactical decision is still revisited after its errors clear.
                at = str(int(episode.get("disagreement_assessed_at_update", -1)))
                windows = use.setdefault(
                    "auxiliary_windows_by_assessment", {}
                ).setdefault(at, {})
                window = f"{first}:{last}"
                windows[window] = windows.get(window, 0) + 1
            episode["auxiliary_registry_id"] = number
            result.append((episode, first, last))
        return result

    def assess_success_disagreements(
        self,
        policy,
        optimizer_update,
        priority_only=False,
        prioritize_disagreements=False,
        sequence_length=32,
        required_conditions=(),
    ):
        """Current full-history fit, including refreshable required sources.

        Resolving a reservation must match the sampler after a learner's
        independent success replaces its coach. Priority flags alone do not
        identify those new ordinary slots. No target or game action changes.
        """
        if prioritize_disagreements and priority_only:
            raise ValueError("Disagreement priority requires full-history assessment")
        required = set(self.required_success_sources(required_conditions))
        total = assessed = required_assessed = observations = 0
        for number, episode in self.episodes.items():
            if not prioritize_disagreements:
                episode.pop("auxiliary_disagreement_priority", None)
                self.retention_usage.get(number, {}).pop(
                    "auxiliary_disagreement_priority", None
                )
            if (
                priority_only
                and not episode.get("auxiliary_priority", False)
                and number not in required
            ):
                continue
            scores = full_values(policy, episode, sequence_length=sequence_length)[:-1]
            observations += len(episode["actions"]) + 1
            target = episode["actions"]
            if episode.get("retention_role") == "qualified_function_reference":
                # A frozen policy was queried on another controller's real
                # history. Preserve those soft labels without rewriting the
                # recorded controller's actual actions or greedy proposals.
                target = episode["policy_probabilities"].argmax(1)
            errors = (
                torch.nonzero(
                    scores.argmax(1) != torch.as_tensor(target, device=scores.device)
                )
                .flatten()
                .cpu()
                .numpy()
            )
            self.retention_usage[number]["assessment_target"] = (
                "frozen_physical_distribution"
                if episode.get("retention_role") == "qualified_function_reference"
                else "executed_actions"
            )
            errors = errors[errors >= episode.get("supervision_start", 0)]
            errors = errors[
                errors < episode.get("supervision_end", len(episode["actions"]))
            ]
            excluded = episode.get(
                (
                    "retention_excluded_decisions"
                    if episode.get("retention_target") == "physical_distribution"
                    else "uncertain_action_decisions"
                ),
                (),
            )
            if any(
                type(at) is not int or not 0 <= at < len(episode["actions"])
                for at in excluded
            ):
                raise ValueError(
                    "Excluded assessment labels must refer to actual decisions"
                )
            # A deliberately unprotected or non-unique action must not become
            # the permanently unfittable frontier of error-driven sampling.
            errors = errors[~np.isin(errors, excluded)]
            if episode.get("retention_allowed_actions"):
                from .teacher import verified_reference_sets

                predicted = scores.argmax(1)
                for at, choices in verified_reference_sets(
                    episode, 0, len(target), scores.shape[1]
                ):
                    if int(predicted[at]) in choices:
                        errors = errors[errors != at]
            self.bytes -= self.episode_bytes(episode)
            episode["supervised_disagreements"] = errors
            episode["disagreement_assessed_at_update"] = int(optimizer_update)
            if prioritize_disagreements:
                # A replay-coverage risk, not an inferred loss or bad action.
                episode["auxiliary_disagreement_priority"] = bool(len(errors))
                if number in self.retention_usage:
                    self.retention_usage[number]["auxiliary_disagreement_priority"] = (
                        bool(len(errors))
                    )
            self.bytes += self.episode_bytes(episode)
            total += len(errors)
            assessed += 1
            required_assessed += int(number in required)
        if self.bytes > self.byte_capacity:
            raise ValueError("Disagreement metadata exceeds the bounded replay budget")
        return dict(
            optimizer_update=int(optimizer_update),
            episodes=len(self.episodes),
            disagreements=total,
            assessed_episodes=assessed,
            required_sources=len(required),
            required_sources_assessed=required_assessed,
            assessed_observations=observations,
            encoding_sequence_length=sequence_length,
            disagreement_priority_episodes=sum(
                e.get("auxiliary_disagreement_priority", False)
                for e in self.episodes.values()
            ),
            failed_conditions=sum(
                e.get("current_complete_useful") is False
                for e in self.episodes.values()
            ),
        )


def audit_success_history_labels(
    episodes,
    use_soft_references=False,
    maximum_examples=32,
    hard_failed_conditions=False,
):
    """Find disagreeing labels after exactly equal recorded legal prefixes.

    Whole-game collectors establish the initial continuity reset. Compare
    raw observations and actually executed history directly; never use a
    stored teacher hidden state to equate two different network histories.
    A disagreement is a fitting conflict, not evidence either action loses.
    """
    episodes = list(episodes)
    matches = hard_conflicts = 0
    examples = []
    for i, left in enumerate(episodes):
        for j, right in enumerate(episodes[i + 1 :], i + 1):
            count = min(len(left["actions"]), len(right["actions"]))
            different = np.flatnonzero(
                left["actions"][:count] != right["actions"][:count]
            )
            if not len(different):
                continue
            at = int(different[0])
            if not np.array_equal(
                left["positions"][: at + 1], right["positions"][: at + 1]
            ):
                continue
            if not np.array_equal(left["frames"][: at + 4], right["frames"][: at + 4]):
                continue
            if any(
                key in left
                and key in right
                and not np.array_equal(left[key][: at + 1], right[key][: at + 1])
                for key in ("previous", "input_durations")
            ):
                continue
            if any(
                at < e.get("supervision_start", 0)
                or at >= e.get("supervision_end", len(e["actions"]))
                for e in (left, right)
            ):
                continue
            if any(
                at in e.get("uncertain_action_decisions", ()) for e in (left, right)
            ):
                continue
            soft = [
                use_soft_references
                and e.get("retention_target") == "physical_distribution"
                and not (
                    hard_failed_conditions and e.get("current_complete_useful") is False
                )
                for e in (left, right)
            ]
            matches += 1
            hard_conflicts += not any(soft)
            if len(examples) < maximum_examples:
                examples.append(
                    dict(
                        episode_indices=[i, j],
                        decision=at,
                        actions=[int(e["actions"][at]) for e in (left, right)],
                        soft_targets=soft,
                        opponents=[e.get("opponent_config") for e in (left, right)],
                        ports=[e.get("player") for e in (left, right)],
                        offsets=[e.get("offset") for e in (left, right)],
                        sources=[e.get("policy_version") for e in (left, right)],
                    )
                )
    return dict(
        episodes=len(episodes),
        exact_prefix_action_disagreements=matches,
        both_hard_target_conflicts=hard_conflicts,
        examples=examples,
        scope="Exact recorded observation/executed-input prefixes only; no file digests, optimal-action labels or global aliasing claim.",
    )


def successful_episode(episode):
    return (
        episode.get("ko_win", False)
        if episode["opponent_config"]["kind"] == "cpu"
        else episode.get("result") == "win"
    )


def legal_rule_demonstration(episode):
    """Verified actual-history rule source; never an independent learned win."""
    first = int(episode.get("supervision_start", 0))
    count = len(episode["actions"])
    return (
        episode.get("legal_controller_recording") is True
        and episode.get("validated_complete")
        and type(episode.get("legal_controller_queries")) is int
        and 0 <= first < count
        and episode["legal_controller_queries"] == count - first
        and episode.get("independent_learned_control") is False
    )


def same_episode_condition(first, second):
    return (
        first.get("opponent_config") == second.get("opponent_config")
        and first.get("player") == second.get("player")
        and first.get("offset") == second.get("offset")
    )


def episode_values(
    policy,
    episode,
    first,
    last,
    hidden=None,
    burn=0,
    return_features=False,
    detach_features=False,
):
    """Read a contiguous legal interval with current visual/recurrent weights."""
    images = torch.as_tensor(
        np.stack([episode["frames"][i : i + 4] for i in range(first, last)])
    )[None]
    extra = (
        {}
        if "input_durations" not in episode
        else dict(
            input_durations=torch.as_tensor(
                episode["input_durations"][first:last][None]
            )
        )
    )
    if return_features:
        extra["return_features"] = True
    if detach_features:
        extra["detach_features"] = True
    return policy.observation_sequence_values(
        images,
        torch.as_tensor(episode["positions"][first:last][None]),
        torch.as_tensor(episode["previous"][first:last][None]),
        hidden,
        burn,
        **extra,
    )


def full_values(policy, episode, *, sequence_length=32):
    """Deployed physical-action scores along a complete actual history.

    These Q scores or PPO logits already include facing/output-map handling.
    Pair their columns with recorded physical actions, not cached-head
    canonical labels from the readout-fitting contract.
    """
    if type(sequence_length) is not int or not 1 <= sequence_length <= 32:
        raise ValueError(
            "History encoding needs a declared sequence length from one to32"
        )
    values = []
    hidden = None
    with torch.no_grad():
        for first in range(0, len(episode["actions"]) + 1, sequence_length):
            scores, hidden = episode_values(
                policy,
                episode,
                first,
                min(first + sequence_length, len(episode["actions"]) + 1),
                hidden,
            )
            values.append(scores[0])
    return torch.cat(values)
