import copy
from pathlib import Path
import torch
from torch import nn


def policy_output_parameter_names(policy):
    """Final physical-action and critic readouts, excluding learned features."""
    linear = [
        (name, module)
        for name, module in policy.action_net.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear:
        raise ValueError("A physical action readout needs its final linear layer")
    name = linear[-1][0]
    prefix = "action_net." + (name + "." if name else "")
    return {
        key
        for key, _ in policy.named_parameters()
        if key.startswith("value_net.") or key in (prefix + "weight", prefix + "bias")
    }


def set_policy_learning_scope(policy, scope="all", *, critic_active=True):
    """One actor/critic training contract for PPO and auxiliary-only updates."""
    if scope not in (
        "all",
        "readout",
        "encoder_readout",
        "hidden_readout",
        "output",
    ) or not isinstance(critic_active, bool):
        raise ValueError(
            "Declare the actor scope and whether the critic has a learning objective"
        )
    if scope == "hidden_readout":
        outputs = policy_output_parameter_names(policy)
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(
                (name.startswith("action_net.") and name not in outputs)
                or (critic_active and name.startswith("value_net."))
            )
    elif scope in ("readout", "encoder_readout"):
        prefixes = (
            ("action_net.", "features_extractor.")
            if scope == "encoder_readout"
            else ("action_net.",)
        )
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(
                name.startswith(prefixes)
                or (critic_active and name.startswith("value_net."))
            )
    elif scope == "output":
        outputs = policy_output_parameter_names(policy)
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name in outputs)
    elif not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("All inference weights must learn")
    if not critic_active:
        # Categorical losses cancel a common value baseline mathematically,
        # but zero gradients still let restored Adam moments move it.
        for name, parameter in policy.named_parameters():
            if name.startswith("value_net."):
                parameter.requires_grad_(False)


def policy_optimizer_step(policy, representation_lr_scale=1.0):
    """Adam with a fixed relative learning rate for learned representations.

    Scaling the parameter displacement is equivalent to a smaller per-block
    Adam learning rate (zero weight decay). Moments still follow the actual
    gradients. No parameter group or inference/checkpoint schema changes.
    """
    if not 0 < representation_lr_scale <= 1:
        raise ValueError(
            "Representation learning rate must remain positive and at most the output rate"
        )
    if representation_lr_scale == 1.0:
        policy.optimizer.step()
        return
    if any(group.get("weight_decay", 0) for group in policy.optimizer.param_groups):
        raise ValueError("Displacement scaling assumes zero Adam weight decay")
    outputs = policy_output_parameter_names(policy)
    representations = [
        p
        for name, p in policy.named_parameters()
        if name not in outputs and p.grad is not None
    ]
    before = [p.detach().clone() for p in representations]
    policy.optimizer.step()
    with torch.no_grad():
        for p, old in zip(representations, before):
            p.copy_(old + representation_lr_scale * (p - old))


def restore_optimizer(optimizer, state):
    """Own the moments: loading on the same device may alias snapshot tensors."""
    optimizer.load_state_dict(copy.deepcopy(state))


def save(path, policy, definition, training=None, optimizer=None):
    path = Path(path)
    from counter_ryu.game import ROOT

    if path.resolve().is_relative_to((ROOT / "models").resolve()):
        raise ValueError(
            "Research saving cannot modify the retained reference directory"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        definition=definition,
        state_dict={k: v.detach().cpu() for k, v in policy.state_dict().items()},
        training=training or {},
    )
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    temporary.replace(path)
