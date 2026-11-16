import copy
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import copy
from collections import deque
from torch.nn import functional as F
import gymnasium as gym

def spaces(config):
    c = config['observation_config']
    bound = np.float32(65535 / 1024)
    return gym.spaces.Dict(dict(image=gym.spaces.Box(0, 255, (4, c['height'], c['width']), np.uint8), position=gym.spaces.Box(np.array([0, 0, 0, 0, -bound, -bound], np.float32), np.full(6, bound, np.float32), dtype=np.float32)))

def compiled(actions):
    result = []
    allowed = {'LEFT', 'RIGHT', 'UP', 'DOWN', 'A', 'B', 'C', 'X', 'Y', 'Z'}
    for action in actions:
        sequence = []
        for part in action['segments']:
            if not set(part['keys']) <= allowed:
                raise ValueError('Unsupported control key')
            sequence.extend([tuple(part['keys'])] * int(part['frames']))
        if len(sequence) != 6:
            raise ValueError('Only six-native-frame inputs are supported')
        result.append(tuple(sequence))
    return result

def advance_input_durations(previous, command, cap=120):
    """Continuous physical LEFT/RIGHT/DOWN keys, not native charge readiness."""
    counts = np.asarray(previous, dtype=np.int16).copy()
    if counts.shape != (3,) or not 0 < int(cap) <= 255:
        raise ValueError('Expected3 bounded physical input durations')
    for keys in command:
        held = np.asarray([key in keys for key in ('LEFT', 'RIGHT', 'DOWN')])
        counts = np.minimum(np.where(held, counts + 1, 0), int(cap))
    return counts.astype(np.uint8)

def continuity_reset(previous, current):
    if previous is None or previous.shape != current.shape:
        return np.ones(len(current), bool)
    return ~np.all(previous[:, 1:] == current[:, :-1], axis=(1, 2, 3))

def mirror_indices(actions):
    ids = {a['name']: i for i, a in enumerate(actions)}
    result = []
    for a in actions:
        name = a['name'].replace('RIGHT', 'TEMP').replace('LEFT', 'RIGHT').replace('TEMP', 'LEFT')
        name = name.replace('_R_', '_TEMP_').replace('_L_', '_R_').replace('_TEMP_', '_L_')
        result.append(ids[name])
    return result

class VisibleHistoryEncoder(BaseFeaturesExtractor):

    def __init__(self, observation_space):
        super().__init__(observation_space, 160)
        cnn = nn.Sequential(nn.Conv2d(7, 16, 8, 4), nn.ReLU(), nn.Conv2d(16, 32, 4, 2), nn.ReLU(), nn.Conv2d(32, 32, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            width = cnn(torch.zeros(1, 7, *observation_space['image'].shape[-2:])).shape[1]
        self.visual = nn.Sequential(cnn, nn.Linear(width, 128), nn.Tanh())
        self.coordinates = nn.Sequential(nn.Linear(6, 32), nn.Tanh())

    def forward(self, obs):
        self.current_observations = {k: v.detach() for k, v in obs.items()}
        image = obs['image']
        pos = obs['position'].clone()
        flip = pos[:, 4] < 0
        image = torch.where(flip[:, None, None, None], image.flip(-1), image)
        pos[:, 0] = torch.where(flip, 0.5 - pos[:, 0], pos[:, 0])
        pos[:, 2] = torch.where(flip, 0.5 - pos[:, 2], pos[:, 2])
        pos[:, 4] = pos[:, 4].abs()
        history = torch.cat((image, image[:, 1:] - image[:, :-1]), dim=1)
        spatial = self.visual[0](history)
        visual = self.visual[2](self.visual[1](spatial))
        coordinates = self.coordinates(pos * 8)
        if getattr(self, 'expose_spatial_context', False):
            context = torch.cat((spatial, coordinates), dim=1)
            self.current_spatial_context = F.layer_norm(context, (context.shape[1],))
        return torch.cat((visual, coordinates), dim=1)

class SequenceQPolicy(MultiInputActorCriticPolicy):
    """Single recurrent learner over every physical input, with no expert votes.

    New models train the visual stem and use physical previous actions;
    historical frozen/canonical models remain loadable. Q heads start
    independently of PPO values/logits.
    Recurrent resets use only continuity of the legal four-image observation.
    """

    def __init__(self, *args, action_mirror=None, memory_hidden=128, trainable_visual=False, physical_actions=False, signed_relative_position=False, input_history=None, physical_commands=None, facing_canonical=False, visual_normalization=None, action_readout=None, deployment_action_indices=None, internal_action_count=None, internal_action_mirror=None, **kwargs):
        self.action_mirror_list = action_mirror
        self.memory_hidden = int(memory_hidden)
        self.trainable_visual = bool(trainable_visual)
        self.physical_actions = bool(physical_actions)
        self.signed_relative_position = bool(signed_relative_position)
        self.facing_canonical = bool(facing_canonical)
        if visual_normalization not in (None, 'layer'):
            raise ValueError('Unknown visual normalization')
        self.visual_normalization = visual_normalization
        if action_readout not in (None, 'relu'):
            raise ValueError('Unknown action readout')
        self.action_readout = action_readout
        self.deployment_action_indices = None if deployment_action_indices is None else list(deployment_action_indices)
        self.internal_action_count = internal_action_count
        self.internal_action_mirror = internal_action_mirror
        if self.facing_canonical and (not self.physical_actions):
            raise ValueError('Facing normalization expects recorded physical inputs')
        self.input_history = copy.deepcopy(input_history)
        self.physical_commands = physical_commands
        if self.input_history is not None:
            if not self.physical_actions or self.input_history['mode'] not in ('real', 'zero') or (not 0 < int(self.input_history['cap_native_frames']) <= 255) or (self.input_history['normalization'] <= 0) or (physical_commands is None):
                raise ValueError('Input durations require explicit physical commands and a real/zero configuration')
        super().__init__(*args, **kwargs)
        self.reset_inference_state()

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.visual_normalization == 'layer':
            self.features_extractor.visual[2] = nn.Sequential(nn.LayerNorm(self.features_extractor.visual[1].out_features), nn.Tanh())
        self.register_buffer('physical_mirror', torch.tensor(self.action_mirror_list, dtype=torch.long))
        count = self.action_space.n if self.internal_action_count is None else int(self.internal_action_count)
        if self.deployment_action_indices is not None:
            indices = self.deployment_action_indices
            if len(indices) != self.action_space.n or len(set(indices)) != len(indices) or min(indices) != 0 or (max(indices) >= count):
                raise ValueError('Invalid external-to-internal deployment action catalog')
        elif count != self.action_space.n:
            raise ValueError('An internal action width requires its explicit physical mapping')
        self.sequence_gru = nn.GRU(160 + count, self.memory_hidden, batch_first=True)
        self.action_net = nn.Linear(160 + self.memory_hidden, count)
        if self.action_readout == 'relu':
            width = 160 + self.memory_hidden
            self.action_net = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, count))
        self.value_net = nn.Linear(160 + self.memory_hidden, 1)
        output = self.action_net if self.action_readout is None else self.action_net[-1]
        nn.init.normal_(output.weight, std=0.01)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(self.value_net.weight)
        nn.init.zeros_(self.value_net.bias)
        if self.input_history is not None:
            self.input_duration_projection = nn.Linear(3, 160, bias=False)
            nn.init.zeros_(self.input_duration_projection.weight)
        for parameter in self.features_extractor.visual[0].parameters():
            parameter.requires_grad_(self.trainable_visual)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad], lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        values = super()._get_constructor_parameters()
        values.update(action_mirror=self.action_mirror_list, memory_hidden=self.memory_hidden, trainable_visual=self.trainable_visual, physical_actions=self.physical_actions, signed_relative_position=self.signed_relative_position, input_history=self.input_history, physical_commands=self.physical_commands, facing_canonical=self.facing_canonical, visual_normalization=self.visual_normalization, action_readout=self.action_readout, deployment_action_indices=self.deployment_action_indices, internal_action_count=self.internal_action_count, internal_action_mirror=self.internal_action_mirror)
        return values

    def encode_maps(self, observations):
        image = observations['image'].float() / 255.0
        flip = (observations['position'][:, 4] < 0) & (not self.physical_actions or self.facing_canonical)
        image = torch.where(flip[:, None, None, None], image.flip(-1), image)
        history = torch.cat((image, image[:, 1:] - image[:, :-1]), 1)
        if self.trainable_visual:
            return self.features_extractor.visual[0](history)
        with torch.no_grad():
            return self.features_extractor.visual[0](history).to(torch.float16)

    def observation_sequence_values(self, images, positions, previous, hidden=None, burn_in=0, input_durations=None, return_features=False):
        """Re-encode legal replay bytes with current filters, including burn-in.

        The recurrent input is the actual preceding physical action for new
        models. Historical canonical models retain their original semantics.
        """
        batch, length = images.shape[:2]
        maps = self.encode_maps(dict(image=images.flatten(0, 1), position=positions.flatten(0, 1)))
        return self.sequence_values(maps.reshape(batch, length, -1), positions, previous, hidden, burn_in, input_durations, return_features)

    def sequence_values(self, maps, positions, previous, hidden=None, burn_in=0, input_durations=None, return_features=False):
        batch, length, width = maps.shape
        visual = self.features_extractor.visual[2](self.features_extractor.visual[1](maps.float().reshape(-1, width)))
        raw_pos = positions.reshape(-1, 6)
        flip = (raw_pos[:, 4] < 0) & (not self.physical_actions or self.facing_canonical)
        relative_x = raw_pos[:, 4]
        if self.facing_canonical or not (self.physical_actions and self.signed_relative_position):
            relative_x = relative_x.abs()
        pos = torch.stack((torch.where(flip, 0.5 - raw_pos[:, 0], raw_pos[:, 0]), raw_pos[:, 1], torch.where(flip, 0.5 - raw_pos[:, 2], raw_pos[:, 2]), raw_pos[:, 3], relative_x, raw_pos[:, 5]), 1)
        features = torch.cat((visual, self.features_extractor.coordinates(pos * 8)), 1).reshape(batch, length, 160)
        if self.input_history is not None:
            if input_durations is None or tuple(input_durations.shape) != (batch, length, 3):
                raise ValueError('Enabled input history requires real contiguous recorded durations')
            duration = input_durations.to(features.device).float() / self.input_history['normalization']
            if self.facing_canonical:
                duration = torch.where(flip.reshape(batch, length, 1), duration[:, :, [1, 0, 2]], duration)
            if self.input_history['mode'] == 'zero':
                duration = torch.zeros_like(duration)
            features = features + self.input_duration_projection(duration)
        if self.facing_canonical:
            previous = torch.where(flip.reshape(batch, length), self.physical_mirror[previous.long()], previous)
        internal_previous = previous.long()
        if self.deployment_action_indices is not None:
            internal_previous = torch.as_tensor(self.deployment_action_indices, device=previous.device)[internal_previous]
        count = self.action_space.n if self.internal_action_count is None else int(self.internal_action_count)
        inputs = torch.cat((features, F.one_hot(internal_previous, count).float()), 2)
        if burn_in:
            with torch.no_grad():
                prefix, hidden = self.sequence_gru(inputs[:, :burn_in].detach(), hidden)
            tail, hidden = self.sequence_gru(inputs[:, burn_in:], hidden.detach())
            recurrent = torch.cat((prefix, tail), 1)
            features = torch.cat((features[:, :burn_in].detach(), features[:, burn_in:]), 1)
        else:
            recurrent, hidden = self.sequence_gru(inputs, hidden)
        combined = torch.cat((features, recurrent), 2)
        advantage = self.action_net(combined)
        q = self.value_net(combined) + advantage - advantage.mean(2, keepdim=True)
        if self.deployment_action_indices is not None:
            if self.facing_canonical:
                if self.internal_action_mirror is None:
                    raise ValueError('Missing original canonical action map')
                q = torch.where(flip.reshape(batch, length, 1), q[:, :, self.internal_action_mirror], q)
            self.last_internal_critic = q.mean(2)
            q = q[:, :, self.deployment_action_indices]
        elif self.facing_canonical:
            q = torch.where(flip.reshape(batch, length, 1), q[:, :, self.physical_mirror], q)
        return (q, hidden, combined) if return_features else (q, hidden)

    def reset_inference_state(self):
        self._sequence_hidden = None
        self._sequence_previous = None
        self._sequence_images = None
        self._sequence_input_durations = None
        self._input_durations_before = None
        self.inference_history_resets = 0

    def get_inference_state(self):
        if self._sequence_hidden is None:
            return None
        result = dict(hidden=self._sequence_hidden.detach().cpu().clone(), previous=self._sequence_previous.copy(), images=self._sequence_images.copy(), reset_count=self.inference_history_resets)
        if self.input_history is not None:
            result.update(input_durations=self._sequence_input_durations.copy(), input_durations_before=self._input_durations_before.copy())
        return result

    def set_inference_state(self, state):
        if state is None:
            self.reset_inference_state()
            return
        self._sequence_hidden = state['hidden'].clone().to(self.device)
        self._sequence_previous = state['previous'].copy()
        self._sequence_images = state['images'].copy()
        self.inference_history_resets = int(state['reset_count'])
        if self.input_history is not None:
            self._sequence_input_durations = state['input_durations'].copy()
            self._input_durations_before = state['input_durations_before'].copy()

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.set_training_mode(False)
        if state is not None:
            self.set_inference_state(state)
        tensor, vectorized = self.obs_to_tensor(observation)
        images = tensor['image'].cpu().numpy()
        batch = len(images)
        starts = continuity_reset(self._sequence_images, images)
        if episode_start is not None:
            starts |= np.asarray(episode_start, bool).reshape(-1)
        if self._sequence_hidden is None or self._sequence_hidden.shape[1] != batch:
            self._sequence_hidden = torch.zeros(1, batch, self.memory_hidden, device=self.device)
            self._sequence_previous = np.zeros(batch, np.int64)
            if self.input_history is not None:
                self._sequence_input_durations = np.zeros((batch, 3), np.uint8)
            starts[:] = True
        self._sequence_hidden[:, starts] = 0
        self._sequence_previous[starts] = 0
        duration_argument = {}
        if self.input_history is not None:
            self._sequence_input_durations[starts] = 0
            self._input_durations_before = self._sequence_input_durations.copy()
            self.last_input_durations_before = self._input_durations_before.copy()
            duration_argument['input_durations'] = torch.as_tensor(self._input_durations_before, device=self.device)[:, None]
        self.inference_history_resets += int(starts.sum())
        self.last_hidden_before = self._sequence_hidden.detach().clone()
        self.last_previous_before = self._sequence_previous.copy()
        with torch.no_grad():
            maps = self.encode_maps(tensor)
            q, hidden = self.sequence_values(maps[:, None], tensor['position'][:, None], torch.as_tensor(self._sequence_previous, device=self.device)[:, None], self._sequence_hidden, **duration_argument)
            canonical_q = q[:, 0]
            flip = (tensor['position'][:, 4] < 0) & (not self.physical_actions)
            physical_q = torch.where(flip[:, None], canonical_q[:, self.physical_mirror], canonical_q)
            canonical = canonical_q.argmax(1)
            if not deterministic:
                canonical = torch.where(torch.rand(batch, device=self.device) < 0.05, torch.randint(self.action_space.n, (batch,), device=self.device), canonical)
            actions = torch.where(flip, self.physical_mirror[canonical], canonical)
            self._sequence_previous = canonical.cpu().numpy().copy()
            self._sequence_hidden = hidden.detach()
            self._sequence_images = images.copy()
            if getattr(self, 'probability_policy', False):
                self.last_policy_logits = physical_q.detach()
            else:
                self.last_context_q_values = physical_q.detach()
            self.last_memory_values = physical_q.max(1).values.detach()
            self.last_memory_probabilities = torch.softmax(physical_q / 0.2, 1).detach()
            result = actions.cpu().numpy()
            if self.input_history is not None:
                self._record_input_durations(result)
        return (result if vectorized else result[0], None)

    def observe_and_force(self, observation, action):
        self.predict(observation, deterministic=True)
        tensor, _ = self.obs_to_tensor(observation)
        self.record_executed_actions(action, tensor['position'].cpu().numpy())

    def record_executed_actions(self, action, positions):
        """Keep recurrent action history faithful after training exploration."""
        actual = np.broadcast_to(np.asarray(action, np.int64), (len(self._sequence_previous),))
        flip = (np.asarray(positions)[:, 4] < 0) & (not self.physical_actions)
        mirror = self.physical_mirror.cpu().numpy()
        self._sequence_previous = np.where(flip, mirror[actual], actual).copy()
        if self.input_history is not None:
            self._record_input_durations(actual)

    def _record_input_durations(self, actual):
        self._sequence_input_durations = np.stack([advance_input_durations(before, self.physical_commands[int(action)], self.input_history['cap_native_frames']) for before, action in zip(self._input_durations_before, actual)])

class SequencePPOPolicy(SequenceQPolicy):
    """Same legal recurrent encoder and parameter capacity, PPO objectives.

    Only fresh, shared random parameters initialize this policy. Dueling
    scores become categorical logits; their action mean is the critic. The
    common baseline cancels in the categorical distribution. No learned Q
    estimate or replay return is treated as a PPO probability/value target.
    """
    probability_policy = True
    action_temperature = 1.0

    def extract_features(self, observations, features_extractor=None):
        return torch.cat((self.encode_maps(observations), observations['position']), 1)

    def sequence_step(self, features, hidden, previous, starts, flip, input_durations=None):
        hidden = hidden * (~starts[:, None]).float()
        previous = torch.where(starts, torch.zeros_like(previous), previous)
        if input_durations is not None:
            input_durations = torch.where(starts[:, None], torch.zeros_like(input_durations), input_durations)
        score, next_hidden = self.sequence_values(features[:, :-6][:, None], features[:, -6:][:, None], previous[:, None], hidden[None], input_durations=None if input_durations is None else input_durations[:, None])
        score = score[:, 0]
        physical_flip = flip & (not self.physical_actions)
        logits = torch.where(physical_flip[:, None], score[:, self.physical_mirror], score)
        critic = score.mean(1) if self.deployment_action_indices is None else self.last_internal_critic[:, 0]
        return (torch.distributions.Categorical(logits=logits / self.action_temperature), critic, next_hidden[0])

    def canonical_actions(self, actions, flip):
        return torch.where(flip, self.physical_mirror[actions], actions) if not self.physical_actions else actions

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        _, _ = super().predict(observation, state, episode_start, deterministic=True)
        distribution = torch.distributions.Categorical(logits=self.last_policy_logits / self.action_temperature)
        actions = distribution.probs.argmax(1) if deterministic else distribution.sample()
        tensor, vectorized = self.obs_to_tensor(observation)
        self.record_executed_actions(actions.cpu().numpy(), tensor['position'].cpu().numpy())
        self.last_memory_probabilities = distribution.probs.detach()
        self.last_memory_values = (self.last_policy_logits.mean(1) if self.deployment_action_indices is None else self.last_internal_critic[:, 0]).detach()
        result = actions.cpu().numpy()
        return (result if vectorized else result[0], None)

def sequence_q_kwargs(definition):
    return dict(features_extractor_class=VisibleHistoryEncoder, net_arch=dict(pi=[], vf=[]), action_mirror=list(range(len(definition['actions']))) if definition.get('physical_actions') and (not definition.get('facing_canonical')) else mirror_indices(definition['actions']), memory_hidden=definition.get('memory_hidden', 128), trainable_visual=definition.get('trainable_visual', False), physical_actions=definition.get('physical_actions', False), signed_relative_position=definition.get('signed_relative_position', False), input_history=definition.get('input_history'), facing_canonical=definition.get('facing_canonical', False), visual_normalization=definition.get('visual_normalization'), action_readout=definition.get('action_readout'), deployment_action_indices=definition.get('deployment_action_indices'), internal_action_count=definition.get('internal_action_count'), internal_action_mirror=definition.get('internal_action_mirror'), physical_commands=compiled(definition['actions']) if definition.get('input_history') is not None else None)

def construct_sequence_q(definition):
    import gymnasium as gym
    cls = SequencePPOPolicy if definition['kind'] == 'sequence_ppo' else SequenceQPolicy
    policy = cls(spaces(definition), gym.spaces.Discrete(len(definition['actions'])), lambda _: 0.0001, **sequence_q_kwargs(definition))
    if cls is SequencePPOPolicy:
        policy.action_temperature = float(definition.get('action_temperature', 1.0))
    return policy
POLICY_DEFINITION = {'observation_source': '"""Student-editable image input. Default: four 192 x 168 grayscale frames.\n\nOnly transform the supplied RGB image. Never operate the game or change rewards.\nThe coordinate branch still uses the six original self-first position values.\nSB3 divides these uint8 images by 255 once, inside its policy preprocessor.\n"""\nimport cv2\nimport numpy as np\n\nINPUT_CONFIG = {\'color\': \'gray\', \'width\': 96, \'height\': 84, \'stack_frames\': 4, \'raw_shape\': [224, 320, 3], \'crop_xyxy\': [0, 0, 256, 224], \'verify_right_padding\': True, \'black_threshold\': 5, \'max_nonblack_fraction\': 0.005}\n\n\ndef preprocess(rgb):\n    """One RGB uint8 frame -> (channels, height, width) uint8 array."""\n    a = np.asarray(rgb)\n    c = INPUT_CONFIG\n    if a.dtype != np.uint8 or tuple(a.shape) != tuple(c["raw_shape"]):\n        raise ValueError(f"Unexpected raw image: {a.shape} {a.dtype}; expected {c[\'raw_shape\']} uint8.")\n    x0, y0, x1, y1 = c["crop_xyxy"]\n    if c["verify_right_padding"] and x1 < a.shape[1]:\n        excluded = a[:, x1:, :]\n        fraction = float(np.mean(np.max(excluded, axis=2) > c["black_threshold"]))\n        if fraction > c["max_nonblack_fraction"]:\n            raise ValueError(\n                f"CROP_GUARD_FAILED: right-side nonblack fraction={fraction:.6f}. "\n                "The fixed viewport is not valid for this observed frame. "\n                "Stop and inspect the raw image; do not train with a silently changing crop."\n            )\n    view = a[y0:y1, x0:x1, :]\n    if c["color"] == "gray":\n        gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)\n        result = cv2.resize(gray, (c["width"], c["height"]), interpolation=cv2.INTER_AREA)[None, :, :]\n    else:\n        result = cv2.resize(view, (c["width"], c["height"]), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)\n    return np.ascontiguousarray(result, dtype=np.uint8)\n', 'observation_config': {'color': 'gray', 'width': 96, 'height': 84, 'stack_frames': 4, 'raw_shape': [224, 320, 3], 'crop_xyxy': [0, 0, 256, 224], 'verify_right_padding': True, 'black_threshold': 5, 'max_nonblack_fraction': 0.005}, 'kind': 'sequence_ppo', 'character': 'Ryu', 'actions': [{'name': 'NO_OP', 'segments': [{'keys': [], 'frames': 6}]}, {'name': 'LEFT', 'segments': [{'keys': ['LEFT'], 'frames': 6}]}, {'name': 'RIGHT', 'segments': [{'keys': ['RIGHT'], 'frames': 6}]}, {'name': 'DOWN', 'segments': [{'keys': ['DOWN'], 'frames': 6}]}, {'name': 'UP', 'segments': [{'keys': ['UP'], 'frames': 6}]}, {'name': 'UP_LEFT', 'segments': [{'keys': ['UP', 'LEFT'], 'frames': 6}]}, {'name': 'UP_RIGHT', 'segments': [{'keys': ['UP', 'RIGHT'], 'frames': 6}]}, {'name': 'DOWN_LEFT', 'segments': [{'keys': ['DOWN', 'LEFT'], 'frames': 6}]}, {'name': 'DOWN_RIGHT', 'segments': [{'keys': ['DOWN', 'RIGHT'], 'frames': 6}]}, {'name': 'A', 'segments': [{'keys': ['A'], 'frames': 6}]}, {'name': 'B', 'segments': [{'keys': ['B'], 'frames': 6}]}, {'name': 'C', 'segments': [{'keys': ['C'], 'frames': 6}]}, {'name': 'X', 'segments': [{'keys': ['X'], 'frames': 6}]}, {'name': 'Y', 'segments': [{'keys': ['Y'], 'frames': 6}]}, {'name': 'Z', 'segments': [{'keys': ['Z'], 'frames': 6}]}, {'name': 'NO_OP_X_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['X'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'NO_OP_Y_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['Y'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'NO_OP_Z_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['Z'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'NO_OP_A_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['A'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'NO_OP_B_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['B'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'NO_OP_C_tap', 'segments': [{'keys': [], 'frames': 2}, {'keys': ['C'], 'frames': 2}, {'keys': [], 'frames': 2}]}, {'name': 'LEFT_X_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'X'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'LEFT_Y_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'Y'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'LEFT_Z_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'Z'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'LEFT_A_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'A'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'LEFT_B_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'B'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'LEFT_C_tap', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['LEFT', 'C'], 'frames': 2}, {'keys': ['LEFT'], 'frames': 2}]}, {'name': 'RIGHT_X_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'X'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'RIGHT_Y_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'Y'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'RIGHT_Z_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'Z'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'RIGHT_A_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'A'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'RIGHT_B_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'B'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'RIGHT_C_tap', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'C'], 'frames': 2}, {'keys': ['RIGHT'], 'frames': 2}]}, {'name': 'DOWN_X_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'X'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'DOWN_Y_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'Y'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'DOWN_Z_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'Z'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'DOWN_A_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'A'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'DOWN_B_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'B'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'DOWN_C_tap', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'C'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}]}, {'name': 'UP_X_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'X'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_Y_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'Y'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_Z_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'Z'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_A_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'A'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_B_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'B'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_C_tap', 'segments': [{'keys': ['UP'], 'frames': 2}, {'keys': ['UP', 'C'], 'frames': 2}, {'keys': ['UP'], 'frames': 2}]}, {'name': 'UP_LEFT_X_tap', 'segments': [{'keys': ['UP', 'LEFT'], 'frames': 2}, {'keys': ['UP', 'LEFT', 'X'], 'frames': 2}, {'keys': ['UP', 'LEFT'], 'frames': 2}]}, {'name': 'UP_LEFT_Y_tap', 'segments': [{'keys': ['UP', 'LEFT'], 'frames': 2}, {'keys': ['UP', 'LEFT', 'Y'], 'frames': 2}, {'keys': ['UP', 'LEFT'], 'frames': 2}]}, {'name': 'UP_LEFT_Z_tap', 'segments': [{'keys': ['UP', 'LEFT'], 'frames': 2}, {'keys': ['UP', 'LEFT', 'Z'], 'frames': 2}, {'keys': ['UP', 'LEFT'], 'frames': 2}]}, {'name': 'UP_RIGHT_X_tap', 'segments': [{'keys': ['UP', 'RIGHT'], 'frames': 2}, {'keys': ['UP', 'RIGHT', 'X'], 'frames': 2}, {'keys': ['UP', 'RIGHT'], 'frames': 2}]}, {'name': 'UP_RIGHT_Y_tap', 'segments': [{'keys': ['UP', 'RIGHT'], 'frames': 2}, {'keys': ['UP', 'RIGHT', 'Y'], 'frames': 2}, {'keys': ['UP', 'RIGHT'], 'frames': 2}]}, {'name': 'UP_RIGHT_Z_tap', 'segments': [{'keys': ['UP', 'RIGHT'], 'frames': 2}, {'keys': ['UP', 'RIGHT', 'Z'], 'frames': 2}, {'keys': ['UP', 'RIGHT'], 'frames': 2}]}, {'name': 'NO_OP_PPP', 'segments': [{'keys': ['X', 'Y', 'Z'], 'frames': 2}, {'keys': [], 'frames': 4}]}, {'name': 'QCF_RIGHT_Y', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'Y'], 'frames': 2}]}, {'name': 'QCF_RIGHT_B', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'B'], 'frames': 2}]}, {'name': 'QCF_RIGHT_C', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT'], 'frames': 2}, {'keys': ['RIGHT', 'C'], 'frames': 2}]}, {'name': 'DP_RIGHT_X', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT', 'X'], 'frames': 2}]}, {'name': 'DP_RIGHT_Z', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT', 'Z'], 'frames': 2}]}, {'name': 'DP_RIGHT_B', 'segments': [{'keys': ['RIGHT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'RIGHT', 'B'], 'frames': 2}]}, {'name': 'QCF_LEFT_Y', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT'], 'frames': 2}, {'keys': ['LEFT', 'Y'], 'frames': 2}]}, {'name': 'QCF_LEFT_B', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT'], 'frames': 2}, {'keys': ['LEFT', 'B'], 'frames': 2}]}, {'name': 'QCF_LEFT_C', 'segments': [{'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT'], 'frames': 2}, {'keys': ['LEFT', 'C'], 'frames': 2}]}, {'name': 'DP_LEFT_X', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT', 'X'], 'frames': 2}]}, {'name': 'DP_LEFT_Z', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT', 'Z'], 'frames': 2}]}, {'name': 'DP_LEFT_B', 'segments': [{'keys': ['LEFT'], 'frames': 2}, {'keys': ['DOWN'], 'frames': 2}, {'keys': ['DOWN', 'LEFT', 'B'], 'frames': 2}]}], 'memory_hidden': 128, 'trainable_visual': True, 'physical_actions': True, 'signed_relative_position': True, 'tap_delay_frames': 2, 'input_history': {'mode': 'real', 'cap_native_frames': 120, 'normalization': 120}, 'action_temperature': 1.0, 'facing_canonical': True, 'action_readout': 'relu', 'deployment_action_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 69, 85, 88, 89, 91, 93, 95, 106, 109, 110, 112, 114, 116], 'internal_action_count': 125, 'internal_action_mirror': [0, 2, 1, 3, 4, 6, 5, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 21, 22, 23, 24, 25, 26, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 52, 53, 54, 55, 56, 45, 46, 47, 48, 49, 50, 63, 64, 65, 66, 67, 68, 57, 58, 59, 60, 61, 62, 69, 71, 70, 72, 73, 75, 74, 76, 78, 77, 79, 80, 82, 81, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]}

def build_model(env,config,seed,device='auto'):
    raise RuntimeError('Use the registered sequence trainer for this policy. The SB3 container does not identify its learning algorithm.')
