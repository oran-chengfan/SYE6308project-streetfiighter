"""Rule opponents for controlled tests; never the deployed learner.

Function copied from the existing project without changing its behavior.
"""
def choose(style, raw, player, count, actions, config=None):
    """A reactive six-frame option; direction is recomputed each decision."""
    config = config or {}
    if style == 'switching':
        schedule = config['styles']
        style = schedule[(count - 1) // int(config['switch_every']) % len(schedule)]
    dx = raw[f'p{3 - player}_x'] - raw[f'p{player}_x']
    distance = abs(dx)
    own_air = raw[f'p{player}_y'] < 184
    other_air = raw[f'p{3 - player}_y'] < 184
    face = 'R' if dx >= 0 else 'L'
    forward, back = ('RIGHT', 'LEFT') if dx >= 0 else ('LEFT', 'RIGHT')
    guard = 'DOWN_' + back
    phase = (count + int(config.get('phase', 0))) % int(config.get('period', 12))
    close = int(config.get('close', 62))
    far = int(config.get('far', 120))
    if style == 'neutral':
        name = 'NO_OP'
    elif style == 'advance_kick':
        name = forward if distance > 85 else 'C_tap' if phase % 3 else 'B_tap'
        if own_air:
            name = 'C_tap'
    elif style == 'jump_kick':
        name = 'UP_' + forward if distance > 70 else 'CROUCH_KICK_C'
        if own_air:
            name = 'C_tap'
    elif style == 'sweep_pressure':
        name = forward if distance > close else 'CROUCH_KICK_C' if phase % 4 else guard
    elif style == 'guard_counter':
        if other_air and distance < 100:
            name = f'SHORYUKEN_{face}_X'
        elif distance > far:
            name = forward
        elif distance > close:
            name = guard if phase < 6 else 'C_tap'
        else:
            name = 'CROUCH_KICK_B' if phase % 3 else guard
    elif style == 'zone_guard':
        if distance > far:
            name = f'HADOUKEN_{face}_Y' if phase % 4 else guard
        elif other_air or distance < 55:
            name = f'SHORYUKEN_{face}_X'
        else:
            name = back if phase < 6 else guard
    elif style == 'bait_counter':
        if other_air and distance < 105:
            name = guard if phase % 3 else f'SHORYUKEN_{face}_X'
        elif distance > far:
            name = forward
        elif distance > close:
            name = guard if phase < 8 else forward
        else:
            name = 'CROUCH_KICK_C' if phase % 4 else guard
    elif style == 'walk_guard':
        name = forward if distance > 80 else 'CROUCH_KICK_B' if phase >= 6 else guard
        if other_air:
            name = guard
    elif style == 'mixed_tactics':
        if other_air and distance < 95:
            name = f'SHORYUKEN_{face}_X'
        elif own_air:
            name = 'C_tap'
        elif distance > far:
            name = f'HADOUKEN_{face}_Y' if phase < 4 else 'UP_' + forward
        elif distance > close:
            name = forward if phase < 6 else guard
        else:
            name = 'CROUCH_KICK_C' if phase % 3 else 'B_tap'
    else:
        raise ValueError(style)
    if name not in actions and name.startswith('CROUCH_KICK_'):
        name = name[-1] + '_tap'
    return actions[name]
