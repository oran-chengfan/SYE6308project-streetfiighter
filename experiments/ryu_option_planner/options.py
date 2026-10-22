PLANS=[('baseline',[]),('walk4',[('toward',4)]),('walk8',[('toward',8)]),('retreat8',[('away',8)]),('retreat16',[('away',16)]),
       ('jump_strike',[('jump',3),('kick',1)]),('walk_sweep',[('toward',4),('sweep',1)]),
       ('guard_fireball',[('guard',8),('fireball',1)]),('wait8',[('neutral',8)])]


def plan_input(mode,dx,names):
    forward='RIGHT' if dx>=0 else 'LEFT';back='LEFT' if dx>=0 else 'RIGHT'
    name=(forward if mode=='toward' else back if mode=='away' else 'UP_'+forward if mode=='jump' else 'DOWN_'+back if mode=='guard'
          else 'C_tap' if mode=='kick' else 'CROUCH_KICK_C' if mode=='sweep' else f'HADOUKEN_{"R" if forward=="RIGHT" else "L"}_Y' if mode=='fireball' else 'NO_OP')
    return names[name]
