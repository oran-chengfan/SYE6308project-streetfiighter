"""Shared roster, physical six-frame inputs and training-only input proposals.

No character or opponent identifier is supplied to the deployed network.
Long charges are repeated decisions, not extended environment steps.
"""
import numpy as np

CHARACTERS = ('Ryu','EHonda','Blanka','Guile','Ken','ChunLi','Zangief','Dhalsim',
              'MBison','Sagat','Balrog','Vega')
GRID = ((0,1,2,3,10,11),(4,5,6,7,9,8))
CELLS = {c:(r,k) for r,row in enumerate(GRID) for k,c in enumerate(row)}
PUNCHES = ('X','Y','Z')
KICKS = ('A','B','C')
DIRECTIONS = ((),('LEFT',),('RIGHT',),('DOWN',),('UP',),
              ('UP','LEFT'),('UP','RIGHT'),('DOWN','LEFT'),('DOWN','RIGHT'))


def advance_input_durations(previous,command,cap=120):
    """Continuous physical LEFT/RIGHT/DOWN keys, not native charge readiness."""
    counts=np.asarray(previous,dtype=np.int16).copy()
    if counts.shape!=(3,) or not 0<int(cap)<=255:raise ValueError('Expected3 bounded physical input durations')
    for keys in command:
        held=np.asarray([key in keys for key in ('LEFT','RIGHT','DOWN')])
        counts=np.minimum(np.where(held,counts+1,0),int(cap))
    return counts.astype(np.uint8)


def physical_actions(tap_delay=0):
    if tap_delay not in (0,2):raise ValueError('Supported in-decision press delays are0or2nativeframes')
    actions = []
    def add(name, segments):
        actions.append(dict(name=name,segments=[dict(keys=list(k),frames=n) for k,n in segments]))
    # The fixed course action ABI occupies indices0..14. Preserve it so an
    # exported CUSTOM_ACTIONS suffix has the same physical meaning as training.
    for direction in DIRECTIONS:add('_'.join(direction) or 'NO_OP',[(direction,6)])
    for button in KICKS+PUNCHES:add(button,[((button,),6)])
    for direction in DIRECTIONS:
        label='_'.join(direction) or 'NO_OP'
        for button in PUNCHES+KICKS:
            segments=([(direction,tap_delay)] if tap_delay else [])+[(direction+(button,),2),(direction,4-tap_delay)]
            add(label+'_'+button+'_tap',segments)
    for buttons,label in [(PUNCHES,'PPP'),(KICKS,'KKK')]:
        for direction in DIRECTIONS[:7]:
            add(('_'.join(direction) or 'NO_OP')+'_'+label,[(direction+buttons,2),(direction,4)])
    for forward,back in [('RIGHT','LEFT'),('LEFT','RIGHT')]:
        paths = {
            'QCF':(('DOWN',),('DOWN',forward),(forward,)),
            'DP':((forward,),('DOWN',),('DOWN',forward)),
            # Historical cardinal-sweep atom. Its label does not establish a
            # half-circle special: complete directions can span decisions.
            'HCF':((back,),('DOWN',),(forward,))}
        for motion,path in paths.items():
            # Motion without a premature attack permits a following UP+P/K
            # decision (rotation/charge release) and other longer sequences.
            add(f'{motion}_{forward}',[(part,2) for part in path])
            for button in PUNCHES+KICKS:
                add(f'{motion}_{forward}_{button}',[(path[0],2),(path[1],2),(path[2]+(button,),2)])
    return actions


def proposal_sequence(actions, raw, player, rng, style='mixed', length=None,charge_decisions=10,complete_half_circle=False):
    """Training/analysis only: legal input sequences, not an acting override.

    Shared by both arms' exploration. Opponent controllers respond again at
    each decision. No future opponent input is recorded or played back.
    """
    ids={a['name']:i for i,a in enumerate(actions)}
    dx=raw[f'p{3-player}_x']-raw[f'p{player}_x']
    front,back=('RIGHT','LEFT') if dx>=0 else ('LEFT','RIGHT')
    char=CHARACTERS[raw[f'p{player}_char']]
    attack=str(rng.choice(PUNCHES+KICKS))
    choice=int(rng.integers(7))
    if style=='guard':choice=1
    if style=='guile_boom':choice=3
    if style=='guile_flash':choice=4
    if style=='guile_hold':choice=1
    if choice==0: names=[front]*3+[front+'_'+attack+'_tap','NO_OP']
    elif choice==1:names=(['DOWN_'+back]*(length or int(charge_decisions)) if style=='guile_hold' else ['DOWN_'+back]*4+['NO_OP',back+'_C_tap'])
    elif choice==2:names=['UP_'+front,front+'_C_tap','NO_OP','DOWN_Y_tap']
    elif choice==3:
        names=([('DOWN_'+back)]*int(charge_decisions)+[front+'_'+str(rng.choice(PUNCHES))+'_tap','NO_OP']
               if char=='Guile' and style.startswith('guile_') else [f'QCF_{front}_{attack}','NO_OP']*3)
    elif choice==4:
        # Real native checks:48frames fail several charge moves,60 works;
        # Bison's kick release is verified at96 (60/72 failed in the probe).
        # No decision advances more than6.
        charge='DOWN_'+back
        duration=int(charge_decisions)
        if char=='MBison' and attack in KICKS:duration=max(duration,16)
        release=('UP_'+front) if char in ('ChunLi','Vega') or char in ('Guile','Blanka') and attack in KICKS else front
        names=[charge]*duration+[release+'_'+attack+'_tap','NO_OP']
        if char=='Guile' and style.startswith('guile_'):
            names=[charge]*duration+['UP_'+str(rng.choice(KICKS))+'_tap','NO_OP']
        if char=='Zangief':names=[front]*3+[f'HCF_{front}','UP_'+str(rng.choice(PUNCHES))+'_tap','NO_OP']
        if char=='Dhalsim' and complete_half_circle:
            # Both ports were checked at the same middle distance: this real
            # three-decision direction history produces the flame, unlike the
            # short cardinal atom. It remains exploration, never deployment.
            names=[back,'DOWN_'+back,f'QCF_{front}_Y','NO_OP']
    elif choice==5:names=[front+'_Z_tap',front+'_C_tap','NO_OP']*3
    else:names=[str(rng.choice(list(ids))),'NO_OP']*3
    if length is None:return [ids[n] for n in names]
    return [ids[n] for n in (names*((length+len(names)-1)//len(names)))[:length]]


def generic_opponent(config, raw, player, count, actions, previous=None):
    """Responsive legal character controls for initial training opponents.

    This is an opponent, never part of deployed model inference. Each call
    uses current distance/facing. Profiles distinguish charge/projectile tools;
    their effectiveness is measured, not inferred from a macro name.
    """
    dx=raw[f'p{3-player}_x']-raw[f'p{player}_x'];distance=abs(dx)
    front,back=('RIGHT','LEFT') if dx>=0 else ('LEFT','RIGHT')
    char=CHARACTERS[raw[f'p{player}_char']]
    period=int(config.get('period',24));phase=(count+int(config.get('phase',0)))%period
    style=config.get('style','mixed')
    if style=='reactive_intercept':
        if char not in ('Ryu','Sagat'):raise ValueError('The tested feedback proposal uses motion projectile/uppercut roles only')
        # Training opponent/proposal only. These coordinates and their prior
        # values are legal observations; no native move/AI/difficulty label or
        # future controller input supplies a decision. Existing profiles below
        # keep their exact behavior.
        own_y=raw[f'p{player}_y'];enemy_y=raw[f'p{3-player}_y']
        closing=0. if previous is None else abs(previous[f'p{3-player}_x']-previous[f'p{player}_x'])-distance
        if own_y<184:name='NO_OP'
        elif enemy_y<185:name=f'DP_{front}_X' if distance<=110 else back
        elif closing>14 and distance<125:name=back
        elif distance<45:name=front+'_Z_tap'
        elif distance<100:name='DOWN_C_tap'
        else:name=f'QCF_{front}_Y'
        return actions[name]
    if style=='switching':style=config.get('styles',['pressure','zoning'])[((count-1)//int(config.get('switch_every',47)))%2]
    if style=='guard':name='DOWN_'+back if phase<period-3 else 'DOWN_C_tap'
    elif distance<38:name=front+'_Z_tap' if phase%4<2 else 'DOWN_C_tap'
    elif style=='pressure':
        name=front if distance>70 else front+'_C_tap'
        if distance>100 and phase<3:name='UP_'+front
    elif char in ('Guile','MBison','Balrog','EHonda','Blanka','ChunLi','Vega'):
        hold=int(config.get('charge_decisions',8))
        if phase<hold+1:name='DOWN_'+back if char!='Balrog' or 'charge_decisions' in config else back
        elif phase==hold+1:
            release='UP' if char in ('ChunLi','Vega') else front
            button='B' if char in ('ChunLi','Vega') else 'Y'
            # Missing charge_decisions denotes the frozen initial screen's
            # original input profile, retained for exact old-condition replay.
            if 'charge_decisions' not in config:
                release='UP' if char in ('ChunLi','Blanka','Vega') else front
                button='B' if char in ('MBison','ChunLi','Vega') else 'Y'
            name=release+'_'+button+'_tap'
        elif distance>95:name=front
        else:name='DOWN_C_tap' if phase%2 else 'NO_OP'
    elif char in ('Ryu','Ken','Sagat','Dhalsim'):
        name=f'QCF_{front}_Y' if phase<3 else 'DOWN_'+back
        if distance<70:name=f'DP_{front}_Z' if phase<5 else 'DOWN_C_tap'
    else:name=front if distance>60 else front+'_Z_tap'
    return actions[name]
