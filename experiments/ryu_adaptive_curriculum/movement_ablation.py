"""Ablate whole requested walking options in frozen complete games.

This is a diagnostic intervention, not a deployed restriction or a training
reward. The actor remembers the actual substituted input and its duration.
"""
import argparse
import copy
import json
from pathlib import Path

from experiments.league_adaptive.runtime import load_actor,setup,write
from experiments.league_adaptive.environment import FightEnv,family


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--agent',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--pool',default='configs/focus_pool.json')
    parser.add_argument('--case',action='append',nargs=3,metavar=('OPPONENT','PORT','OFFSET'))
    args=parser.parse_args();setup()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);actor=load_actor(args.agent)
    rivals=json.loads(Path(args.pool).read_text());env=FightEnv(actor.definition,rivals)
    ids={a['name']:i for i,a in enumerate(actor.actions)};rows=[]
    requested=args.case or [('tactical_walk_guard',1,12),('heldout_fireball_timing',1,0),('strong_response',2,0)]
    cases=[(next(i for i,r in enumerate(rivals) if r['id']==name),int(port),int(offset)) for name,port,offset in requested]
    try:
        for index,port,offset in cases:
            env.reset(options=dict(opponent=index,player=port,offset=offset));actor.reset()
            saved=env.save();actor_state=copy.deepcopy(actor.get_state())
            baseline=None
            for intervention in ('original','neutral_instead_of_walk','uppercut_instead_of_walk'):
                obs=env.restore(saved);actor.set_state(copy.deepcopy(actor_state));replacements=0;walk_calls=0;ground_motion=0
                while not env.ended:
                    before=copy.deepcopy(actor.get_state());raw=dict(env.game.raw)
                    action=int(actor.predict(obs,True)[0]);name=actor.actions[action]['name']
                    option_start=bool(actor.policy.last_option_starts[0]) if hasattr(actor.policy,'last_option_starts') else True
                    if option_start and name in ('LEFT','RIGHT') and intervention!='original':
                        duration=int(actor.policy.last_duration_choices[0]) if hasattr(actor.policy,'last_duration_choices') else 1
                        dx=raw[f'p{3-port}_x']-raw[f'p{port}_x']
                        replacement='NO_OP' if intervention=='neutral_instead_of_walk' else f"SHORYUKEN_{'R' if dx>=0 else 'L'}_X"
                        action=ids[replacement];actor.set_state(before)
                        if hasattr(actor.policy,'observe_and_force_option'):
                            actor.policy.observe_and_force_option(obs,action,duration)
                        elif before is not None:
                            raise ValueError('Stateful policies require an explicit actual-input memory update')
                        replacements+=1
                    actual_name=actor.actions[action]['name'];walk_calls+=int(actual_name in ('LEFT','RIGHT'))
                    obs,_,_,_,info=env.step(action)
                    if actual_name in ('LEFT','RIGHT') and raw[f'p{port}_y']==192 and env.game.raw[f'p{port}_y']==192 and not info['damage_taken']:
                        ground_motion+=abs(env.game.raw[f'p{port}_x']-raw[f'p{port}_x'])
                row=dict(opponent=rivals[index]['id'],player=port,offset=offset,intervention=intervention,replaced_whole_options=replacements,
                         executed_walking_calls=walk_calls,ground_displacement_during_walking_without_hp_loss=ground_motion,
                         **{k:info[k] for k in ('result','self_hp','opponent_hp','native_frames','end_reason')})
                if baseline is None:baseline=row
                if not replacements and any(row[k]!=baseline[k] for k in ('result','self_hp','opponent_hp','native_frames')):
                    raise RuntimeError('Negative control with no replaced options changed')
                rows.append(row);write(out/'progress.json',dict(rows=rows));print('MOVEMENT_ABLATION',json.dumps(row),flush=True)
        write(out/'summary.json',dict(agent=args.agent,rows=rows,native_frames=env.native_frames,
            scope='Same weights and starts; replace every requested whole walking option by neutral or a facing uppercut of the same duration, with corrected own actual-input memory. Measures this policy-level intervention, not a single-action effect or learned-agent performance. Ground-displacement count is diagnostic, not a move detector.'))
    finally:env.close()


if __name__=='__main__':main()
