"""Test observable history continuity under the official inference call pattern."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.league_adaptive.runtime import load_actor,setup,write
from experiments.ryu_direct_league.environment import RyuEnv


def continuity_reset(previous,current):
    if previous is None or previous.shape!=current.shape:return np.ones(len(current),bool)
    return ~np.all(previous[:,1:]==current[:,:-1],axis=(1,2,3))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--agent',required=True);parser.add_argument('--pool',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();setup();actor=load_actor(args.agent);pool=json.loads(Path(args.pool).read_text());env=RyuEnv(actor.definition,pool,0,gamma=1.)
    c=actor.definition['observation_config'];previous=np.zeros((1,4,c['height'],c['width']),np.uint8);rows=[]
    false_resets=0;missed_starts=0;decisions=0
    try:
        for rival_index,rival in enumerate(pool):
            for player in ((1,) if rival['kind']=='cpu' else (1,2)):
                for offset in ((0,) if rival['kind']=='cpu' else (0,12)):
                    obs,_=env.reset(options=dict(opponent=rival_index,player=player,offset=offset));first=True;actor.reset()
                    while True:
                        current=obs['image'][None];reset=bool(continuity_reset(previous,current)[0])
                        if first and not reset:missed_starts+=1
                        if not first and reset:false_resets+=1
                        previous=current.copy();decisions+=1
                        obs,_,done,truncated,info=env.step(int(actor.predict(obs,True)[0]))
                        if done or truncated:break
                        first=False
                    rows.append({k:info[k] for k in ('opponent','player','offset','result','native_frames')})
        static=np.full_like(previous,97)
        static_resets=int(sum(bool(continuity_reset(static,static)[0]) for _ in range(32)))
        # This is a genuine information boundary, retained explicitly rather
        # than treating any four identical frames as a reliable start signal.
        indistinguishable_restart=not bool(continuity_reset(static,static)[0])
        write(args.out,dict(games=len(rows),decisions=decisions,missed_observed_full_match_starts=missed_starts,
            unexpected_within_match_resets=false_resets,synthetic_static_scene_false_resets=static_resets,
            identical_observation_restart_cannot_be_identified=indistinguishable_restart,rows=rows,
            status='TESTED_FULL_MATCH_CALLS_PASS' if not (missed_starts or false_resets or static_resets) else 'FAILED',
            scope='Legal image-stack continuity only, with an initial zero-image warmup. Tested complete native games and static-frame negative control. An external reset to an identical observation remains indistinguishable without a protocol signal; this is not a formal guarantee for arbitrary resets.'))
        print('HISTORY_CONTINUITY',len(rows),decisions,missed_starts,false_resets,static_resets,flush=True)
    finally:env.close();actor.close()


if __name__=='__main__':main()
