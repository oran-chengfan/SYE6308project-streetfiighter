"""Identify actual intervened inputs in completed native prefix trajectories."""
import argparse,json,math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();folder=Path(a.data)
    report=json.loads((folder/'summary.json').read_text())
    lengths=dict(baseline=0,walk4=4,walk8=8,retreat8=8,retreat16=16,jump_strike=4,walk_sweep=5,guard_fireball=9,wait8=8)
    with np.load(folder/'data.npz',allow_pickle=False) as data:
        start=data['transition_start'].copy();end=data['transition_end'].copy();done=data['transition_done'].copy();win=data['transition_win'].copy()
    intervention=np.zeros(len(start),bool);beneficial=np.zeros(len(start),bool);cursor=0
    for root in report['roots']:
        if not root['baseline_reproduced']:raise ValueError('Unverified baseline')
        base=root['branches'][0]['result']
        for branch in root['branches']:
            count=math.ceil((branch['native_frames']-root['offset'])/6);last=cursor+count
            if last>len(start) or not done[cursor] or done[cursor+1:last].any():raise ValueError('Native trajectory boundary differs from recorded native frames')
            if count>1 and not np.array_equal(end[cursor+1:last],start[cursor:last-1]):raise ValueError('Native reversed trajectory edges do not link')
            positions=np.arange(count-1,-1,-1)
            mask=(positions>=root['root'])&(positions<root['root']+lengths[branch['plan']])
            intervention[cursor:last]=mask
            beneficial[cursor:last]=mask&(branch['result']=='win')&(base!='win')
            if not np.all(win[cursor:last]==(branch['result']=='win')):raise ValueError('Native outcome labels differ')
            cursor=last
    if cursor!=len(start):raise ValueError('Unaccounted recovery transitions')
    np.savez_compressed(folder/'prefix_annotations.npz',start=start,end=end,intervention=intervention,beneficial=beneficial)
    output=dict(transitions=len(start),intervention_transitions=int(intervention.sum()),outcome_improving_prefix_transitions=int(beneficial.sum()),
                scope='Actual input steps inside the complete intervention only. Improvement means a nonwinning baseline and a winning whole-prefix continuation. No claim that each isolated input is optimal.')
    (folder/'prefix_summary.json').write_text(json.dumps(output,indent=2)+'\n');print(output)


if __name__=='__main__':main()
