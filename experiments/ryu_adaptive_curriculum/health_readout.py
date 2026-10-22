"""Can the frozen legal representation read health from real recorded games?

Entire new diagnostic videos are held out. This is perception/readout evidence,
not a claim that an actor uses health or that a new fighting policy improved.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.league_adaptive.runtime import ROOT,load_actor,setup,write


def audit_orientation(directory):
    directory=Path(directory)
    with np.load(directory/'data.npz',allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    report=json.loads((directory/'summary.json').read_text());flips=[]
    for video in report['videos']:
        rows=json.loads((Path(video['video']).parent/'decisions.json').read_text())[4::2]
        if len(rows)!=video['observations']:raise RuntimeError('Readout pose/label alignment changed')
        flips.extend(r['opponent_x']<r['self_x'] for r in rows)
    flips=np.asarray(flips,bool);train=~data['validation'];test=data['validation'];health=data['health']
    canonical=np.where(flips[:,None],health[:,::-1],health)
    features=data['features'];gated=np.concatenate((features*(~flips)[:,None],features*flips[:,None]),1)
    versions=[('canonical_features_canonical_bar_targets',features,canonical),('features_with_legal_raw_side_interaction_absolute_targets',gated,health)]
    result=[]
    for name,array,labels in versions:
        x=torch.as_tensor(array,dtype=torch.float64);target=torch.as_tensor(labels,dtype=torch.float64)
        mean=x[train].mean(0);std=x[train].std(0).clamp_min(.01);z=(x-mean)/std
        z=torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1);regularizer=torch.eye(z.shape[1],dtype=z.dtype);regularizer[-1,-1]=0
        coefficients=torch.linalg.solve(z[train].T@z[train]+regularizer,z[train].T@target[train])
        predicted=(z[test]@coefficients).clamp(0,1);error=(predicted-target[test])*176;mask=(target[test,0]-target[test,1]).abs()*176>=16
        row=dict(input=name,mean_absolute_error_hp=error.abs().mean(0).tolist(),rmse_hp=error.square().mean(0).sqrt().tolist(),
                 lead_sign_accuracy_gap16=float(((predicted[:,0]>predicted[:,1])==(target[test,0]>target[test,1]))[mask].double().mean()))
        result.append(row);print('ORIENTATION_CORRECTED_HEALTH',row,flush=True)
    write(directory/'orientation_audit.json',dict(rows=result,validation_samples=int(test.sum()),
        correction='The original absolute P1/P2 readout conflates health decoding with left-right assignment, because the actor encoder mirrors its image according to the supplied relative coordinates. These tests align targets or restore that legal side information. A poor original readout does not by itself establish missing visual health information.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out');parser.add_argument('--audit-existing')
    parser.add_argument('--agent',default='artifacts/ryu_movement_20261022/learned_duration/checkpoint_24576.pt')
    args=parser.parse_args();setup()
    if args.audit_existing:audit_orientation(args.audit_existing);return
    if not args.out:raise ValueError('Specify an output directory')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    actor=load_actor(args.agent)
    folders=[ROOT/'models/counter_ryu_final/videos',ROOT/'models/counter_ryu_memory_candidate/videos',
             ROOT/'artifacts/ryu_direct_league_20261021/videos',ROOT/'artifacts/ryu_movement_20261022/diagnostic_videos']
    data={key:[] for key in ('hud','features','health','validation','trajectory')};provenance=[];seen=set()
    for folder in folders:
        for path in sorted(folder.glob('*/decisions.json')):
            report_path=path.parent/'summary.json';video=path.parent/'replay.mp4'
            if not report_path.exists() or not video.exists():continue
            report=json.loads(report_path.read_text());rival=report['opponent']
            if rival.get('character','Ryu')!='Ryu':continue
            identity=(report['agent'],json.dumps(rival,sort_keys=True),report['player'],report['offset'],report['native_frames'])
            if identity in seen:continue
            seen.add(identity);rows=json.loads(path.read_text());capture=cv2.VideoCapture(str(video))
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));offset=int(report['offset'])
            if not rows or abs(frame_count-(int(rows[-1]['frame'])-offset))>1:
                capture.release();raise RuntimeError('Recorded native-frame alignment changed: '+str(video))
            targets={int(row['frame'])-offset-1:(i,row) for i,row in enumerate(rows)}
            stack=deque(maxlen=4);observations=[];health=[]
            for frame_index in range(frame_count):
                ok,bgr=capture.read()
                if not ok:raise RuntimeError('Video decode ended before the labelled frame')
                if frame_index not in targets:continue
                i,row=targets[frame_index]
                # The recording contains a 2x nearest-neighbour native image
                # above its diagnostic footer. Undo that resize; use exactly
                # the legal 256x224 viewport and 96x84 grayscale transform.
                native=cv2.resize(bgr[:448],(320,224),interpolation=cv2.INTER_AREA)
                gray=cv2.cvtColor(native[:,:256],cv2.COLOR_BGR2GRAY)
                frame=cv2.resize(gray,(96,84),interpolation=cv2.INTER_AREA)
                stack.append(frame)
                if len(stack)<4 or i%2:continue
                image=np.stack(tuple(stack))
                x,y,ox,oy=[row[k] for k in ('self_x','self_y','opponent_x','opponent_y')]
                position=np.asarray([x,y,ox,oy,ox-x,oy-y],np.float32)/1024.
                observations.append(dict(image=image,position=position))
                own,other=max(0,row['self_hp'])/176.,max(0,row['opponent_hp'])/176.
                health.append((own,other) if report['player']==1 else (other,own))
            capture.release()
            features=[]
            with torch.no_grad():
                for begin in range(0,len(observations),64):
                    batch={k:np.stack([r[k] for r in observations[begin:begin+64]]) for k in ('image','position')}
                    tensor,_=actor.policy.obs_to_tensor(batch)
                    features.append(actor.policy.extract_features(tensor).cpu().numpy())
            if not features:continue
            images=np.stack([r['image'] for r in observations]);hud=images[:,:,:22,:]
            # Average pooling matches the standalone linear sensor model.
            pooled=hud.reshape(len(hud),4,11,2,48,2).mean((3,5)).reshape(len(hud),-1).astype(np.float32)/255.
            trajectory=len(provenance);validation='diagnostic_videos' in str(path)
            fields=dict(hud=pooled,features=np.concatenate(features),health=np.asarray(health,np.float32),
                        validation=np.full(len(hud),validation,bool),trajectory=np.full(len(hud),trajectory,np.int32))
            for key,value in fields.items():data[key].append(value)
            provenance.append(dict(video=str(video),observations=len(hud),validation=validation,player=report['player'],result=report['result']))
            print('HEALTH_DATA',video,len(hud),'heldout',validation,flush=True)
    data={key:np.concatenate(value) for key,value in data.items()}
    np.savez_compressed(out/'data.npz',**data)
    train=~data['validation'];test=data['validation'];target=torch.as_tensor(data['health'],dtype=torch.float64)
    if not train.any() or not test.any():raise RuntimeError('A whole-video train/validation split is required')
    reports=[]
    for name in ('features','hud'):
        x=torch.as_tensor(data[name],dtype=torch.float64);mean=x[train].mean(0);std=x[train].std(0).clamp_min(.01)
        z=(x-mean)/std;z=torch.cat((z,torch.ones(len(z),1,dtype=z.dtype)),1)
        regularizer=torch.eye(z.shape[1],dtype=z.dtype);regularizer[-1,-1]=0
        coefficients=torch.linalg.solve(z[train].T@z[train]+regularizer,z[train].T@target[train])
        predicted=(z[test]@coefficients).clamp(0,1);error=(predicted-target[test])*176
        informative=(target[test,0]-target[test,1]).abs()*176>=16
        row=dict(input=name,validation_samples=int(test.sum()),mean_absolute_error_hp=error.abs().mean(0).tolist(),
                 rmse_hp=error.square().mean(0).sqrt().tolist(),
                 lead_sign_accuracy_when_gap_at_least16hp=float(((predicted[:,0]>predicted[:,1])==(target[test,0]>target[test,1]))[informative].double().mean()),
                 informative_samples=int(informative.sum()))
        reports.append(row);print('HEALTH_READOUT',row,flush=True)
        if name=='hud':
            weight=(coefficients[:-1]/std[:,None]).T.float();bias=(coefficients[-1]-mean@(coefficients[:-1]/std[:,None])).float()
            torch.save(dict(weight=weight,bias=bias,input_shape=[4,22,96],pool=[2,2],targets='Absolute P1 and P2 health /176; clip predictions to[0,1].'),out/'hud_linear.pt')
    write(out/'summary.json',dict(agent=args.agent,training_samples=int(train.sum()),validation_samples=int(test.sum()),videos=provenance,rows=reports,
        scope='Linear ridge readouts from a fixed actor encoder versus a dedicated four-frame HUD crop, using decoded native recordings and exact recorded HP labels. New diagnostic videos are held out by whole match. Decoded pixels differ slightly from native observations; this is a perception diagnostic, not fighting-policy evaluation or proof that the actor uses the decodable information.'))


if __name__=='__main__':main()
