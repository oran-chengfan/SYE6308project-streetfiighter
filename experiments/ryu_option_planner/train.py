"""Fit an ensemble of paired option advantages from real complete outcomes."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.league_adaptive.runtime import setup,write
from experiments.ryu_direct_league.policy import save
from experiments.ryu_option_planner.policy import construct_planner


def load_roots(folder):
    files=sorted(Path(folder).glob('shard_*/condition_*.npz'))
    seed=Path(folder)/'seed36.npz'
    if seed.exists():files=[seed]+files
    parts=[]
    for file in files:
        with np.load(file,allow_pickle=False) as d:parts.append({key:d[key].copy() for key in d.files})
    if not parts:raise ValueError('No completed paired native roots')
    data={key:np.concatenate([d[key] for d in parts]) for key in parts[0]};keep=[];known={}
    for i in range(len(data['lengths'])):
        key=(str(data['opponent'][i]),int(data['player'][i]),int(data['offset'][i]),int(data['root_decision'][i]))
        if key in known:
            old=known[key]
            if not np.array_equal(data['utility'][old],data['utility'][i]) or not np.allclose(data['margin'][old],data['margin'][i]):raise ValueError('Duplicate native root outcomes differ')
        else:known[key]=i;keep.append(i)
    return {key:value[keep] for key,value in data.items()},files


def context(data,indices,offsets):
    selected=[]
    for i in indices:
        n=int(data['lengths'][i]);selected.append([n-1-age for age in sorted(offsets,reverse=True) if age<n])
    counts=np.asarray([len(v) for v in selected]);width=int(counts.max());batch=len(indices)
    arrays=[np.zeros((batch,width,3584),np.float16),np.zeros((batch,width,6),np.float32),np.zeros((batch,width,12),np.float32),np.zeros((batch,width),np.int64)]
    for b,(i,positions) in enumerate(zip(indices,selected)):
        for dest,key in zip(arrays,('maps','positions','extras','previous')):dest[b,:len(positions)]=data[key][i,positions]
    return tuple(torch.as_tensor(v) for v in arrays)+(torch.as_tensor(counts),)


def policy_gain(prediction,utility,beta=1.,margin=.2):
    advantage=prediction[:,:,1:];score=advantage.mean(1)-beta*advantage.std(1)-margin
    best=score.argmax(1)+1;chosen=np.where(score.max(1)>0,best,0);gain=utility[np.arange(len(chosen)),chosen]-utility[:,0]
    return dict(mean_gain=float(gain.mean()),win_rate_change=float(gain.mean()/2),switches=int((chosen>0).sum()),
                improved=int((gain>0).sum()),worsened=int((gain<0).sum()),roots=len(chosen))


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--updates',type=int,default=2400);p.add_argument('--seed',type=int,default=3881);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--heads-only',action='store_true')
    p.add_argument('--semantic-features',action='store_true')
    a=p.parse_args();setup();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=torch.load(a.initial,map_location='cpu',weights_only=False);definition=copy.deepcopy(source['definition'])
    if a.semantic_features:definition['planner_semantics']=True
    policy=construct_planner(definition);state=policy.state_dict()
    for key,value in source['state_dict'].items():
        if key.startswith('planner_heads.') and key.endswith('.0.weight') and state[key].shape!=value.shape:
            state[key].zero_();state[key][:,:value.shape[1]]=value
        else:state[key]=value.clone()
    policy.load_state_dict(state,strict=True)
    if a.heads_only:
        for name,parameter in policy.named_parameters():parameter.requires_grad_(name.startswith('planner_heads.'))
        policy.optimizer=torch.optim.Adam([v for v in policy.parameters() if v.requires_grad],lr=a.lr)
    for group in policy.optimizer.param_groups:group['lr']=a.lr
    data,files=load_roots(a.data);groups=np.unique(data['opponent']);rng.shuffle(groups)
    if len(groups)<3:raise ValueError('At least three opponent groups are required for a grouped validation')
    validation_groups=groups[:max(1,len(groups)//4)];validation=np.flatnonzero(np.isin(data['opponent'],validation_groups));training=np.flatnonzero(~np.isin(data['opponent'],validation_groups))
    targets=np.concatenate((data['utility'][:,:1],data['utility'][:,1:]-data['utility'][:,:1]),1).astype(np.float32)
    group_names=np.unique(data['opponent']);bootstrap=rng.poisson(1.,(policy.planner_ensemble,len(group_names)))
    train_group_indices=[int(np.flatnonzero(group_names==v)[0]) for v in np.unique(data['opponent'][training])]
    for e in range(len(bootstrap)):
        if not bootstrap[e,train_group_indices].any():bootstrap[e,int(rng.choice(train_group_indices))]=1
    group_index=np.asarray([int(np.flatnonzero(group_names==v)[0]) for v in data['opponent']]);history=[];best=None;started=time.monotonic()
    write(out/'config.json',dict(arguments=vars(a),files=[str(f) for f in files],roots=len(targets),training_roots=len(training),validation_roots=len(validation),validation_opponents=validation_groups.tolist(),
         model='Retained learned PPO motor controller plus trainable legal-history GRU/encoder and five bootstrap option-outcome heads.',
         target='Predict baseline win(+1/nonwin-1) and paired complete-option outcome differences. Every queried option followed by the same retained continuation. No fabricated unqueried action labels.',
         selection='Grouped heldout-opponent development roots; conditional option gain is not complete-agent win rate. Complete native-match acceptance remains mandatory.'))

    feature_cache=None
    if a.heads_only:
        features=[];policy.set_training_mode(False)
        with torch.no_grad():
            for begin in range(0,len(targets),32):
                features.append(policy.planner_features(*context(data,np.arange(begin,min(begin+32,len(targets))),definition['context_offsets'])))
        feature_cache=torch.cat(features)

    def predictions(indices):
        policy.set_training_mode(False);result=[]
        with torch.no_grad():
            for begin in range(0,len(indices),32):
                ids=indices[begin:begin+32]
                result.append((policy.values_from_features(feature_cache[ids]) if feature_cache is not None else policy.option_values(*context(data,ids,definition['context_offsets']))).numpy())
        return np.concatenate(result)

    train_groups={str(g):training[data['opponent'][training]==g] for g in np.unique(data['opponent'][training])};train_names=list(train_groups)
    for iteration in range(a.updates):
        if (out/'STOP').exists():break
        indices=np.asarray([int(rng.choice(train_groups[train_names[int(rng.integers(len(train_names)))]])) for _ in range(32)])
        policy.set_training_mode(True);value=policy.values_from_features(feature_cache[indices]) if feature_cache is not None else policy.option_values(*context(data,indices,definition['context_offsets']))
        desired=torch.as_tensor(targets[indices]);weight=torch.as_tensor(bootstrap[:,group_index[indices]].T,dtype=torch.float32)
        per=(value[:,:,0]-desired[:,None,0]).square()+ (value[:,:,1:]-desired[:,None,1:]).square().mean(2)
        loss=(per*weight).sum()/weight.sum().clamp_min(1)
        policy.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([v for v in policy.parameters() if v.requires_grad],1.);policy.optimizer.step()
        if (iteration+1)%200==0 or iteration+1==a.updates:
            pred=predictions(validation);gain=policy_gain(pred,data['utility'][validation],definition['planner_beta'],definition['planner_margin'])
            row=dict(updates=iteration+1,seconds=time.monotonic()-started,loss=float(loss.detach()),validation=gain,validation_mse=float(((pred-targets[validation,None,:])**2).mean()))
            history.append(row);write(out/'live.json',row);print('OPTION_PLANNER_FIT',json.dumps(row),flush=True)
            # Select by heldout paired gain; ties prefer lower mean prediction error.
            rank=(gain['mean_gain'],-row['validation_mse'])
            if best is None or rank>best[0]:best=(rank,{k:v.detach().clone() for k,v in policy.state_dict().items()},iteration+1)
    if best is None:raise RuntimeError('No fitted checkpoint')
    policy.load_state_dict(best[1],strict=True);save(out/'agent.pt',policy,definition,dict(stage='paired_option_advantages',selected_update=best[2],roots=len(targets),validation_groups=validation_groups.tolist()))
    validation_pred=predictions(validation);grid=[]
    for beta in (0.,.5,1.,2.):
        for margin in (0.,.2,.5):grid.append(dict(beta=beta,margin=margin,**policy_gain(validation_pred,data['utility'][validation],beta,margin)))
    write(out/'result.json',dict(status='FIT_REQUIRES_COMPLETE_NATIVE_MATCHES',roots=len(targets),selected_update=best[2],history=history,calibration_grid=grid,
         scope='Conditional paired outcomes on grouped development roots. Repeated interventions by the deployed planner and unseen-opponent robustness must be evaluated in full native games.'))


if __name__=='__main__':main()
