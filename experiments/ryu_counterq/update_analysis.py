"""Captured-update attribution and bounded action-readout feasibility.

Parameter interventions and explicitly marked supervised fits share the
existing policy/game APIs. This is not another RL trainer or evaluator;
native complete games remain in league_adaptive.evaluate.
"""
import argparse
import copy
import time
from pathlib import Path

import torch

from experiments.league_adaptive.runtime import ROOT,Actor,load_actor,setup,write
from experiments.league_adaptive.evaluate import evaluate,exact_opponent_pool
from experiments.ryu_counterq.policy import construct_sequence_q
from experiments.ryu_counterq.replay import SequenceReplay,full_values,episode_values,audit_success_history_labels,same_episode_condition
from experiments.ryu_counterq.stabilization import read
from experiments.ryu_counterq.training_state import restore_optimizer
from experiments.ryu_direct_league.policy import save


def readout_data(policy,episodes,verification=None):
    """Current legal raw histories, not cached teacher recurrent states."""
    features=[];targets=[];weights=[];slices=[];offset=0;reconstruction_error=0.
    logit_scale=1.;probability_error=0.;greedy_changes=0;full_error=0.;head_probability_error=0.
    temperature=float(policy.action_temperature) if getattr(policy,'probability_policy',False) else 1.
    with torch.no_grad():
        for episode in episodes:
            parts=[];hidden=None;n=len(episode['actions'])
            for first in range(0,n,32):
                scores,hidden,combined=episode_values(policy,episode,first,min(first+32,n),hidden,return_features=True)
                head=policy.action_net(combined)
                direct=policy.value_net(combined)+head-head.mean(2,keepdim=True)
                if policy.facing_canonical:
                    flip=torch.as_tensor(episode['positions'][first:min(first+32,n),4]<0)[None,:,None]
                    direct=torch.where(flip,direct[:,:,policy.physical_mirror],direct)
                    head=torch.where(flip,head[:,:,policy.physical_mirror],head)
                reconstruction_error=max(reconstruction_error,float(((scores-scores.mean(2,keepdim=True))-
                    (head-head.mean(2,keepdim=True))).abs().max()))
                logit_scale=max(logit_scale,float(scores.abs().max()),float(direct.abs().max()))
                full_error=max(full_error,float((scores-direct).abs().max()))
                probability_error=max(probability_error,float(((scores/temperature).softmax(2)-(direct/temperature).softmax(2)).abs().max()))
                head_probability_error=max(head_probability_error,float(((scores/temperature).softmax(2)-(head/temperature).softmax(2)).abs().max()))
                greedy_changes+=int((scores.argmax(2)!=head.argmax(2)).sum())
                parts.append(combined[0])
            minimum=episode.get('supervision_start',0);x=torch.cat(parts)[minimum:]
            y=torch.as_tensor(episode['actions'][minimum:]).long()
            if policy.facing_canonical:
                y=torch.where(torch.as_tensor(episode['positions'][minimum:n,4]<0),policy.physical_mirror[y],y)
            features.append(x);targets.append(y);weights.append(torch.full((len(y),),1./(len(episodes)*len(y))))
            slices.append(dict(first=offset,last=offset+len(y),opponent=episode['opponent_config'],
                player=episode['player'],offset=episode['offset'],supervision_start=minimum,
                source=episode['policy_version']))
            offset+=len(y)
    tolerance=8*torch.finfo(torch.float32).eps*logit_scale
    check=dict(centered_logit_absolute_error=reconstruction_error,logit_scale=logit_scale,
        scale_aware_roundoff_tolerance=tolerance,maximum_probability_error=probability_error,
        full_dueling_reconstruction_error=full_error,head_only_probability_error=head_probability_error,
        probability_temperature=temperature,greedy_action_changes=greedy_changes)
    if verification is not None:verification.update(check)
    if reconstruction_error>tolerance or full_error>tolerance or probability_error>1e-6 or greedy_changes:
        raise RuntimeError(f'Readout reconstruction changes the physical policy: {check}')
    return torch.cat(features).double(),torch.cat(targets),torch.cat(weights).double(),slices,reconstruction_error


def readout_geometry(spec,root):
    """Inspect exact feature aliases using existing torch, without a solver."""
    from collections import Counter
    rule=spec['readout_geometry'];out=root/rule['phase'];out.mkdir(exist_ok=False)
    source=torch.load(ROOT/rule['source'],map_location='cpu',weights_only=False)
    fitted=torch.load(ROOT/rule['fitted'],map_location='cpu',weights_only=False)
    saved=torch.load(ROOT/rule['bank'],map_location='cpu',weights_only=False)
    policy=construct_sequence_q(source['definition']);policy.load_state_dict(source['state_dict']);policy.set_training_mode(False)
    raw,target,_,_,error=readout_data(policy,list(saved.get('bank',saved)['episodes'].values()))
    if any(not torch.equal(v,fitted['state_dict'][k]) for k,v in source['state_dict'].items() if not k.startswith('action_net.')):
        raise ValueError('Geometry comparison requires identical feature and recurrent parameters')
    policy.load_state_dict(fitted['state_dict'])
    with torch.no_grad():predicted=policy.action_net(raw.float()).argmax(1)
    _,inverse=torch.unique(raw.float(),dim=0,return_inverse=True)
    low=torch.full((int(inverse.max())+1,),policy.action_space.n,dtype=torch.long)
    high=torch.full_like(low,-1);low.scatter_reduce_(0,inverse,target,reduce='amin');high.scatter_reduce_(0,inverse,target,reduce='amax')
    conflicting=low!=high
    confusions=Counter(tuple(sorted((int(a),int(b)))) for a,b in zip(target,predicted) if a!=b)
    value=dict(exact_feature_groups=len(low),exact_groups_with_conflicting_actions=int(conflicting.sum()),
        actions_in_conflicting_groups=int(conflicting[inverse].sum()),
        largest_current_confusions=[dict(actions=pair,names=[source['definition']['actions'][a]['name'] for a in pair],count=count)
            for pair,count in confusions.most_common(rule['maximum_pairs'])],
        reconstruction_error=error,native_frames=0,optimizer_updates=0,
        unavailable_linear_solver='The planned scipy linear program was not executed because scipy is absent. No dependency installed; no linear infeasibility claim.',
        scope='Exact current-feature equality only. No exact conflict does not establish linear separability or prove capacity sufficiency; fitted errors and complete games remain the evidence.')
    write(out/'summary.json',value);print('READOUT_GEOMETRY',value,flush=True);return value


def readout_feasibility(spec,root,variant=None):
    """Bounded existing or explicitly converted readout fit, then full games.

    Re-encode all raw legal histories with the frozen current backbone. This
    isolates readout fitting; it is not PPO, a Q target or a new RL trainer.
    """
    rule=dict(spec['readout_feasibility'])
    if variant is not None:rule.update(rule['variants'][variant])
    out=root/rule['phase']
    if (out/'summary.json').exists():return read(out/'summary.json')
    out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'fit.json').exists() and (out/'agent.pt').exists():
        previous=read(out/'configuration.json')
        fixed=lambda r:{k:v for k,v in r.items() if k not in ('variants','evaluation_native_frame_cap')}
        if fixed(previous)!=fixed(rule) or rule['evaluation_native_frame_cap']<previous['evaluation_native_frame_cap']:
            raise ValueError('Evaluation-only resume changed the completed training declaration')
        saved=torch.load(out/'bank.pt' if (out/'bank.pt').exists() else ROOT/rule['bank'],map_location='cpu',weights_only=False)
        episodes=list(saved.get('bank',saved)['episodes'].values())
        write(out/'evaluation_resume.json',dict(previous_native_frame_cap=previous['evaluation_native_frame_cap'],
            native_frame_cap=rule['evaluation_native_frame_cap'],optimizer_updates=0,weights_changed=False,
            reason='Complete the declared enlarged continuous-game panel using the already frozen fitted checkpoint.'))
        return evaluate_readout_endpoint(spec,rule,out,episodes,read(out/'fit.json'),resume=True)
    if any((out/name).exists() for name in ('agent.pt','fit_progress.json','initial_actor.pt')):
        raise RuntimeError('Preserve partial fitted results; do not silently repeat their optimization')
    configuration=out/'configuration.json'
    if configuration.exists() and read(configuration)!=rule:raise ValueError('Readout declaration differs from the interrupted preflight')
    write(configuration,rule)
    source=torch.load(ROOT/rule['source'],map_location='cpu',weights_only=False)
    saved=torch.load(ROOT/rule['bank'],map_location='cpu',weights_only=False)
    state=saved.get('bank',saved)
    if rule.get('verified_episode_sources') or rule.get('independent_success_sources') or rule.get('bank_capacity'):
        capacity,byte_capacity,burn,learn,nstep,alpha=state['settings']
        capacity=rule.get('bank_capacity',{}).get('transitions',capacity)
        byte_capacity=rule.get('bank_capacity',{}).get('bytes',byte_capacity)
        bank=SequenceReplay(capacity=capacity,byte_capacity=byte_capacity,burn=burn,learn=learn,nstep=nstep,alpha=alpha)
        bank.load_state_dict(state,allow_capacity_expansion=bool(rule.get('bank_capacity')));prepared=[]
        preparation=dict(original_settings=list(state['settings']),current_settings=list(bank.state_dict()['settings']),
            native_frames=0,source_acquisition='Previously recorded real full games; their native collection costs are separate.')
        original_conditions=list(bank.episodes.values())
        for acquisition in rule.get('independent_success_sources',[]):
            path=acquisition['path'] if isinstance(acquisition,dict) else acquisition
            scope=acquisition.get('scope','all') if isinstance(acquisition,dict) else 'all'
            if scope not in ('all','pvp'):raise ValueError('Unknown extra success acquisition scope')
            actual=torch.load(ROOT/path,map_location='cpu',weights_only=False)['episodes']
            if scope=='pvp':actual=[e for e in actual if e['opponent_config']['kind']!='cpu']
            preparation.setdefault('success_sources',[]).append(dict(path=path,scope=scope,recorded_games=len(actual)))
            preparation.setdefault('independent_success_refresh',[]).extend(bank.refresh_self_successes(actual,0,
                rule['maximum_episodes'],rule['phase']))
        if any(not any(same_episode_condition(old,new) for new in bank.episodes.values()) for old in original_conditions):
            raise ValueError('Independent success refresh removed a retained original condition')
        preparation['all_original_conditions_retained']=True
        for path in rule.get('verified_episode_sources',[]):
            prepared.extend(torch.load(ROOT/path,map_location='cpu',weights_only=False)['episodes'])
        if prepared:preparation.update(bank.append_verified_successes(prepared,rule['maximum_episodes'],
            include_actual_prefix=rule.get('include_actual_verified_prefix',False),
            replace_existing_conditions=rule.get('replace_verified_conditions',False)))
        preparation.update(episodes=len(bank.episodes),transitions=bank.transitions,bytes=bank.bytes)
        write(out/'demonstration_preparation.json',preparation)
        state=bank.state_dict();torch.save(state,out/'bank.pt')
    episodes=list(state['episodes'].values())
    audit=audit_success_history_labels(episodes,use_soft_references=False)
    write(out/'label_preflight.json',audit)
    if audit['both_hard_target_conflicts']:raise ValueError('Readout targets disagree after identical legal histories')
    definition=copy.deepcopy(source['definition']);initialization_frames=0
    if rule.get('nonlinear_readout',False):
        from experiments.ryu_counterq.policy import initialize_action_readout
        policy,definition,conversion=initialize_action_readout(source)
        save(out/'initial_actor.pt',policy,definition,dict(purpose='Behavior-preserving readout initialization',**conversion))
        cfg=read(ROOT/spec['base_config']);conditions=rule['initialization_conditions']
        pool=[r for r in cfg['pool'] if r['id'] in {c['opponent'] for c in conditions}];before=None
        for name,path in (('original',ROOT/rule['source']),('converted',out/'initial_actor.pt')):
            actor=load_actor(path)
            try:tested=evaluate(actor,pool,out/f'initialization_{name}',cpu=False,conditions=conditions,
                arena_call_sequence=True,progress=False,reward_kind=cfg['learner']['reward'])
            finally:actor.close()
            initialization_frames+=tested['native_frames']
            fields=('opponent','player','offset','result','end_reason','self_hp','opponent_hp','native_frames','ko_win')
            rows=[{k:r[k] for k in fields} for r in tested['rows']]
            if before is None:before=rows
            elif rows!=before:raise RuntimeError('Readout conversion changed zero-update complete games')
        write(out/'initialization.json',dict(**conversion,complete_games_equal=True,
            native_frames=initialization_frames,conditions=conditions))
    else:
        policy=construct_sequence_q(definition);policy.load_state_dict(source['state_dict'],strict=True)
    policy.set_training_mode(False)
    verification={};raw,target,weight,slices,reconstruction_error=readout_data(policy,episodes,verification)
    write(out/'numerical_contract.json',verification)
    mean=raw.mean(0);scale=raw.std(0).clamp_min(1e-4);x=(raw-mean)/scale
    head=copy.deepcopy(policy.action_net).double()
    first=head if isinstance(head,torch.nn.Linear) else head[0]
    with torch.no_grad():
        original_weight=first.weight.clone();first.weight.mul_(scale[None]);first.bias.add_(original_weight@mean)
        if rule.get('nonlinear_readout',False):
            old=source['state_dict'];linear=raw@old['action_net.weight'].double().T+old['action_net.bias'].double()
            converted=head(x);conversion_error=float((linear-converted).abs().max())
            if conversion_error>1e-5 or not torch.equal(linear.argmax(1),converted.argmax(1)):
                raise RuntimeError('Readout conversion changed recorded legal-history greedy actions')
    def fit_metrics(logits):
        correct=logits.gather(1,target[:,None])[:,0];other=logits.clone();other.scatter_(1,target[:,None],-torch.inf)
        margin=correct-other.max(1).values
        return dict(actions=len(target),matching_actions=int((margin>0).sum()),minimum_margin=float(margin.min()),
            weighted_cross_entropy=float((torch.nn.functional.cross_entropy(logits,target,reduction='none')*weight).sum()),
            trajectories=[dict(item,matching_actions=int((margin[item['first']:item['last']]>0).sum()),
                actions=item['last']-item['first'],minimum_margin=float(margin[item['first']:item['last']].min())) for item in slices])
    with torch.no_grad():initial=fit_metrics(head(x))
    write(out/'initial_fit.json',dict(metrics=initial,reconstruction_error=reconstruction_error,
        feature_encoding_seconds=time.monotonic()-started,episodes=len(episodes),dimensions=x.shape[1]))
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1.,max_iter=rule['maximum_iterations'],
        max_eval=rule['maximum_closure_calls'],history_size=20,line_search_fn='strong_wolfe',
        tolerance_grad=1e-9,tolerance_change=1e-12)
    closures=0;history=[];optimization_started=time.monotonic()
    def closure():
        nonlocal closures
        optimizer.zero_grad();logits=head(x)
        loss=(torch.nn.functional.cross_entropy(logits,target,reduction='none')*weight).sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite readout fitting loss')
        loss.backward();closures+=1
        if closures==1 or closures%32==0:
            history.append(dict(closure_calls=closures,loss=float(loss.detach()),
                matching_actions=int((logits.argmax(1)==target).sum()),seconds=time.monotonic()-optimization_started))
            write(out/'fit_progress.json',dict(points=history))
            print('READOUT_FIT',history[-1],flush=True)
        return loss
    previous_threads=torch.get_num_threads()
    optimization_threads=int(rule.get('optimization_threads',previous_threads))
    if optimization_threads<1:raise ValueError('Readout optimization threads must be positive')
    try:
        torch.set_num_threads(optimization_threads)
        optimizer.step(closure)
    finally:
        torch.set_num_threads(previous_threads)
    with torch.no_grad():
        first.weight.div_(scale[None]);first.bias.sub_(first.weight@mean)
        policy.action_net.load_state_dict(head.float().state_dict())
        logits=policy.action_net(raw.float())
        deployed=policy.value_net(raw.float())+logits-logits.mean(1,keepdim=True)
        if not torch.equal(deployed.argmax(1),logits.argmax(1)):
            raise RuntimeError('Float32 deployed dueling scores change the fitted action ranking')
        fitted=fit_metrics(deployed.double())
    changes={name:float((value-source['state_dict'][name]).abs().max()) for name,value in policy.state_dict().items()
        if value.is_floating_point() and not name.startswith('action_net.')}
    if max(changes.values())!=0.:raise RuntimeError('Readout diagnostic changed the backbone')
    model=out/'agent.pt'
    save(model,policy,definition,dict(purpose='Bounded fixed-backbone readout feasibility, no online RL',
        source=rule['source'],closure_calls=closures,optimizer_restart_required=True))
    optimization_seconds=time.monotonic()-optimization_started
    write(out/'fit.json',dict(initial=initial,final=fitted,closure_calls=closures,
        accepted_iterations=next(iter(optimizer.state.values())).get('n_iter',0),
        optimization_samples=closures*len(target),optimization_seconds=optimization_seconds,
        optimization_threads=optimization_threads,
        initialization_native_frames=initialization_frames,readout=definition.get('action_readout','linear'),
        readout_only=True,backbone_parameter_change=max(changes.values()),float32_deployment=True,
        last_gradient_infinity_norm=max(float(p.grad.abs().max()) for p in head.parameters()),
        predeclared_iteration_limit=rule['maximum_iterations']))
    return evaluate_readout_endpoint(spec,rule,out,episodes,read(out/'fit.json'))


def evaluate_readout_endpoint(spec,rule,out,episodes,fit,resume=False):
    """Complete the frozen readout panel without repeating its optimization."""
    model=out/'agent.pt'
    conditions=[dict(opponent=e['opponent_config']['id'],player=e['player'],offset=e['offset']) for e in episodes]
    if len({(c['opponent'],c['player'],c['offset']) for c in conditions})!=len(conditions):
        raise ValueError('Readout panel requires exact distinct conditions')
    base=read(ROOT/spec['base_config'])
    pool=exact_opponent_pool([e['opponent_config'] for e in episodes]);previous={}
    partial=out/'independent_complete_games'/'partial.json'
    completed=read(partial)['rows'] if resume and partial.exists() else []
    spent=sum(r['native_frames'] for r in completed)
    deployment_audit=None
    def observe(actor,observation,info):
        nonlocal spent
        key=(info['opponent'],info['player'],info['offset']);now=info['native_frames']
        spent+=now-previous.get(key,0);previous[key]=now
        if spent+6>rule['evaluation_native_frame_cap'] and info['result']=='ongoing':
            raise TimeoutError('Readout diagnostic full-game budget reached')
        if deployment_audit is not None:deployment_audit(actor,observation,info)
    actor=load_actor(model)
    if rule.get('evaluation_action_indices'):
        from experiments.ryu_counterq.policy import restrict_sequence_actions
        from experiments.league_adaptive.evaluate import InferenceIdentityAudit
        indices=read(ROOT/rule['evaluation_action_indices'])
        policy,definition,_=restrict_sequence_actions(dict(definition=actor.definition,state_dict=actor.policy.state_dict()),indices)
        actor.close();actor=Actor(policy,definition);deployment_audit=InferenceIdentityAudit(model)
    deployment_identity=None
    try:
        tested=evaluate(actor,pool,out/'independent_complete_games',cpu=False,conditions=conditions,
            arena_call_sequence=True,progress=True,reward_kind=base['learner']['reward'],transition_observer=observe,
            completed_rows=completed)
        if deployment_audit is not None:deployment_identity=deployment_audit.result()
    finally:
        actor.close()
        if deployment_audit is not None:deployment_audit.close()
    cpu=[r for r in tested['rows'] if r['opponent_kind']=='cpu'];pvp=[r for r in tested['rows'] if r['opponent_kind']!='cpu']
    result=dict(source=rule['source'],bank=rule['bank'],episodes=len(episodes),actions=fit['final']['actions'],
        fit_record=(out/'fit.json').relative_to(ROOT).as_posix(),closure_calls=fit['closure_calls'],
        optimization_samples=fit['optimization_samples'],optimization_seconds=fit['optimization_seconds'],
        initialization_native_frames=fit['initialization_native_frames'],readout=fit['readout'],
        CPU_KO=sum(r['ko_win'] for r in cpu),CPU_conditions=len(cpu),PvP_wins=sum(r['result']=='win' for r in pvp),
        PvP_conditions=len(pvp),PvP_by_port={str(p):sum(r['result']=='win' for r in pvp if r['player']==p) for p in (1,2)},
        evaluation_native_frames=spent,evaluation_setup_native_frames=tested.get('setup_native_frames',0),rows=tested['rows'],goal_complete=False,
        scope='Training/development fixed-backbone readout fit followed by independent full games. Hard action labels are actual verified continuations, not optimal Q. No online PPO updates or final holdout. One endpoint alone does not establish transfer, continued learning stability or completion of the overall capability goal.')
    if resume:
        result['evaluation_resume']=read(out/'evaluation_resume.json')
        interrupted=out/'evaluation_interruption.json'
        if interrupted.exists():result['evaluation_interruption']=read(interrupted)
    if deployment_identity is not None:
        result.update(deployment_action_count=len(actor.actions),deployment_identity=deployment_identity,
            deployment_scope='Actual reduced-catalog control with original full-catalog inference as a read-only observer on every real input. Probabilities compared conditional on retained actions; no teacher takeover or extra native games.')
    write(out/'summary.json',result)
    return result


def analyze(snapshot,actual_post,reference_bank,pool,conditions,out,native_cap,base_results=None,
            components=('base','joint','actor','critic','auxiliary')):
    """Reapply actual gradients from the same Adam state, then play full games."""
    out.mkdir(parents=True,exist_ok=False)
    captured=torch.load(snapshot,map_location='cpu',weights_only=False)
    actual=torch.load(actual_post,map_location='cpu',weights_only=False)
    bank={'episodes':{}} if reference_bank is None else torch.load(reference_bank,map_location='cpu',weights_only=False)
    if 'bank' in bank:bank=bank['bank']
    wanted={c['opponent'] for c in conditions}
    histories=[e for e in bank['episodes'].values() if e['opponent_config']['id'] in wanted]
    models={};records={}
    for label in components:
        policy=construct_sequence_q(copy.deepcopy(captured['definition']))
        policy.load_state_dict(captured['state_dict'],strict=True)
        restore_optimizer(policy.optimizer,copy.deepcopy(captured['optimizer']))
        if label=='joint_fresh_moments':policy.optimizer.state.clear()
        norm=None
        if label!='base':
            policy.optimizer.zero_grad(set_to_none=True)
            for name,parameter in policy.named_parameters():
                if label in ('joint','joint_fresh_moments','momentum_only'):
                    first=captured['gradients']['ppo'][name];second=captured['gradients']['auxiliary'][name]
                    gradient=second if first is None else first if second is None else first+second
                else:gradient=captured['gradients'][label][name]
                if label=='momentum_only' and gradient is not None:gradient=torch.zeros_like(gradient)
                parameter.grad=None if gradient is None else gradient.clone()
            norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),.5))
            policy.optimizer.step()
        preferences=[]
        for episode in histories:
            scores=full_values(policy,episode)[:-1].cpu();target=torch.as_tensor(episode['actions'])
            wrong=torch.nonzero(scores.argmax(1)!=target).flatten()
            chosen=scores.gather(1,target[:,None])[:,0];other=scores.clone()
            other.scatter_(1,target[:,None],-torch.inf)
            preferences.append(dict(opponent=episode['opponent_config'],player=episode['player'],offset=episode['offset'],
                reference_actions=len(target),matching_actions=int((scores.argmax(1)==target).sum()),
                first_reference_difference=None if not len(wrong) else int(wrong[0]),
                minimum_logit_margin=float((chosen-other.max(1).values).min())))
        changes={name:float((value-captured['state_dict'][name]).abs().max()) for name,value in policy.state_dict().items()
            if value.is_floating_point()}
        records[label]=dict(gradient_norm_before_clip=norm,reference_history_preferences=preferences,
            maximum_parameter_change=max(changes.values()))
        if label=='joint':
            error=max(float((v-actual['state_dict'][k]).abs().max()) for k,v in policy.state_dict().items() if v.is_floating_point())
            records[label]['actual_joint_update_parameter_error']=error
            if error>1e-6:
                write(out/'parameter_probe.json',dict(records=records))
                raise RuntimeError('Joint update differs from actual update; stop attribution')
        path=out/label/'agent.pt'
        save(path,policy,captured['definition'],dict(component=label,purpose='Captured-gradient diagnostic only; not a trained candidate',
            snapshot=snapshot.relative_to(ROOT).as_posix()))
        models[label]=path
    spent=0;completed=[]
    expected=None if base_results is None else read(base_results)['rows']
    for label,path in models.items():
        if spent+6>native_cap:break
        previous={}
        def observe(actor,observation,info):
            nonlocal spent
            key=(info['opponent'],info['player'],info['offset']);now=info['native_frames']
            spent+=now-previous.get(key,0);previous[key]=now
            if spent+6>native_cap and info['result']=='ongoing':raise TimeoutError('Captured-update native budget reached')
        actor=load_actor(path)
        try:
            result=evaluate(actor,pool,out/label/'complete_games',cpu=False,conditions=conditions,
                arena_call_sequence=True,progress=False,reward_kind='pve_ko_priority',transition_observer=observe)
        except TimeoutError:
            records[label]['incomplete']=True;break
        finally:actor.close()
        fields=('opponent','player','offset','result','end_reason','self_hp','opponent_hp','native_frames','ko_win')
        records[label]['games']=[{k:row[k] for k in fields} for row in result['rows']]
        completed.append(label)
        if label=='base' and expected is not None:
            before={(r['opponent'],r['player'],r['offset']):r for r in expected}
            if any(any(row[k]!=before[(row['opponent'],row['player'],row['offset'])][k] for k in fields)
                    for row in result['rows']):raise RuntimeError('Base games differ from recorded complete results')
        write(out/'progress.json',dict(records=records,native_frames=spent,complete_models=completed))
        print('UPDATE_COMPONENT',label,records[label]['games'],flush=True)
    value=dict(records=records,native_frames=spent,complete_models=completed,
        optimizer_update=captured['optimizer_update'],source_snapshot=snapshot.relative_to(ROOT).as_posix(),
        scope='Conditional one-update interventions. Reference-history disagreement alone is not failure; complete games decide. Not global regret, new policy training or proof of all later regression.')
    write(out/'summary.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','actual-post','out'):parser.add_argument('--'+key,required=True)
    for key in ('reference-bank','pool','conditions'):parser.add_argument('--'+key)
    parser.add_argument('--base-results');parser.add_argument('--native-cap',type=int,default=24000)
    parser.add_argument('--components',nargs='+',choices=('base','joint','actor','critic','auxiliary','momentum_only','joint_fresh_moments'),
        default=['base','joint','actor','critic','auxiliary'])
    args=parser.parse_args();setup()
    resolve=lambda value:(ROOT/value).resolve()
    out=resolve(args.out)
    if not out.is_relative_to(ROOT/'artifacts'):raise ValueError('Diagnostic data must stay under artifacts')
    fixture=torch.load(resolve(args.snapshot),map_location='cpu',weights_only=False)
    pool=read(resolve(args.pool)) if args.pool else fixture['analysis_pool']
    conditions=read(resolve(args.conditions)) if args.conditions else fixture['analysis_conditions']
    wanted={c['opponent'] for c in conditions}
    analyze(resolve(args.snapshot),resolve(args.actual_post),None if args.reference_bank is None else resolve(args.reference_bank),
        [r for r in pool if r['id'] in wanted],conditions,out,args.native_cap,
        None if args.base_results is None else resolve(args.base_results),args.components)


if __name__=='__main__':main()
