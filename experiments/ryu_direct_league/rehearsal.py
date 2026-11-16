"""Training-only successful observations and an explicit auxiliary policy loss.

No replay reward, advantage, value target or hidden state enters PPO rollouts.
Physical policy targets are evaluated through the current actor, including
trainable perception or complete skill branches when explicitly configured.
"""
import copy
from collections import Counter
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.league_adaptive.runtime import ROOT, load_actor, restore_fight, write


class SuccessCollector:
    """Evaluator observer: retain at most a fixed number of states per won round."""
    def __init__(self, maximum=64, seed=7301):
        self.maximum = maximum
        self.rng = np.random.default_rng(seed)
        self.pending, self.trajectories, self.outcomes = [], [], []
        self.seen = 0

    def __call__(self, actor, observation, info):
        # Called after stepping; observation is the legal pre-action observation.
        # Frozen stateless reference output is recomputed, not router-only logits.
        with torch.no_grad():
            tensors, _ = actor.policy.obs_to_tensor(observation)
            probabilities = actor.policy.get_distribution(tensors).distribution.probs[0].cpu().numpy().copy()
        item = (observation['image'].copy(), observation['position'].copy(), probabilities)
        self.seen += 1
        if len(self.pending) < self.maximum:
            self.pending.append(item)
        else:
            index = int(self.rng.integers(self.seen))
            if index < self.maximum:
                self.pending[index] = item
        if info['result'] != 'ongoing':
            row = {k: info[k] for k in ('opponent', 'opponent_kind', 'player', 'offset', 'result', 'self_hp', 'opponent_hp', 'native_frames')}
            self.outcomes.append(row)
            if info['result'] == 'win':
                group = 'pve' if info['opponent_kind'] == 'cpu' else f"pvp_p{info['player']}"
                self.trajectories.append(dict(group=group, condition=row, observations=self.pending))
            self.pending, self.seen = [], 0

    def save(self, directory):
        records, labels, descriptions = [], [], []
        for number, trajectory in enumerate(self.trajectories):
            records.extend(trajectory['observations'])
            labels.extend([number] * len(trajectory['observations']))
            descriptions.append(dict(id=number, group=trajectory['group'], condition=trajectory['condition'], observations=len(trajectory['observations'])))
        groups = Counter(t['group'] for t in self.trajectories)
        if any(groups[g] < 8 for g in ('pve', 'pvp_p1', 'pvp_p2')):
            raise ValueError(f'Insufficient successful training coverage for fixed trajectory cap: {groups}')
        np.savez_compressed(directory/'success_observations.npz',
            image=np.stack([x[0] for x in records]), position=np.stack([x[1] for x in records]),
            probabilities=np.stack([x[2] for x in records]), trajectory=np.asarray(labels, np.int32))
        result = dict(trajectories=descriptions, groups=dict(groups), observations=len(records),
            maximum_trajectory_probability=max(1/(3*v) for v in groups.values()),
            outcomes=self.outcomes, scope='Successful training conditions only. Opponent/group/trajectory labels are sampler metadata, never policy input.')
        write(directory/'success_observations.json', result)
        return result


def physical_kl(policy, observations, target):
    current = policy.get_distribution(observations).distribution.probs
    return (target * (target.clamp_min(1e-12).log() - current.clamp_min(1e-12).log())).sum(1).mean()


def retemperature_success_references(episodes,target_temperature,source_temperature=1.):
    """Express stored physical PPO distributions at the declared temperature.

    p_new is proportional to p_old ** (T_old / T_new). This operates only on
    explicit probability references, never Q scores or executed-action labels.
    It does not claim that the reference action is optimal or guarantees a win.
    """
    if not np.isfinite(target_temperature) or target_temperature<=0:raise ValueError('Positive target temperature required')
    changed=rows=0
    for episode in episodes:
        if episode.get('retention_target')!='physical_distribution':continue
        old=float(episode.get('reference_temperature',source_temperature))
        probabilities=np.asarray(episode['policy_probabilities'],np.float64)
        if not np.isfinite(old) or old<=0 or probabilities.ndim!=2 or not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities.sum(1)<=0).any():
            raise ValueError('Invalid recorded PPO probability reference')
        with np.errstate(divide='ignore'):logits=np.log(probabilities)*(old/target_temperature)
        logits-=logits.max(1,keepdims=True);probabilities=np.exp(logits)
        probabilities/=probabilities.sum(1,keepdims=True)
        episode['policy_probabilities']=probabilities.astype(np.float32)
        episode['reference_temperature']=float(target_temperature)
        episode['reference_transform']='Stored PPO physical probabilities reexpressed at the current constant policy temperature'
        changed+=old!=target_temperature;rows+=len(probabilities)
    return dict(episodes_changed=changed,probability_rows=rows,target_temperature=target_temperature)


def recurrent_success_loss(policy,replay,count=4,length=16,disagreement_fraction=0.,disagreement_scope='failed',error_balance=0.,priority_fraction=0.,gradient_context=0,use_soft_references=False,hard_failed_conditions=False,disagreement_selection='random'):
    """Finite action supervision on verified complete self-successes.

    Reconstruct the entire actual prefix with current visual/GRU weights.
    Only the selected suffix supplies labels. An explicitly configured
    preceding context can carry gradients through its actual recurrent inputs.
    There are no replay returns,
    critic targets, hidden-state labels or PPO advantage samples here.
    """
    from experiments.ryu_counterq.replay import episode_values
    if not getattr(policy,'probability_policy',False) or not policy.physical_actions:
        raise ValueError('Sequence success supervision requires physical categorical actions')
    if not 0<=error_balance<=1:raise ValueError('Invalid supervised error balance')
    if int(gradient_context)!=gradient_context or gradient_context<0:raise ValueError('Invalid gradient context')
    losses=[];samples=0
    for episode,first,last in replay.success_sequences(count,length,disagreement_fraction,disagreement_scope,priority_fraction,disagreement_selection):
        hidden=None
        gradient_start=max(0,first-int(gradient_context))
        with torch.no_grad():
            for start in range(0,gradient_start,32):
                _,hidden=episode_values(policy,episode,start,min(start+32,gradient_start),hidden)
        logits,_=episode_values(policy,episode,gradient_start,last,hidden)
        logits=logits[:,first-gradient_start:]
        action=torch.as_tensor(episode['actions'][first:last],device=logits.device)
        use=replay.retention_usage[episode['auxiliary_registry_id']]
        context=first-gradient_start
        use['gradient_context_transitions']=use.get('gradient_context_transitions',0)+context
        actual_bad=int((logits[0].argmax(1)!=action).sum())
        assessed_errors=episode.get('supervised_disagreements',[])
        if len(assessed_errors) and first<=int(assessed_errors[0])<last:
            use['phase_assessed_frontier_windows']=use.get('phase_assessed_frontier_windows',0)+1
        use['actual_disagreements_trained']=use.get('actual_disagreements_trained',0)+actual_bad
        use['phase_actual_disagreements_trained']=use.get('phase_actual_disagreements_trained',0)+actual_bad
        reference_soft=use_soft_references and episode.get('retention_target')=='physical_distribution'
        repair=reference_soft and hard_failed_conditions and episode.get('current_complete_useful') is False
        if reference_soft and 'policy_probabilities' not in episode:raise ValueError('Missing physical reference distribution')
        soft=reference_soft and not repair
        if soft:
            if 'policy_probabilities' not in episode:raise ValueError('A soft reference must contain actual frozen physical probabilities')
            target=torch.as_tensor(episode['policy_probabilities'][first:last],device=logits.device,dtype=logits.dtype)
            if target.shape!=logits[0].shape or not torch.isfinite(target).all() or (target<0).any() or (target.sum(1)<=0).any():
                raise ValueError('Invalid physical probability reference')
            target=target/target.sum(1,keepdim=True)
            loss=(target*(target.clamp_min(1e-12).log()-F.log_softmax(logits[0]/policy.action_temperature,1))).sum(1).mean()
        else:loss=F.cross_entropy(logits[0]/policy.action_temperature,action)
        kind='soft_reference' if soft else 'hard_action'
        if repair:use['outcome_gated_hard_transitions']=use.get('outcome_gated_hard_transitions',0)+len(action)
        use[kind+'_transitions']=use.get(kind+'_transitions',0)+len(action)
        use[kind+'_loss_weighted_sum']=use.get(kind+'_loss_weighted_sum',0.)+float(loss.detach())*len(action)
        if error_balance and not soft:
            wrong=logits[0].detach().argmax(1)!=action
            if wrong.any():
                loss=(1-error_balance)*loss+error_balance*F.cross_entropy(logits[0,wrong]/policy.action_temperature,action[wrong])
            use['error_balanced_samples']=use.get('error_balanced_samples',0)+int(wrong.sum())
        losses.append(loss)
        samples+=len(action)
    return torch.stack(losses).mean(),samples


def retention_kl(logits, target, allowed=None):
    """Physical KL, optionally pooling only a verified outcome-equivalent set.

    Pooling removes preference *within* that set, without rewarding its use
    or granting permission to any other action. A singleton is ordinary KL.
    """
    logp = F.log_softmax(logits, 1)
    terms = target * (target.clamp_min(1e-12).log() - logp)
    if allowed is None:
        return terms.sum(1).mean()
    mass = (target * allowed).sum(1)
    pooled = mass * (mass.clamp_min(1e-12).log() - logp.masked_fill(~allowed, -torch.inf).logsumexp(1))
    return (terms.masked_fill(allowed, 0).sum(1) + pooled).mean()


def validate_action_sets(bundle_path, continuation_path, directory, label_version, include_diagnostic=False,
                         maximum_native_frames=200000,maximum_seconds=1200):
    """Training-only crossed first-action/complete-continuation validation.

    Restored roots have both players' image histories, opponent controller
    state and all RNGs. Only stateless mixtures are supported here; no hidden
    policy history may be silently discarded. Results are conditional on the
    explicitly named immutable continuation, never oracle action values.
    """
    from experiments.league_adaptive.environment import FightEnv
    from experiments.ryu_counterq.collect_demonstrations import advance, begin_episode, continue_episode, terminal
    directory=ROOT/directory;directory.mkdir(parents=True,exist_ok=False)
    bundle=torch.load(ROOT/bundle_path,map_location='cpu',weights_only=False)
    actor=load_actor(continuation_path)
    if actor.definition['kind']!='neural_mixture' or actor.get_state() is not None:
        raise ValueError('Selective audit requires the declared stateless full-action mixture')
    others={name:load_actor(path) for name,path in bundle['additional_actors'].items()}
    if any(a.get_state() is not None or a.actions!=actor.actions for a in others.values()):
        raise ValueError('Cannot discard actor history or change physical action mapping')
    env=FightEnv(actor.definition,bundle['pool'])
    rows=[];traces=[];started=time.monotonic();exhausted=False
    try:
        for number,root in enumerate(bundle['roots']):
            # Training relaxations use only the preselected parent-success
            # roots; mandatory failure diagnoses remain in the initial audit.
            if not root['retention_eligible'] and not include_diagnostic:continue
            results=[]
            for action in (root['old_action'],root['alternative_action']):
                if exhausted:break
                obs=restore_fight(env,actor,root['saved'],others)
                before=dict(env.game.raw);raw=begin_episode(obs);trace=[]
                def observe(environment,previous,info,executed):
                    trace.append(dict(decision=environment.episode_steps-1,action=int(executed),
                        x=[environment.game.raw[f'p{p}_x'] for p in (1,2)],
                        y=[environment.game.raw[f'p{p}_y'] for p in (1,2)],
                        hp=[environment.game.raw[f'p{p}_hp'] for p in (1,2)],
                        dealt=info['damage_dealt'],taken=info['damage_taken']))
                    if environment.native_frames>=maximum_native_frames or time.monotonic()-started>=maximum_seconds:
                        raise TimeoutError('Counterfactual budget reached; unfinished branch remains unknown')
                try:
                    previous=obs;obs,info=advance(env,obs,action,raw)
                    observe(env,previous,info,action)
                    if not env.ended:
                        _,info=continue_episode(env,actor,obs,raw,observer=observe)
                except TimeoutError:
                    exhausted=True
                    results.append(dict(first_action=action,continuation=str(continuation_path),result='incomplete',
                        counts_as_complete=False,native_frames=env.game.elapsed))
                    traces.append(dict(root=number,first_action=action,incomplete=True,trace=trace));break
                result=terminal(info)
                results.append(dict(first_action=action,first_action_name=actor.actions[action]['name'],
                    continuation=str(continuation_path),**result))
                traces.append(dict(root=number,first_action=action,before={k:before[k] for k in
                    ('p1_x','p2_x','p1_y','p2_y','p1_hp','p2_hp')},trace=trace))
            def acceptable(r):
                return r['result']=='win' and (root['condition'][1]!='cpu' or
                    r['opponent_hp']<=0 and r['end_reason'] in ('signed_hp_ko','round_counter'))
            safe=len(results)==2 and all(acceptable(r) for r in results)
            rows.append(dict(root=number,condition=root['condition'],decision=root['decision'],
                retention_eligible=root['retention_eligible'],
                old_action=root['old_action'],alternative_action=root['alternative_action'],
                allowed_actions=[root['old_action'],root['alternative_action']],safe=safe,
                status='conditional_safe' if safe else 'protected_unsafe_or_unknown',
                continuation=str(continuation_path),branches=results))
            write(directory/'progress.json',dict(completed=len(rows),safe=sum(r['safe'] for r in rows),native_frames=env.native_frames))
        document=dict(rows=rows,continuation=str(continuation_path),label_version=label_version,
            status='budget_exhausted' if exhausted else 'complete',
            source_bundle=str(bundle_path),native_frames=env.native_frames,seconds=time.monotonic()-started,
            snapshot_roots=len(rows),restores=2*len(rows),
            environment_calls=sum(len(t['trace']) for t in traces),
            validity='Current checkpoint block only; expire and refresh under next frozen policy; failures remain protected.',
            scope='Complete conditional continuations, not independent wins, on-policy rollouts or oracle labels.')
        write(directory/'labels.json',document);write(directory/'traces.json',traces)
        return document
    finally:
        env.close();actor.close()
        for other in others.values():other.close()


def sample_replay(groups, indices, rng, categories=None, new_fraction=0.):
    """Balanced visible-mode coverage, with a hard 8/128 trajectory cap."""
    group_ids = rng.permutation((np.arange(128)+int(rng.integers(3))) % 3)
    counts, trajectory_ids = Counter(), []
    for group in group_ids:
        eligible = [t for t in groups[group] if counts[t] < 8]
        if categories is not None and new_fraction:
            want_new=bool(rng.random()<new_fraction)
            subset=[t for t in eligible if categories[t]==want_new]
            if subset:eligible=subset
        chosen = int(rng.choice(eligible))
        trajectory_ids.append(chosen)
        counts[chosen] += 1
    return [int(rng.choice(indices[t])) for t in trajectory_ids]


def continuous_contexts(policy, paths, include_proposals=False,old_conditions=None):
    """Compact frozen legal contexts from genuine contiguous demonstrations.

    Metadata selects training data only. Acting always recomputes the same
    features and physical logits from the legal observation; no cache deploys.
    """
    if not policy.correction_only:raise ValueError('Cached contexts require an immutable original actor')
    features=[];bases=[];flips=[];targets=[];actions=[];labels=[];descriptions=[];proposals=[];state_keys=[]
    mirror=policy.correction_physical_mirror.cpu().numpy()
    started=time.monotonic()
    policy.set_training_mode(False)
    for directory,is_new in paths:
        directory=ROOT/directory
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for entry in manifest['episodes']:
            if not is_new and old_conditions is not None:
                key=tuple(entry['condition'][k] for k in ('opponent','opponent_kind','player','offset'))
                if key not in old_conditions:continue
            root=entry.get('condition',{}).get('root',16)
            if is_new and root<16:continue
            filename=(directory/entry['file']).resolve()
            if filename.parent!=directory.resolve():raise ValueError('Invalid continuous demonstration path')
            with np.load(filename,allow_pickle=False) as archive:
                data={k:archive[k] for k in ('frames','positions','actions','supervised','targets','dones')}
                n=len(data['actions'])
                if (len(data['frames'])!=n+4 or len(data['positions'])!=n+1 or
                    not data['dones'][-1] or data['dones'][:-1].any()):
                    raise ValueError('Continuous demonstrations must contain one complete actual-action history')
                indices=np.flatnonzero(data['supervised'])
                if not len(indices):continue
                number=len(descriptions)
                for begin in range(0,len(indices),128):
                    ix=indices[begin:begin+128]
                    observations=dict(image=np.stack([data['frames'][i:i+4] for i in ix]),position=data['positions'][ix])
                    flip=observations['position'][:,4]<0
                    actual=np.where(flip,mirror[data['actions'][ix]],data['actions'][ix])
                    with torch.no_grad():
                        tensor,_=policy.obs_to_tensor(observations)
                        predicted=policy.get_distribution(tensor).distribution.probs.argmax(1).cpu().numpy()
                        context=[x.detach().cpu().clone() for x in policy.last_correction_context]
                        proposed=policy.last_neural_proposals.cpu().clone() if include_proposals else None
                        # Batch-size roundoff at near ties must not corrupt
                        # the exact executed-action anchor. Recompute only
                        # these contexts using the actual one-call protocol.
                        if not is_new:
                            for j in np.flatnonzero(predicted!=actual):
                                single={k:v[j] for k,v in observations.items()}
                                t,_=policy.obs_to_tensor(single)
                                chosen=int(policy.get_distribution(t).distribution.probs.argmax(1))
                                if chosen!=actual[j]:raise ValueError('Original success action does not match inherited actor')
                                for k,value in enumerate(policy.last_correction_context):context[k][j]=value[0].cpu()
                                if include_proposals:proposed[j]=policy.last_neural_proposals[0].cpu()
                    features.append(context[0]);bases.append(context[1]);flips.append(context[2])
                    if include_proposals:proposals.append(proposed)
                    target=data['targets'][ix].astype(np.float32)
                    target=np.where(flip[:,None],target[:,mirror],target)
                    target/=target.sum(1,keepdims=True)
                    targets.append(target);actions.extend(actual.tolist());labels.extend([number]*len(ix))
                    state_keys.extend((entry['condition']['opponent'],entry['condition'].get('opponent_kind'),
                        entry['condition']['player'],entry['condition']['offset'],int(i)+entry.get('start_decision',0)) for i in ix)
                descriptions.append(dict(id=number,group={'p1':'pvp_p1','p2':'pvp_p2','pve':'pve'}[entry['group']],
                    observations=len(indices),new_tactic=is_new,origin=entry.get('origin'),condition=entry['condition']))
    return dict(context=(torch.cat(features),torch.cat(bases),torch.cat(flips)),
        probabilities=np.concatenate(targets),actions=np.asarray(actions,np.int64),
        labels=np.asarray(labels,np.int32),trajectories=descriptions,seconds=time.monotonic()-started,
        state_keys=state_keys,
        proposals=torch.cat(proposals) if include_proposals else None)


@torch.no_grad()
def demonstration_fit(policy, paths, selected_files=None):
    """Fit on stored legal prefix histories, not closed-loop success evidence."""
    if not getattr(policy,'correction_only',False):
        replay=LegalTrajectoryReplay(policy,[(p,True) for p in paths],selected_files=selected_files)
        reports=[];total=matches=outside=chosen_outside=0;nll=0.;conflicts=0;ceiling_correct=0
        total_kl=0.
        policy.set_training_mode(False)
        for trajectory in replay.trajectories:
            indices=replay.indices[trajectory['id']]
            correct=[];labels={};trajectory_kl=0.
            for start in range(0,len(indices),64):
                ix=indices[start:start+64];observation=replay.observations(ix)
                tensor,_=policy.obs_to_tensor(observation)
                probabilities=policy.get_distribution(tensor).distribution.probs.cpu().numpy()
                soft=replay.probabilities[ix]
                trajectory_kl+=float((soft*(np.log(np.maximum(soft,1e-30))-
                    np.log(np.maximum(probabilities,1e-30)))).sum())
                target=replay.probabilities[ix].argmax(1);predicted=probabilities.argmax(1)
                correct.extend((predicted==target).tolist())
                nll+=float(-np.log(np.maximum(probabilities[np.arange(len(ix)),target],1e-30)).sum())
                if hasattr(policy,'last_neural_proposals'):
                    proposed=policy.last_neural_proposals.cpu().numpy()
                    outside+=int((~(proposed==target[:,None]).any(1)).sum())
                    chosen_outside+=int((~(proposed==predicted[:,None]).any(1)).sum())
                for j,action in enumerate(target):
                    key=observation['image'][j].tobytes()+observation['position'][j].tobytes()
                    labels.setdefault(key,Counter())[int(action)]+=1
            conflicting=sum(len(v)>1 for v in labels.values())
            ceiling=sum(max(v.values()) for v in labels.values())
            conflicts+=conflicting;ceiling_correct+=ceiling;total+=len(indices);matches+=sum(correct);total_kl+=trajectory_kl
            different=np.flatnonzero(~np.asarray(correct))
            reports.append(dict(condition=trajectory['condition'],labeled=len(indices),greedy_matches=sum(correct),
                first_mode_difference=None if not len(different) else int(replay.local[indices[different[0]]]),
                physical_soft_kl=trajectory_kl/len(indices),
                whole_stored_prefix_matches=all(correct),exact_observation_label_conflicts=conflicting,
                deterministic_fit_ceiling=ceiling/len(indices)))
        return dict(labeled=total,greedy_matches=matches,target_negative_log_probability=nll/max(1,total),
            physical_soft_kl=total_kl/max(1,total),
            target_actions_outside_reference_proposals=outside,chosen_actions_outside_reference_proposals=chosen_outside,
            whole_stored_prefix_matches=sum(r['whole_stored_prefix_matches'] for r in reports),
            exact_observation_label_conflicts=conflicts,deterministic_fit_ceiling=ceiling_correct/max(1,total),
            trajectories=reports,native_frames=0,scope='Fit on actual legal continuous demonstration histories, not independent closed-loop ability.')
    data=continuous_contexts(policy,[(p,True) for p in paths],include_proposals=True)
    context=tuple(x.to(policy.device) for x in data['context'])
    probabilities=policy.corrected_logits(*context).softmax(1)
    actual=torch.as_tensor(data['actions'],device=policy.device)
    predicted=probabilities.argmax(1);rows=torch.arange(len(actual),device=policy.device)
    correct=(predicted==actual).cpu().numpy()
    trajectories=[]
    for trajectory in data['trajectories']:
        ix=data['labels']==trajectory['id']
        trajectories.append(dict(id=trajectory['id'],condition=trajectory['condition'],
            labeled=int(ix.sum()),greedy_matches=int(correct[ix].sum()),whole_stored_prefix_matches=bool(correct[ix].all())))
    return dict(labeled=len(actual),greedy_matches=int(correct.sum()),
        target_actions_outside_reference_proposals=int((~(data['proposals'].to(policy.device)==actual[:,None]).any(1)).sum()),
        chosen_actions_outside_reference_proposals=int((~(data['proposals'].to(policy.device)==predicted[:,None]).any(1)).sum()),
        target_negative_log_probability=float(-probabilities[rows,actual].clamp_min(1e-30).log().mean()),
        whole_stored_prefix_matches=sum(t['whole_stored_prefix_matches'] for t in trajectories),
        trajectories=trajectories,encoding_seconds=data['seconds'],native_frames=0,
        scope='Teacher-path diagnostic only. Actual deployed choices and complete student-controlled games remain the ability tests.')


class LegalTrajectoryReplay:
    """Compact contiguous storage for a changing encoder; no frozen latents.

    Actions and targets on disk are facing-canonical. The live policy consumes
    physical probabilities, so every sampled target is mapped back exactly.
    Failed search trajectories supply conditional value targets only.
    """
    def __init__(self,policy,paths,old_conditions=None,selected_files=None):
        from experiments.league_adaptive.policy import mirror_indices
        mirror=getattr(policy,'correction_physical_mirror',None)
        if mirror is None:mirror=getattr(policy,'physical_mirror',None)
        self.mirror=np.asarray(mirror_indices(policy.definition['actions']) if hasattr(policy,'definition') else
            mirror.cpu().numpy())
        self.episodes=[];self.trajectories=[];self.labels=[];self.local=[];self.probabilities=[]
        self.indices={};self.failed=[];self.new_all=[]
        selected_files=None if selected_files is None else {str(Path(p).resolve()) for p in selected_files}
        loaded=set()
        for directory_number,(directory,is_new) in enumerate(paths):
            directory=ROOT/directory
            document=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            for entry in document['episodes']:
                if not is_new and old_conditions is not None and directory_number==0:
                    key=tuple(entry['condition'][k] for k in ('opponent','opponent_kind','player','offset'))
                    if key not in old_conditions:continue
                filename=(directory/entry['file']).resolve()
                if filename.parent!=directory.resolve():raise ValueError('Invalid replay episode path')
                if selected_files is not None and str(filename) not in selected_files:continue
                loaded.add(str(filename))
                with np.load(filename,allow_pickle=False) as archive:
                    data={k:archive[k].copy() for k in ('frames','positions','actions','targets','supervised','dones','returns')}
                n=len(data['actions'])
                if (len(data['frames'])!=n+4 or len(data['positions'])!=n+1 or
                    not n or not data['dones'][-1] or data['dones'][:-1].any() or
                    data['frames'].dtype!=np.uint8 or data['positions'].dtype!=np.float32):
                    raise ValueError('Replay must contain a complete continuous legal history')
                number=len(self.episodes);start=len(self.labels)
                self.episodes.append(data)
                self.labels.extend([number]*n);self.local.extend(range(n))
                target=data['targets'].astype(np.float32)
                flip=data['positions'][:-1,4]<0
                target=np.where(flip[:,None],target[:,self.mirror],target)
                target/=target.sum(1,keepdims=True)
                self.probabilities.append(target)
                supervised=data['supervised'].copy()
                tactical=bool(is_new or entry.get('tactical_priority',False))
                if tactical:
                    supervised[:int(entry.get('supervision_start',0))]=False
                    if not entry.get('policy_supervised',True):supervised[:]=False
                selected=np.flatnonzero(supervised)+start
                if len(selected):
                    self.indices[number]=selected
                    self.trajectories.append(dict(id=number,
                        group={'p1':'pvp_p1','p2':'pvp_p2','pve':'pve'}[entry['group']],
                        observations=len(selected),new_tactic=tactical,condition=entry['condition'],
                        episode_path=filename.relative_to(ROOT).as_posix(),origin=entry.get('origin')))
                if tactical:
                    self.new_all.extend(range(start,start+n))
                    result=entry['condition'].get('result')
                    if result not in ('win','loss','draw'):raise ValueError('Search experience needs a complete outcome')
                    # Stored quality return = terminal quality - HP potential.
                    # Translate to the unchanged PPO win reward exactly. The
                    # continuation is recorded in the episode metadata; this
                    # is not a claim of optimal or current-policy value.
                    c=entry['condition']
                    if result=='win':
                        ko=c['opponent_hp']<=0 and c['end_reason'] in ('signed_hp_ko','round_counter')
                        quality=2.+.2*np.clip(c['self_hp']/176.,0,1)+.1*ko+.05*(1-min(c['native_frames']/7200.,1.))
                    else:quality=-1. if result=='draw' else -2.
                    data['conditional_returns']=data['returns']+(1. if result=='win' else -1.)-quality
                    if result!='win':self.failed.extend(range(start,start+n))
        self.labels=np.asarray(self.labels,np.int32);self.local=np.asarray(self.local,np.int32)
        self.probabilities=np.concatenate(self.probabilities)
        if selected_files is not None and loaded!=selected_files:raise ValueError('Common replay catalog has unavailable trajectories')

    def observations(self,indices):
        images=[];positions=[]
        for index in indices:
            episode=self.episodes[self.labels[index]];local=self.local[index]
            images.append(episode['frames'][local:local+4]);positions.append(episode['positions'][local])
        return dict(image=np.stack(images),position=np.stack(positions))

    def conditional_returns(self,indices):
        return np.asarray([self.episodes[self.labels[i]]['conditional_returns'][self.local[i]] for i in indices],np.float32)


class ReplayTasks:
    """Explicit training tasks over one shared catalog, with bounded samples.

    Task/source metadata never reaches the policy. Weighted losses have equal
    total mass across arms; gradients are measured before clipping/Adam, not
    represented as exact attribution of Adam's eventual parameter changes.
    """
    names=('p1_retention','p2_tactic','general')

    def __init__(self,trajectories,indices,spec,arm,catalog):
        self.trajectories={t['id']:t for t in trajectories};self.indices=indices
        entries={e['episode_path']:e for e in catalog['episodes']}
        if len(entries)!=len(catalog['episodes']):raise ValueError('Duplicate common trajectory path')
        self.tasks={t:entries[d['episode_path']]['task'] for t,d in self.trajectories.items()}
        settings=spec['arms'][arm]
        self.recovery_root_priority=bool(settings.get('recovery_root_priority',spec.get('recovery_root_priority',False)))
        self.recovery_roots={t:int(indices[t][0]) for t,d in self.trajectories.items()
                             if self.recovery_root_priority and entries[d['episode_path']].get('supervision_start',0)>0}
        self.recovery_root_samples=Counter()
        if set(self.tasks.values())!=set(self.names):raise ValueError('All three replay tasks need actual trajectories')
        self.arm=arm;self.counts=settings['counts'];raw=settings['weights']
        if set(self.counts)!=set(self.names) or sum(self.counts.values())!=128 or min(self.counts.values())<=0:
            raise ValueError('Task counts must be positive and sum to128')
        self.cap=int(spec['maximum_trajectory_samples']);self.interval=int(spec['gradient_interval'])
        if self.cap<=0 or self.interval<=0:raise ValueError('Positive trajectory/gradient limits required')
        self.priority_states={t:[i] for t,i in self.recovery_roots.items()}
        self.priority_offsets={(t,i):0 for t,i in self.recovery_roots.items()}
        self.priority_context_samples=Counter()
        for t,description in self.trajectories.items():
            offsets=entries[description['episode_path']].get('priority_replay_offsets',[])
            if len(set(offsets))!=len(offsets) or any(type(o) is not int or not 0<=o<len(indices[t]) for o in offsets):
                raise ValueError('Priority offsets must identify distinct supervised states')
            if not self.recovery_root_priority:continue
            for offset in offsets:
                index=int(indices[t][offset]);points=self.priority_states.setdefault(t,[])
                if index not in points:points.append(index)
                self.priority_offsets[t,index]=offset
            if len(self.priority_states.get(t,[]))>self.cap:raise ValueError('Context priorities exceed the trajectory cap')
        self.scale=128*float(spec['common_loss_mass'])/sum(self.counts[k]*float(raw[k]) for k in self.names)
        self.weights={k:float(raw[k])*self.scale for k in self.names}
        self.ids={k:[t for t in self.tasks if self.tasks[t]==k] for k in self.names}
        for task in self.names:
            if self.cap*len(self.ids[task])<self.counts[task]:raise ValueError('Insufficient unique trajectories for task cap')
        self.samples=Counter();self.groups=Counter();self.trajectory_samples=Counter();self.losses=Counter();self.weighted_losses=Counter()
        self.batches=0;self.maximum_observed=0;self.gradient_reports=0;self.gradient_seconds=0.;self.latest_gradients=None

    def sample(self,rng):
        batch=[];labels=[];used=Counter();used_states={}
        for task in self.names:
            groups=('pve','pvp_p1','pvp_p2') if task=='general' else (None,)
            order=rng.permutation((np.arange(self.counts[task])+int(rng.integers(len(groups))))%len(groups))
            for group_index in order:
                group=groups[group_index]
                eligible=[t for t in self.ids[task] if used[t]<self.cap and
                          (group is None or self.trajectories[t]['group']==group)]
                if not eligible:raise ValueError('Task subgroup lacks capped trajectory coverage')
                trajectory=int(rng.choice(eligible));candidates=self.indices[trajectory]
                available=[int(i) for i in candidates if int(i) not in used_states.setdefault(trajectory,set())]
                priorities=self.priority_states.get(trajectory,[])
                if used[trajectory]<len(priorities):
                    # A verified handoff may depend on this first actual
                    # student state. Retain the trajectory cap and sample the
                    # rest normally; root metadata never enters the policy.
                    chosen=priorities[used[trajectory]]
                    self.priority_context_samples[f'{trajectory}:{self.priority_offsets[trajectory,chosen]}']+=1
                    if chosen==self.recovery_roots.get(trajectory):self.recovery_root_samples[trajectory]+=1
                else:chosen=int(rng.choice(available if available else candidates))
                used_states[trajectory].add(chosen);used[trajectory]+=1;batch.append(chosen);labels.append(task)
                self.samples[task]+=1;self.groups[task+'|'+self.trajectories[trajectory]['group']]+=1
                self.trajectory_samples[trajectory]+=1
        self.maximum_observed=max(self.maximum_observed,max(used.values()));self.batches+=1
        return batch,labels

    def record_losses(self,per,labels):
        for task in self.names:
            values=per[np.asarray(labels)==task]
            self.losses[task]+=float(values.sum())
            self.weighted_losses[task]+=float(values.sum())*self.weights[task]/128

    def gradients(self,policy,components):
        started=time.monotonic();named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
        groups={'perception':('features_extractor.',),'actor_router':('mlp_extractor.policy_net.','action_net.'),
                'full_action_head':('correction_action_net.',)}
        if getattr(policy,'full_distribution_experts',False):
            groups.update(tactical_perception=('experts.1.features_extractor.',),
                tactical_actor=('experts.1.mlp_extractor.policy_net.','experts.1.action_net.','experts.1.correction_action_net.'))
        gradients={k:torch.autograd.grad(components[k],[p for _,p in named],retain_graph=True,allow_unused=True) for k in self.names}
        norms={};total={}
        for task,values in gradients.items():
            norms[task]={g:sum(float(v.square().sum()) for (n,_),v in zip(named,values)
                if v is not None and n.startswith(prefixes))**.5 for g,prefixes in groups.items()}
            total[task]=sum(float(v.square().sum()) for v in values if v is not None)**.5
        cosine={}
        for i,a in enumerate(self.names):
            for b in self.names[i+1:]:
                dot=sum(float((x*y).sum()) for x,y in zip(gradients[a],gradients[b]) if x is not None and y is not None)
                cosine[a+'|'+b]=dot/max(total[a]*total[b],1e-30)
        self.gradient_seconds+=time.monotonic()-started;self.gradient_reports+=1
        self.latest_gradients=dict(batch=self.batches,weighted_norms=norms,weighted_total_norms=total,cosines=cosine,
            scope='Actual weighted auxiliary component gradients before global clipping and Adam; no gradient surgery.')

    def metadata(self):
        return dict(arm=self.arm,counts=self.counts,effective_weights=self.weights,common_mass=sum(self.counts[k]*self.weights[k] for k in self.names)/128,
            recovery_root_priority=self.recovery_root_priority,prioritized_recovery_root_samples=dict(self.recovery_root_samples),
            prioritized_context_samples=dict(self.priority_context_samples),
            trajectory_cap=self.cap,maximum_observed_trajectory_samples=self.maximum_observed,
            trajectory_counts={k:len(self.ids[k]) for k in self.names},actual_samples=dict(self.samples),
            actual_group_samples=dict(self.groups),actual_trajectory_samples=dict(self.trajectory_samples),
            mean_unweighted_kl={k:self.losses[k]/max(1,self.samples[k]) for k in self.names},
            mean_weighted_batch_contribution={k:self.weighted_losses[k]/max(1,self.batches) for k in self.names},
            batches=self.batches,gradient_interval=self.interval,gradient_reports=self.gradient_reports,
            extra_backward_queries=3*self.gradient_reports,gradient_measurement_seconds=self.gradient_seconds,
            latest_gradients=self.latest_gradients)


def trajectory_catalog(path, ancestors=()):
    """Immutable stage additions reference earlier metadata, not copied data."""
    path=(ROOT/path).resolve()
    if path in ancestors:raise ValueError('Cyclic trajectory catalog inheritance')
    document=json.loads(path.read_text(encoding='utf-8'))
    if document.get('base_catalog'):
        base=trajectory_catalog(document['base_catalog'],ancestors+(path,))
        excluded=set(document.get('exclude_base_tasks',()))
        replaced=set(document.get('exclude_base_episodes',()))
        available={e['episode_path'] for e in base['episodes']}
        if not replaced<=available:raise ValueError('Superseded replay trajectory is absent from the base catalog')
        document['episodes']=[e for e in base['episodes'] if e.get('task') not in excluded
                              and e['episode_path'] not in replaced]+document['episodes']
    return document


class Rehearsal:
    """Separate, bounded policy cloning after each unmodified PPO update."""
    def __init__(self, path, policy, seed, coefficient, steps=8, learning_rate=.01,
                 new_paths=(),new_fraction=0.,preserve_actions=False,projection_passes=32,
                 preservation_margin=.05,preservation_gap_source='prior',replay_reference_report=None,
                 selective_labels=None,optimizer='sgd',failure_value_coefficient=0.,demonstration_coefficient=1.,old_paths=(),
                 task_config=None,task_arm=None,telemetry_path=None):
        self.path, self.coefficient, self.steps = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)), coefficient, steps
        self.rng = np.random.default_rng(seed + 67001)
        self.replay_reference_report=None if replay_reference_report is None else str(replay_reference_report)
        self.old_paths=[str(p) for p in old_paths]
        self.context=None;self.dynamic=None;self.categories=None;self.new_fraction=float(new_fraction);self.cache_seconds=0.
        self.failure_value_coefficient=float(failure_value_coefficient)
        self.demonstration_coefficient=float(demonstration_coefficient)
        self.tasks=None;self.telemetry_path=telemetry_path
        spec=None;catalog=None;selected_files=None
        if task_config is not None:
            spec=json.loads((ROOT/task_config).read_text(encoding='utf-8'))
            catalog=trajectory_catalog(spec['catalog'])
            selected_files=[ROOT/e['episode_path'] for e in catalog['episodes']]
        if path.is_dir():
            conditions=None
            if replay_reference_report is not None:
                rows=json.loads((ROOT/replay_reference_report).read_text(encoding='utf-8'))['rows']
                conditions={tuple(r[k] for k in ('opponent','opponent_kind','player','offset')) for r in rows if r['result']=='win'}
            if not getattr(policy,'correction_only',False):
                if preserve_actions or selective_labels:raise ValueError('A changing body uses soft experience, never linear action locks')
                self.dynamic=LegalTrajectoryReplay(policy,[(path,False)]+[(p,False) for p in old_paths]+[(p,True) for p in new_paths],conditions,selected_files)
                self.probabilities,self.labels=self.dynamic.probabilities,self.dynamic.labels
                metadata=dict(trajectories=self.dynamic.trajectories);data={}
            else:
                if old_paths:raise ValueError('Refreshed raw-history replay requires a trainable encoder')
                data=continuous_contexts(policy,[(path,False)]+[(p,True) for p in new_paths],old_conditions=conditions)
                self.context,self.probabilities,self.labels=data['context'],data['probabilities'],data['labels']
                self.actions=data['actions']
                metadata=dict(trajectories=data['trajectories']);self.cache_seconds=data['seconds']
            self.categories={t['id']:t['new_tactic'] for t in metadata['trajectories']}
        else:
            with np.load(path, allow_pickle=False) as data:
                self.observations = {k: data[k].copy() for k in ('image', 'position')}
                self.probabilities, self.labels = data['probabilities'].copy(), data['trajectory'].copy()
            metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if self.observations['image'].dtype != np.uint8 or self.observations['position'].dtype != np.float32:
                raise ValueError('Replay must contain legal deployment observations')
        if not np.allclose(self.probabilities.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid physical-action targets')
        self.groups = [[t['id'] for t in metadata['trajectories'] if t['group'] == g]
                       for g in ('pve', 'pvp_p1', 'pvp_p2')]
        if any(len(g) < 8 for g in self.groups):
            raise ValueError('Replay group lacks trajectory coverage')
        self.indices = (self.dynamic.indices if self.dynamic is not None else
            {int(t): np.flatnonzero(self.labels == t) for group in self.groups for t in group})
        self.parameters = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer_name=optimizer
        self.optimizer = (torch.optim.Adam if optimizer=='adam' else torch.optim.SGD)(self.parameters, lr=learning_rate)
        self.updates, self.seconds, self.examples = 0, 0., 0
        self.latest = {}
        self.trajectories=metadata['trajectories']
        if spec is not None:
            if self.dynamic is None:raise ValueError('Explicit replay tasks require trainable legal-history replay')
            self.tasks=ReplayTasks(self.trajectories,self.indices,spec,task_arm,catalog)
        if preserve_actions and self.context is None:raise ValueError('Action constraints require continuous original success data')
        self.constraint=None
        self.allowed=None;self.selective_metadata=None;self.state_keys=data.get('state_keys',[]) if path.is_dir() else []
        if preserve_actions:
            old=np.asarray([not self.categories[int(t)] for t in self.labels])
            self.constraint=ActionConstraints(policy,tuple(x[old] for x in self.context),data['actions'][old],
                                             projection_passes,preservation_margin,preservation_gap_source)
            self.constraint_indices=np.flatnonzero(old)
        if selective_labels is not None:self.set_selective_labels(policy,selective_labels)

    def set_selective_labels(self,policy,path):
        if self.constraint is None:raise ValueError('Selective retention requires continuous action constraints')
        document=json.loads((ROOT/path).read_text(encoding='utf-8'))
        allowed=torch.zeros((len(self.labels),self.probabilities.shape[1]),dtype=torch.bool)
        # Unknown and harmful states retain their original singleton action.
        allowed[torch.arange(len(self.labels)),torch.as_tensor(self.actions)]=True
        # The stored actual action, rather than a rounded soft-target argmax,
        # is authoritative at near ties.
        allowed[self.constraint_indices]=False
        allowed[self.constraint_indices, self.constraint.actions.cpu()]=True
        lookup={tuple(k):i for i,k in enumerate(self.state_keys)}
        if len(lookup)!=len(self.state_keys):raise ValueError('Ambiguous replay state identity')
        active=[]
        for row in document['rows']:
            if not row.get('safe',False) or not row.get('retention_eligible',True):continue
            key=tuple(row['condition'])+(int(row['decision']),)
            if key not in lookup:continue
            index=lookup[key]
            if index not in self.constraint_indices:raise ValueError('Cannot relax an unprotected replay state')
            original=int(self.constraint.actions[np.flatnonzero(self.constraint_indices==index)[0]])
            if original!=row['old_action']:raise ValueError('Outcome label and executed replay action differ')
            permitted=row['allowed_actions']
            if original not in permitted or len(permitted)!=2 or len(set(permitted))!=2:
                raise ValueError('Only the original and one fully validated alternative may be pooled')
            allowed[index,permitted]=True;active.append(dict(index=index,condition=list(key[:-1]),decision=key[-1],actions=permitted))
        self.allowed=allowed
        self.constraint.set_allowed(policy,allowed[self.constraint_indices])
        self.selective_metadata=dict(labels=str(path),continuation=document['continuation'],
            label_version=document['label_version'],active_states=len(active),active=active,
            expiry='Next checkpoint gate; no inference lookup or future-action information.')

    def update(self, policy, logger):
        started = time.monotonic()
        values = []
        policy.set_training_mode(True)
        for _ in range(self.steps):
            task_labels=None
            if self.tasks is not None:
                indices,task_labels=self.tasks.sample(self.rng)
            elif self.dynamic is not None:
                old_groups=[[t for t in g if not self.categories[t]] for g in self.groups]
                indices=sample_replay(old_groups,self.indices,self.rng)
                new_ids=[t['id'] for t in self.trajectories if t['new_tactic']]
                count=round(128*self.new_fraction) if new_ids else 0
                if count:
                    indices=indices[:128-count]+[int(self.rng.choice(self.indices[int(self.rng.choice(new_ids))])) for _ in range(count)]
                new_mask=torch.as_tensor([self.categories[int(self.labels[i])] for i in indices],device=policy.device)
            else:
                indices = sample_replay(self.groups, self.indices, self.rng,self.categories,self.new_fraction)
            target = torch.as_tensor(self.probabilities[indices], device=policy.device)
            components={};per_values=None
            if self.dynamic is not None:
                observations,_=policy.obs_to_tensor(self.dynamic.observations(indices))
                def current_loss():
                    nonlocal components,per_values
                    current=policy.get_distribution(observations).distribution.probs
                    per=(target*(target.clamp_min(1e-12).log()-current.clamp_min(1e-12).log())).sum(1)
                    if self.tasks is not None:
                        per_values=per.detach().cpu().numpy()
                        components={k:per[torch.as_tensor(np.asarray(task_labels)==k,device=policy.device)].sum()*self.tasks.weights[k]/128
                                    for k in self.tasks.names}
                        return sum(components.values())
                    return (per*torch.where(new_mask,self.demonstration_coefficient,self.coefficient)).mean()
            elif self.context is None:
                observations, _ = policy.obs_to_tensor({k: v[indices] for k, v in self.observations.items()})
                def current_loss():return physical_kl(policy, observations, target)
            else:
                context=tuple(x[indices].to(policy.device) for x in self.context)
                def current_loss():
                    logits=policy.corrected_logits(*context)
                    allowed=None if self.allowed is None else self.allowed[indices].to(policy.device)
                    return retention_kl(logits,target,allowed)
            self.optimizer.zero_grad(set_to_none=True)
            before = current_loss()
            loss = before if self.dynamic is not None else self.coefficient * before
            value_loss=torch.zeros((),device=policy.device)
            if self.dynamic is not None and self.failure_value_coefficient and self.dynamic.new_all:
                vi=self.rng.choice(self.dynamic.new_all,size=32,replace=True)
                vo,_=policy.obs_to_tensor(self.dynamic.observations(vi))
                vt=torch.as_tensor(self.dynamic.conditional_returns(vi),device=policy.device)
                value_loss=F.mse_loss(policy.predict_values(vo).flatten(),vt)
                loss=loss+self.failure_value_coefficient*value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite rehearsal loss')
            if self.tasks is not None:
                self.tasks.record_losses(per_values,task_labels)
                if self.updates%self.tasks.interval==0:self.tasks.gradients(policy,components)
            loss.backward()
            if self.dynamic is not None:
                self.gradient_groups={prefix:float(sum(float(p.grad.square().sum()) for name,p in policy.named_parameters()
                    if name.startswith(prefix) and p.grad is not None)**.5)
                    for prefix in ('features_extractor.','mlp_extractor.policy_net.','action_net.','correction_action_net.','value_net.',
                                   'experts.1.features_extractor.','experts.1.correction_action_net.')}
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, .5)
            self.optimizer.step()
            with torch.no_grad():
                after = current_loss()
            values.append((before.item(), after.item(), float(norm),float(value_loss.detach())))
            self.updates += 1
            self.examples += len(indices)
        self.seconds += time.monotonic()-started
        before, after, gradient, value_loss = np.mean(values, axis=0)
        self.latest = dict(kl_before=float(before), kl_after=float(after), weighted_gradient_norm=float(gradient),
                           optimizer_steps=self.updates, examples=self.examples, seconds=self.seconds,
                           conditional_value_loss=float(value_loss))
        for key, value in self.latest.items():
            logger.record('rehearsal/'+key, value)
        if self.tasks is not None and self.telemetry_path is not None:
            with Path(self.telemetry_path).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(auxiliary_updates=self.updates,tasks=self.tasks.metadata()))+'\n')
        if self.constraint is not None:
            result=self.constraint.project(policy)
            for key,value in result.items():logger.record('preservation/'+key,value)

    def metadata(self):
        return dict(dataset=self.path,replay_reference_report=self.replay_reference_report,replay_refresh=self.old_paths,coefficient=self.coefficient, optimizer=self.optimizer_name,
                    learning_rate=self.optimizer.param_groups[0]['lr'], steps_per_ppo_update=self.steps,
                    batch_size=128, maximum_samples_per_old_trajectory_per_batch=8,
                    maximum_samples_per_new_trajectory_per_batch=(self.tasks.cap if self.tasks is not None else
                        round(128*self.new_fraction) if self.dynamic is not None else 8),
                    optimizer_steps=self.updates, examples=self.examples, seconds=self.seconds, latest=copy.deepcopy(self.latest),
                    continuous_contexts=self.context is not None,cache_seconds=self.cache_seconds,
                    trainable_encoder_legal_replay=self.dynamic is not None,
                    gradients=getattr(self,'gradient_groups',None),failure_value_coefficient=self.failure_value_coefficient,
                    demonstration_coefficient=self.demonstration_coefficient,
                    failed_search_observations=len(self.dynamic.failed) if self.dynamic is not None else 0,
                    new_fraction=self.new_fraction,trajectory_count=len(self.trajectories),
                    new_trajectory_count=sum(t.get('new_tactic',False) for t in self.trajectories),
                    replay_tasks=None if self.tasks is None else self.tasks.metadata(),
                    selective=self.selective_metadata,
                    action_constraints=None if self.constraint is None else self.constraint.metadata())


class ActionConstraints:
    """Training-only half-space projections for a frozen-body linear head.

    These constraints preserve sampled original success actions, which is
    stronger than preserving outcomes and can impede useful learning. They
    provide no guarantee on unsampled states. Complete native games remain
    the acceptance test. No observations or answers are consulted at inference.
    """
    def __init__(self,policy,context,actions,passes=32,margin_cap=.05,gap_source='prior'):
        self.context=tuple(x.to(policy.device) for x in context)
        self.actions=torch.as_tensor(actions,device=policy.device).long()
        if margin_cap<0:raise ValueError('Preservation margin must be nonnegative')
        if gap_source not in ('prior','initial_policy'):raise ValueError('Unknown preservation gap source')
        self.passes=int(passes);self.margin_cap=float(margin_cap);self.anchor=self._parameters(policy)
        self.initial_anchor=[p.clone() for p in self.anchor]
        self.gap_source=gap_source;self.initial_margins=None
        self.allowed=F.one_hot(self.actions,policy.action_space.n if hasattr(policy,'action_space') else
                               self.context[1].shape[1]).bool()
        self.updates=0;self.seconds=0.;self.projections=0;self.backtracks=0;self.rejected=0;self.latest={}
        with torch.no_grad():
            initial=policy.corrected_logits(*self.context)
            if (initial.argmax(1)!=self.actions).any():raise ValueError('Initial old success actions are not faithfully inherited')
            if gap_source=='initial_policy':
                rows=torch.arange(len(self.actions),device=policy.device)
                self.initial_margins=((initial[rows,self.actions,None]-initial)*.5).clamp(0,self.margin_cap)

    def set_allowed(self,policy,allowed):
        allowed=allowed.to(policy.device)
        if allowed.shape!=self.allowed.shape or not allowed[torch.arange(len(self.actions)),self.actions].all():
            raise ValueError('Every safe set must contain the protected executed action')
        revoked=bool((self.allowed&~allowed).any())
        self.allowed=allowed.clone()
        # A refreshed label can expire. Reproject before collecting new data;
        # the immutable original head is the fallback if all passes fail.
        if revoked:self.anchor=[p.clone() for p in self.initial_anchor]
        # Projection is performed once after the ordinary auxiliary update,
        # exactly as in the control arm, before the next rollout is collected.

    @staticmethod
    def _parameters(policy):return [p.detach().clone() for p in policy.correction_action_net.parameters()]

    @torch.no_grad()
    def permission_witness(self,policy,maximum_passes=64,maximum_seconds=180,maximum_delta=.05):
        """Bounded single-target use of the existing physical projection.

        This is a capacity diagnosis, not an RL update or optimality claim.
        Target selection happens once, before fitting. Failed attempts restore
        the complete initial head; no witness initializes formal training.
        """
        started=time.monotonic();initial=self._parameters(policy)
        logits=policy.corrected_logits(*self.context)
        safe=(self.allowed.sum(1)>1).nonzero().flatten()
        if not len(safe):raise ValueError('No validated alternative for feasibility witness')
        alternate=self.allowed[safe].clone();alternate[torch.arange(len(safe)),self.actions[safe]]=False
        targets=alternate.long().argmax(1)
        gaps=logits[safe,self.actions[safe]]-logits[safe,targets]
        choice=int(gaps.argmin());index=int(safe[choice]);target=int(targets[choice])
        features,base,flip=self.context;mirror=policy.correction_physical_mirror
        head=policy.correction_action_net;attempts=[];success=False
        for attempt in range(maximum_passes):
            if time.monotonic()-started>=maximum_seconds:break
            scores=policy.corrected_logits(features[index:index+1],base[index:index+1],flip[index:index+1])[0]
            other=int(scores.argmax());amount=max(0.,float(scores[other]-scores[target])+1e-4)
            canonical_target=int(mirror[target]) if flip[index] else target
            canonical_other=int(mirror[other]) if flip[index] else other
            if other!=target:
                step=amount/(2*(float(features[index].square().sum())+1))
                head.weight[canonical_target].add_(features[index],alpha=step)
                head.weight[canonical_other].add_(features[index],alpha=-step)
                head.bias[canonical_target]+=step;head.bias[canonical_other]-=step
            change=max(float((parameter-old).abs().max()) for parameter,old in zip(head.parameters(),initial))
            if change>maximum_delta:break
            result=self.project(policy)
            chosen=int(policy.corrected_logits(features[index:index+1],base[index:index+1],flip[index:index+1])[0].argmax())
            attempts.append(dict(pass_number=attempt+1,target_chosen=chosen==target,maximum_parameter_delta=change,projection=result))
            if chosen==target and result['unsafe_action_flips_after']==0:
                success=True;break
        if not success:
            for parameter,value in zip(head.parameters(),initial):parameter.copy_(value)
            self.anchor=[p.clone() for p in initial]
        return dict(success=success,chosen_state=index,old_action=int(self.actions[index]),alternative_action=target,
            initial_logit_gap=float(gaps[choice]),target_halfspace_pushes=len(attempts),attempts=attempts,
            seconds=time.monotonic()-started,optimizer_steps=0,native_frames=0,
            scope='One fixed minimum-gap target, bounded existing projections; failure is not proof that the entire head is infeasible. No formal initialization or Q targets.')

    @torch.no_grad()
    def project(self,policy):
        started=time.monotonic();features,base,flip=self.context
        head=policy.correction_action_net;mirror=policy.correction_physical_mirror
        rows=torch.arange(len(self.actions),device=policy.device)
        prior=((1-policy.correction_smoothing)*base+policy.correction_smoothing/base.shape[1]).clamp_min(1e-12).log()/policy.correction_temperature
        margins=(self.initial_margins if self.initial_margins is not None else
                 ((prior[rows,self.actions,None]-prior)*.5).clamp(0,self.margin_cap))
        def violations():
            logits=policy.corrected_logits(features,base,flip)
            chosen=logits.masked_fill(~self.allowed,-torch.inf).argmax(1)
            errors=logits-logits[rows,chosen,None]+margins
            errors[self.allowed]=-torch.inf
            wrong=~self.allowed[rows,logits.argmax(1)]
            maximum=errors.max(1).values
            maximum=torch.where(wrong,maximum.clamp_min(1e-6),maximum)
            return errors,maximum,wrong
        errors,maximum,wrong=violations();before=int((policy.corrected_logits(*self.context).argmax(1)!=self.actions).sum());count=0
        for _ in range(self.passes):
            ix=maximum.topk(min(64,len(maximum))).indices
            if maximum[ix[0]]<=1e-7 and not wrong.any():break
            for index in ix:
                if maximum[index]<=1e-7:continue
                scores=policy.corrected_logits(features[index:index+1],base[index:index+1],flip[index:index+1])[0]
                teacher=int(scores.masked_fill(~self.allowed[index],-torch.inf).argmax())
                error=scores-scores[teacher]+margins[index]
                error[self.allowed[index]]=-torch.inf
                other=int(error.argmax());amount=float(error[other])
                if not self.allowed[index,int(scores.argmax())]:amount=max(amount,1e-6)
                if amount<=1e-7:continue
                canonical_other=int(mirror[other]) if flip[index] else other
                canonical_teacher=int(mirror[teacher]) if flip[index] else teacher
                step=amount/(2*(float(features[index].square().sum())+1))
                head.weight[canonical_teacher].add_(features[index],alpha=step)
                head.weight[canonical_other].add_(features[index],alpha=-step)
                head.bias[canonical_teacher]+=step;head.bias[canonical_other]-=step;count+=1
            errors,maximum,wrong=violations()
        pre_backtrack_flips=int(wrong.sum())
        pre_backtrack_margin_violation=float(maximum.max())
        proposed=self._parameters(policy);fraction=1.
        for attempt in range(18):
            errors,maximum,wrong=violations()
            if not wrong.any() and maximum.max()<=1e-6:break
            fraction=.5**(attempt+1)
            for parameter,old,new in zip(head.parameters(),self.anchor,proposed):parameter.copy_(old+(new-old)*fraction)
            self.backtracks+=1
        errors,maximum,wrong=violations()
        if wrong.any() or maximum.max()>1e-6:
            for parameter,old in zip(head.parameters(),self.anchor):parameter.copy_(old)
            fraction=0.;self.rejected+=1
            _,maximum,wrong=violations()
        self.anchor=self._parameters(policy);self.updates+=1;self.projections+=count;self.seconds+=time.monotonic()-started
        actual_flips=int((policy.corrected_logits(*self.context).argmax(1)!=self.actions).sum())
        self.latest=dict(old_action_flips_before=before,old_action_flips_after=actual_flips,
                         unsafe_action_flips_after=int(wrong.sum()),released_action_choices=actual_flips-int(wrong.sum()),
                         accepted_fraction=fraction,halfspace_projections=count,
                         old_action_flips_before_backtracking=pre_backtrack_flips,
                         max_margin_violation_before_backtracking=pre_backtrack_margin_violation)
        return self.latest

    def metadata(self):
        return dict(old_states=len(self.actions),updates=self.updates,seconds=self.seconds,
            validated_nonsingleton_states=int((self.allowed.sum(1)>1).sum()),
            margin_cap=self.margin_cap,gap_source=self.gap_source,maximum_passes=self.passes,
            halfspace_projections=self.projections,backtracks=self.backtracks,rejected_updates=self.rejected,
            latest=self.latest,scope='Training-state action preservation only; native outcome gates and unseen final tests remain mandatory.')
