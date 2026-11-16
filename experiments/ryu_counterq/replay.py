"""Continuous legal replay; alternative histories share their actual prefix."""
import copy
from collections import defaultdict,deque,Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class Episode:
    """One actual consecutive observation/action history, including its prefix."""
    def __init__(self,obs,group,family):
        self.frames=[x.copy() for x in obs['image']]
        self.positions=[obs['position'].copy()]
        self.actions=[];self.rewards=[];self.dones=[];self.hidden=[];self.previous=[];self.greedy_actions=[]
        self.input_durations=[]
        self.policy_probabilities=[]
        self.group=group;self.family=family;self.learn_start=0

    def before(self,policy):
        values=getattr(policy,'last_context_q_values',getattr(policy,'last_policy_logits',None))
        self.greedy_actions.append(-1 if values is None else int(values[0].argmax()))
        self.hidden.append(policy.last_hidden_before[0,0].cpu().numpy().copy())
        self.previous.append(int(policy.last_previous_before[0]))
        if getattr(policy,'input_history',None) is not None:self.input_durations.append(policy.last_input_durations_before[0].copy())
        if getattr(policy,'probability_policy',False):
            self.policy_probabilities.append(policy.last_memory_probabilities[0].detach().cpu().numpy().copy())

    def append(self,action,reward,done,obs):
        self.actions.append(action);self.rewards.append(reward);self.dones.append(done)
        self.frames.append(obs['image'][-1].copy());self.positions.append(obs['position'].copy())

    def prefix(self,burn):
        start=max(0,len(self.actions)-burn);e=copy.copy(self)
        for name in ('actions','rewards','dones','hidden','previous','positions','input_durations','greedy_actions','policy_probabilities'):
            setattr(e,name,list(getattr(self,name)[start:]))
        e.frames=list(self.frames[start:]);e.learn_start=len(e.actions)
        return e

    def pack(self,policy):
        if len(self.hidden)!=len(self.actions):raise RuntimeError('Actor history is not aligned with executed inputs')
        if policy._sequence_hidden is None:
            if self.actions:raise RuntimeError('Active episode lost its actual action history')
            h=np.zeros(policy.memory_hidden,np.float32);previous=0
        else:
            h=policy._sequence_hidden[0,0].detach().cpu().numpy().copy()
            previous=int(policy._sequence_previous[0])
        result=dict(frames=np.asarray(self.frames,np.uint8),positions=np.asarray(self.positions,np.float32),
            actions=np.asarray(self.actions,np.int64),rewards=np.asarray(self.rewards,np.float32),
            greedy_actions=np.asarray(self.greedy_actions,np.int64),
            dones=np.asarray(self.dones,bool),hidden=np.asarray(self.hidden+[h],np.float16),
            previous=np.asarray(self.previous+[previous],np.int64),group=self.group,family=self.family,
            learn_start=self.learn_start)
        if getattr(policy,'input_history',None) is not None:
            if len(self.input_durations)!=len(self.actions):raise RuntimeError('Input duration replay is not aligned')
            result['input_durations']=np.asarray(self.input_durations+[policy._sequence_input_durations[0]],np.uint8)
        if self.policy_probabilities:
            if len(self.policy_probabilities)!=len(self.actions):raise RuntimeError('Soft action targets are not aligned with actual history')
            result['policy_probabilities']=np.asarray(self.policy_probabilities,np.float32)
        return result


class TrajectoryRecorder:
    """Capture complete independent greedy matches through the shared evaluator."""
    def __init__(self,rivals,policy_version):
        self.rivals={r['id']:copy.deepcopy(r) for r in rivals}
        self.policy_version=policy_version;self.current=None;self.episodes=[]

    def record_transition(self,actor,observation,next_observation,info):
        key=tuple(info[k] for k in ('opponent','player','offset'))
        if self.current is None:
            group='pve' if info['opponent_kind']=='cpu' else f"p{info['player']}"
            self.current=Episode(observation,group,f'{self.policy_version}/{key}')
            self.current_key=key
        if key!=self.current_key:raise RuntimeError('Incomplete recorded fight crossed a reset')
        self.current.before(actor.policy)
        ended=info['end_reason'] is not None
        self.current.append(info['action'],info['step_reward'],ended and info['end_reason']!='frame_cap',next_observation)
        if ended:
            packed=self.current.pack(actor.policy)
            packed.update(source='factual',data_phase='historical_seed',
                opponent_config=copy.deepcopy(self.rivals[info['opponent']]),player=info['player'],offset=info['offset'],
                character=actor.definition.get('character','Ryu'),result=info['result'],end_reason=info['end_reason'],
                validated_complete=info['result']!='cutoff',
                ko_win=info['result']=='win' and info['opponent_hp']<=0 and info['end_reason'] in ('signed_hp_ko','round_counter'),
                final_self_hp=info['self_hp'],final_opponent_hp=info['opponent_hp'],
                policy_version=self.policy_version)
            if getattr(actor.policy,'probability_policy',False):
                packed['reference_temperature']=float(actor.policy.action_temperature)
            self.episodes.append(packed);self.current=None


class SequenceReplay:
    """Contiguous real histories, current visual encoding and sequence TD.

    Raw replay uses stored actor state plus real burn-in; representational
    drift is reduced, not eliminated. Cuts bootstrap at their last real
    observation and padding is masked. Historical feature replay is retained
    only for old data contracts. Sampling balances PvE/P1/P2 with priorities.
    """
    def __init__(self, capacity=60000, seed=1, burn=40, learn=40, nstep=5,
                 byte_capacity=402653184,alpha=.6):
        self.capacity, self.burn, self.learn, self.nstep = int(capacity), int(burn), int(learn), int(nstep)
        self.rng = np.random.default_rng(seed)
        self.episodes = {}
        self.priorities = {}
        self.count = self.transitions = 0
        self.byte_capacity=int(byte_capacity);self.bytes=0;self.alpha=float(alpha)
        self.history_ids=set();self.history_fraction=0.;self.history_sample_fraction=0.
        self.retention_usage={};self.retention_events=[]
        self.auxiliary_sampling_totals={}

    @staticmethod
    def episode_bytes(episode):
        return sum(v.nbytes for v in episode.values() if isinstance(v,np.ndarray))

    def _remove(self,number):
        removed=self.episodes.pop(number);self.transitions-=len(removed['actions'])
        self.bytes-=self.episode_bytes(removed);self.history_ids.discard(number)
        for key in [k for k in self.priorities if k[0]==number]:del self.priorities[key]

    def _trim(self):
        while self.episodes:
            history=[n for n in self.episodes if n in self.history_ids]
            recent=[n for n in self.episodes if n not in self.history_ids]
            hn=sum(len(self.episodes[n]['actions']) for n in history)
            hb=sum(self.episode_bytes(self.episodes[n]) for n in history)
            if self.history_fraction and (hn>int(self.capacity*self.history_fraction) or hb>int(self.byte_capacity*self.history_fraction)):
                self._remove(history[0]);continue
            # The historical cap is a maximum, not empty reserved storage.
            # Recent data can use unused space so both arms share total caps.
            rc=self.capacity-hn
            rb=self.byte_capacity-hb
            if recent and (self.transitions-hn>rc or self.bytes-hb>rb):
                self._remove(recent[0]);continue
            break

    def configure_history(self,numbers,capacity_fraction,sample_fraction):
        if not 0<=capacity_fraction<1 or not 0<=sample_fraction<1:raise ValueError('Invalid bounded history mixture')
        if not set(numbers)<=self.episodes.keys():raise ValueError('Historical sequence is missing')
        self.history_fraction=float(capacity_fraction)
        self.history_sample_fraction=float(sample_fraction) if capacity_fraction else 0.
        self.history_ids=set(numbers) if capacity_fraction else set()
        self._trim()

    def add(self, episode):
        n = len(episode['actions'])
        raw='frames' in episode
        if n < 1 or (len(episode['frames'])!=n+4 if raw else len(episode['maps'])!=n+1) or len(episode['positions']) != n+1:
            raise ValueError('Continuous replay requires one more legal observation than actions')
        if (not raw and not episode['dones'][-1]) or np.any(episode['dones'][:-1]):
            raise ValueError('Replay episodes must end once and cannot cross resets')
        number = self.count
        self.count += 1
        self.episodes[number] = episode
        self.transitions += n
        self.bytes+=self.episode_bytes(episode)
        initial_priority=max(self.priorities.values(), default=1.)
        for anchor in range(int(episode.get('learn_start',0)), n, self.learn):
            self.priorities[(number, anchor)] = initial_priority
        self._trim()
        return number

    def state_dict(self):
        """Bounded local replay snapshot, including real histories and RNG."""
        result=dict(settings=(self.capacity,self.byte_capacity,self.burn,self.learn,self.nstep,self.alpha),
            episodes=self.episodes,priorities=self.priorities,count=self.count,transitions=self.transitions,
            bytes=self.bytes,rng=self.rng.bit_generator.state,
            auxiliary_sampling_totals=copy.deepcopy(self.auxiliary_sampling_totals),
            retention=dict(ids=sorted(self.history_ids),capacity_fraction=self.history_fraction,sample_fraction=self.history_sample_fraction))
        if self.retention_usage or self.retention_events:
            result.update(retention_usage=self.retention_usage,retention_events=self.retention_events)
        return result

    def load_state_dict(self,state,*,allow_capacity_expansion=False):
        expected=(self.capacity,self.byte_capacity,self.burn,self.learn,self.nstep,self.alpha)
        recorded=tuple(state['settings'])
        expanded=(allow_capacity_expansion and recorded[2:]==expected[2:]
            and self.capacity>=recorded[0] and self.byte_capacity>=recorded[1])
        if recorded!=expected and not expanded:raise ValueError('Replay settings differ from the common initialization')
        self.episodes=dict(state['episodes']);self.priorities=dict(state['priorities'])
        self.count=int(state['count']);self.transitions=int(state['transitions']);self.bytes=int(state['bytes'])
        self.rng.bit_generator.state=state['rng']
        # Older snapshots have no exact lifetime totals. Start here rather
        # than inventing counts from a bounded retired-trajectory registry.
        self.auxiliary_sampling_totals=copy.deepcopy(state.get('auxiliary_sampling_totals',{}))
        retention=state.get('retention',{})
        self.history_ids=set(retention.get('ids',[]));self.history_fraction=float(retention.get('capacity_fraction',0.))
        self.history_sample_fraction=float(retention.get('sample_fraction',0.))
        self.retention_usage=copy.deepcopy(state.get('retention_usage',{}));self.retention_events=copy.deepcopy(state.get('retention_events',[]))
        if self.transitions!=sum(len(ep['actions']) for ep in self.episodes.values()):
            raise ValueError('Replay transition accounting differs')
        if self.bytes!=sum(self.episode_bytes(ep) for ep in self.episodes.values()) or not self.history_ids<=self.episodes.keys():
            raise ValueError('Replay storage accounting or historical membership differs')
        if self.transitions>self.capacity or self.bytes>self.byte_capacity:
            raise ValueError('Restored replay exceeds its declared storage budget')

    def account_retention_samples(self,data,valid):
        """Count actual TD use, independently of file presence or configured mass."""
        for i,(number,_) in enumerate(data['keys']):
            if number not in self.retention_usage:continue
            item=self.retention_usage[number];n=int(valid[i].sum())
            item['sampled_sequences']+=1;item['TD_transitions']+=n
            if data['retention_sources'][i]=='history':item['history_TD_transitions']+=n

    def insert_verified_episode(self,episode,frame,require_success=True):
        if not episode.get('validated_complete') or (require_success and not successful_episode(episode)):
            raise ValueError('Historical success requires a complete native win/CPU KO')
        if 'greedy_actions' not in episode or not np.array_equal(episode['actions'],episode['greedy_actions']):
            raise ValueError('Refresh collection must be independent greedy control')
        fresh=copy.deepcopy(episode);fresh['verified_frame']=int(frame)
        for number,old in self.episodes.items():
            if same_episode_condition(old,fresh) and np.array_equal(old['actions'],fresh['actions']) and np.array_equal(old['frames'],fresh['frames']) and np.array_equal(old['positions'],fresh['positions']):
                # Same deterministic experience: refresh its genuinely observed
                # actor h/provenance, never count another copy as diversity.
                fresh['retention_role']=old.get('retention_role','discovered_success')
                self.bytes+=self.episode_bytes(fresh)-self.episode_bytes(old);self.episodes[number]=fresh
                return number,False
        fresh.setdefault('retention_role','discovered_success' if successful_episode(fresh) else 'difficult_failure')
        return self.add(fresh),True

    def replace_verified_success(self,episode,frame,maximum_episodes):
        """One bounded, independently won self-trajectory per exact condition.

        This bank can also supply separate policy supervision. Its rewards
        never become PPO advantages; use and retirement are counted explicitly.
        """
        if not successful_episode(episode) or not episode.get('validated_complete'):
            raise ValueError('A success bank requires a complete native win/CPU KO')
        if not np.array_equal(episode['actions'],episode.get('greedy_actions')):
            raise ValueError('A verified self-success must use independent greedy control')
        matches=[n for n,e in self.episodes.items() if same_episode_condition(e,episode)]
        controlled=lambda e:e.get('controlled_replay_equal',False) and not e.get('independent_learned_control',False)
        if sum(not controlled(self.episodes[n]) for n in matches)>1:
            raise ValueError('Success bank contains duplicate independent condition slots')
        if matches:
            number=matches[0];old=self.episodes[number]
            same_reference=(old.get('retention_target','executed_action')==episode.get('retention_target','executed_action')
                and np.array_equal(old.get('policy_probabilities'),episode.get('policy_probabilities')))
            if len(matches)==1 and not controlled(old) and same_reference and all(np.array_equal(old[k],episode[k]) for k in ('actions','frames','positions')):
                if episode.get('auxiliary_priority',False):
                    old['auxiliary_priority']=True;self.retention_usage[number]['auxiliary_priority']=True
                if episode.get('learner_success_owner'):
                    old['learner_success_owner']=episode['learner_success_owner']
                    old['last_self_verification']=dict(policy_version=episode['policy_version'],frame=int(frame))
                return number,False
            # Actual independent mastery replaces its conditional coaches,
            # rather than accumulating permanent parallel teaching versions.
            for number in matches:
                self.retention_usage[number]['retired_frame']=int(frame);self._remove(number)
        elif len(self.episodes)>=maximum_episodes:
            number=min(self.episodes)
            self.retention_usage[number]['retired_frame']=int(frame);self._remove(number)
        episode=copy.deepcopy(episode);episode['independent_learned_control']=True;episode['supervision_start']=0
        if any(self.retention_usage[n].get('auxiliary_priority',False) for n in matches):
            episode['auxiliary_priority']=True
        number,added=self.insert_verified_episode(episode,frame)
        if number not in self.episodes:raise ValueError('A complete success exceeds the configured bank capacity')
        self.retention_usage[number]=dict(first_protected_frame=int(frame),retired_frame=None,
            sampled_sequences=0,auxiliary_transitions=0,TD_transitions=0,history_TD_transitions=0,
            policy_version=episode['policy_version'],opponent=episode['opponent_config'],
            player=episode['player'],offset=episode['offset'],group=episode['group'])
        self.retention_usage[number]['auxiliary_priority']=bool(episode.get('auxiliary_priority',False))
        for n,item in self.retention_usage.items():
            if n not in self.episodes and item['retired_frame'] is None:item['retired_frame']=int(frame)
        retired=[n for n in self.retention_usage if n not in self.episodes]
        for n in retired[:-128]:del self.retention_usage[n]
        return number,added

    def add_verified_demonstration(self,episode,frame=0,*,replace_condition=False):
        """Replayed controlled or independent wins supply auxiliary data only.

        Keep its real source-greedy labels; never invent greedy/on-policy
        status for a search controller. This bank may supervise actions only.
        """
        independent=(episode.get('independent_learned_control',False) and episode.get('validated_complete')
            and np.array_equal(episode['actions'],episode.get('greedy_actions')))
        if not successful_episode(episode) or not (episode.get('controlled_replay_equal') or independent):
            raise ValueError('A demonstration needs a reproduced complete native win/CPU KO')
        if not 0<=episode.get('supervision_start',0)<len(episode['actions']):
            raise ValueError('A demonstration needs a nonempty supervised suffix')
        fresh=copy.deepcopy(episode);fresh['supervision_only']=True
        fresh['source']='verified_independent_success' if independent else 'verified_controlled_demonstration'
        fresh['verified_frame']=int(frame)
        if replace_condition:
            for number,old in list(self.episodes.items()):
                if not same_episode_condition(old,fresh):continue
                self.retention_usage[number]['retired_frame']=int(frame)
                self.retention_usage[number]['retirement_reason']='replaced_by_verified_coherent_response'
                self._remove(number)
        number=self.add(fresh)
        if number not in self.episodes:raise ValueError('The full demonstration exceeds the bounded bank')
        self.retention_usage[number]=dict(first_protected_frame=int(frame),retired_frame=None,
            sampled_sequences=0,auxiliary_transitions=0,TD_transitions=0,history_TD_transitions=0,
            policy_version=fresh['policy_version'],opponent=fresh['opponent_config'],
            player=fresh['player'],offset=fresh['offset'],group=fresh['group'])
        self.retention_usage[number]['auxiliary_priority']=bool(fresh.get('auxiliary_priority',False))
        return number

    def append_verified_successes(self,episodes,maximum_episodes,*,include_actual_prefix=False,replace_existing_conditions=False):
        """First compatible full win per condition, in collection order.

        Controlled actions remain separate supervision, including any actual
        source prefix. Exact conflicting histories stay unresolved; this is
        neither a best-action oracle nor on-policy PPO experience.
        """
        original=list(self.episodes.values());pending=[];selected=[];rejected=[];replaced=[]
        for episode in episodes:
            if not any(same_episode_condition(episode,old) for old in pending):pending.append(episode)
        if replace_existing_conditions:
            # Remove all explicitly repaired teaching versions together;
            # their mutual old labels need not be jointly executable. The
            # complete replacement set must restore every condition below.
            for number,old in list(self.episodes.items()):
                if not any(same_episode_condition(old,e) for e in pending):continue
                replaced.append(dict(opponent=old['opponent_config'],player=old['player'],offset=old['offset'],source=old['policy_version']))
                self.retention_usage[number]['retired_frame']=0
                self.retention_usage[number]['retirement_reason']='verified_common_prefix_repair'
                self._remove(number)
            if audit_success_history_labels(self.episodes.values(),use_soft_references=False)['both_hard_target_conflicts']:
                raise ValueError('Unresolved hard history conflicts remain outside the declared repaired conditions')
        for condition in pending:
            if any(same_episode_condition(condition,old) for old in self.episodes.values()):
                raise ValueError('Appending discovery data must not silently replace an existing condition')
            for episode in episodes:
                if not same_episode_condition(episode,condition):continue
                fresh=copy.deepcopy(episode)
                if include_actual_prefix:
                    if not fresh.get('controlled_replay_equal') or not fresh.get('validated_complete'):
                        raise ValueError('Full-prefix labels require a complete physically reproduced game')
                    fresh['original_supervision_start']=fresh.get('supervision_start',0)
                    fresh['supervision_start']=0
                    fresh['prefix_label_origin']='Actually executed source-policy prefix of this complete winning continuation; not teacher/oracle labels.'
                fresh.pop('policy_probabilities',None)
                fresh.pop('reference_temperature',None)
                fresh['retention_target']='executed_action'
                audit=audit_success_history_labels([*self.episodes.values(),fresh],use_soft_references=False)
                if audit['both_hard_target_conflicts']:
                    rejected.append(dict(source=fresh['policy_version'],opponent=fresh['opponent_config'],
                        player=fresh['player'],offset=fresh['offset'],conflicts=audit['examples']))
                    continue
                if len(self.episodes)>=maximum_episodes:raise ValueError('Verified success episode capacity reached')
                number=self.add_verified_demonstration(fresh)
                selected.append(dict(episode=number,opponent=fresh['opponent_config'],player=fresh['player'],
                    offset=fresh['offset'],source=fresh['policy_version'],decisions=len(fresh['actions']),
                    original_supervision_start=fresh.get('original_supervision_start',fresh.get('supervision_start',0)),
                    supervision_start=fresh.get('supervision_start',0),end_reason=fresh.get('end_reason'),
                    ko_win=fresh.get('ko_win',False)))
                break
            else:raise ValueError('No conflict-free verified full-game continuation for a declared discovery condition')
        if any(not any(same_episode_condition(old,new) for new in self.episodes.values()) for old in original):
            raise ValueError('Verified repair lost an original complete condition')
        return dict(selected=selected,rejected=rejected,replaced_conditions=replaced,
            all_original_conditions_retained=True,all_original_episodes_retained=not replaced,
            episodes=len(self.episodes),transitions=self.transitions,bytes=self.bytes,native_frames=0,
            selection='First physically verified compatible win per exact condition in original order; no health-based selection.',
            scope='Training/development experience only; controlled continuations are not independent student mastery or PPO advantage samples.')

    def refresh_self_successes(self,episodes,frame,maximum_episodes,owner):
        """One shared refresh for actual complete greedy learner victories.

        Conditional coaches never certify mastery. PPO probabilities are
        physical reference distributions; Q scores never become such labels.
        These episodes supply separate supervision, never PPO advantages.
        """
        changes=[]
        for episode in episodes:
            if not successful_episode(episode):continue
            if (not episode.get('validated_complete') or episode.get('controlled_replay_equal')
                    or not np.array_equal(episode['actions'],episode.get('greedy_actions'))):
                raise ValueError('Self refresh needs an autonomous complete greedy game')
            fresh=copy.copy(episode)
            fresh.update(learner_success_owner=owner,supervision_only=True,
                retention_target='physical_distribution' if 'policy_probabilities' in episode else 'executed_action')
            number,changed=self.replace_verified_success(fresh,frame,maximum_episodes)
            changes.append(dict(episode=number,changed=changed,opponent=fresh['opponent_config']['id'],
                player=fresh['player'],offset=fresh['offset'],policy_version=fresh['policy_version']))
        return changes

    def success_sequences(self,count,length,disagreement_fraction=0.,disagreement_scope='failed',priority_fraction=0.,disagreement_selection='random'):
        """Group-balanced, distinct actual histories for an auxiliary loss."""
        if disagreement_scope not in ('failed','verified_success'):raise ValueError('Unknown success-disagreement scope')
        if disagreement_selection not in ('random','first'):raise ValueError('Unknown success frontier selection')
        if not 0<=priority_fraction<=1:raise ValueError('Invalid within-group new-skill sampling fraction')
        if len(self.episodes)<count:raise ValueError('Distinct success trajectories are missing')
        groups=['pve','p1','p2'];self.rng.shuffle(groups);chosen=[]
        for i in range(count):
            available=[n for n,e in self.episodes.items() if n not in chosen and e['group']==groups[i%3]]
            if not available:available=[n for n in self.episodes if n not in chosen]
            if priority_fraction:
                new=[n for n in available if self.episodes[n].get('auxiliary_priority',False)]
                old=[n for n in available if n not in new]
                if new and old:available=new if self.rng.random()<priority_fraction else old
            number=int(self.rng.choice(available));chosen.append(number)
        result=[]
        for number in chosen:
            episode=self.episodes[number];minimum=int(episode.get('supervision_start',0))
            first=int(self.rng.integers(minimum,len(episode['actions'])))
            errors=episode.get('supervised_disagreements',np.empty(0,np.int64))
            eligible=(episode.get('current_complete_useful') is False if disagreement_scope=='failed' else successful_episode(episode))
            focused=(disagreement_fraction>0 and eligible
                and len(errors)>0 and self.rng.random()<disagreement_fraction)
            if focused:
                center=int(self.rng.choice(errors))
                if disagreement_selection=='first':center=int(errors[0])
                first=max(minimum,min(center-length//2,len(episode['actions'])-length))
            last=min(first+length,len(episode['actions']))
            use=self.retention_usage[number];use['sampled_sequences']+=1;use['auxiliary_transitions']+=last-first
            total=self.auxiliary_sampling_totals.setdefault(episode['group'],dict(sequences=0,transitions=0))
            total['sequences']+=1;total['transitions']+=last-first
            use['phase_auxiliary_transitions']=use.get('phase_auxiliary_transitions',0)+last-first
            key='disagreement_sequences' if focused else 'uniform_sequences';use[key]=use.get(key,0)+1
            use['covered_disagreements']=use.get('covered_disagreements',0)+int(((errors>=first)&(errors<last)).sum())
            episode['auxiliary_registry_id']=number
            result.append((episode,first,last))
        return result

    def note_complete_results(self,rows,*,self_success_owner=None,prioritize_failures=False):
        """Update sampler outcomes; identify full self-successes needing capture.

        A teacher/controlled success does not certify this learning run's
        mastery. Also refresh a regained win, rather than treating an old
        condition slot as a permanent substitute for current experience.
        """
        needed=[]
        for row in rows:
            matches=[(number,episode) for number,episode in self.episodes.items() if episode['opponent_config']==row['opponent_config']
                and episode['player']==row['player'] and episode['offset']==row['offset']]
            useful=bool(row['result']=='win' and (row['opponent_kind']!='cpu' or row['ko_win']))
            if useful and (not any(e.get('current_complete_useful') is True for _,e in matches)
                    or self_success_owner is not None and not any(e.get('learner_success_owner')==self_success_owner for _,e in matches)):
                needed.append(row)
            for number,episode in matches:
                episode['current_complete_useful']=useful
                if prioritize_failures:
                    episode['auxiliary_priority']=not useful
                    if number in self.retention_usage:self.retention_usage[number]['auxiliary_priority']=not useful
        return needed

    def assess_success_disagreements(self,policy,optimizer_update):
        """Current full-history action fit; no old hidden-state or Q labels."""
        total=0
        for episode in self.episodes.values():
            scores=full_values(policy,episode)[:-1]
            errors=torch.nonzero(scores.argmax(1)!=torch.as_tensor(episode['actions'],device=scores.device)).flatten().cpu().numpy()
            errors=errors[errors>=episode.get('supervision_start',0)]
            self.bytes-=self.episode_bytes(episode)
            episode['supervised_disagreements']=errors
            episode['disagreement_assessed_at_update']=int(optimizer_update)
            self.bytes+=self.episode_bytes(episode);total+=len(errors)
        if self.bytes>self.byte_capacity:raise ValueError('Disagreement metadata exceeds the bounded replay budget')
        return dict(optimizer_update=int(optimizer_update),episodes=len(self.episodes),disagreements=total,
            failed_conditions=sum(e.get('current_complete_useful') is False for e in self.episodes.values()))

    def refresh_history(self,frame,spec):
        """Bounded old/new-success consolidation; no action or Q-value lock."""
        used=[];size=byte_count=0;maximum=int(self.capacity*spec['capacity_fraction'])
        byte_maximum=int(self.byte_capacity*spec['capacity_fraction'])
        def take(number,required=False):
            nonlocal size,byte_count
            if number in used:return True
            e=self.episodes[number];n=len(e['actions']);b=self.episode_bytes(e)
            fits=(len(used)<spec['maximum_episodes'] and size+n<=maximum and byte_count+b<=byte_maximum)
            if not fits:
                if required:raise ValueError('Declared bounded bank cannot fit its old/starting anchors')
                return False
            used.append(number);size+=n;byte_count+=b;return True
        for role in ('original_success','starting_success'):
            for number,e in self.episodes.items():
                if e.get('retention_role')==role:take(number,required=True)
        failures={g:[] for g in ('pve','p1','p2')};new={g:[] for g in failures}
        for number,e in self.episodes.items():
            if e.get('retention_role')=='difficult_failure' and e.get('validated_complete'):failures[e['group']].append(number)
            if e.get('retention_role')=='discovered_success':new[e['group']].append(number)
        for g in failures:
            failures[g].sort(key=lambda n:(-self.episodes[n].get('final_opponent_hp',0),n))
            for number in failures[g][:spec['minimum_failures_per_group']]:take(number,required=True)
            new[g].sort(key=lambda n:(-self.episodes[n].get('verified_frame',0),-n))
            unique=[]
            for number in new[g]:
                if not any(same_episode_condition(self.episodes[number],self.episodes[n]) for n in unique):unique.append(number)
            new[g]=unique
        for row in zip_longest(*(new[g] for g in ('pve','p1','p2'))):
            for number in row:
                if number is not None:take(number)
        for row in zip_longest(*(failures[g] for g in ('pve','p1','p2'))):
            for number in row:
                if number is not None:take(number)
        before=set(self.history_ids);after=set(used);events=[]
        for number in sorted(before-after):
            item=self.retention_usage.get(number)
            if item is not None:
                item['retired_frame']=int(frame)
                events.append(dict(event='retired',episode=number,**copy.deepcopy(item)))
        for number in used:
            e=self.episodes[number]
            if number not in self.retention_usage:
                self.retention_usage[number]=dict(first_protected_frame=int(frame),retired_frame=None,
                    sampled_sequences=0,TD_transitions=0,history_TD_transitions=0,role=e['retention_role'],
                    family=e['family'],opponent=e['opponent_config'],player=e['player'],offset=e['offset'])
            if number not in before:
                self.retention_usage[number]['retired_frame']=None
                events.append(dict(event='protected',episode=number,**copy.deepcopy(self.retention_usage[number])))
        self.configure_history(used,spec['capacity_fraction'],spec['sample_fraction'])
        if self.history_ids!=after:raise RuntimeError('History capacity logic silently discarded selected evidence')
        retired=[n for n,v in self.retention_usage.items() if n not in after]
        for number in retired[:-256]:del self.retention_usage[number]
        self.retention_events=events
        return dict(frame=int(frame),events=events,protected_episodes=len(after),protected_transitions=size,
            protected_bytes=byte_count,roles=dict(Counter(self.episodes[n]['retention_role'] for n in after)),
            actual_TD_use={str(n):copy.deepcopy(self.retention_usage[n]) for n in after})

    def sample(self, count, *, early=False, beta=.4, group=None):
        if any(e.get('supervision_only',False) for e in self.episodes.values()):
            raise ValueError('Auxiliary demonstration banks are not ordinary TD/on-policy replay')
        if 'frames' in next(iter(self.episodes.values())):
            return self._sample_raw(count,beta=beta,group=group)
        burn = 0 if early else self.burn
        candidates = defaultdict(list)
        for key in self.priorities:
            if (key[1] == 0) == bool(early) and (group is None or self.episodes[key[0]]['group']==group):
                candidates[self.episodes[key[0]]['group']].append(key)
        if not candidates:
            raise ValueError('No complete sequences of the requested type')
        groups = sorted(candidates)
        selected, probabilities, uniform_probabilities = [], [], []
        for _ in range(count):
            group = groups[int(self.rng.integers(len(groups)))]
            keys = candidates[group]
            mass = np.asarray([self.priorities[k] for k in keys], np.float64)
            mass /= mass.sum()
            index = int(self.rng.choice(len(keys), p=mass))
            selected.append(keys[index])
            probabilities.append(float(mass[index])/len(groups))
            uniform_probabilities.append(1./(len(groups)*len(keys)))
        length = burn+self.learn+self.nstep
        width = next(iter(self.episodes.values()))['maps'].shape[1]
        maps = np.zeros((count, length, width), np.float16)
        positions = np.zeros((count, length, 6), np.float32)
        previous = np.zeros((count, length), np.int64)
        actions = np.zeros((count, length), np.int64)
        rewards = np.zeros((count, length), np.float32)
        dones = np.ones((count, length), bool)
        valid = np.zeros((count, length), bool)
        action_count=next((ep['targets'].shape[-1] for ep in self.episodes.values() if 'targets' in ep),0)
        targets = np.zeros((count, length, action_count), np.float32)
        supervised = np.zeros((count, length), bool)
        returns = np.zeros((count, length), np.float32)
        for i, (number, anchor) in enumerate(selected):
            ep = self.episodes[number]
            begin = max(0,anchor-burn)
            obs_count = min(length, len(ep['maps'])-begin)
            count_actions = min(length, len(ep['actions'])-begin)
            maps[i, :obs_count] = ep['maps'][begin:begin+obs_count]
            positions[i, :obs_count] = ep['positions'][begin:begin+obs_count]
            prev = np.concatenate(([0], ep['actions']))
            previous[i, :obs_count] = prev[begin:begin+obs_count]
            actions[i, :count_actions] = ep['actions'][begin:begin+count_actions]
            rewards[i, :count_actions] = ep['rewards'][begin:begin+count_actions]
            dones[i, :count_actions] = ep['dones'][begin:begin+count_actions]
            valid[i, :count_actions] = True
            if 'targets' in ep:
                targets[i, :count_actions] = ep['targets'][begin:begin+count_actions]
                supervised[i, :count_actions] = ep['supervised'][begin:begin+count_actions]
                returns[i, :count_actions] = ep['returns'][begin:begin+count_actions]
        # Preserve raw target/sampling ratios across the learner's mixed
        # singleton draws. Per-draw normalization would silently make all1.
        # The target balances groups, then samples windows uniformly within
        # each group. The learner normalizes the combined weighted TD loss.
        weights = np.power(np.asarray(uniform_probabilities)/np.asarray(probabilities), float(beta))
        result = {k:torch.as_tensor(v) for k,v in dict(maps=maps, positions=positions, previous=previous,
            actions=actions, rewards=rewards, dones=dones, valid=valid, targets=targets,
            supervised=supervised, returns=returns, weights=weights.astype(np.float32)).items()}
        result.update(keys=selected, burn=burn)
        return result

    def _sample_raw(self,count,beta=.4,group=None):
        """Current visual encoding, stored actor h, real burn-in and cut masks.

        Branches carry up to burn preceding *actual* transitions. Only their
        alternative suffix is eligible for TD; source families are retained.
        Short nonterminal cuts bootstrap at their actual last observation.
        """
        groups=defaultdict(list)
        for key in self.priorities:
            ep=self.episodes[key[0]]
            if group is None or ep['group']==group:groups[ep['group']].append(key)
        # A start window must not remove warm-up from unrelated late windows
        # in the same batch. Use a homogeneous available-burn bucket instead
        # of silently taking the minimum anchor over a heterogeneous batch.
        g=sorted(groups)[int(self.rng.integers(len(groups)))];all_keys=groups[g]
        historical=[k for k in all_keys if k[0] in self.history_ids]
        recent=[k for k in all_keys if k[0] not in self.history_ids]
        partition='recent'
        if historical:
            use_history=not recent or self.rng.random()<self.history_sample_fraction
            all_keys=historical if use_history else recent;partition='history' if use_history else 'recent'
        buckets=defaultdict(list)
        for key in all_keys:buckets[min(self.burn,key[1])].append(key)
        lengths=sorted(buckets)
        sizes=np.asarray([len(buckets[n]) for n in lengths],np.float64)
        burn=int(self.rng.choice(lengths,p=sizes/sizes.sum()));keys=buckets[burn]
        mass=np.asarray([self.priorities[k]**self.alpha for k in keys]);mass/=mass.sum()
        # Group and bucket probabilities cancel against the desired uniform
        # window target. PER operates within the selected real-history bucket.
        selected=[];weights=[];families=[]
        for _ in range(count):
            i=int(self.rng.choice(len(keys),p=mass));selected.append(keys[i]);weights.append((1./(len(keys)*mass[i]))**beta)
        length=burn+self.learn+self.nstep
        image_shape=next(iter(self.episodes.values()))['frames'].shape[-2:]
        arrays=dict(images=np.zeros((count,length,4,*image_shape),np.uint8),
            positions=np.zeros((count,length,6),np.float32),previous=np.zeros((count,length),np.int64),
            actions=np.zeros((count,length),np.int64),rewards=np.zeros((count,length),np.float32),
            dones=np.zeros((count,length),bool),valid=np.zeros((count,length),bool),
            collector_greedy=np.zeros((count,length),bool),collector_known=np.zeros((count,length),bool))
        has_durations='input_durations' in next(iter(self.episodes.values()))
        if has_durations:arrays['input_durations']=np.zeros((count,length,3),np.uint8)
        hidden=[];sources=[];phases=[]
        for i,(number,anchor) in enumerate(selected):
            ep=self.episodes[number];begin=anchor-burn;n=len(ep['actions']);end=min(n+1,begin+length)
            for j,t in enumerate(range(begin,end)):
                arrays['images'][i,j]=ep['frames'][t:t+4]
                arrays['positions'][i,j]=ep['positions'][t]
                arrays['previous'][i,j]=ep['previous'][t]
                if has_durations:arrays['input_durations'][i,j]=ep['input_durations'][t]
            # Repeat the last real observation only for numerical padding;
            # its transitions remain invalid and never become fake terminals.
            if end-begin<length:
                for field in ('images','positions','previous')+(('input_durations',) if has_durations else ()):
                    arrays[field][i,end-begin:]=arrays[field][i,end-begin-1]
            m=min(length,n-begin)
            for field in ('actions','rewards','dones'):arrays[field][i,:m]=ep[field][begin:begin+m]
            if 'greedy_actions' in ep:
                reference=ep['greedy_actions'][begin:begin+m]
                arrays['collector_known'][i,:m]=reference>=0
                arrays['collector_greedy'][i,:m]=ep['actions'][begin:begin+m]==reference
            arrays['valid'][i,:m]=True
            hidden.append(ep['hidden'][begin]);families.append(ep['family']);sources.append(ep.get('source','factual'))
            phases.append(ep.get('data_phase','online'))
        result={k:torch.as_tensor(v) for k,v in arrays.items()}
        result.update(hidden=torch.as_tensor(np.stack(hidden))[None].float(),weights=torch.tensor(weights,dtype=torch.float32),
                      keys=selected,burn=burn,families=families,sources=sources,data_phases=phases,sample_groups=[g]*count,
                      retention_sources=[partition]*count,
                      collector_update_versions=[self.episodes[n].get('actor_update_version',-1) for n,_ in selected])
        return result

    def update(self, keys, td, valid):
        for key, error, mask in zip(keys, td.detach().abs().cpu().numpy(), valid.cpu().numpy()):
            values = error[mask]
            if len(values):
                self.priorities[key] = float(.9*values.max()+.1*values.mean()+1e-4)

    def save_examples(self,directory,limit=8,max_bytes=67108864):
        """Keep bounded continuous evidence, not an unbounded replay dump."""
        import json
        from experiments.league_adaptive.runtime import write
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        selected={}
        for number,episode in reversed(list(self.episodes.items())):
            if 'frames' not in episode:continue
            key=(episode['group'],episode.get('source','factual'),episode.get('result','unknown'))
            selected.setdefault(key,(number,episode))
        entries=[];used=0
        # Round-robin groups and sources, preserving failures as well as wins.
        buckets=defaultdict(list)
        for value in selected.values():buckets[value[1]['group']].append(value)
        for values in buckets.values():
            values.sort(key=lambda item:(item[1].get('source','factual')!='factual',
                        {'win':0,'loss':1}.get(item[1].get('result'),2)))
        ordered=[item for row in zip_longest(*(buckets[g] for g in sorted(buckets))) for item in row if item is not None]
        for number,episode in ordered:
            if len(entries)>=limit:break
            path=directory/f'episode_{number}.npz'
            arrays={k:v for k,v in episode.items() if isinstance(v,np.ndarray)}
            metadata={k:v for k,v in episode.items() if not isinstance(v,np.ndarray)}
            np.savez_compressed(path,**arrays,metadata=np.frombuffer(json.dumps(metadata).encode(),np.uint8))
            size=path.stat().st_size
            if used+size>max_bytes:
                path.unlink();continue
            used+=size;entries.append(dict(file=path.name,bytes=size,transitions=len(episode['actions']),**metadata))
        write(directory/'manifest.json',dict(entries=entries,bytes=used,limit=limit,
             scope='Actual consecutive legal frames, executed inputs, rewards and actor histories. Branch suffixes are off-policy transitions, not independent victories or imitation labels.'))
        return dict(episodes=len(entries),bytes=used)


def audit_success_history_labels(episodes,use_soft_references=False,maximum_examples=32,hard_failed_conditions=False):
    """Find disagreeing labels after exactly equal recorded legal prefixes.

    Whole-game collectors establish the initial continuity reset. Compare
    raw observations and actually executed history directly; never use a
    stored teacher hidden state to equate two different network histories.
    A disagreement is a fitting conflict, not evidence either action loses.
    """
    episodes=list(episodes);matches=hard_conflicts=0;examples=[]
    for i,left in enumerate(episodes):
        for right in episodes[i+1:]:
            count=min(len(left['actions']),len(right['actions']))
            different=np.flatnonzero(left['actions'][:count]!=right['actions'][:count])
            if not len(different):continue
            at=int(different[0])
            if not np.array_equal(left['positions'][:at+1],right['positions'][:at+1]):continue
            if not np.array_equal(left['frames'][:at+4],right['frames'][:at+4]):continue
            if any(key in left and key in right and not np.array_equal(left[key][:at+1],right[key][:at+1])
                    for key in ('previous','input_durations')):continue
            if any(at<e.get('supervision_start',0) for e in (left,right)):continue
            soft=[use_soft_references and e.get('retention_target')=='physical_distribution'
                and not (hard_failed_conditions and e.get('current_complete_useful') is False) for e in (left,right)]
            matches+=1;hard_conflicts+=not any(soft)
            if len(examples)<maximum_examples:
                examples.append(dict(decision=at,actions=[int(e['actions'][at]) for e in (left,right)],
                    soft_targets=soft,opponents=[e.get('opponent_config') for e in (left,right)],
                    ports=[e.get('player') for e in (left,right)],offsets=[e.get('offset') for e in (left,right)],
                    sources=[e.get('policy_version') for e in (left,right)]))
    return dict(episodes=len(episodes),exact_prefix_action_disagreements=matches,
        both_hard_target_conflicts=hard_conflicts,examples=examples,
        scope='Exact recorded observation/executed-input prefixes only; no file digests, optimal-action labels or global aliasing claim.')


def successful_episode(episode):
    return episode.get('ko_win',False) if episode['opponent_config']['kind']=='cpu' else episode.get('result')=='win'


def same_episode_condition(first,second):
    return (first.get('opponent_config')==second.get('opponent_config') and
        first.get('player')==second.get('player') and first.get('offset')==second.get('offset'))


def select_history(replay,quotas):
    """Deterministic capped successes and diverse hard failures, not Q labels."""
    chosen=[];conditions=set()
    for group,quota in quotas.items():
        available=[n for n,e in replay.episodes.items() if e['group']==group and e.get('validated_complete')]
        success=lambda e:e.get('ko_win',False) if e['opponent_config']['kind']=='cpu' else e.get('result')=='win'
        wins=[n for n in available if success(replay.episodes[n])]
        failures=sorted([n for n in available if n not in wins],
            key=lambda n:(-replay.episodes[n].get('final_opponent_hp',0),n))
        used_roles=set();ordered=list(wins)
        for n in failures:
            role=replay.episodes[n]['opponent_config'].get('character','Ryu')
            if role not in used_roles:ordered.append(n);used_roles.add(role)
        ordered.extend(n for n in failures if n not in ordered)
        selected=0
        for n in ordered:
            e=replay.episodes[n];key=(e['opponent_config']['id'],e['player'],e['offset'])
            if key in conditions:continue
            chosen.append(n);conditions.add(key);selected+=1
            if selected>=quota:break
    return chosen


def reconstruct_input_durations(episode,commands,cap=120):
    """Rebuild legal counters from a complete actual issued-command history."""
    from experiments.league_adaptive.characters import advance_input_durations
    if episode.get('learn_start',0)!=0:raise ValueError('A cut prefix requires its actual preceding duration state')
    values=[np.zeros(3,np.uint8)]
    for action in episode['actions']:values.append(advance_input_durations(values[-1],commands[int(action)],cap))
    return np.stack(values)


def episode_values(policy,episode,first,last,hidden=None,burn=0,return_features=False):
    """Read a contiguous legal interval with current visual/recurrent weights."""
    images=torch.as_tensor(np.stack([episode['frames'][i:i+4] for i in range(first,last)]))[None]
    extra={} if 'input_durations' not in episode else dict(input_durations=torch.as_tensor(episode['input_durations'][first:last][None]))
    if return_features:extra['return_features']=True
    return policy.observation_sequence_values(images,torch.as_tensor(episode['positions'][first:last][None]),
        torch.as_tensor(episode['previous'][first:last][None]),hidden,burn,**extra)


def full_values(policy,episode):
    """Current scores (Q or PPO logits) along one complete actual history."""
    values=[];hidden=None
    with torch.no_grad():
        for first in range(0,len(episode['actions'])+1,32):
            scores,hidden=episode_values(policy,episode,first,min(first+32,len(episode['actions'])+1),hidden)
            values.append(scores[0])
    return torch.cat(values)


def trajectory_policy_audit(policy,episode):
    """Teacher-history preference and conditional return; no optimal-Q label."""
    was_training=policy.training;policy.set_training_mode(False)
    try:
        q=full_values(policy,episode)[:-1].cpu()
    finally:policy.set_training_mode(was_training)
    actions=torch.as_tensor(episode['actions']);selected=q.gather(1,actions[:,None])[:,0]
    gap=q.max(1).values-selected
    observed=torch.as_tensor(np.cumsum(episode['rewards'][::-1],dtype=np.float64)[::-1].copy(),dtype=torch.float32)
    return dict(decisions=len(actions),recorded_action_is_current_greedy=int((q.argmax(1)==actions).sum()),
        mean_gap_to_current_greedy=float(gap.mean()),maximum_gap_to_current_greedy=float(gap.max()),
        recorded_action_mean_q=float(selected.mean()),current_greedy_mean_q=float(q.max(1).values.mean()),
        recorded_continuation_mean_return=float(observed.mean()),
        recorded_action_return_absolute_gap=float((selected-observed).abs().mean()),
        scope='Current network re-encodes the full recorded legal history and executed previous inputs. Return describes the recorded controller at gamma1; it is neither optimal Q truth nor current-greedy continuation value. Action disagreement alone is not tactical error.')


def recurrent_drift_audit(policy,replay,count=12):
    """Current full-prefix reconstruction versus stored or zero h plus burn-in."""
    import time
    began=time.monotonic();was_training=policy.training;policy.set_training_mode(False)
    groups=defaultdict(list)
    for number,e in replay.episodes.items():
        anchors=[a for n,a in replay.priorities if n==number and a>=replay.burn]
        if anchors:
            anchor=min(anchors,key=lambda a:abs(a-len(e['actions'])/2))
            groups[(e['group'],number in replay.history_ids)].append((number,anchor))
    selected=[]
    for row in zip_longest(*(groups[g] for g in sorted(groups))):
        selected.extend(k for k in row if k is not None)
        if len(selected)>=count:break
    results=[]
    try:
        with torch.no_grad():
            for number,anchor in selected[:count]:
                e=replay.episodes[number];begin=anchor-replay.burn;end=min(len(e['actions']),anchor+replay.learn)
                def values(first,last,hidden,burn=0):
                    return episode_values(policy,e,first,last,hidden,burn)
                hidden=None
                for first in range(0,begin,32):_,hidden=values(first,min(begin,first+32),hidden)
                exact,_=values(begin,end,hidden);stored=torch.as_tensor(e['hidden'][begin])[None,None].float()
                approximate,_=values(begin,end,stored,replay.burn)
                zero,_=values(begin,end,torch.zeros_like(stored),replay.burn)
                reference=exact[:,replay.burn:]
                results.append(dict(group=e['group'],retention='history' if number in replay.history_ids else 'recent',
                    family=e['family'],anchor=anchor,decisions=end-anchor,
                    stored_h_max_q_error=float((approximate[:,replay.burn:]-reference).abs().max()),
                    stored_h_action_differences=int((approximate[:,replay.burn:].argmax(-1)!=reference.argmax(-1)).sum()),
                    zero_h_max_q_error=float((zero[:,replay.burn:]-reference).abs().max()),
                    zero_h_action_differences=int((zero[:,replay.burn:].argmax(-1)!=reference.argmax(-1)).sum())))
    finally:policy.set_training_mode(was_training)
    return dict(rows=results,seconds=time.monotonic()-began,native_frames=0,optimizer_updates=0,
        scope='Read-only current-encoder full-prefix comparison. Drift is measured, not assumed eliminated by stored hidden state and burn-in.')


def sequence_double_q_targets(online,target,rewards,dones,burn,learn,nstep,gamma=1.,valid=None,
                              *,actions=None,cut_non_greedy=False,trace_mask=None,return_details=False):
    """Finite-horizon Double Q with optional Watkins-style greedy cuts.

    A nongreedy *intermediate* executed input cuts before its reward and
    bootstraps at that state. The root action may be off-policy. Real game
    terminals suppress bootstrap; a finite recording cut does not. With a
    nonlinear approximator this is not an unbiased optimal-Q label.
    An explicit trace_mask selects a frozen collector-policy prefix rather
    than the current greedy policy; its policy-lag bias remains.
    """
    if cut_non_greedy and actions is None:raise ValueError('Greedy cuts require actual executed inputs')
    if nstep<1:raise ValueError('A backup must contain at least one transition')
    valid=torch.ones_like(dones,dtype=torch.bool) if valid is None else valid
    batch=rewards.shape[0];device=online.device
    roots=torch.arange(burn,burn+learn,device=device)[None].expand(batch,-1)
    total=rewards.new_zeros((batch,learn));steps=torch.zeros_like(roots)
    alive=torch.ones_like(roots,dtype=torch.bool);active=alive.clone();greedy_cut=~alive
    greedy=online.argmax(-1)
    for k in range(nstep):
        idx=(roots+k).clamp_max(rewards.shape[1]-1)
        real=active&valid.gather(1,idx)
        if k and cut_non_greedy:
            allowed=(actions.gather(1,idx)==greedy.gather(1,idx)) if trace_mask is None else trace_mask.gather(1,idx)
            mismatch=real&~allowed
            greedy_cut|=mismatch;real&=~mismatch
        total+=(gamma**k)*rewards.gather(1,idx)*real
        steps+=real.long();alive&=~(real&dones.gather(1,idx));active=real&alive
    indices=(roots+steps).clamp_max(online.shape[1]-1)
    oq=online.gather(1,indices[:,:,None].expand(-1,-1,online.shape[-1]))
    tq=target.gather(1,indices[:,:,None].expand_as(oq))
    next_q=tq.gather(2,oq.argmax(2,keepdim=True)).squeeze(2)
    bootstrap=(gamma**steps)*alive*next_q;desired=total+bootstrap
    if return_details:
        return desired,dict(steps=steps,reward_sum=total,bootstrap=bootstrap,
            next_target_q=next_q,bootstrap_indices=indices,greedy_cut=greedy_cut,terminal=~alive)
    return desired


class Replay:
    def __init__(self,nodes=150000,transitions=32768,width=3584,context=16,seed=2231,offsets=None):
        self.capacity=int(nodes);self.transition_capacity=int(transitions);self.context=int(context);self.rng=np.random.default_rng(seed)
        self.offsets=None if offsets is None else tuple(int(v) for v in offsets)
        self.maps=np.empty((nodes,width),np.float16);self.positions=np.empty((nodes,6),np.float32);self.extras=np.empty((nodes,12),np.float32)
        self.previous=np.empty(nodes,np.int16);self.parent=np.full(nodes,-1,np.int64);self.node_stamp=np.full(nodes,-1,np.int64);self.node_count=0
        self.starts=np.empty(transitions,np.int64);self.ends=np.empty(transitions,np.int64);self.actions=np.empty(transitions,np.int64)
        self.rewards=np.empty(transitions,np.float32);self.dones=np.empty(transitions,bool);self.durations=np.empty(transitions,np.int16)
        self.transition_stamp=np.full(transitions,-1,np.int64);self.transition_count=0;self.groups=defaultdict(lambda:deque(maxlen=4096))

    def add_node(self,parent,feature_map,position,extras,previous):
        number=self.node_count;index=number%self.capacity;self.node_count+=1
        self.maps[index]=feature_map;self.positions[index]=position;self.extras[index]=extras;self.previous[index]=previous
        self.parent[index]=parent;self.node_stamp[index]=number
        return number

    def add_transition(self,start,action,reward,end,done,duration,group):
        number=self.transition_count;index=number%self.transition_capacity;self.transition_count+=1
        self.starts[index]=start;self.ends[index]=end;self.actions[index]=action;self.rewards[index]=reward;self.dones[index]=done;self.durations[index]=duration
        self.transition_stamp[index]=number;self.groups[group].append(number)

    def history(self,number):
        values=[]
        while number>=0 and len(values)<self.context:
            index=number%self.capacity
            if self.node_stamp[index]!=number:return None
            values.append(number);number=int(self.parent[index])
        return values[::-1]

    def contexts(self,numbers,device='cpu'):
        histories=[self.history(int(n)) for n in numbers]
        if any(not h for h in histories):raise ValueError('A requested context is no longer in RAM replay')
        if self.offsets is not None:
            histories=[[h[len(h)-1-age] for age in sorted(self.offsets,reverse=True) if age<len(h)] for h in histories]
        lengths=np.asarray([len(h) for h in histories],np.int64);length=int(lengths.max());batch=len(histories)
        ids=np.zeros((batch,length),np.int64);mask=np.arange(length)[None]<lengths[:,None]
        for i,h in enumerate(histories):ids[i,:len(h)]=h
        maps=np.zeros((batch,length,self.maps.shape[1]),np.float16);positions=np.zeros((batch,length,6),np.float32)
        extras=np.zeros((batch,length,12),np.float32);previous=np.zeros((batch,length),np.int64)
        ix=ids[mask]%self.capacity
        maps[mask]=self.maps[ix];positions[mask]=self.positions[ix];extras[mask]=self.extras[ix];previous[mask]=self.previous[ix]
        return tuple(torch.as_tensor(x,device=device) for x in (maps,positions,extras,previous))+(torch.as_tensor(lengths),)

    def sample(self,batch=64):
        n=min(self.transition_count,self.transition_capacity)
        if n<batch:return None
        chosen=[];groups=list(self.groups);attempts=0
        while len(chosen)<batch and attempts<batch*40:
            attempts+=1
            if self.rng.random()<.5:
                group=groups[int(self.rng.integers(len(groups)))];values=self.groups[group]
                number=values[int(self.rng.integers(len(values)))];i=number%self.transition_capacity
                if self.transition_stamp[i]!=number:continue
            else:i=int(self.rng.integers(n))
            if self.history(int(self.starts[i])) is None or self.history(int(self.ends[i])) is None:continue
            chosen.append(i)
        if len(chosen)<batch:return None
        i=np.asarray(chosen)
        return dict(observation=self.contexts(self.starts[i]),next_observation=self.contexts(self.ends[i]),
                    action=torch.as_tensor(self.actions[i]),reward=torch.as_tensor(self.rewards[i]),done=torch.as_tensor(self.dones[i]),
                    duration=torch.as_tensor(self.durations[i],dtype=torch.float32))


class Demonstrations(Replay):
    """Permanent, group-balanced real trajectories, separate from the ring."""
    def __init__(self,path,context=16,seed=2231,objective='graded',offsets=None,prefix_priority=False):
        with np.load(path,allow_pickle=False) as source:
            nodes=len(source['maps']);transitions=len(source['transition_start'])
            super().__init__(nodes=nodes,transitions=transitions,width=source['maps'].shape[1],context=context,seed=seed,offsets=offsets)
            for key in ('maps','positions','extras','previous','parent'):setattr(self,key,source[key].copy())
            self.node_stamp=np.arange(nodes);self.node_count=nodes
            for dest,key in [('starts','start'),('ends','end'),('actions','action'),('rewards','reward'),('dones','done'),('durations','duration')]:
                setattr(self,dest,source['transition_'+key].copy())
            self.returns=source['transition_returns'].copy();self.wins=source['transition_win'].copy()
            self.labels=source['transition_group'].copy();self.transition_count=transitions
            self.transition_stamp=np.arange(transitions)
            self.demo_groups={str(g):np.flatnonzero(self.labels==g) for g in np.unique(self.labels)}
        self.prefix=np.zeros(transitions,bool);self.imitation=self.wins.copy();self.priority_groups=None
        self.supervision_group=np.zeros(transitions,np.int64)
        annotation=Path(path).parent/'prefix_annotations.npz'
        if annotation.is_file():
            with np.load(annotation,allow_pickle=False) as data:
                if not np.array_equal(data['start'],self.starts) or not np.array_equal(data['end'],self.ends):raise ValueError('Recovery prefix annotations and native transition IDs differ')
                self.prefix=data['intervention'].copy()
                if prefix_priority:
                    self.imitation=data['beneficial'].copy()
                    self.supervision_group[:]=1
                    self.priority_groups={g:ids[self.prefix[ids]] for g,ids in self.demo_groups.items() if self.prefix[ids].any()}
        if objective=='win':
            # Original native reward is terminal(+1/-1/0)+0.25*netHPdelta.
            # Recover its exact potential from actual parent/child edges.
            terminals=np.where(self.dones,np.where(self.rewards>.5,1.,np.where(self.rewards<-.5,-1.,0.)),0.)
            increments=np.full(nodes,np.nan,np.float64)
            for end,change in zip(self.ends,self.rewards-terminals):
                if np.isfinite(increments[end]) and not np.isclose(increments[end],change):raise ValueError('Conflicting native HP changes on a shared edge')
                increments[end]=change
            potential=np.zeros(nodes,np.float64)
            for node,parent in enumerate(self.parent):
                if parent<0:continue
                if parent>=node or not np.isfinite(increments[node]):raise ValueError('Incomplete native potential path')
                potential[node]=potential[parent]+increments[node]
            terminal=np.where(self.dones,np.where(self.wins,1.,-1.),0.)
            self.rewards=(terminal+np.where(self.dones,0.,potential[self.ends])-potential[self.starts]).astype(np.float32)
            self.returns=(np.where(self.wins,1.,-1.)-potential[self.starts]).astype(np.float32)
        # Collectors can append an episode forwards or backwards. Follow only
        # contiguous rows whose native end/start IDs match, never another fork.
        self.nstep_ends=self.ends.copy();self.nstep_rewards=self.rewards.astype(np.float32).copy()
        self.nstep_dones=self.dones.copy();self.nstep_durations=self.durations.astype(np.float32).copy()
        for i in range(transitions):
            cursor=i
            while self.nstep_durations[i]<32 and not self.nstep_dones[i]:
                successors=[j for j in (cursor+1,cursor-1) if 0<=j<transitions and self.starts[j]==self.ends[cursor]]
                if not successors:break
                if len(successors)!=1:raise ValueError('Ambiguous contiguous demonstration continuation')
                cursor=successors[0];self.nstep_ends[i]=self.ends[cursor];self.nstep_rewards[i]+=self.rewards[cursor]
                self.nstep_dones[i]=self.dones[cursor];self.nstep_durations[i]+=self.durations[cursor]

    def sample(self,batch=64):
        groups=list(self.demo_groups)
        chosen=[]
        for row in range(batch):
            grouped=self.priority_groups if self.priority_groups is not None and row<batch//2 else self.demo_groups
            names=list(grouped);chosen.append(int(self.rng.choice(grouped[names[int(self.rng.integers(len(names)))]])))
        i=np.asarray(chosen)
        return dict(observation=self.contexts(self.starts[i]),next_observation=self.contexts(self.ends[i]),
                    action=torch.as_tensor(self.actions[i]),reward=torch.as_tensor(self.rewards[i],dtype=torch.float32),
                    done=torch.as_tensor(self.dones[i]),duration=torch.as_tensor(self.durations[i],dtype=torch.float32),
                    returns=torch.as_tensor(self.returns[i],dtype=torch.float32),win=torch.as_tensor(self.wins[i]),
                    imitation=torch.as_tensor(self.imitation[i]),prefix=torch.as_tensor(self.prefix[i]),
                    supervision_group=torch.as_tensor(self.supervision_group[i]),
                    nstep_next_observation=self.contexts(self.nstep_ends[i]),nstep_reward=torch.as_tensor(self.nstep_rewards[i]),
                    nstep_done=torch.as_tensor(self.nstep_dones[i]),nstep_duration=torch.as_tensor(self.nstep_durations[i]))


def concatenate_contexts(parts):
    max_length=max(int(part[0].shape[1]) for part in parts);joined=[]
    for component in range(4):
        values=[]
        for part in parts:
            value=part[component];missing=max_length-value.shape[1]
            values.append(F.pad(value,(0,missing)) if component==3 else F.pad(value,(0,0,0,missing)))
        joined.append(torch.cat(values,0))
    return tuple(joined)+(torch.cat([part[4] for part in parts]),)


class DemonstrationMix:
    """Equal source mass, group balance within each native source."""
    def __init__(self,paths,context=16,seed=2231,objective='graded',offsets=None,prefix_priority=False):
        self.sources=[Demonstrations(path,context,seed+53*i,objective,offsets,prefix_priority) for i,path in enumerate(paths)]
        self.transition_count=sum(source.transition_count for source in self.sources)

    def sample(self,batch=64):
        counts=[batch//len(self.sources)+(i<batch%len(self.sources)) for i in range(len(self.sources))]
        parts=[source.sample(count) for source,count in zip(self.sources,counts) if count]
        return {key:concatenate_contexts([part[key] for part in parts]) if key in ('observation','next_observation','nstep_next_observation') else torch.cat([part[key] for part in parts]) for key in parts[0]}


def demonstration_objective(q,data,margin=.1,return_anchor=True):
    """Retain successful actions and achievable returns; defeats remain TD data.

    Observed teacher returns are a lower-bound training anchor, not an oracle
    or a target that says a losing teacher action is the best available action.
    """
    chosen=q.gather(1,data['action'][:,None]).flatten();success=data.get('imitation',data['win'])
    if not bool(success.any()):return q.sum()*0.,dict(margin=0.,return_lower_bound=0.,accuracy=None)
    costs=torch.full_like(q,float(margin));costs.scatter_(1,data['action'][:,None],0.)
    group=data.get('supervision_group',torch.zeros_like(data['action']))
    per_example=(q+costs).max(1).values-chosen
    masks=[success&(group==g) for g in torch.unique(group[success])]
    # Equal teaching-source mass prevents hundreds of continuation attacks
    # from washing out the rare whole-prefix interventions that changed a loss.
    imitation=torch.stack([per_example[mask].mean() for mask in masks]).mean()
    lower_bound=torch.stack([F.smooth_l1_loss(chosen[mask],torch.maximum(chosen[mask].detach(),data['returns'][mask])) for mask in masks]).mean() if return_anchor else q.sum()*0.
    return imitation+lower_bound,dict(margin=float(imitation.detach()),return_lower_bound=float(lower_bound.detach()),
                                    accuracy=float((q.argmax(1)[success]==data['action'][success]).float().mean()),
                                    prefix_samples=int(data['prefix'].sum()) if 'prefix' in data else 0,imitation_samples=int(success.sum()))


def double_q_target(online_next,target_next,rewards,dones,durations,gamma):
    selected=online_next.argmax(1)
    continuation=target_next.gather(1,selected[:,None]).flatten()
    return rewards+(~dones).float()*(float(gamma)**durations)*continuation
