"""Orchestrate shared DDQN training/evaluation stages without another trainer."""
import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from itertools import zip_longest
from pathlib import Path
from experiments.league_adaptive.runtime import ROOT,load_actor,write,setup
from experiments.league_adaptive.evaluate import evaluate
from experiments.ryu_counterq.train import train_run,initial_pool

REPORT=ROOT/'reports/counterfactual_dqn'
ARTIFACT=ROOT/'artifacts/counterfactual_dqn'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def report_update(**fields):
    path=REPORT/'summary.json';data=read(path) if path.exists() else {}
    data.update(fields);write(path,data)


def rank(result):
    pvp=result['overall'];ports=result['by_port']
    return (pvp['wins']+result['pve']['wins'],min(ports[str(p)]['wins'] for p in (1,2)),
            min(x['wins']/max(1,x['episodes']) for x in result['by_opponent'].values()),pvp['hp_margin'])


def evaluated(path,pool,out,**kwargs):
    summary=Path(out)/'summary.json'
    if summary.exists():return read(summary)
    actor=load_actor(path)
    try:return evaluate(actor,pool,out,cpu=False,arena_call_sequence=True,progress=False,**kwargs)
    finally:actor.close()


def completed_training(out,config,character,seed,arm,budget,pool):
    """Resume only the declared run; configuration edits need a fresh root."""
    if not (out/'summary.json').exists():
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Incomplete run is preserved: {out}. Use a fresh root or explicitly archive it before restarting.')
        return False
    saved=read(out/'config.json')
    expected=dict(character=character,seed=seed,arm=arm,budget=budget,pool=pool)
    if any(saved.get(key)!=value for key,value in expected.items()) or any(
            saved['config'][key]!=config[key] for key in ('learner','network','actions','counterfactual')):
        raise ValueError(f'Completed run configuration differs: {out}. Choose a fresh run root; no results were overwritten.')
    if saved['config'].get('active_replay_policy')!=config.get('active_replay_policy'):
        raise ValueError('Completed run uses another replay retention policy')
    return True


def screening(config):
    full=initial_pool();small=[r for r in full if r['id'] in ('cpu_Ryu_1','cpu_Guile_1','mixed_Ryu','mixed_Guile')]
    results={}
    for character in config['characters']:
        out=ARTIFACT/'screen'/character
        if not (out/'summary.json').exists():
            train_run(config,character,config['screen']['seed'],'ordinary',config['screen']['frames_per_character'],out,
                      checkpoints=config['screen']['checkpoints'],dev_pool=small)
        result=evaluated(out/'agent.pt',full,out/'panel',offsets=(0,))
        results[character]=dict(training=read(out/'summary.json'),evaluation=result)
        report_update(stage='all_character_screen',screen=results,
                      screening_scope='Random initialization, same DDQN inputs/architecture/budget and 36 full matches per role. Provisional, not a permanent character ranking.')
        print('SCREEN',character,rank(result),flush=True)
    ordered=sorted(results,key=lambda c:rank(results[c]['evaluation']),reverse=True)
    shortlist=ordered[:config['screen']['shortlist']]
    report_update(screen_order=ordered,shortlist=shortlist,stage='shortlist_training')
    return shortlist


def shortlist_training(config,characters):
    full=initial_pool();results={}
    for character in characters:
        results[character]={}
        for seed in (config['screen']['seed'],config['screen']['replication_seed']):
            out=ARTIFACT/'shortlist'/character/str(seed)
            source=ARTIFACT/'screen_refinement'/character/'agent.pt' if seed==config['screen']['seed'] else None
            prior=config['screen']['frames_per_character']+config['screen']['refinement_frames_per_character']
            budget=config['screen']['shortlist_total_frames']-(prior if source else 0)
            if not (out/'summary.json').exists():
                train_run(config,character,seed,'ordinary',budget,out,start=source,checkpoints=[budget//2,budget])
            ev=evaluated(out/'agent.pt',full,out/'panel',offsets=(0,))
            results[character][str(seed)]=dict(training=read(out/'summary.json'),evaluation=ev)
            report_update(shortlist_results=results)
            print('SHORTLIST',character,seed,rank(ev),flush=True)
    def score(c):
        ranks=[rank(v['evaluation']) for v in results[c].values()]
        return tuple(sum(r[i] for r in ranks)/len(ranks) for i in range(4))
    selected=max(characters,key=score)
    report_update(selected_character=selected,character_selection_scores={c:score(c) for c in characters},
                  stage='paired_training',character_selection_scope='One first-seed continuation with refreshed replay plus a genuinely independent random second seed. Primary A/B starts independently from scratch.')
    return selected


def refined_screening(config):
    """Equal additional learning after measured invalid charge durations."""
    state=read(REPORT/'summary.json')
    if 'initial_screen' not in state:report_update(initial_screen=state['screen'])
    full=initial_pool();small=[r for r in full if r['id'] in ('cpu_Ryu_1','cpu_Guile_1','mixed_Ryu','mixed_Guile')]
    results={};budget=config['screen']['refinement_frames_per_character']
    for character in config['characters']:
        out=ARTIFACT/'screen_refinement'/character
        if not (out/'summary.json').exists():
            train_run(config,character,config['screen']['seed'],'ordinary',budget,out,
                      start=ARTIFACT/'screen'/character/'agent.pt',pool=full,checkpoints=[30000,60000,budget],dev_pool=small)
        ev=evaluated(out/'agent.pt',full,out/'panel',offsets=(0,))
        results[character]=dict(training=read(out/'summary.json'),evaluation=ev)
        report_update(stage='verified_input_refinement',refined_screen=results,
            refinement_scope='Every role receives the same additional120k from its own original120k weights. Verified duration and responsive input profiles replace the measured incomplete charge coverage; original results are preserved separately.')
        print('REFINED_SCREEN',character,rank(ev),flush=True)
    order=sorted(results,key=lambda c:rank(results[c]['evaluation']),reverse=True)
    chosen=order[:config['screen']['shortlist']]
    report_update(refined_screen_order=order,shortlist=chosen,refined_screen_complete=True,stage='shortlist_training')
    return chosen


def comparison(config,character):
    pool=initial_pool(stars=(1,4,8),styles=('mixed','pressure','guard'))
    results={}
    for seed in config['comparison']['seeds']:
        results[str(seed)]={}
        for arm in config['comparison']['arms']:
            out=ARTIFACT/'comparison'/str(seed)/arm
            if not completed_training(out,config,character,seed,arm,config['comparison']['frames_per_arm'],pool):
                train_run(config,character,seed,arm,config['comparison']['frames_per_arm'],out,pool=pool,
                          checkpoints=config['comparison']['checkpoints'])
            ev=evaluated(out/'agent.pt',initial_pool(),out/'panel',offsets=(0,))
            results[str(seed)][arm]=dict(training=read(out/'summary.json'),evaluation=ev)
            report_update(comparison=results,stage='paired_training')
            print('COMPARISON',character,seed,arm,rank(ev),flush=True)
    report_update(stage='paired_training_complete')
    return results


def endpoint_paths(config,stage='comparison'):
    return {f'{arm}_{seed}':ARTIFACT/stage/str(seed)/arm/'agent.pt'
            for seed in config['comparison']['seeds'] for arm in config['comparison']['arms']}


def counter_role(results):
    counts={c:0 for c in results[next(iter(results))]['ordinary']['evaluation']['by_opponent'] if c.startswith('mixed_')}
    for seed in results.values():
        for arm in seed.values():
            for name in counts:counts[name]+=arm['evaluation']['by_opponent'][name]['wins']
    return min(counts,key=lambda name:(counts[name],list(initial_pool()[i]['id'] for i in range(len(initial_pool()))).index(name))).removeprefix('mixed_')


def feedback_training(config):
    """One measured counter -> shared opponent pool -> paired learning cycle."""
    state=read(REPORT/'summary.json');character=state['selected_character'];parents=endpoint_paths(config)
    if state.get('final_frozen'):raise RuntimeError('Final weights are frozen; do not train from final-test feedback')
    if any(str(seed) not in state.get('comparison',{}) or any(arm not in state['comparison'][str(seed)] for arm in config['comparison']['arms'])
           for seed in config['comparison']['seeds']):raise RuntimeError('Finish both paired training seeds first')
    eligible=False
    for seed in state['comparison'].values():
        for result in seed.values():
            ev=result['evaluation']
            roles={r['opponent_character'] for r in ev['rows'] if r['opponent_kind']!='cpu' and r['result']=='win'}
            eligible|=len(roles)>=3 and all(ev['by_port'][str(p)]['wins']>0 for p in (1,2))
    if not eligible:
        report_update(stage='learning_diagnosis_required',reason='No parent yet has independent PvP wins against3roles and on both ports; diagnose learning before spending counter-search budget.')
        return
    role=counter_role(state['comparison']);spec=config['league_feedback'];snapshots=[]
    report_update(stage='training_counter_search',training_counter_character=role)
    for target,path in parents.items():
        pool=[dict(id='training_target_'+target,kind='agent',character=character,path=path.relative_to(ROOT).as_posix())]
        for seed in spec['counter_seeds']:
            out=ARTIFACT/'training_counters'/target/str(seed)
            if not completed_training(out,config,role,seed,'ordinary',spec['counter_frames'],pool):
                train_run(config,role,seed,'ordinary',spec['counter_frames'],out,pool=pool,checkpoints=spec['counter_checkpoints'])
            run=read(out/'summary.json')
            for frame in spec['counter_checkpoints']:
                actual=next(c['actual_frames'] for c in run['curve'] if c['threshold']==frame)
                snapshots.append(dict(id=f'training_counter_{target}_{seed}_{frame}',kind='agent',character=role,
                    path=(out/f'checkpoint_{frame}.pt').relative_to(ROOT).as_posix(),
                    target=target,training_seed=seed,checkpoint_target_frames=frame,training_frames=actual))
    # Every parent sees every generated snapshot before selection. A midpoint
    # and endpoint of the same search are dependent, never extra replicates.
    measurements={};coverage={s['id']:set() for s in snapshots}
    for target,path in parents.items():
        ev=evaluated(path,snapshots,ARTIFACT/'training_counter_panels'/target,offsets=(0,))
        measurements[target]=ev
        for row in ev['rows']:
            if row['result']=='loss':coverage[row['opponent']].add((target,row['player'],row['offset']))
    # Small set cover over actually exposed losses, rather than one file per
    # nominal tactic. Retain at most4 useful policies for bounded main replay.
    remaining=list(snapshots);retained=[];covered=set()
    while remaining and len(retained)<4:
        best=max(remaining,key=lambda s:(len(coverage[s['id']]-covered),len(coverage[s['id']])))
        if not coverage[best['id']]-covered:break
        retained.append(best);covered.update(coverage[best['id']]);remaining.remove(best)
    report_update(training_counter_measurements=measurements,training_counter_snapshots=snapshots,
                  retained_training_counters=retained,training_counter_scope='Two independent counter seeds per target; snapshots are dependent. Retention uses measured marginal loss coverage, not a claim of distinct tactics.')
    if not retained:
        report_update(stage='counter_diagnosis_required',reason='No generated frozen counter won a complete development game; no unmeasured counter is labelled effective.')
        return
    cpu=[r for star in range(1,9) for r in initial_pool(stars=(star,),styles=())]
    generic=[r for r in initial_pool(stars=(),styles=('mixed','pressure','guard'))]
    historical=[dict(id='historical_strong_response',kind='agent',character='Ryu',path='models/opponents/strong_response.pt'),
                dict(id='historical_formal_reference',kind='agent',character='Ryu',path=config['reference'])]
    # Expose neural counters from the first coverage cycle. Avoid placing all
    # counter games after a long block of CPU difficulties.
    pool=[]
    for group in zip_longest(cpu,generic,retained+historical):pool.extend(r for r in group if r is not None)
    write(ARTIFACT/'feedback_pool.json',pool)
    paired_feedback_training(config)


def paired_feedback_training(config,phase='feedback'):
    """Matched continuations; each declared phase starts at its primary parent."""
    state=read(REPORT/'summary.json')
    if state.get('final_frozen'):raise RuntimeError('Final weights are frozen; candidate learning has ended')
    phases=selection_stages(config)
    if phase not in phases[1:]:raise ValueError(phase)
    for prior in phases[:phases.index(phase)]:
        if any(arm not in state.get(prior,{}).get(str(seed),{})
            for seed in config['comparison']['seeds'] for arm in config['comparison']['arms']):
            raise RuntimeError(f'Finish every paired seed in {prior} before {phase}')
    character=state['selected_character'];parents=endpoint_paths(config)
    pool=read(ARTIFACT/'feedback_pool.json');results={}
    spec=feedback_spec(config,phase)
    main_config=copy.deepcopy(config)
    main_config['learner'].update(config['league_feedback'].get('main_learner_overrides',{}))
    main_config['learner'].update(spec.get('main_learner_overrides',{}))
    for seed in config['comparison']['seeds']:
        results[str(seed)]={}
        for arm in config['comparison']['arms']:
            out=ARTIFACT/phase/str(seed)/arm
            if not completed_training(out,main_config,character,seed,arm,spec['main_frames_per_arm_seed'],pool):
                train_run(main_config,character,seed,arm,spec['main_frames_per_arm_seed'],out,pool=pool,
                          checkpoints=spec['main_checkpoints'],start=parents[f'{arm}_{seed}'])
            ev=evaluated(out/'agent.pt',initial_pool(),out/'panel',offsets=(0,))
            results[str(seed)][arm]=dict(training=read(out/'summary.json'),evaluation=ev)
            report_update(stage=phase+'_training',**{phase:results},feedback_pool=pool)
            print(phase.upper(),seed,arm,rank(ev),flush=True)
    report_update(stage=phase+'_complete')


def selection_stages(config):
    if config.get('own_character'):
        return ['paired']+(['league_adaptation'] if config.get('league_adaptation',{}).get('enabled') else [])
    phases=['comparison','feedback']
    if config.get('continuation_check',{}).get('enabled'):phases.append('low_exploration')
    if config.get('data_mix_check',{}).get('enabled'):phases.append('interleaved_feedback')
    return phases


def feedback_spec(config,phase):
    return config[dict(feedback='league_feedback',low_exploration='continuation_check',
                       interleaved_feedback='data_mix_check')[phase]]


def final_rule_pool(config):
    rules=[]
    for character in config.get('opponent_characters',config.get('characters',())):
        for period in config['final']['novel_rule_periods']:
            rules.append(dict(id=f'new_timing_{character}_{period}',kind='generic',character=character,
                              style='mixed',period=period,phase=7,charge_decisions=10))
        for switch in config['final']['novel_switch_every']:
            rules.append(dict(id=f'new_switch_{character}_{switch}',kind='generic',character=character,
                              style='switching',styles=['pressure','guard'],period=31,switch_every=switch,phase=11,charge_decisions=10))
    return rules


def initialization_baselines(config):
    """Do not attribute a lucky untrained attack pattern to RL acquisition."""
    import torch
    from experiments.ryu_counterq.policy import random_sequence_q
    from experiments.ryu_direct_league.policy import save
    state=read(REPORT/'summary.json')
    if not state.get('refined_screen_complete'):raise RuntimeError('Finish verified-input role screening first')
    from experiments.league_adaptive.runtime import NativeGame
    game=NativeGame();starts=[]
    try:
        for own in state['shortlist']:
            for stars in (4,8):
                for rival in config['characters']:
                    game.open('pve',stars,character=rival,p1_character=own)
                    starts.append(dict(character=own,opponent=rival,stars=stars,readings=dict(game.raw)))
        write(ARTIFACT/'paired_start_preparation.json',dict(rows=starts,setup_native_frames=game.setup_frames,
             scope='Prepare identical immutable starts before paired learning so both arms have zero menu-frame cost. These setup frames are separate from training.'))
    finally:game.close()
    pool=initial_pool();results={}
    for character in state['shortlist']:
        results[character]={}
        for seed in config['comparison']['seeds']:
            path=ARTIFACT/'untrained'/character/str(seed)/'agent.pt'
            if not path.exists():
                torch.manual_seed(seed);policy,definition=random_sequence_q(character,config['network'])
                save(path,policy,definition,training=dict(initialization='random',native_training_frames=0,seed=seed))
            results[character][str(seed)]=evaluated(path,pool,path.parent/'panel',offsets=(0,))
            write(ARTIFACT/'untrained_baselines.json',dict(untrained_greedy_baselines=results))
    reference=evaluated(config['reference'],pool,ARTIFACT/'known_reference_panel',offsets=(0,))
    write(ARTIFACT/'untrained_baselines.json',dict(untrained_greedy_baselines=results,known_scope_reference=reference,
         scope='Greedy random networks, not uniformly random actions. These known conditions do not affect role selection and are not final holdout.'))


def plot_screening(config):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    state=read(REPORT/'summary.json');data=state['refined_screen'];characters=config['characters']
    if len(data)!=len(characters):raise RuntimeError('Do not present an incomplete screen as a full matrix')
    matrices=[np.zeros((12,12)) for _ in range(3)]
    for i,own in enumerate(characters):
        for row in data[own]['evaluation']['rows']:
            j=characters.index(row['opponent_character'])
            if row['opponent_kind']=='cpu':matrices[0][i,j]=1. if row['ko_win'] else .5 if row['result']=='win' else 0.
            else:matrices[row['player']][i,j]=1. if row['result']=='win' else .5 if row['result']=='draw' else 0.
    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    for ax,matrix,title in zip(axes,matrices,['Native CPU star1: KO=1, timeout win=.5','PvP controller P1','PvP controller P2']):
        ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlGn');ax.set_title(title)
        ax.set_xticks(range(12),characters,rotation=70);ax.set_yticks(range(12),characters)
        ax.set_xlabel('Opponent character');ax.set_ylabel('Learner character')
        for i in range(12):
            for j in range(12):ax.text(j,i,str(matrix[i,j]).rstrip('0').rstrip('.'),ha='center',va='center',fontsize=8)
    fig.suptitle('All12:120k basic warmup +120k validated-input continuation; seed51031; known36-game panel')
    fig.savefig(REPORT/'character_screening.png',dpi=160);plt.close(fig)


def monitor_development(config,stage='comparison'):
    """One read-only evaluator, shared conditions for all paired checkpoints."""
    pool=[r for r in initial_pool() if r['id'] in ('cpu_Ryu_1','cpu_Guile_1','mixed_Ryu','mixed_Guile')]
    if stage!='comparison':
        while 'retained_training_counters' not in read(REPORT/'summary.json'):time.sleep(15)
        retained=read(REPORT/'summary.json')['retained_training_counters']
        if not retained:raise RuntimeError('No effective training counter; diagnose before feedback monitoring')
        pool+=retained[:2]
    spec=feedback_spec(config,stage) if stage!='comparison' else {}
    thresholds=(config['comparison']['checkpoints'] if stage=='comparison' else spec['main_checkpoints'])[:-1]
    result_path=ARTIFACT/('light_development.json' if stage=='comparison' else stage+'_light_development.json')
    write(result_path.with_name(result_path.stem+'_pool.json'),pool)
    seen={}
    while True:
        for seed in config['comparison']['seeds']:
            for arm in config['comparison']['arms']:
                for threshold in thresholds:
                    key=f'{seed}/{arm}/{threshold}'
                    checkpoint=ARTIFACT/stage/str(seed)/arm/f'checkpoint_{threshold}.pt'
                    if key in seen or not checkpoint.exists():continue
                    ev=evaluated(checkpoint,pool,checkpoint.parent/f'light_development_{threshold}',offsets=(0,))
                    seen[key]=dict(evaluation=ev,scope='Known read-only greedy trend check, with the same pool across arms/seeds. Includes up to2 retained training counters in feedback. Not full regression or final acceptance.')
                    write(result_path,seen)
                    print('LIGHT_DEVELOPMENT',stage,seed,arm,threshold,ev['overall']['wins'],ev['pve']['wins'],flush=True)
        if len(seen)==len(config['comparison']['seeds'])*len(config['comparison']['arms'])*len(thresholds):return
        time.sleep(15)


def plot_learning(config):
    """Keep per-seed full-game trends and panel denominators visible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if config.get('own_character'):
        phases=['pretrain']+selection_stages(config)
        fig,axes=plt.subplots(3,len(phases),figsize=(4.5*len(phases),8),squeeze=False)
        def values(ev):
            return [sum(r['ko_win'] for r in ev['rows'] if r['opponent_kind']=='cpu')]+[ev['by_port'][str(p)]['wins'] for p in (1,2)]
        for column,phase in enumerate(phases):
            arms=['ordinary'] if phase=='pretrain' else config['comparison']['arms']
            for seed in config['comparison']['seeds']:
                for arm in arms:
                    source=ARTIFACT/phase/str(seed)/arm/'summary.json'
                    if not source.exists():continue
                    points=[]
                    if phase=='pretrain':
                        initial=ARTIFACT/'untrained'/str(seed)/'development/summary.json'
                        if initial.exists():points.append((0,values(read(initial))))
                    else:
                        parent=config[phase]['parent_stage'];parent_arm='ordinary' if parent=='pretrain' else arm
                        if parent=='replay_prefill':parent_arm='ordinary'
                        initial=ARTIFACT/parent/str(seed)/parent_arm/'summary.json'
                        if initial.exists():
                            initial_ev,_=fixed_role_endpoint(config,initial.parent/'agent.pt')
                            points.append((0,values(initial_ev)))
                    for checkpoint in read(source)['curve']:
                        ev=checkpoint.get('development')
                        if not ev:continue
                        points.append((checkpoint['actual_frames'],values(ev)))
                    for row in range(3):
                        axes[row,column].plot([p[0]/1e6 for p in points],[p[1][row] for p in points],
                            color='tab:blue' if arm=='ordinary' else 'tab:orange',
                            linestyle='-' if seed==config['comparison']['seeds'][0] else '--',marker='o',label=f'{arm}, {seed}')
            for row,denominator in enumerate((24,8,8)):
                axis=axes[row,column];axis.set_ylim(-.2,denominator+.2);axis.grid(alpha=.25)
                if row==0:axis.set_title(phase)
                if column==0:axis.set_ylabel(('CPU KO /24','PvP P1 wins /8','PvP P2 wins /8')[row])
                if row==2:axis.set_xlabel('Local training native frames (million)')
        axes[0,-1].legend(fontsize=7)
        fig.suptitle('Fixed known complete games; branch simulation included in counterfactual training frames')
        fig.tight_layout();fig.savefig(REPORT/'learning_curves.png',dpi=150);plt.close(fig)
        return
    state=read(REPORT/'summary.json');character=state['selected_character']
    phases=selection_stages(config);fig,axes=plt.subplots(3,len(phases),figsize=(4.3*len(phases),8),squeeze=False)
    base_ids={'mixed_Ryu','mixed_Guile'}
    counter_ids={r['id'] for r in state.get('retained_training_counters',[])[:2]}
    cpu_ids={'cpu_Ryu_1','cpu_Guile_1'}
    def counts(rows):
        return [sum(r['result']=='win' for r in rows if r['opponent'] in names)
                for names in (base_ids,counter_ids,cpu_ids)]
    for column,phase in enumerate(phases):
        for seed in config['comparison']['seeds']:
            for arm in config['comparison']['arms']:
                if phase=='comparison':
                    source=ARTIFACT/'untrained'/character/str(seed)/'panel/summary.json'
                    rows=read(source)['rows']
                else:
                    rows=read(ARTIFACT/'comparison'/str(seed)/arm/'panel/summary.json')['rows']
                    rows+=read(ARTIFACT/'training_counter_panels'/f'{arm}_{seed}'/'summary.json')['rows']
                points=[(0,counts(rows))]
                path=ARTIFACT/phase/str(seed)/arm
                spec=config['comparison'] if phase=='comparison' else feedback_spec(config,phase)
                thresholds=spec['checkpoints'] if phase=='comparison' else spec['main_checkpoints']
                for threshold in thresholds[:-1]:
                    summary=path/f'light_development_{threshold}/summary.json'
                    if summary.exists():points.append((threshold,counts(read(summary)['rows'])))
                if (path/'panel/summary.json').exists():
                    rows=read(path/'panel/summary.json')['rows']
                    if phase!='comparison':
                        extra=ARTIFACT/'selection'/phase/f'{arm}_{seed}'/'additional/summary.json'
                        if extra.exists():rows+=read(extra)['rows']
                    points.append((thresholds[-1],counts(rows)))
                for row in range(3):
                    if row==1 and phase=='comparison':continue
                    # Endpoint counters are displayed only when their full
                    # additional panel exists; missing is never zero wins.
                    values=points
                    if row==1 and (path/'panel/summary.json').exists():
                        extra=ARTIFACT/'selection'/phase/f'{arm}_{seed}'/'additional/summary.json'
                        if not extra.exists():values=points[:-1]
                    axes[row,column].plot([p[0]/1e6 for p in values],[p[1][row] for p in values],
                        color='tab:blue' if arm=='ordinary' else 'tab:orange',
                        linestyle='-' if seed==config['comparison']['seeds'][0] else '--',marker='o',
                        label=f'{arm}, seed{seed}')
        for row,denominator in enumerate((4,2*len(counter_ids),2)):
            axis=axes[row,column];axis.set_ylim(-.1,max(1,denominator)+.1);axis.set_yticks(range(max(1,denominator)+1));axis.grid(alpha=.25)
            axis.set_xlim(0,thresholds[-1]/1e6)
            if row==0:axis.set_title(phase.replace('_',' '))
            if column==0:axis.set_ylabel(('Base PvP wins /4','Training-counter wins /'+str(denominator),'CPU wins /2')[row])
            if row==2:axis.set_xlabel('Declared checkpoint (million native frames)')
    axes[1,0].text(.5,.5,'Counters generated after primary training',ha='center',va='center',transform=axes[1,0].transAxes,fontsize=8)
    axes[0,0].legend(fontsize=7,loc='best')
    fig.suptitle('Known development cases; greedy complete games; continuations are not extra independent seeds')
    fig.tight_layout();REPORT.mkdir(parents=True,exist_ok=True);fig.savefig(REPORT/'learning_curves.png',dpi=150);plt.close(fig)


def freeze_representatives(config):
    if config.get('own_character'):return fixed_role_freeze(config)
    """Choose only on known development conditions, then keep weights fixed."""
    from experiments.ryu_direct_league.policy import save
    state=read(REPORT/'summary.json')
    if state.get('final_frozen'):return state['frozen_models']
    for stage in selection_stages(config):
        runs=state.get(stage,{})
        if any(arm not in runs.get(str(seed),{})
               for seed in config['comparison']['seeds'] for arm in config['comparison']['arms']):
            raise RuntimeError(f'Complete every declared arm and seed in {stage} before selecting representatives')
    extra=state.get('retained_training_counters',[])+[
        dict(id='historical_strong_response',kind='agent',character='Ryu',path='models/opponents/strong_response.pt'),
        dict(id='historical_formal_reference',kind='agent',character='Ryu',path=config['reference'])]
    measurements={};choices={}
    for stage in selection_stages(config):
        for key,path in endpoint_paths(config,stage).items():
            base=evaluated(path,initial_pool(),path.parent/'panel',offsets=(0,))
            additional_dir=ARTIFACT/'selection'/stage/key/'additional'
            additional=evaluated(path,extra,additional_dir,offsets=(0,))
            parts=(base,additional)
            # Every complete case starts from the same legal observation and
            # h=0 through detected image discontinuity, without a host reset.
            # Reuse immutable completed base cases rather than replaying36
            # identical games for every selection candidate.
            if any(row.get('model_memory_resets')!=1 for part in parts for row in part['rows']):
                raise RuntimeError('Cannot compose cached cases with persistent cross-match policy history')
            arm=key.rsplit('_',1)[0]
            score=(sum(p['overall']['wins']+p['pve']['wins'] for p in parts),
                   min(sum(part['by_port'][str(port)]['wins'] for part in parts) for port in (1,2)),
                   sum(row['ko_win'] for part in parts for row in part['rows'] if row['opponent_kind']=='cpu'))
            measurements[stage+'/'+key]=dict(score=score,episodes=sum(len(p['rows']) for p in parts),
                source_evaluations=[(path.parent/'panel/summary.json').relative_to(ROOT).as_posix(),
                                    (additional_dir/'summary.json').relative_to(ROOT).as_posix()],
                scope='Composition of complete immutable base cases and new additional games; no extra independent victories or repeated base simulation.')
            if arm not in choices or score>choices[arm][0]:choices[arm]=(score,path,stage,key)
    models={'reference':config['reference']};lineage={}
    for arm,(score,path,stage,key) in choices.items():
        model_dir=ROOT/'models/counterfactual_dqn' if ARTIFACT==ROOT/'artifacts/counterfactual_dqn' else ARTIFACT/'frozen_models'
        destination=model_dir/f'{arm}.pt';actor=load_actor(path)
        save(destination,actor.policy,actor.definition,training=dict(source=path.relative_to(ROOT).as_posix(),stage=stage,run=key,
             selection_panel='Known common base plus retained training counters and historical agents',selection_score=score))
        models[arm]=destination.relative_to(ROOT).as_posix();lineage[arm]=dict(source=path.relative_to(ROOT).as_posix(),stage=stage,run=key,score=score)
    best=max(choices,key=lambda arm:choices[arm][0])
    report_update(final_frozen=True,final_frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  frozen_models=models,frozen_lineage=lineage,selection_measurements=measurements,
                  representative_model=models[best],research_representative=best,stage='frozen_before_final',
                  frozen_scope='Research representatives, not nominated replacements. No subsequent candidate training is permitted using final results.')
    return models


def protection_from_evaluation(evaluation,manifest_rows):
    """Score the fixed manifest using already completed exact configurations.

    Display aliases may differ. Weight paths and every other configuration
    field remain part of identity; same-name neural versions stay separate.
    This permits a larger complete panel to cover the gate without replaying
    its identical games, and refuses missing or contradictory evidence.
    """
    def identity(configuration,player,offset):
        settings={k:v for k,v in configuration.items() if k!='id'}
        return json.dumps(settings,sort_keys=True),int(player),int(offset)
    available={}
    for row in evaluation['rows']:
        available.setdefault(identity(row['opponent_config'],row['player'],row['offset']),[]).append(row)
    kept=[];lost=[];required=[r for r in manifest_rows if r['protected']]
    if not required:raise ValueError('Protection manifest has no protected conditions')
    fields=('result','end_reason','self_hp','opponent_hp','native_frames','ko_win')
    for original in required:
        matches=available.get(identity(original['opponent_configuration'],original['player'],original['offset']),[])
        if not matches:raise ValueError('Complete evaluation is missing a protected exact configuration')
        row=matches[0]
        if any(any(other[k]!=row[k] for k in fields) for other in matches[1:]):
            raise RuntimeError('Equivalent configuration aliases have contradictory complete outcomes')
        if row['result'] not in ('win','loss','draw') or row['end_reason']=='frame_cap':
            raise ValueError('An unfinished branch cannot satisfy complete-game protection')
        condition=dict(opponent=row['opponent'],player=row['player'],offset=row['offset'],original=original)
        good=row['result']=='win' and (row['opponent_kind']!='cpu' or row['ko_win'])
        (kept if good else lost).append(dict(condition=condition,result=row))
    return dict(protected_wins_kept=len(kept),required=len(required),passed=len(kept)==len(required),
        evaluated_consumed_conditions=len(evaluation['rows']),all_consumed_wins=sum(r['result']=='win' for r in evaluation['rows']),
        original52_kept=sum(x['condition']['original']['original52'] for x in kept),
        expanded64_kept=sum(x['condition']['original']['expanded64'] for x in kept),
        reference76_kept=sum(x['condition']['original']['reference_result']=='win' for x in kept),
        pve_ko=sum(x['result']['opponent_kind']=='cpu' for x in kept),
        protected_PvP_by_port={str(p):dict(
            conditions=sum(x['result']['opponent_kind']!='cpu' and x['result']['player']==p for x in kept+lost),
            wins=sum(x['result']['opponent_kind']!='cpu' and x['result']['player']==p for x in kept)) for p in (1,2)},
        lost=lost,evaluation=evaluation)


def old_protection(models,*,publish=True,full_known=False,output=None,record_histories=False):
    """Exact consumed configurations, preserving weight paths and panel labels."""
    manifest=read(ROOT/'reports/selective_retention/protection_manifest.json')
    if sum(r['protected'] for r in manifest['rows'])!=90:
        raise ValueError('The fixed protection union is not the declared90 conditions')
    original=[r for r in manifest['rows'] if full_known or r['protected']]
    pool=[];mapping=[];configs={}
    for row in original:
        config=copy.deepcopy(row['opponent_configuration'])
        identity=json.dumps(config,sort_keys=True)
        if identity not in configs:
            config['id']=f'protected_{len(pool)}_'+config['id'];configs[identity]=config['id'];pool.append(config)
        mapping.append(dict(opponent=configs[identity],player=row['player'],offset=row['offset'],original=row))
    conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in mapping]
    results={}
    for name,path in models.items():
        directory=(ARTIFACT/'old_protection' if output is None else Path(output))/name
        recorder=None
        if record_histories:
            from experiments.ryu_counterq.replay import TrajectoryRecorder
            if (directory/'summary.json').exists() and not (directory/'actual_episodes.pt').exists():
                raise ValueError('Cached protection scores have no raw histories; use an explicit new collection directory')
            if not (directory/'summary.json').exists():recorder=TrajectoryRecorder(pool,str(path))
        ev=evaluated(path,pool,directory,conditions=conditions,
            **(dict(transition_observer=recorder) if recorder is not None else {}))
        if recorder is not None:
            import torch
            torch.save(dict(episodes=recorder.episodes),directory/'actual_episodes.pt')
        by_key={(r['opponent'],r['player'],r['offset']):r for r in ev['rows']}
        results[name]=protection_from_evaluation(ev,manifest['rows'])
        if record_histories:results[name]['history_record']=(directory/'actual_episodes.pt').relative_to(ROOT).as_posix()
        if name=='reference':
            results[name]['historical_result_mismatches']=[condition for condition in mapping
                if by_key[tuple(condition[k] for k in ('opponent','player','offset'))]['result']!=condition['original']['reference_result']]
    if publish:
        report_update(old_protection=results,old_protection_scope=f'Consumed{len(original)}-condition panel; fixed90-condition protection union, original52 and expanded64 subsets. Own character is an explicit experimental variable; this is not literal inheritance of Ryu trajectories.')
    return results


def frozen_final(config):
    models=freeze_representatives(config);state=read(REPORT/'summary.json')
    rules=final_rule_pool(config);cpu=initial_pool(stars=range(1,9),styles=())
    focus=[r for r in read(ROOT/config['final']['known_focus_pool']) if r['kind']!='cpu']
    write(ARTIFACT/'final_rules.json',rules)
    results={};floors={}
    # Reference measured first on exact frozen conditions; numerical floors
    # are fixed before any candidate result on these conditions is inspected.
    for name,path in models.items():
        r=evaluated(path,rules,ARTIFACT/'final_rules'/name,offsets=tuple(config['final']['offsets']))
        c=evaluated(path,cpu,ARTIFACT/'cpu_coverage'/name,offsets=(0,))
        f=evaluated(path,focus,ARTIFACT/'known_focus'/name,offsets=(0,12))
        results[name]=dict(rules=r,cpu=c,known_focus=f)
        if name=='reference':
            floors['rules']={str(p):r['by_port'][str(p)]['wins'] for p in (1,2)}
            floors['cpu_ko_by_star']={str(s):sum(row['ko_win'] for row in c['rows'] if row['cpu_stars']==s) for s in range(1,9)}
            report_update(final_floors=floors,final_reference_first=True)
        report_update(final_results=results,stage='frozen_final_evaluation')
    if config.get('own_character'):
        counter_runs,snapshots=matched_counter_search(config,models,config['final'],'final')
    else:
        role=state.get('training_counter_character') or counter_role(state['comparison'])
        snapshots=[];counter_runs={}
        for target,path in models.items():
            actor=load_actor(path);character=actor.definition.get('character','Ryu')
            opponent=[dict(id='final_target_'+target,kind='agent',character=character,path=path)]
            for seed in config['final']['counter_seeds']:
                out=ARTIFACT/'final_counter_training'/target/str(seed)
                if not completed_training(out,config,role,seed,'ordinary',config['final']['counter_frames_per_target_seed'],opponent):
                    train_run(config,role,seed,'ordinary',config['final']['counter_frames_per_target_seed'],out,
                              pool=opponent,checkpoints=config['final']['counter_snapshots'])
                counter_runs[target+'/'+str(seed)]=read(out/'summary.json')
                for frame in config['final']['counter_snapshots']:
                    actual=next(c['actual_frames'] for c in counter_runs[target+'/'+str(seed)]['curve'] if c['threshold']==frame)
                    snapshots.append(dict(id=f'final_counter_{target}_{seed}_{frame}',kind='agent',character=role,
                        path=(out/f'checkpoint_{frame}.pt').relative_to(ROOT).as_posix(),target=target,training_seed=seed,
                        checkpoint_target_frames=frame,training_frames=actual))
    write(ARTIFACT/'final_counter_pool.json',snapshots)
    for name,path in models.items():
        ev=evaluated(path,snapshots,ARTIFACT/'final_counter_panels'/name,offsets=tuple(config['final']['offsets']))
        results[name]['counters']=ev
        if name=='reference':
            floors['counters']={str(p):ev['by_port'][str(p)]['wins'] for p in (1,2)}
            report_update(final_floors=floors)
    protections=old_protection(models,full_known=config['acceptance'].get('full_consumed_regression',False));acceptance={}
    reference=results['reference']
    for name,data in results.items():
        if name=='reference':continue
        ports=all(data[panel]['by_port'][str(p)]['wins']>=floors[panel][str(p)] for panel in ('rules','counters') for p in (1,2))
        ko=all(sum(r['ko_win'] for r in data['cpu']['rows'] if r['cpu_stars']==s)>=floors['cpu_ko_by_star'][str(s)] for s in range(1,9))
        complete_ko=sum(r['ko_win'] for r in data['cpu']['rows'])>=config['acceptance'].get('pve_ko_required',0)
        new_wins=sum(c['result']=='win' and r['result']!='win' for panel in ('rules','counters')
                     for r,c in zip(reference[panel]['rows'],data[panel]['rows']))
        acceptance[name]=dict(old_protection_passed=protections[name]['passed'],new_port_floors_passed=ports,
                              cpu_ko_floors_passed=ko,required_cpu_ko_passed=complete_ko,new_complete_wins=new_wins,
                              score_gate_passed=protections[name]['passed'] and ports and ko and complete_ko and new_wins>0
                                  and not protections['reference']['historical_result_mismatches'],
                              nomination_eligible=False,
                              pending=['Final-weight independent export and continuous dual-port complete matches','Actual behavioral evidence and final process cleanup'])
    report_update(final_results=results,final_floors=floors,final_counter_runs=counter_runs,final_counter_pool=snapshots,
        acceptance=acceptance,stage='frozen_final_complete',
        official_arena='Not executed. Local raw/SB3 reload and game checks do not certify Arena.',
        final_scope='Novel rule timing/combinations and independently trained final counters were unused by candidate training. CPU coverage and old protections are known regression. Finite tests do not imply universal robustness.')


def validate_exported_policy(source,out,selected):
    """One standalone-export audit on previously scored complete conditions.

    Both the older fixed-role study and subsequent consolidated policies use
    this implementation. The observer only compares outputs; it never acts.
    """
    from experiments.league_adaptive.evaluate import InferenceIdentityAudit
    out=Path(out);selected=list(selected)
    if not selected:raise ValueError('Select complete identity matches before exporting')
    keys=('opponent','opponent_kind','player','offset','result','end_reason','self_hp','opponent_hp','native_frames')
    conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in selected]
    if len({tuple(r.values()) for r in conditions})!=len(conditions):
        raise ValueError('Duplicate export identity condition')
    rivals=[]
    for row in selected:
        prior=next((r for r in rivals if r['id']==row['opponent']),None)
        if prior is not None and prior!=row['opponent_config']:
            raise ValueError('Export identity aliases different opponent versions')
        if prior is None:rivals.append(row['opponent_config'])
    has_memory=any('model_memory_resets' in r for r in selected)
    if has_memory and any(r.get('model_memory_resets')!=1 for r in selected):
        raise ValueError('Raw identity case retained memory from a preceding match')
    if not out.exists():
        subprocess.run([sys.executable,'-m','experiments.ryu_direct_league.portable_export',
            '--agent',str(source),'--out',str(out)],cwd=ROOT,check=True)
    reload=read(out/'reload_result.json')
    source_alias=None
    if Path(reload['source_agent']).resolve()!=(ROOT/source).resolve():
        # Publishing an unchanged checkpoint may give it a new path. Keep the
        # original reload receipt and verify the two policies directly rather
        # than relabeling that receipt or repeating synthetic export checks.
        from experiments.ryu_counterq.report import model_identity
        source_alias=model_identity(reload['source_agent'],source)
    checked=out/'complete_match_identity.json'
    if checked.exists():
        saved=read(checked)
        prior_conditions=saved.get('conditions')
        if prior_conditions is None:
            cached=read(out/'complete_matches/summary.json')
            prior_conditions=[{k:r[k] for k in ('opponent','player','offset')} for r in cached['rows']]
        if prior_conditions!=conditions:raise ValueError('Export identity conditions changed')
        return saved
    audit=InferenceIdentityAudit(source);actor=load_actor(out/'policy.zip')
    try:
        ev=evaluate(actor,rivals,out/'complete_matches',cpu=False,
            ports=tuple(sorted({r['player'] for r in selected})),
            offsets=tuple(sorted({r['offset'] for r in selected})),progress=False,
            arena_call_sequence=True,conditions=conditions,transition_observer=audit)
        identity=audit.result()
    finally:actor.close();audit.close()
    expected={(r['opponent'],r['player'],r['offset']):r for r in selected}
    differences=[dict(condition=[r['opponent'],r['player'],r['offset']],field=k,
        expected=expected[(r['opponent'],r['player'],r['offset'])][k],actual=r[k])
        for r in ev['rows'] for k in keys if r[k]!=expected[(r['opponent'],r['player'],r['offset'])][k]]
    if differences:raise RuntimeError(f'Export changed complete matches: {differences}')
    if has_memory and any(r.get('model_memory_resets')!=1 for r in ev['rows']):
        raise RuntimeError('Export cross-match continuity reset differs')
    result=dict(isolated_reload=reload,source_alias_identity=source_alias,
        actual_input_identity=identity,conditions=conditions,
        complete_matches=len(selected),terminal_fields_compared=len(keys),terminal_differences=differences,
        memory_resets=[r.get('model_memory_resets') for r in ev['rows']],
        native_frames=ev['native_frames'],seconds=ev['seconds'],
        scope='Technical raw/export identity on selected already-scored complete games; warmup and no explicit cross-game reset. Not additional ability evidence or official Arena acceptance.')
    write(checked,result);return result


def validate_exports(config):
    """Audit final weights through the shared standalone-export implementation."""
    state=read(REPORT/'summary.json')
    if not state.get('final_frozen') or 'final_counter_pool' not in state:
        raise RuntimeError('Complete frozen evaluation before final export validation')
    results={}
    for arm in config['comparison']['arms']:
        source=state['frozen_models'][arm];out=ARTIFACT/'exports'/arm
        lineage=state['frozen_lineage'][arm]
        base={'rows':[]} if config.get('own_character') else read(ARTIFACT/lineage['stage']/lineage['run'].rsplit('_',1)[1]/arm/'panel/summary.json')
        cases=config['final']['export_identity_conditions']
        available={(r['opponent'],r['player'],r['offset']):r
            for r in base['rows']+state['final_results'][arm]['cpu']['rows']+state['final_results'][arm]['rules']['rows']}
        selected=[available[tuple(c[k] for k in ('opponent','player','offset'))] for c in cases]
        results[arm]=validate_exported_policy(source,out,selected)
    report_update(export_validation=results)
    return results


def record_frozen_behavior(config):
    """Render preregistered complete cases with the shared native recorder."""
    state=read(REPORT/'summary.json')
    if not state.get('final_frozen') or 'final_counter_pool' not in state:
        raise RuntimeError('Complete frozen evaluation before behavior recordings')
    pools=dict(cpu=initial_pool(stars=range(1,9),styles=()),rules=final_rule_pool(config),
               counters=state['final_counter_pool'])
    for name,pool in pools.items():write(ARTIFACT/f'behavior_{name}_pool.json',pool)
    results=[]
    for number,case in enumerate(config['final']['behavior_cases']):
        for name in case['models']:
            if name=='research':name=state['research_representative']
            source=state['frozen_models'][name];out=ARTIFACT/'final_behavior'/f'{number}_{name}'
            if not (out/'summary.json').exists():
                subprocess.run([sys.executable,'-m','experiments.league_adaptive.record','--agent',source,
                    '--opponent-pool',str(ARTIFACT/f"behavior_{case['panel']}_pool.json"),
                    '--opponent-id',case['opponent'],'--player',str(case['player']),'--offset',str(case['offset']),
                    '--out',str(out),'--label',name],cwd=ROOT,check=True)
            recorded=read(out/'summary.json')
            if Path(recorded['agent']).resolve()!=(ROOT/source).resolve():raise ValueError('Behavior movie source differs')
            expected=next(r for r in state['final_results'][name][case['panel']]['rows']
                if all(r[k]==case[k] for k in ('opponent','player','offset')))
            if recorded['opponent']!=expected['opponent_config'] or any(recorded[k]!=expected[k]
                for k in ('player','offset','result','self_hp','opponent_hp','native_frames','end_reason')):
                raise RuntimeError('Full behavior movie does not reproduce its scored frozen case')
            results.append(dict(case=case,model=name,out=out.relative_to(ROOT).as_posix(),
                                result=recorded,terminal_fields_equal=True))
            report_update(behavior=results)
    return results


def fixed_role_training(config,stage):
    """Fixed-role phases share the existing learner and complete evaluator."""
    if (REPORT/'summary.json').exists() and read(REPORT/'summary.json').get('final_frozen'):
        raise RuntimeError('Main candidate weights are frozen; final conditions cannot feed continued training')
    role=config['own_character'];spec=config[stage]
    cfg=copy.deepcopy(config)
    cfg['learner'].update(spec.get('learner_overrides',{}))
    cfg['counterfactual'].update(spec.get('counterfactual_overrides',{}))
    budget=spec.get('frames_per_seed',spec.get('frames_per_arm_seed',spec.get('smoke_frames')))
    seeds=[config['pretrain']['smoke_seed']] if stage=='short' else spec.get('seeds',config['comparison']['seeds'])
    arms=['ordinary'] if stage in ('short','pretrain','replay_prefill') else config['comparison']['arms']
    pool=read(REPORT/'summary.json')['league_pool'] if spec.get('use_league_pool') else config['pool']
    if stage=='replay_prefill':
        retained=read(REPORT/'summary.json')['retained_training_counters']
        pool=[r for r in pool if r['kind']=='cpu' and r['stars'] in spec['cpu_stars']
              or r['id'] in spec['pvp_ids']]+retained
    dev=[p for p in config['pool'] if p['id'] in config['notebook']['demo_opponents']] if stage=='short' else config['development_pool']
    if stage=='replay_prefill':dev=None
    learning=read(REPORT/'summary.json').get('learning',{}) if (REPORT/'summary.json').exists() else {}
    results={}
    for seed in seeds:
        results[str(seed)]={}
        for arm in arms:
            source=None
            if spec.get('parent_stage'):
                parent_arm='ordinary' if spec['parent_stage'] in ('pretrain','replay_prefill') else arm
                source=ARTIFACT/spec['parent_stage']/str(seed)/parent_arm/'agent.pt'
                if not source.exists():raise FileNotFoundError(source)
            out=ARTIFACT/stage/str(seed)/arm
            if not completed_training(out,cfg,role,seed,arm,budget,pool):
                train_run(cfg,role,seed,arm,budget,out,pool=pool,start=source,
                          checkpoints=spec['checkpoints'],dev_pool=dev)
            training=read(out/'summary.json')
            if stage=='replay_prefill':
                if (not training['coverage_completed'] or training['costs'].get('optimizer_updates',0)
                        or any(not r['complete'] or r['result']=='cutoff' for r in training['rows'])):
                    raise RuntimeError('Common replay collection must cover every declared case without parameter updates')
                import torch
                from experiments.league_adaptive.runtime import same_state
                before=torch.load(source,map_location='cpu',weights_only=False)
                after=torch.load(out/'agent.pt',map_location='cpu',weights_only=False)
                if not same_state(before['state_dict'],after['state_dict']) or not same_state(before['optimizer_state'],after['optimizer_state']):
                    raise RuntimeError('Replay collection changed parent weights or optimizer')
                parent_training=read(source.parent/'summary.json')
                terminal=parent_training['curve'][-1]
                training['inherited_development']=dict(source=(source.parent/f"development_{terminal['threshold']}/summary.json").relative_to(ROOT).as_posix(),
                    reason='Parent policy and optimizer remain numerically identical; common replay collection performed no updates.')
                write(out/'summary.json',training)
            if source is not None and Path(read(out/'config.json')['source']).resolve()!=source.resolve():
                raise ValueError('A completed continuation names a different parent checkpoint')
            results[str(seed)][arm]=dict(agent=(out/'agent.pt').relative_to(ROOT).as_posix(),
                actual_training_frames=training['actual_training_frames'],costs=training['costs'],
                curve=[dict(threshold=point['threshold'],actual_frames=point['actual_frames'],
                    checkpoint=point['checkpoint'],development_score=fixed_role_score(point['development']) if point.get('development') else None,
                    full_record=(out/f"development_{point['threshold']}/summary.json").relative_to(ROOT).as_posix() if point.get('development') else None)
                    for point in training['curve']],branch_summary=training['branch_summary'],
                parent=None if source is None else source.relative_to(ROOT).as_posix())
            if stage=='replay_prefill':
                results[str(seed)][arm].update(parent_policy_unchanged=True,parent_optimizer_unchanged=True,
                    replay_transitions=training['replay_transitions'],replay_bytes=training['replay_bytes'],
                    complete_conditions=len(training['rows']),
                    source_scope='Capacity-retention data from the frozen parent, not newly discovered ability. Full legal trajectories; no imitation loss or on-policy PPO advantage samples.')
            learning[stage]=results
            report_update(stage=stage+'_running',selected_character=role,learning=learning)
    if stage=='league_adaptation':
        from experiments.league_adaptive.evaluate import paired_regression
        state=read(REPORT/'summary.json');changes={};snapshots=state['training_counter_snapshots']
        retained={r['id'] for r in state['retained_training_counters']}
        for name,path in endpoint_paths(config,stage).items():
            baseline='parent_'+name.rsplit('_',1)[1] if spec['parent_stage']=='replay_prefill' else name
            before=read(ARTIFACT/'training_counter_panels'/baseline/'summary.json')
            after=evaluated(path,snapshots,ARTIFACT/'training_counter_panels'/(stage+'_'+name),offsets=(0,))
            identity=lambda r:(r['opponent'],r['player'],r['offset'],r['opponent_config'])
            if [identity(r) for r in before['rows']]!=[identity(r) for r in after['rows']]:
                raise ValueError('Counter feedback comparison changed an opponent version or condition')
            delta=paired_regression(before['rows'],after['rows'])
            training=read(path.parent/'summary.json')
            seen=[r for r in training['rows'] if r['opponent'] in retained]
            changes[name]=dict(before_pvp=before['overall'],after_pvp=after['overall'],
                before_ports=before['by_port'],after_ports=after['by_port'],
                old_wins_kept=delta['old_wins_kept'],new_wins=[r['condition'] for r in delta['new_wins']],
                lost_wins=[r['condition'] for r in delta['old_wins_lost']],
                actual_counter_training_episodes=len(seen),actual_counter_training_complete=sum(r['complete'] for r in seen),
                scope='Same complete responsive-counter cases before/after actual shared-pool learning; training outcomes include exploration. This is consumed development and is not an extra independent seed.')
        report_update(counter_feedback_changes=changes)
    report_update(stage=stage+'_complete',selected_character=role,learning=learning)
    return results


def fixed_role_score(evaluation):
    """Preregistered known-development order, independent of final results."""
    return (sum(r['ko_win'] for r in evaluation['rows'] if r['opponent_kind']=='cpu'),
            min(evaluation['by_port'][str(p)]['wins'] for p in (1,2)),evaluation['overall']['wins'])


def fixed_role_endpoint(config,path):
    """Reuse the actual terminal checkpoint evaluation; no duplicate games."""
    training=read(path.parent/'summary.json')
    if not training['curve'] and training.get('inherited_development'):
        source=ROOT/training['inherited_development']['source'];return read(source),source
    last=training['curve'][-1]
    if last['actual_frames']==training['actual_training_frames'] and last.get('development'):
        pool=[r['opponent_config'] for r in last['development']['rows']]
        if all(r in pool for r in config['development_pool']):
            return last['development'],path.parent/f"development_{last['threshold']}/summary.json"
    out=path.parent/'panel'
    return evaluated(path,config['development_pool'],out,offsets=(0,)),out/'summary.json'


def matched_counter_search(config,models,spec,phase,*,artifact_root=None):
    """Shared counter orchestration; both learners keep their existing trainer.

    PPO counters use one fresh initialization per role/seed across targets.
    This never loads a learned Q estimate as a policy or value target.
    """
    root=ARTIFACT if artifact_root is None else Path(artifact_root).resolve()
    if not root.resolve().is_relative_to(ROOT/'artifacts'):
        raise ValueError('Counter process data must remain under artifacts')
    algorithm=spec.get('algorithm','DDQN')
    if algorithm not in ('DDQN','PPO'):raise ValueError('Unknown counter algorithm')
    progress_name=spec.get('progress_record',phase+'_counter_runs.json')
    if Path(progress_name).name!=progress_name or Path(progress_name).suffix!='.json':
        raise ValueError('Counter progress record must be a local JSON filename')
    snapshots=[];runs={}
    for target,path in models.items():
        actor=load_actor(path)
        try:own=actor.definition.get('character','Ryu')
        finally:actor.close()
        pool=[dict(id='counter_target_'+target,kind='agent',character=own,path=path)]
        for role in spec['counter_roles']:
            cfg=copy.deepcopy(config)
            if algorithm=='DDQN':
                cfg['network'].update(tap_delay_frames=2 if role=='Guile' else 0,signed_relative_position=role=='Guile')
            elif cfg['network'].get('input_history') is None:
                raise ValueError('Sequence PPO counters require their declared legal input-history configuration')
            cfg['learner']['proposal_style']='guile_charge' if role=='Guile' else 'mixed'
            for seed in spec['counter_seeds']:
                budget=spec['counter_frames_per_target_role_seed'];out=root/(phase+'_counter_training')/target/role/str(seed)
                key=f'{target}/{role}/{seed}'
                if algorithm=='DDQN':
                    if not completed_training(out,cfg,role,seed,'ordinary',budget,pool):
                        train_run(cfg,role,seed,'ordinary',budget,out,pool=pool,checkpoints=spec['counter_snapshots'])
                    training=read(out/'summary.json')
                    runs[key]={k:training[k] for k in ('actual_training_frames','costs','algorithm')}
                    checkpoints={c['threshold']:(out/f"checkpoint_{c['threshold']}.pt",c['actual_frames'])
                        for c in training['curve']}
                else:
                    import torch
                    from experiments.ryu_counterq.policy import random_sequence_q,construct_sequence_q
                    from experiments.ryu_direct_league.policy import save
                    if cfg['learner']['gamma']!=1.:raise ValueError('Declare a common discounted counter experiment first')
                    cfg.update(own_character=role,pool=pool,development_pool=[])
                    cfg.pop('success_consolidation',None)
                    cfg['learner'].pop('training_focus',None)
                    cfg['learner'].update(sampling_mode='fixed_groups',coverage_offsets=[0,12],
                        training_group_weights=dict(pve=0.,p1=.5,p2=.5))
                    initial=root/(phase+'_counter_initialization')/role/f'{seed}.pt'
                    if not initial.exists():
                        torch.manual_seed(seed);fresh,definition=random_sequence_q(role,cfg['network'])
                        definition['kind']='sequence_ppo';policy=construct_sequence_q(definition)
                        policy.load_state_dict(fresh.state_dict(),strict=True)
                        save(initial,policy,definition,dict(algorithm='PPO',seed=seed,network=cfg['network'],
                            initialization='Fresh random tensors shared across targets, no learned Q transfer'))
                    else:
                        source=torch.load(initial,map_location='cpu',weights_only=False)
                        if (source['training']['seed']!=seed or source['training']['network']!=cfg['network']
                                or source['definition']['kind']!='sequence_ppo' or source['definition']['character']!=role):
                            raise ValueError('Cached counter initialization has a different protocol')
                    out.mkdir(parents=True,exist_ok=True);configuration=out/'configuration.json';pool_file=out/'pool.json'
                    if configuration.exists() and read(configuration)!=cfg:
                        raise ValueError('Counter target or training protocol changed')
                    write(configuration,cfg);write(pool_file,pool);run=out/'PPO'
                    if not (run/'result.json').exists():
                        if run.exists():raise RuntimeError('Preserve interrupted PPO counter; explicitly account for its continuation')
                        subprocess.run([sys.executable,'-m','experiments.ryu_adaptive_curriculum.train_direct_sequence',
                            '--config',str(configuration),'--pool',str(pool_file),'--initial',str(initial),
                            '--seed',str(seed),'--out',str(run),'--native-frames',str(budget),
                            '--checkpoints',*map(str,spec['counter_snapshots']),
                            '--max-seconds','0','--family-entropy','0','--capture-updates'],cwd=ROOT,check=True)
                    training=read(run/'result.json');arguments=training['arguments']
                    if (arguments['seed']!=seed or arguments['native_frames']!=budget
                            or Path(arguments['initial']).resolve()!=initial.resolve()):
                        raise ValueError('Completed PPO counter has a different declared budget/initialization')
                    costs={k:training[k] for k in ('optimizer_updates','optimization_samples','optimization_seconds',
                        'environment_calls','seconds','evaluation_native_frames')}
                    runs[key]=dict(actual_training_frames=training['native_frames'],costs=costs,algorithm='PPO',
                        requested_training_frames=budget,native_frame_overrun=training['native_frames']-budget,
                        initialization=initial.relative_to(ROOT).as_posix(),target=path,
                        condition_exposure=read(run/'condition_exposure.json'),
                        scope='Same fresh initialization and requested native budget across targets; actual rollout overshoot is reported.')
                    checkpoints={}
                    for record in sorted(run.glob('checkpoint_*.json')):
                        item=read(record)
                        for frame in item['requested_native_frames']:
                            if frame in checkpoints:raise ValueError('Duplicate requested counter snapshot')
                            checkpoints[frame]=(record.with_suffix('.pt'),item['native_frames'])
                for frame in spec['counter_snapshots']:
                    checkpoint,actual=checkpoints[frame]
                    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
                    snapshot=dict(id=f'{phase}_counter_{target}_{role}_{seed}_{frame}',kind='agent',character=role,
                        path=checkpoint.relative_to(ROOT).as_posix(),target=target,
                        training_seed=seed,checkpoint_target_frames=frame,training_frames=actual)
                    if algorithm=='PPO':snapshot['algorithm']=algorithm
                    snapshots.append(snapshot)
                write(root/progress_name,dict(runs=runs,snapshots=snapshots))
    return runs,snapshots


def fixed_role_league(config):
    """Measured counters enter the common pool before main adaptation."""
    state=read(REPORT/'summary.json')
    if state.get('final_frozen'):raise RuntimeError('Final weights are frozen')
    candidates=endpoint_paths(config,'paired');measurements={}
    for name,path in candidates.items():
        if not (path.parent/'summary.json').exists():raise FileNotFoundError(path)
        ev,_=fixed_role_endpoint(config,path)
        measurements[name]=fixed_role_score(ev)
    target=max(measurements,key=lambda name:(measurements[name],name))
    runs,snapshots=matched_counter_search(config,{target:candidates[target].relative_to(ROOT).as_posix()},config['training_counters'],'training')
    panels={name:evaluated(path,snapshots,ARTIFACT/'training_counter_panels'/name,offsets=(0,)) for name,path in candidates.items()}
    if config.get('replay_prefill'):
        for seed in config['comparison']['seeds']:
            path=ARTIFACT/'pretrain'/str(seed)/'ordinary/agent.pt'
            evaluated(path,snapshots,ARTIFACT/'training_counter_panels'/('parent_'+str(seed)),offsets=(0,))
    retained=[];effectiveness={}
    for role in config['training_counters']['counter_roles']:
        group=[]
        for rival in snapshots:
            if rival['character']!=role:continue
            rows=[r for panel in panels.values() for r in panel['rows'] if r['opponent']==rival['id']]
            losses=sum(r['result']=='loss' for r in rows)
            effectiveness[rival['id']]=dict(main_losses=losses,main_games=len(rows),main_hp_margin=sum(r['self_hp']-r['opponent_hp'] for r in rows))
            if losses:group.append(rival)
        if group:
            selected=copy.deepcopy(max(group,key=lambda r:(effectiveness[r['id']]['main_losses'],-effectiveness[r['id']]['main_hp_margin'],r['id'])))
            selected['training_weight']=config['training_counters'].get('retained_sampling_weight',1.)
            retained.append(selected)
    if not retained:raise RuntimeError('Counters exposed no complete-game weakness; inspect before spending adaptation budget')
    report_update(training_counter_runs=runs,training_counter_snapshots=snapshots,retained_training_counters=retained,
        training_counter_effectiveness=effectiveness,training_counter_target=target,league_pool=config['pool']+retained,
        stage='training_counters_measured',training_counter_scope='At most one effective snapshot per role. All four paired models faced all snapshots; role names alone do not prove tactical diversity. Retained controllers enter both arms\' subsequent training.')
    return retained


def fixed_role_freeze(config):
    from experiments.ryu_direct_league.policy import save
    state=read(REPORT/'summary.json')
    if state.get('final_frozen'):return state['frozen_models']
    choices={};measurements={}
    for phase in selection_stages(config):
        for name,path in endpoint_paths(config,phase).items():
            if not (path.parent/'summary.json').exists():raise FileNotFoundError(path)
            ev,source_evaluation=fixed_role_endpoint(config,path)
            score=fixed_role_score(ev);arm=name.rsplit('_',1)[0]
            measurements[phase+'/'+name]=dict(score=score,episodes=len(ev['rows']),evaluation=source_evaluation.relative_to(ROOT).as_posix())
            if arm not in choices or score>choices[arm][0]:choices[arm]=(score,path,phase,name)
    models={'reference':config['reference']};lineage={}
    for arm,(score,path,phase,name) in choices.items():
        destination=ROOT/'models'/config['experiment']/(arm+'.pt')
        if destination.exists():raise FileExistsError(destination)
        actor=load_actor(path)
        try:save(destination,actor.policy,actor.definition,training=dict(source=path.relative_to(ROOT).as_posix(),stage=phase,run=name,selection_score=score))
        finally:actor.close()
        models[arm]=destination.relative_to(ROOT).as_posix()
        lineage[arm]=dict(source=path.relative_to(ROOT).as_posix(),stage=phase,run=name,score=score)
    best=max(choices,key=lambda arm:(choices[arm][0],arm))
    report_update(final_frozen=True,final_frozen_at_utc=datetime.now(timezone.utc).isoformat(),frozen_models=models,
        frozen_lineage=lineage,selection_measurements=measurements,representative_model=models[best],research_representative=best,
        selected_character=config['own_character'],stage='frozen_before_final',
        frozen_scope='Research representatives only; selection uses known development endpoints. No final result may change these weights.')
    return models


def fixed_role_interface(config,args):
    """Notebook stages call the original evaluation/export/record entries."""
    agent=args.agent
    if agent is None:
        state=read(REPORT/'summary.json') if (REPORT/'summary.json').exists() else {}
        agent=state.get('representative_model')
    if agent is None:
        agent=(ARTIFACT/'short'/str(config['pretrain']['smoke_seed'])/'ordinary/agent.pt').relative_to(ROOT).as_posix()
    checked=load_actor(agent)
    try:
        if checked.definition.get('character')!=config['own_character']:
            raise ValueError('The selected agent does not control the fixed own role')
        algorithm='PPO' if checked.definition['kind'] in ('sequence_ppo','recurrent_direct_ppo','memory_ppo','neural_mixture') else 'DDQN'
    finally:checked.close()
    if args.stage=='evaluate':
        pool=read(ROOT/args.pool) if args.pool else [p for p in config['pool'] if p['id'] in config['notebook']['demo_opponents']]
        conditions=read(ROOT/args.conditions) if args.conditions else None
        return evaluated(agent,pool,ARTIFACT/'evaluation',offsets=tuple(args.offsets),
                         ports=tuple(args.players),conditions=conditions)
    if args.stage=='export':
        return subprocess.run([sys.executable,'-m','experiments.ryu_direct_league.portable_export',
            '--agent',agent,'--out',str(ARTIFACT/'standalone')],cwd=ROOT,check=True)
    rivals=read(ROOT/args.pool) if args.pool else config['pool']
    opponent=next(p for p in rivals if p['id']==args.opponent)
    command=[sys.executable,'-m','experiments.league_adaptive.record','--agent',agent,
        '--player',str(args.players[0]),'--offset',str(args.offsets[0]),'--out',str(ARTIFACT/'video'),
        '--label',config['own_character']+' '+algorithm]
    if opponent['kind']=='cpu':
        command+=['--cpu-character',opponent['character'],'--cpu-stars',str(opponent['stars'])]
    else:
        write(ARTIFACT/'interface_pool.json',rivals)
        command+=['--opponent-pool',str(ARTIFACT/'interface_pool.json'),'--opponent-id',opponent['id']]
    subprocess.run(command,cwd=ROOT,check=True)
    (ARTIFACT/'video/playback.html').write_text('<!doctype html><meta charset="utf-8"><video controls src="replay.mp4" style="max-width:100%"></video>',encoding='utf-8')


def main():
    global ARTIFACT,REPORT
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('screen','refine','shortlist','compare','discovery','feedback','low-exploration','interleaved-feedback','freeze','final','validate','behavior','initial-baselines','monitor','plot-screen','plot-learning','short','pretrain','paired','replay_prefill','league','league_adaptation','evaluate','export','video','stabilization-short','stabilization-resume-check','stabilization-prepare','stabilization-run','stabilization-analyze','stabilization-input-history'),required=True)
    parser.add_argument('--config',default='configs/counterfactual_dqn.json')
    parser.add_argument('--character')
    parser.add_argument('--agent');parser.add_argument('--pool');parser.add_argument('--conditions')
    parser.add_argument('--players',nargs='+',type=int,choices=(1,2),default=(1,2))
    parser.add_argument('--offsets',nargs='+',type=int,default=(0,))
    parser.add_argument('--opponent',default='cpu_Ryu_1')
    parser.add_argument('--run-root')
    parser.add_argument('--report-root')
    parser.add_argument('--monitor-stage',choices=('comparison','feedback','low_exploration','interleaved_feedback'),default='comparison')
    args=parser.parse_args();setup();config=read(ROOT/args.config)
    name=config.get('experiment','counterfactual_dqn') if config.get('own_character') else 'counterfactual_dqn'
    ARTIFACT=(ROOT/(args.run_root or 'artifacts/'+name)).resolve()
    REPORT=(ROOT/(args.report_root or 'reports/'+name)).resolve()
    if config.get('protocol')=='stabilization':
        from experiments.ryu_counterq.stabilization import dispatch
        return dispatch(config,args,ARTIFACT,REPORT,report_update,completed_training,fixed_role_interface)
    if config.get('own_character'):
        if args.stage in ('short','pretrain','paired','replay_prefill','league_adaptation'):
            fixed_role_training(config,args.stage);return
        if args.stage in ('evaluate','export','video'):
            fixed_role_interface(config,args);return
        stages={'league':fixed_role_league,'freeze':freeze_representatives,'final':frozen_final,
                'validate':validate_exports,'behavior':record_frozen_behavior,'plot-learning':plot_learning}
        if args.stage in stages:return stages[args.stage](config)
        raise ValueError('Fixed-role research never invokes character screening or legacy study phases')
    if args.stage=='feedback':feedback_training(config);return
    if args.stage=='low-exploration':paired_feedback_training(config,'low_exploration');return
    if args.stage=='interleaved-feedback':paired_feedback_training(config,'interleaved_feedback');return
    if args.stage=='freeze':freeze_representatives(config);return
    if args.stage=='final':frozen_final(config);return
    if args.stage=='validate':validate_exports(config);return
    if args.stage=='behavior':record_frozen_behavior(config);return
    if args.stage=='initial-baselines':initialization_baselines(config);return
    if args.stage=='monitor':monitor_development(config,args.monitor_stage);return
    if args.stage=='plot-screen':plot_screening(config);return
    if args.stage=='plot-learning':plot_learning(config);return
    if args.stage=='refine':refined_screening(config);return
    if args.stage in ('screen','discovery'):
        chars=screening(config)
        chars=refined_screening(config)
        if args.stage=='screen':return
    else:chars=read(REPORT/'summary.json')['shortlist']
    if args.stage in ('shortlist','discovery'):
        character=shortlist_training(config,chars)
        if args.stage=='shortlist':return
    else:character=args.character or read(REPORT/'summary.json').get('selected_character')
    if args.stage in ('compare','discovery'):comparison(config,character)


if __name__=='__main__':main()
