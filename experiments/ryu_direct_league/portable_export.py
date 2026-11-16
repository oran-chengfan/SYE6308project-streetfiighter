"""Export raw SB3 weights and self-contained student sources, without course APIs."""
import argparse,ast,json,subprocess,sys,tempfile,types
from pathlib import Path
import cloudpickle,gymnasium as gym,numpy as np,torch
from stable_baselines3 import PPO
from experiments.league_adaptive.runtime import ROOT,load_actor,setup,spaces,write


def inference_support(out):
    import importlib.metadata,platform
    (out/'inference.py').write_text('''"""Load standalone SB3 inference; this does not certify course Arena compatibility."""
from pathlib import Path
from stable_baselines3 import PPO

def load_model(device='cpu'):
    model=PPO.load(Path(__file__).resolve().parent/'policy.zip',device=device)
    model.policy.set_training_mode(False)
    return model
''')
    versions={name:importlib.metadata.version(name) for name in ('stable-baselines3','torch','numpy','gymnasium','cloudpickle')}
    write(out/'runtime_versions.json',dict(python=platform.python_version(),packages=versions,
        observation='image uint8[4,84,96], position float32[6], self-first and divided by 1024',
        action='deterministic discrete input-template index; see policy_definition.json',
        scope='Versions used for local CPU reload. The emulator and game setup are separate from policy inference.'))


def inspect_course_contract(definition,out,notebook):
    """Read the real notebook; execute only pure action and coordinate code.

    Package/descriptor/ROM verification and installers are never imported or
    executed. This is local interface evidence, not an official acceptance.
    """
    document=json.loads(Path(notebook).read_text(encoding='utf-8'))
    core=next(''.join(c['source']) for c in document['cells']
        if ''.join(c['source']).startswith('# Public core:'))
    files=next(ast.literal_eval(n.value) for n in ast.parse(core).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CORE_FILES' for t in n.targets))
    action_tree=ast.parse(files['sf2_actions.py']);nodes=[]
    constants={'KEYS','QUANTUM'};functions={'validate_library','compiled'}
    for node in action_tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:nodes.append(node)
    import re
    namespace=dict(re=re,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<read-only course action functions>','exec'),namespace)
    commands=namespace['compiled'](definition['actions'])
    from experiments.league_adaptive.runtime import compiled,Side
    if commands!=tuple(compiled(definition['actions'])):raise RuntimeError('Course six-frame physical commands differ')
    observation_tree=ast.parse(files['sf2_observation.py'])
    position=next(n for n in observation_tree.body if isinstance(n,ast.FunctionDef) and n.name=='position')
    coordinates=dict(np=np)
    exec(compile(ast.Module(body=[position],type_ignores=[]),'<read-only course coordinates>','exec'),coordinates)
    course_side=next(n for n in observation_tree.body if isinstance(n,ast.ClassDef) and n.name=='Side')
    observe=next(n for n in course_side.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
    concatenations=[n for n in ast.walk(observe) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='concatenate']
    if len(concatenations)!=1 or ast.unparse(concatenations[0])!='np.concatenate(tuple(self.frames), axis=0)':
        raise ValueError('Re-review the changed course chronological image-stack contract')
    raw=dict(p1_x=310,p1_y=192,p2_x=140,p2_y=170);checks=[]
    rng=np.random.default_rng(107);images=[np.zeros((224,320,3),np.uint8)]
    for _ in range(8):
        rgb=np.zeros((224,320,3),np.uint8);rgb[:,:256]=rng.integers(0,256,(224,256,3),dtype=np.uint8);images.append(rgb)
    for player in (1,2):
        side=Side(definition,player);side.reset(np.zeros((224,320,3),np.uint8))
        observed=side.observe(raw);expected=coordinates['position'](raw,player,1024.)
        if not np.array_equal(observed['position'],expected):raise RuntimeError('Course own-first coordinates differ')
        if observed['image'].shape!=(4,84,96) or observed['image'].dtype!=np.uint8:raise RuntimeError('Course image contract differs')
        expected_frames=[side.module.preprocess(images[0]).copy() for _ in range(4)]
        for rgb in images[1:]:
            side.update(rgb);expected_frames=(expected_frames+[side.module.preprocess(rgb).copy()])[-4:]
            if not np.array_equal(side.observe(raw)['image'],np.concatenate(tuple(expected_frames),axis=0)):
                raise RuntimeError('Chronological preprocessing differs from the course image-stack rule')
        checks.append(dict(player=player,coordinates_equal=True,image_shape=list(observed['image'].shape),
            chronological_image_updates_checked=8,image_differences=0))
    bundle=ast.parse(files['sf2_bundle.py']);loader=next(n for n in bundle.body if isinstance(n,ast.ClassDef) and n.name=='LoadedAgent')
    model_calls=[n for n in ast.walk(loader) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='predict']
    if len(model_calls)!=1 or any(k.arg in ('state','episode_start') for k in model_calls[0].keywords):
        raise ValueError('Re-review the changed public recurrent calling convention')
    arena=ast.parse(files['sf2_arena_actor.py'])
    arena_calls=[n for n in ast.walk(arena) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='predict']
    if not arena_calls or any(any(k.arg in ('state','episode_start') for k in n.keywords)
            or not any(k.arg=='deterministic' and isinstance(k.value,ast.Constant)
                and k.value.value is True for k in n.keywords) for n in arena_calls):
        raise ValueError('Re-review the changed actual Arena prediction convention')
    result=dict(notebook=str(Path(notebook).relative_to(ROOT)),course_loader='sf2_bundle.LoadedAgent',
        loader_class_lines=[loader.lineno,loader.end_lineno],model_load='PPO.load(policy.zip)',
        prediction='model.predict(side.observe(raw), deterministic=deterministic); no external recurrent state or episode_start',
        action_count=len(definition['actions']),course_action_limit=64,all_commands_six_native_frames=all(len(x)==6 for x in commands),
        actual_course_action_expansion_equal=True,coordinate_checks=checks,
        arena_predict_calls=[dict(line=n.lineno,deterministic=True,external_recurrent_state=False) for n in arena_calls],
        image_check_scope='Actual exported student preprocessing and chronological four-frame arrays, with the course Side.observe concatenate expression inspected unchanged. The original descriptor loader was never executed.',
        student_source_slots=['model.py','observation.py','reward.py','spec.json','common.json','training.json','policy.zip'],
        submission_boundary='The notebook requires a complete v3 inner agent and an outer submission container with mandatory verification records. A raw policy.zip is insufficient. Those prohibited verification paths were not executed and no records were fabricated.',
        executed_course_nodes=['sf2_actions.validate_library','sf2_actions.compiled','sf2_observation.position'],
        official_arena_executed=False,scope='Pure-code and SB3 runtime compatibility only; full original package validation remains unavailable under the no-file-verification constraint.')
    write(Path(out)/'course_interface.json',result)
    return result


def source_for(definition):
    code=ROOT/'experiments/ryu_direct_league/policy.py'
    tree=ast.parse(code.read_text());names={'DirectEncoder','MotionEncoder','DirectPolicy','policy_kwargs'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    other=ast.parse((ROOT/'experiments/league_adaptive/policy.py').read_text())
    mirror=next(n for n in other.body if isinstance(n,ast.FunctionDef) and n.name=='mirror_indices')
    imports='''import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
'''
    source=imports+'\n\n'+ast.unparse(ast.Module(body=[mirror]+nodes,type_ignores=[]))
    source+='\n\nPOLICY_DEFINITION = '+repr({k:definition[k] for k in ('actions','canonical','encoder') if k in definition})+'\n'
    source+='''
def build_model(env, config, seed, device='auto'):
    return PPO(DirectPolicy, env, policy_kwargs=policy_kwargs(POLICY_DEFINITION),
        learning_rate=float(config['learning_rate']), n_steps=int(config['n_steps']),
        batch_size=int(config['batch_size']), n_epochs=int(config['n_epochs']),
        gamma=float(config['gamma']), gae_lambda=float(config.get('gae_lambda', .95)),
        clip_range=float(config['clip_range']), ent_coef=float(config.get('ent_coef', .005)),
        target_kl=config.get('target_kl', .03), seed=int(seed), device=device)
'''
    return source


def verify_instance_isolation(out):
    """Interleave two independent loaded actors against a third clean control."""
    child='''
import copy,json,sys
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
torch.set_num_threads(1)
p=Path(sys.argv[1]);actors=[PPO.load(p/'policy.zip',device='cpu') for _ in range(3)]
for actor in actors:actor.policy.set_training_mode(False);actor.policy.reset_inference_state()
assert len({next(a.policy.parameters()).data_ptr() for a in actors})==3
def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 return a==b
with np.load(p/'reload_sequence_inputs.npz',allow_pickle=False) as seq:
 for i in range(len(seq['image'])):
  obs={k:seq[k][i].copy() for k in ('image','position')}
  a0=actors[0].predict(obs,deterministic=True)[0]
  state=copy.deepcopy(actors[0].policy.get_inference_state())
  other=dict(image=obs['image'],position=obs['position'][[2,3,0,1,4,5]].copy())
  other['position'][4:]*=-1
  actors[2].predict(other,deterministic=True)
  assert same(state,actors[0].policy.get_inference_state()),'Another agent changed live history'
  a1=actors[1].predict(obs,deterministic=True)[0]
  assert np.array_equal(a0,a1) and same(state,actors[1].policy.get_inference_state())
  assert torch.equal(actors[0].policy.last_memory_probabilities,actors[1].policy.last_memory_probabilities)
result=dict(independent_agents=2,separate_control=1,continuous_decisions=i+1,
 separate_parameter_storage=True,history_pollution=0,action_probability_and_memory_differences=0,
 external_recurrent_state_passed=False,native_frames=0,official_arena=False)
(p/'instance_isolation.json').write_text(json.dumps(result,indent=2)+'\\n')
print('INDEPENDENT_INSTANCES_MATCHED',result['continuous_decisions'])
'''
    subprocess.run([sys.executable,'-I','-s','-c',child,str(out)],cwd=tempfile.gettempdir(),check=True)
    return json.loads((Path(out)/'instance_isolation.json').read_text())


def mixture_source(definition):
    imports='''import copy,itertools,sys,types
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
_MODULE_IDS=itertools.count()
'''
    selections=[('experiments/league_adaptive/runtime.py',{'source_module','spaces','decode_classes','legacy_policy'}),
        ('experiments/league_adaptive/policy.py',{'mirror_indices','VisibleHistoryEncoder','ResponsePolicy'}),
        ('experiments/ryu_direct_league/policy.py',{'DirectEncoder','MotionEncoder','DirectPolicy','policy_kwargs','construct_direct'}),
        ('experiments/ryu_direct_league/neural_mixture.py',{'NeuralMixturePolicy','mixture_kwargs','construct_mixture'})]
    nodes=[]
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    for relative,names in selections:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    source=imports+'\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='''
def construct_policy(definition):
    kind=definition['kind']
    if kind=='neural_mixture':return construct_mixture(definition)
    if kind=='direct_ppo':return construct_direct(definition)
    if kind=='legacy':return legacy_policy(definition,spaces(definition))
    if kind=='response' and definition.get('reference_definition') is None:
        return ResponsePolicy(spaces(definition),gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,
            features_extractor_class=VisibleHistoryEncoder,net_arch=dict(pi=[128],vf=[128]),
            action_mirror=mirror_indices(definition['actions']),reference_definition=None,
            prior_strength=definition.get('prior_strength',.5))
    raise ValueError('Unsupported neural constituent')
'''
    source+='\nPOLICY_DEFINITION = '+repr(definition)+'\n'
    source+='''
def build_model(env,config,seed,device='auto'):
    return PPO(NeuralMixturePolicy,env,policy_kwargs=mixture_kwargs(POLICY_DEFINITION),
        learning_rate=float(config['learning_rate']),n_steps=int(config['n_steps']),
        batch_size=int(config['batch_size']),n_epochs=int(config['n_epochs']),gamma=float(config['gamma']),
        gae_lambda=float(config.get('gae_lambda',.95)),clip_range=float(config['clip_range']),
        ent_coef=float(config.get('ent_coef',.005)),target_kl=config.get('target_kl',.03),seed=int(seed),device=device)
'''
    return source


def memory_source(definition):
    source=mixture_source(definition)
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    nodes=[]
    for relative,names in [('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
        ('experiments/ryu_adaptive_curriculum/memory_policy.py',{'MemoryPolicy','memory_kwargs','construct_memory'})]:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    source+='\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='''
_construct_feedforward_policy=construct_policy
def construct_policy(definition):
    if definition['kind']=='memory_ppo':return construct_memory(definition)
    return _construct_feedforward_policy(definition)

def build_model(env,config,seed,device='auto'):
    raise RuntimeError('This policy uses contiguous-sequence PPO training. Load policy.zip for inference; use the supplied research trainer to train memory.')
'''
    return source


def recurrent_direct_source(definition):
    source=source_for(definition)+'\nfrom torch.nn import functional as F\n'
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    nodes=[]
    for relative,names in [('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
        ('experiments/ryu_adaptive_curriculum/recurrent_direct.py',{'action_groups','HealthReader','RecurrentDirectPolicy','recurrent_kwargs','construct_recurrent'})]:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    # spaces is a pure Gym-space constructor; no emulator dependency is needed.
    runtime_tree=ast.parse((ROOT/'experiments/league_adaptive/runtime.py').read_text())
    nodes.insert(0,next(n for n in runtime_tree.body if isinstance(n,ast.FunctionDef) and n.name=='spaces'))
    source+='\nimport gymnasium as gym\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='\nPOLICY_DEFINITION = '+repr(definition)+'\n'
    source+='''
def build_model(env,config,seed,device='auto'):
    raise RuntimeError('Use contiguous-sequence PPO to train this recurrent policy. Ordinary PPO.load supports inference.')
'''
    return source


def counterq_source(definition):
    sequence=definition['kind'] in ('sequence_double_q','sequence_ppo')
    source=('''import copy
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
''' if sequence else source_for(definition))+'\nimport copy\nfrom collections import deque\nfrom torch.nn import functional as F\nimport gymnasium as gym\n'
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    nodes=[]
    selections=[('experiments/league_adaptive/runtime.py',{'spaces'}),
        ('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
        ('experiments/ryu_adaptive_curriculum/recurrent_direct.py',{'HealthReader'}),
        ('experiments/ryu_counterq/policy.py',{'make_extras','ContextQPolicy','counterq_kwargs','construct_counterq'})]
    if sequence:
        selections=[('experiments/league_adaptive/runtime.py',{'spaces','compiled'}),
            ('experiments/league_adaptive/characters.py',{'advance_input_durations'}),
            ('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
            ('experiments/league_adaptive/policy.py',{'VisibleHistoryEncoder','mirror_indices'}),
            ('experiments/ryu_counterq/policy.py',{'SequenceQPolicy','SequencePPOPolicy','sequence_q_kwargs','construct_sequence_q'})]
    for relative,names in selections:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    source+='\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='\nPOLICY_DEFINITION = '+repr(definition)+'\n'
    source+='''
def build_model(env,config,seed,device='auto'):
    raise RuntimeError('Use the registered sequence trainer for this policy. The SB3 container does not identify its learning algorithm.')
'''
    return source


def option_planner_source(definition):
    from experiments.ryu_option_planner.options import PLANS
    source=mixture_source(definition)+'\nfrom collections import deque\nPLANS = '+repr(PLANS)+'\n'
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    nodes=[]
    selections=[('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
        ('experiments/ryu_adaptive_curriculum/recurrent_direct.py',{'HealthReader'}),
        ('experiments/ryu_counterq/policy.py',{'make_extras','ContextQPolicy','counterq_kwargs'}),
        ('experiments/ryu_option_planner/options.py',{'plan_input'}),
        ('experiments/ryu_option_planner/policy.py',{'OptionPlannerPolicy','planner_kwargs','construct_planner'})]
    for relative,names in selections:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    source+='\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='\nPOLICY_DEFINITION = '+repr(definition)+'\n'
    source+='''
def build_model(env,config,seed,device='auto'):
    raise RuntimeError('Train the outcome planner using its paired native-root trainer. PPO.load is the inference interface.')
'''
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--agent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--action-indices',help='Explicit external catalog reduction for the course action limit; JSON original-index list. Raw source remains unchanged.')
    p.add_argument('--course-notebook',help='Read-only interface check; never executes notebook setup, packaging, descriptor checks or Arena.')
    a=p.parse_args();setup()
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(out)
    actor=load_actor(a.agent);conversion=None
    if a.action_indices:
        from experiments.ryu_counterq.policy import restrict_sequence_actions
        from experiments.league_adaptive.runtime import Actor
        selected=json.loads(Path(a.action_indices).read_text(encoding='utf-8'))
        policy,definition,conversion=restrict_sequence_actions(
            dict(definition=actor.definition,state_dict=actor.policy.state_dict()),selected)
        actor.close();actor=Actor(policy,definition)
    d=actor.definition
    if d['kind'] not in ('direct_ppo','neural_mixture','memory_ppo','recurrent_direct_ppo','context_double_dqn','sequence_double_q','sequence_ppo','counterfactual_option_planner'):raise ValueError('Unsupported learned policy architecture')
    recurrent=d['kind'] in ('memory_ppo','recurrent_direct_ppo','context_double_dqn','sequence_double_q','sequence_ppo','counterfactual_option_planner')
    out.mkdir(parents=True)
    if conversion:write(out/'catalog_conversion.json',dict(source_agent=a.agent,**conversion))
    if a.course_notebook:inspect_course_contract(d,out,ROOT/a.course_notebook)
    inference_support(out);source=option_planner_source(d) if d['kind']=='counterfactual_option_planner' else counterq_source(d) if d['kind'] in ('context_double_dqn','sequence_double_q','sequence_ppo') else recurrent_direct_source(d) if d['kind']=='recurrent_direct_ppo' else memory_source(d) if d['kind']=='memory_ppo' else mixture_source(d) if d['kind']=='neural_mixture' else source_for(d);(out/'student_model.py').write_text(source)
    (out/'student_actions.py').write_text('CUSTOM_ACTIONS = '+repr(d['actions'][15:])+'\n')
    (out/'student_observation.py').write_text(d['observation_source'])
    write(out/'policy_definition.json',d)
    module=types.ModuleType('counter_ryu_portable_model');module.__file__=str(out/'student_model.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    class SpaceOnly(gym.Env):
        observation_space=spaces(d);action_space=gym.spaces.Discrete(len(d['actions']))
        def reset(self,*,seed=None,options=None):raise RuntimeError('Serialization only')
        def step(self,action):raise RuntimeError('Serialization only')
    cls,kwargs=(module.SequencePPOPolicy,module.sequence_q_kwargs(d)) if d['kind']=='sequence_ppo' else (module.SequenceQPolicy,module.sequence_q_kwargs(d)) if d['kind']=='sequence_double_q' else (module.OptionPlannerPolicy,module.planner_kwargs(d)) if d['kind']=='counterfactual_option_planner' else (module.ContextQPolicy,module.counterq_kwargs(d)) if d['kind']=='context_double_dqn' else (module.RecurrentDirectPolicy,module.recurrent_kwargs(d)) if d['kind']=='recurrent_direct_ppo' else (module.MemoryPolicy,module.memory_kwargs(d)) if d['kind']=='memory_ppo' else (module.NeuralMixturePolicy,module.mixture_kwargs(d)) if d['kind']=='neural_mixture' else (module.DirectPolicy,module.policy_kwargs(d))
    model=PPO(cls,SpaceOnly(),policy_kwargs=kwargs,device='cpu',n_steps=512,batch_size=512,
        gamma=1.,gae_lambda=.95,n_epochs=4,learning_rate=1e-4,clip_range=.1,ent_coef=.005,target_kl=.03)
    model.policy.load_state_dict(actor.policy.state_dict(),strict=True)
    cloudpickle.register_pickle_by_value(module)
    try:model.save(str(out/'policy.zip'))
    finally:cloudpickle.unregister_pickle_by_value(module)
    with np.load(ROOT/'models/counter_ryu_final/reload_inputs.npz',allow_pickle=False) as data:
        ids=np.linspace(0,len(data['image'])-1,80,dtype=int)
        observations={k:data[k][ids].copy() for k in ('image','position')}
    np.savez_compressed(out/'reload_inputs.npz',**observations)
    if recurrent:
        with np.load(ROOT/'assets/validation/sequence_inputs.npz',allow_pickle=False) as sequence:
            np.savez_compressed(out/'reload_sequence_inputs.npz',image=sequence['image'][:96],position=sequence['position'][:96])
    child='''import sys,numpy as np,torch
from stable_baselines3 import PPO
from pathlib import Path
torch.set_num_threads(1)
p=Path(sys.argv[1]);m=PPO.load(p/'policy.zip',device='cpu');m.policy.set_training_mode(False)
sequence_q=sys.argv[2]=='sequence_double_q'
with np.load(p/'reload_inputs.npz',allow_pickle=False) as data:o={k:data[k].copy() for k in ('image','position')}
with torch.inference_mode():action=m.predict(o,deterministic=True)[0]
with torch.no_grad():
 if hasattr(m.policy,'last_memory_probabilities'):
  prob=m.policy.last_memory_probabilities.cpu().numpy();value=m.policy.last_memory_values.cpu().numpy()
 else:
  t,_=m.policy.obs_to_tensor(o);prob=m.policy.get_distribution(t).distribution.probs.cpu().numpy();value=m.policy.predict_values(t).cpu().numpy()
q=m.policy.last_context_q_values.cpu().numpy() if sequence_q else np.empty((0,),np.float32)
np.savez_compressed(p/'reload_predictions.npz',action=action,probability=prob,value=value,q=q)
if (p/'reload_sequence_inputs.npz').exists():
 m.policy.reset_inference_state();actions=[];probabilities=[];values=[];q_vectors=[]
 with np.load(p/'reload_sequence_inputs.npz',allow_pickle=False) as seq:
  for i in range(len(seq['image'])):
   with torch.inference_mode():action=m.predict({k:seq[k][i] for k in ('image','position')},deterministic=True)[0]
   actions.append(action)
   probabilities.append(m.policy.last_memory_probabilities.cpu().numpy()[0]);values.append(m.policy.last_memory_values.cpu().numpy()[0])
   if sequence_q:q_vectors.append(m.policy.last_context_q_values.cpu().numpy()[0])
 np.savez_compressed(p/'reload_sequence_predictions.npz',action=actions,probability=probabilities,value=values,q=q_vectors)
print('Standalone SB3 reload completed in an isolated interpreter.')
'''
    subprocess.run([sys.executable,'-I','-s','-c',child,str(out),d['kind']],cwd=tempfile.gettempdir(),check=True)
    source_actions=actor.predict(observations,True)[0]
    with torch.no_grad():
        if recurrent:
            probability=actor.policy.last_memory_probabilities.cpu().numpy();value=actor.policy.last_memory_values.cpu().numpy()
        else:
            tensor,_=actor.policy.obs_to_tensor(observations)
            probability=actor.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
            value=actor.policy.predict_values(tensor).cpu().numpy()
    with np.load(out/'reload_predictions.npz',allow_pickle=False) as read:
        actions_equal=bool(np.array_equal(source_actions,read['action']))
        probability_error=float(np.max(np.abs(probability-read['probability'])))
        value_error=float(np.max(np.abs(value-read['value'])))
        q_error=float(np.max(np.abs(actor.policy.last_context_q_values.cpu().numpy()-read['q']))) if d['kind']=='sequence_double_q' else None
    if not actions_equal or probability_error>1e-6 or value_error>1e-6:raise RuntimeError('Portable predictions changed')
    if q_error is not None and q_error>1e-6:raise RuntimeError('Portable Q vector changed')
    sequence_result=None
    if recurrent:
        actor.reset();actions=[];probabilities=[];values=[];q_vectors=[]
        with np.load(out/'reload_sequence_inputs.npz',allow_pickle=False) as sequence:
            for i in range(len(sequence['image'])):
                actions.append(actor.predict({k:sequence[k][i] for k in ('image','position')},True)[0])
                probabilities.append(actor.policy.last_memory_probabilities.cpu().numpy()[0]);values.append(actor.policy.last_memory_values.cpu().numpy()[0])
                if d['kind']=='sequence_double_q':q_vectors.append(actor.policy.last_context_q_values.cpu().numpy()[0])
        with np.load(out/'reload_sequence_predictions.npz',allow_pickle=False) as read:
            sequence_result=dict(decisions=len(actions),actions_equal=bool(np.array_equal(actions,read['action'])),
                max_probability_error=float(np.max(np.abs(probabilities-read['probability']))),max_value_error=float(np.max(np.abs(values-read['value']))))
            if d['kind']=='sequence_double_q':sequence_result['max_q_error']=float(np.max(np.abs(q_vectors-read['q'])))
        if not sequence_result['actions_equal'] or sequence_result['max_probability_error']>1e-6 or sequence_result['max_value_error']>1e-6:raise RuntimeError('Recurrent sequence changed after isolated reload')
        if sequence_result.get('max_q_error',0.)>1e-6:raise RuntimeError('Recurrent Q sequence changed after isolated reload')
    write(out/'reload_result.json',dict(source_agent=str(Path(a.agent).resolve()),observations=80,
        greedy_actions_equal=actions_equal,max_action_probability_error=probability_error,max_value_error=value_error,
        max_q_error=q_error,
        recurrent_sequence=sequence_result,
        score_semantics='For sequence Double Q: softmax(Q/.2) is a cloning-link identity score, not the acting distribution; deterministic deployment uses Q argmax.' if d['kind']=='sequence_double_q' else 'Policy distribution and value identity checks.',
        scope='Learned inference weights reload in a separate isolated interpreter with no project modules. Q policies use SB3 only as a serialization container. This is not an official Arena submission package or an Arena acceptance test.'))
    if a.course_notebook and recurrent:verify_instance_isolation(out)
    print('PORTABLE_POLICY_RELOAD_MATCHED',out,flush=True)


if __name__=='__main__':main()
