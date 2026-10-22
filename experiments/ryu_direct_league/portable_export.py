"""Export raw SB3 weights and self-contained student sources, without course APIs."""
import argparse,ast,json,subprocess,sys,tempfile,types
from pathlib import Path
import cloudpickle,gymnasium as gym,numpy as np,torch
from stable_baselines3 import PPO
from experiments.league_adaptive.runtime import ROOT,load_actor,setup,spaces,write


def inference_support(out):
    import importlib.metadata,platform
    (out/'inference.py').write_text('''"""Load Counter-Ryu inference weights through the course-compatible SB3 loader."""
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
    source=source_for(definition)+'\nimport copy\nfrom collections import deque\nfrom torch.nn import functional as F\nimport gymnasium as gym\n'
    class RemoveProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module and node.module.startswith('experiments.') else node
    nodes=[]
    selections=[('experiments/league_adaptive/runtime.py',{'spaces'}),
        ('experiments/ryu_adaptive_curriculum/history_protocol.py',{'continuity_reset'}),
        ('experiments/ryu_adaptive_curriculum/recurrent_direct.py',{'HealthReader'}),
        ('experiments/ryu_counterq/policy.py',{'make_extras','ContextQPolicy','counterq_kwargs','construct_counterq'})]
    for relative,names in selections:
        tree=ast.parse((ROOT/relative).read_text())
        nodes.extend(RemoveProjectImports().visit(n) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    source+='\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))
    source+='\nPOLICY_DEFINITION = '+repr(definition)+'\n'
    source+='''
def build_model(env,config,seed,device='auto'):
    raise RuntimeError('Train this value policy with the counterfactual DoubleDQN trainer. PPO.load is the inference container interface.')
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
    p=argparse.ArgumentParser();p.add_argument('--agent',required=True);p.add_argument('--out',required=True);a=p.parse_args();setup()
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError(out)
    actor=load_actor(a.agent);d=actor.definition
    if d['kind'] not in ('direct_ppo','neural_mixture','memory_ppo','recurrent_direct_ppo','context_double_dqn','counterfactual_option_planner'):raise ValueError('Unsupported learned policy architecture')
    recurrent=d['kind'] in ('memory_ppo','recurrent_direct_ppo','context_double_dqn','counterfactual_option_planner')
    out.mkdir(parents=True);inference_support(out);source=option_planner_source(d) if d['kind']=='counterfactual_option_planner' else counterq_source(d) if d['kind']=='context_double_dqn' else recurrent_direct_source(d) if d['kind']=='recurrent_direct_ppo' else memory_source(d) if d['kind']=='memory_ppo' else mixture_source(d) if d['kind']=='neural_mixture' else source_for(d);(out/'student_model.py').write_text(source)
    (out/'student_actions.py').write_text('CUSTOM_ACTIONS = '+repr(d['actions'][15:])+'\n')
    (out/'student_observation.py').write_text(d['observation_source'])
    write(out/'policy_definition.json',d)
    module=types.ModuleType('counter_ryu_portable_model');module.__file__=str(out/'student_model.py')
    sys.modules[module.__name__]=module;exec(compile(source,module.__file__,'exec'),module.__dict__)
    class SpaceOnly(gym.Env):
        observation_space=spaces(d);action_space=gym.spaces.Discrete(len(d['actions']))
        def reset(self,*,seed=None,options=None):raise RuntimeError('Serialization only')
        def step(self,action):raise RuntimeError('Serialization only')
    cls,kwargs=(module.OptionPlannerPolicy,module.planner_kwargs(d)) if d['kind']=='counterfactual_option_planner' else (module.ContextQPolicy,module.counterq_kwargs(d)) if d['kind']=='context_double_dqn' else (module.RecurrentDirectPolicy,module.recurrent_kwargs(d)) if d['kind']=='recurrent_direct_ppo' else (module.MemoryPolicy,module.memory_kwargs(d)) if d['kind']=='memory_ppo' else (module.NeuralMixturePolicy,module.mixture_kwargs(d)) if d['kind']=='neural_mixture' else (module.DirectPolicy,module.policy_kwargs(d))
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
with np.load(p/'reload_inputs.npz',allow_pickle=False) as data:o={k:data[k].copy() for k in ('image','position')}
action=m.predict(o,deterministic=True)[0]
with torch.no_grad():
 if hasattr(m.policy,'last_memory_probabilities'):
  prob=m.policy.last_memory_probabilities.cpu().numpy();value=m.policy.last_memory_values.cpu().numpy()
 else:
  t,_=m.policy.obs_to_tensor(o);prob=m.policy.get_distribution(t).distribution.probs.cpu().numpy();value=m.policy.predict_values(t).cpu().numpy()
np.savez_compressed(p/'reload_predictions.npz',action=action,probability=prob,value=value)
if (p/'reload_sequence_inputs.npz').exists():
 m.policy.reset_inference_state();actions=[];probabilities=[];values=[]
 with np.load(p/'reload_sequence_inputs.npz',allow_pickle=False) as seq:
  for i in range(len(seq['image'])):
   actions.append(m.predict({k:seq[k][i] for k in ('image','position')},deterministic=True)[0])
   probabilities.append(m.policy.last_memory_probabilities.cpu().numpy()[0]);values.append(m.policy.last_memory_values.cpu().numpy()[0])
 np.savez_compressed(p/'reload_sequence_predictions.npz',action=actions,probability=probabilities,value=values)
print('Standalone SB3 reload completed in an isolated interpreter.')
'''
    subprocess.run([sys.executable,'-I','-s','-c',child,str(out)],cwd=tempfile.gettempdir(),check=True)
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
    if not actions_equal or probability_error>1e-6 or value_error>1e-6:raise RuntimeError('Portable predictions changed')
    sequence_result=None
    if recurrent:
        actor.reset();actions=[];probabilities=[];values=[]
        with np.load(out/'reload_sequence_inputs.npz',allow_pickle=False) as sequence:
            for i in range(len(sequence['image'])):
                actions.append(actor.predict({k:sequence[k][i] for k in ('image','position')},True)[0])
                probabilities.append(actor.policy.last_memory_probabilities.cpu().numpy()[0]);values.append(actor.policy.last_memory_values.cpu().numpy()[0])
        with np.load(out/'reload_sequence_predictions.npz',allow_pickle=False) as read:
            sequence_result=dict(decisions=len(actions),actions_equal=bool(np.array_equal(actions,read['action'])),
                max_probability_error=float(np.max(np.abs(probabilities-read['probability']))),max_value_error=float(np.max(np.abs(values-read['value']))))
        if not sequence_result['actions_equal'] or sequence_result['max_probability_error']>1e-6 or sequence_result['max_value_error']>1e-6:raise RuntimeError('Recurrent sequence changed after isolated reload')
    write(out/'reload_result.json',dict(source_agent=str(Path(a.agent).resolve()),observations=80,
        greedy_actions_equal=actions_equal,max_action_probability_error=probability_error,max_value_error=value_error,
        recurrent_sequence=sequence_result,
        scope='Raw PPO weights reload in a separate isolated interpreter with no project modules. This is not an official Arena submission package or an Arena acceptance test.'))
    print('PORTABLE_POLICY_RELOAD_MATCHED',out,flush=True)


if __name__=='__main__':main()
