"""Research inference and native games, independent of course verification APIs.

Trusted local sources and tensors only. No submission or Arena acceptance claim.
Native observations, six-frame inputs and round outcomes match the course.
"""
from __future__ import annotations
import base64,copy,io,itertools,json,pickle,random,sys,tempfile,time,types,zipfile
from collections import deque
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts'
TARGET=ROOT/'models/counter_ryu_final/agent.pt'
_MODULE_IDS=itertools.count()
GAME='StreetFighterIISpecialChampionEdition-Genesis'
from experiments.league_adaptive.characters import CHARACTERS
COMMON=dict(ruleset='hyper',speed_stars=2,decision_frames=6,position_scale=1024.,max_round_frames=7200)


def same_state(a, b):
    """Direct runtime comparisons; observations and random state are not files."""
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def save_fight(env, actor, additional_actors=None):
    """Simulator, both visible histories, controller and all acting RNG state."""
    result=dict(env=env.save(), actor=copy.deepcopy(actor.get_state()), python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.random.get_rng_state().clone())
    if additional_actors:
        result['additional_actors']={name:copy.deepcopy(a.get_state()) for name,a in additional_actors.items()}
    return result


def restore_fight(env, actor, saved, additional_actors=None):
    observation = env.restore(saved['env'])
    actor.set_state(copy.deepcopy(saved['actor']))
    if set(saved.get('additional_actors',{}))!=set(additional_actors or {}):
        raise ValueError('Restore every saved additional policy history')
    for name,a in (additional_actors or {}).items():
        a.set_state(copy.deepcopy(saved['additional_actors'][name]))
    random.setstate(saved['python']); np.random.set_state(saved['numpy']); torch.random.set_rng_state(saved['torch'])
    restored, original = save_fight(env, actor, additional_actors), copy.deepcopy(saved)
    # NativeGame independently verifies restored RAM, RGB, readings and time.
    # Serialized engine bookkeeping need not be byte-identical after restore.
    restored['env']['game'].pop('state'); original['env']['game'].pop('state')
    if not same_state(original, restored):
        raise RuntimeError('Restored simulator/history/controller/random state differs')
    return observation


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    # Windows readers and indexers can briefly deny replacement. Keep the
    # previous complete document available, and bound retries to one second.
    for attempt in range(21):
        try:
            tmp.replace(path)
            return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32) or attempt==20:
                raise
            time.sleep(.05)


def setup():
    import cv2
    torch.set_num_threads(1);cv2.setNumThreads(1)


def source_module(source):
    name=f'counter_local_source_{next(_MODULE_IDS)}'
    module=types.ModuleType(name);module.__file__='<trusted local model>'
    sys.modules[name]=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def spaces(config):
    c=config['observation_config'];bound=np.float32(65535/1024)
    return gym.spaces.Dict(dict(image=gym.spaces.Box(0,255,(4,c['height'],c['width']),np.uint8),
        position=gym.spaces.Box(np.array([0,0,0,0,-bound,-bound],np.float32),np.full(6,bound,np.float32),dtype=np.float32)))


def encode_classes(v):
    if isinstance(v,type):return {'local_class':v.__name__}
    if isinstance(v,dict):return {k:encode_classes(x) for k,x in v.items()}
    if isinstance(v,tuple):return {'tuple_items':[encode_classes(x) for x in v]}
    if isinstance(v,list):return [encode_classes(x) for x in v]
    return v


def decode_classes(v,module):
    if isinstance(v,dict):
        if set(v)=={'local_class'}:return getattr(module,v['local_class'])
        if set(v)=={'tuple_items'}:return tuple(decode_classes(x,module) for x in v['tuple_items'])
        return {k:decode_classes(x,module) for k,x in v.items()}
    if isinstance(v,list):return [decode_classes(x,module) for x in v]
    return v


def legacy_policy(definition,observation_space):
    module=source_module(definition['model_source'])
    kwargs=decode_classes(definition['policy_kwargs'],module)
    from stable_baselines3.common import policies
    cls=getattr(module,definition['policy_class'],None) or getattr(policies,definition['policy_class'])
    with torch.random.fork_rng(devices=[]):
        policy=cls(observation_space,
            gym.spaces.Discrete(len(definition['actions'])),lambda _:1e-4,**kwargs)
    return policy


class Actor:
    def __init__(self,policy,definition):
        self.policy=policy;self.definition=definition;self.actions=definition['actions']
        self.observation_source=definition['observation_source']
        self.policy.set_training_mode(False)
    def predict(self,obs,deterministic=True):
        result=self.policy.predict(obs,deterministic=deterministic)
        if hasattr(self.policy,'last_route_probabilities'):
            self.last_neural_route_probabilities=self.policy.last_route_probabilities.cpu().numpy().copy()
            self.last_neural_proposals=self.policy.last_neural_proposals.cpu().numpy().copy()
        if hasattr(self.policy,'last_residual_probability'):
            self.last_residual_probability=self.policy.last_residual_probability.cpu().numpy().copy()
        return result
    def reset(self):
        if hasattr(self.policy,'reset_inference_state'):self.policy.reset_inference_state()
    def get_state(self):return self.policy.get_inference_state() if hasattr(self.policy,'get_inference_state') else None
    def set_state(self,state):
        if hasattr(self.policy,'set_inference_state'):self.policy.set_inference_state(state)
    def close(self):pass


def load_legacy(path):
    """Read only inference members of our existing trusted course exports."""
    with zipfile.ZipFile(path) as archive:
        spec=json.loads(archive.read('spec.json'))
        source=archive.read('model.py').decode()
        observation_source=archive.read('observation.py').decode()
        packed=archive.read('policy.zip')
    module=source_module(source)
    class LocalUnpickler(pickle.Unpickler):
        def find_class(self,module_name,name):
            if module_name.startswith('sf2_custom_model_') and hasattr(module,name):return getattr(module,name)
            return super().find_class(module_name,name)
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        data=json.loads(archive.read('data'))
        def item(name):
            value=data[name]
            return LocalUnpickler(io.BytesIO(base64.b64decode(value[':serialized:']))).load() if isinstance(value,dict) and ':serialized:' in value else value
        kwargs=item('policy_kwargs');cls=item('policy_class')
        definition=dict(kind='legacy',policy_class=cls.__name__,policy_kwargs=encode_classes(kwargs),
            model_source=source,actions=spec['actions'],observation_source=observation_source,
            observation_config=spec['observation']['config'])
        policy=legacy_policy(definition,spaces(definition))
        state=torch.load(io.BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
        policy.load_state_dict(state,strict=True)
    for p in policy.parameters():p.requires_grad_(False)
    policy.optimizer=None
    return Actor(policy,definition)


def load_actor(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    if Path(path).suffix=='.zip':
        with zipfile.ZipFile(path) as archive:
            legacy='spec.json' in archive.namelist()
            definition=(json.loads(archive.read('counter_ryu_definition.json'))
                if 'counter_ryu_definition.json' in archive.namelist() else None)
        if legacy:return load_legacy(path)
        from stable_baselines3 import PPO
        definition_path=Path(path).parent/'policy_definition.json'
        if definition is None:definition=json.loads(definition_path.read_text())
        elif definition_path.exists() and definition!=json.loads(definition_path.read_text()):
            raise ValueError('Embedded and companion policy definitions disagree')
        model=PPO.load(path,device='cpu')
        if model.action_space.n!=len(definition['actions']) or model.observation_space!=spaces(definition):
            raise ValueError('Portable policy and local observation/action definition disagree')
        for p in model.policy.parameters():p.requires_grad_(False)
        model.policy.optimizer=None
        return Actor(model.policy,definition)
    from experiments.league_adaptive.policy import construct_policy
    data=torch.load(path,map_location='cpu',weights_only=False)
    policy=construct_policy(data['definition']);policy.load_state_dict(data['state_dict'],strict=True)
    for p in policy.parameters():p.requires_grad_(False)
    policy.optimizer=None
    return Actor(policy,data['definition'])


class Side:
    def __init__(self,definition,player):
        self.player=player;self.module=source_module(definition['observation_source']);self.frames=deque(maxlen=4)
    def reset(self,rgb):
        frame=self.module.preprocess(rgb);self.frames=deque([frame.copy() for _ in range(4)],maxlen=4)
    def update(self,rgb):self.frames.append(self.module.preprocess(rgb).copy())
    def observe(self,raw):
        p=self.player;q=3-p;x,y,ox,oy=(raw[f'p{p}_x'],raw[f'p{p}_y'],raw[f'p{q}_x'],raw[f'p{q}_y'])
        position=np.asarray([x,y,ox,oy,ox-x,oy-y],np.float32)/1024.
        return dict(image=np.concatenate(tuple(self.frames),axis=0),position=position)


class NativeGame:
    def __init__(self):
        self.engine=None;self.starts={};self.raw={};self.elapsed=0;self.key=None;self.on_frame=None
        self.setup_frames=0;self.held=((),())
        for path in (ROOT/'assets/starts').glob('*.npz'):
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(data['metadata'].tobytes().decode())
                key=(meta['mode'],int(meta['stars']),meta.get('p1_character','Ryu'),meta['character'])
                if key in self.starts:raise ValueError('Duplicate native start')
                self.starts[key]=dict(state=data['state'].tobytes(),rgb=data['rgb'].copy(),ram=data['ram'].copy(),
                    readings=meta['readings'],definitions=meta['definitions'],fps=meta['fps'],buttons=meta['buttons'])
        if not all((mode,star,'Ryu','Ryu') in self.starts for mode,star in [('pvp',1),*[('pve',s) for s in range(1,9)]]):
            raise RuntimeError('Required Ryu starts are missing from assets/starts')
    def _create(self):
        import stable_retro as retro
        game_id=None
        for candidate in (GAME,GAME+'-v0'):
            try:
                if Path(retro.data.get_romfile_path(candidate)).is_file():game_id=candidate;break
            except (FileNotFoundError,RuntimeError):pass
        if game_id is None:raise FileNotFoundError('Previously installed game is unavailable')
        self.work=tempfile.TemporaryDirectory(prefix='counter_native_')
        scenario=Path(self.work.name)/'scenario.json';scenario.write_text('{}')
        # The installed make/RetroEnv path reads the existing ROM directly;
        # data.groom_rom and data.verify_hash are separate import/check utilities.
        # A pre-call guard stops execution if a library unexpectedly enters them.
        old_profile=sys.getprofile()
        def prohibit_identity_calculations(frame,event,arg):
            if event=='call' and frame.f_code.co_name in ('groom_rom','verify_hash','get_known_hashes'):
                raise RuntimeError('Unexpected identity-calculation path; stopped before entering it')
            if event=='c_call' and getattr(arg,'__module__','') in ('_hashlib','_sha1','_sha256','_sha512','_md5','_blake2','_sha3'):
                raise RuntimeError('Unexpected digest constructor; stopped before calling it')
        try:
            sys.setprofile(prohibit_identity_calculations)
            self.engine=retro.make(game=game_id,state=retro.State.NONE,players=2,use_restricted_actions=retro.Actions.ALL,
                scenario=str(scenario),render_mode='rgb_array')
            self.engine.reset()
        finally:sys.setprofile(old_profile)
        self.buttons=list(self.engine.buttons);self.fps=float(self.engine.em.get_screen_rate())
        self.cold_state=bytes(self.engine.em.get_state())
    def _physical_restore(self,saved):
        engine=self.engine;engine.em.set_state(saved['state'])
        self.held=tuple(tuple(k) for k in saved.get('held',((),())))
        for player in range(2):
            mask=np.zeros(engine.num_buttons,np.uint8)
            for name in self.held[player]:mask[self.buttons.index(name)]=1
            engine.em.set_button_mask(mask,player)
        engine.data.reset()
        for name,spec in self.definitions.items():engine.data.set_variable('boot_'+name,dict(spec))
        engine.data.update_ram()
        if not np.array_equal(np.asarray(engine.get_ram(),np.uint8),saved['ram']):raise RuntimeError('Restored RAM differs')
        self.raw=self.read()
        if self.raw!=saved['readings']:raise RuntimeError('Restored native readings differ')
        self.rgb=saved['rgb'].copy();engine.img=self.rgb.copy();engine.ram=None
    def open(self,mode='pvp',stars=1,character='Ryu',p1_character='Ryu'):
        if self.engine is None:self._create()
        # Legacy callers specify the P2 character. New callers explicitly name
        # both physical controller identities, independently of screen side.
        key=(mode,int(stars),p1_character,character)
        if mode not in ('pvp','pve') or not 1<=int(stars)<=8 or any(c not in CHARACTERS for c in key[2:]):
            raise ValueError('Invalid native match configuration')
        if key in self.starts:saved=self.starts[key]
        else:
            from experiments.league_adaptive.native_setup import NativeSetup
            self.definitions=self.starts[('pvp',1,'Ryu','Ryu')]['definitions']
            saved=NativeSetup(self,OUT/'native_starts').start(key)
        self.definitions=saved['definitions']
        if self.buttons!=saved['buttons'] or abs(self.fps-saved['fps'])>1e-6:raise RuntimeError('Native controller/fps mismatch')
        self._physical_restore(saved);self.key=key;self.elapsed=0;self.on_frame=None
        self.wins=(self.raw['p1_wins'],self.raw['p2_wins'])
        self._validate()
    def _validate(self):
        if any(self.raw[f'p{p}_char']!=CHARACTERS.index(self.key[p+1]) for p in (1,2)):
            raise RuntimeError('Native character identity mismatch')
        if self.key[0]=='pve' and any(self.raw[k]!=self.key[1]-1 for k in ('difficulty_menu','difficulty_fight')):
            raise RuntimeError('Native CPU difficulty mismatch')
    def read(self):return {name:int(self.engine.data.lookup_value('boot_'+name)) for name in self.definitions}
    def step_keys(self,k1=(),k2=()):
        if self.key[0]=='pve' and k2:raise RuntimeError('Cannot control native CPU')
        action=np.zeros(2*len(self.buttons),np.int8)
        for port,keys in enumerate((k1,k2)):
            for name in keys:action[port*len(self.buttons)+self.buttons.index(name)]=1
        self.held=(tuple(k1),tuple(k2))
        before=self.raw;self.rgb,*_=self.engine.step(action);self.raw=self.read();self.elapsed+=1;self._validate()
        h1,h2=self.raw['p1_hp'],self.raw['p2_hp'];winner=reason=None
        if min(h1,h2)<0:winner=('draw' if h1<0 and h2<0 else 'p2' if h1<0 else 'p1');reason='signed_hp_ko'
        elif any(self.raw[f'p{p}_wins']>self.wins[p-1] for p in (1,2)):
            a,b=[self.raw[f'p{p}_wins']>self.wins[p-1] for p in (1,2)];winner='draw' if a and b else 'p1' if a else 'p2';reason='round_counter'
        elif self.raw['timer_bcd']==0:winner='p1' if h1>h2 else 'p2' if h2>h1 else 'draw';reason='native_time_up'
        elif self.elapsed>=7200:reason='frame_cap'
        if reason is None and any(self.raw[f'p{p}_hp']>before[f'p{p}_hp'] for p in (1,2)):raise RuntimeError('Unexpected HP refill')
        if self.on_frame is not None:self.on_frame(self.rgb,self.raw,k1,k2,self.elapsed)
        return self.raw,winner,reason
    def snapshot(self):
        return dict(state=bytes(self.engine.em.get_state()),ram=np.asarray(self.engine.get_ram(),np.uint8).copy(),
            rgb=self.rgb.copy(),readings=dict(self.raw),elapsed=self.elapsed,key=self.key,wins=self.wins,held=self.held)
    def restore(self,saved):
        if self.key!=saved['key']:
            mode,stars,p1,p2=saved['key'];self.open(mode,stars,p2,p1)
        self._physical_restore(saved);self.elapsed=saved['elapsed'];self.wins=saved['wins']
    def close(self):
        if self.engine is not None:self.engine.close();self.engine=None;self.work.cleanup()


def compiled(actions):
    result=[]
    allowed={'LEFT','RIGHT','UP','DOWN','A','B','C','X','Y','Z'}
    for action in actions:
        sequence=[]
        for part in action['segments']:
            if not set(part['keys'])<=allowed:raise ValueError('Unsupported control key')
            sequence.extend([tuple(part['keys'])]*int(part['frames']))
        if len(sequence)!=6:raise ValueError('Only six-native-frame inputs are supported')
        result.append(tuple(sequence))
    return result


def summary(rows):
    n=len(rows)
    return dict(episodes=n,wins=sum(r['result']=='win' for r in rows),losses=sum(r['result']=='loss' for r in rows),
        draws=sum(r['result']=='draw' for r in rows),cutoffs=sum(r['result']=='cutoff' for r in rows),
        win_rate=sum(r['result']=='win' for r in rows)/n if n else None,
        hp_margin=float(np.mean([max(0,r['self_hp'])-max(0,r['opponent_hp']) for r in rows])) if n else None)
