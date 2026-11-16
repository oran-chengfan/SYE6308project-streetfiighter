"""Native menu recognition and character setup.

Text masks and menu geometry adapted from the supplied course interface;
no course loader, scanner, cache or packaging function is executed.
"""

import base64
import numpy as np
from pathlib import Path
from experiments.league_adaptive.characters import CHARACTERS, GRID, CELLS


class NativeSetup:
    """Navigate real menus, cache only verified native first-round states."""
    def __init__(self, game, cache):
        self.game=game;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.phase='menu';self.stage=None

    def advance(self, n=1, p1=(), p2=()):
        g=self.game
        for _ in range(n):
            if self.stage is not None:g.engine.data.set_value('boot_stage_selector',self.stage)
            action=np.zeros(len(g.buttons)*2,np.int8)
            for port,keys in enumerate((p1,p2)):
                for key in keys:action[port*len(g.buttons)+g.buttons.index(key)]=1
            g.rgb,*_=g.engine.step(action);g.raw=g.read();g.setup_frames+=1
            g.held=(tuple(p1),tuple(p2))

    def tap(self, player, key, release=16):
        self.advance(8,(key,) if player==1 else (), (key,) if player==2 else ())
        self.advance(release)

    def wait_menu(self, kind, budget=2200):
        for index in range(budget):
            if menu_kind(self.game.rgb)==kind and cursor_row(self.game.rgb) is not None:return
            if kind=='rules' and index and index%240==0:self.tap(1,'START',60)
            else:self.advance()
        raise RuntimeError('Native menu not recognized: '+kind)

    def row(self, kind, wanted):
        self.wait_menu(kind)
        for _ in range(4):
            row=cursor_row(self.game.rgb)
            if row==wanted:return
            if row is None:raise RuntimeError('Ambiguous native menu cursor')
            self.tap(1,'DOWN' if row<wanted else 'UP',32)
        raise RuntimeError('Native menu row did not respond')

    def pack(self):
        g=self.game
        return dict(state=bytes(g.engine.em.get_state()),ram=np.asarray(g.engine.get_ram(),np.uint8).copy(),
                    rgb=g.rgb.copy(),readings=dict(g.raw),held=g.held,
                    definitions=g.definitions,buttons=g.buttons,fps=g.fps)

    def cache_write(self,path,data):
        import json
        files=list(self.cache.glob('*.npz'))
        if sum(p.stat().st_size for p in files)>268435456:raise RuntimeError('Native start cache capacity reached')
        metadata={k:v for k,v in data.items() if k not in ('state','ram','rgb')}
        np.savez_compressed(path,state=np.frombuffer(data['state'],np.uint8),ram=data['ram'],rgb=data['rgb'],
                            metadata=np.frombuffer(json.dumps(metadata).encode(),np.uint8))

    @staticmethod
    def cache_read(path):
        import json
        with np.load(path,allow_pickle=False) as data:
            result=json.loads(data['metadata'].tobytes().decode())
            result.update(state=data['state'].tobytes(),ram=data['ram'].copy(),rgb=data['rgb'].copy())
        return result

    def cursor(self, player):
        g=self.game;before=g.raw[f'p{player}_char']
        if before not in CELLS:return False
        row,col=CELLS[before];key='RIGHT' if col<5 else 'LEFT'
        self.tap(player,key);after=g.raw[f'p{player}_char']
        self.tap(player,'LEFT' if key=='RIGHT' else 'RIGHT')
        return after==GRID[row][col+(1 if key=='RIGHT' else -1)] and g.raw[f'p{player}_char']==before

    def menu(self,mode,stars):
        g=self.game;path=self.cache/f'menu_hyper_speed2_{mode}_{stars}.npz'
        if path.exists():g._physical_restore(self.cache_read(path));return
        # With State.NONE, RetroEnv.reset only advances the current machine;
        # it does not reboot it. Return to this instance's actual cold boot.
        g.engine.em.set_state(g.cold_state)
        g.rgb,*_=g.engine.reset()
        g.engine.data.reset()
        for name,spec in g.definitions.items():g.engine.data.set_variable('boot_'+name,dict(spec))
        g.raw=g.read();g.held=((),())
        self.advance(1740);self.tap(1,'START',128)
        self.tap(1,'DOWN');self.tap(1,'DOWN');self.tap(1,'START',240)
        for _ in range(360):
            if visible_options_stars(g.rgb) is not None:break
            self.advance()
        else:raise RuntimeError('Native OPTIONS not recognized')
        visible=visible_options_stars(g.rgb)
        if g.raw['difficulty_menu']!=visible-1:raise RuntimeError('Visible difficulty/readback mismatch')
        for _ in range(8):
            visible=visible_options_stars(g.rgb)
            if visible==stars:break
            self.tap(1,'LEFT' if visible>stars else 'RIGHT',48)
            if visible_options_stars(g.rgb) is None:raise RuntimeError('Difficulty stars not legible')
        if visible_options_stars(g.rgb)!=stars or g.raw['difficulty_menu']!=stars-1:
            raise RuntimeError('Difficulty menu failed')
        self.tap(1,'START',120);self.row('rules',1)
        for _ in range(10):
            speed=visible_speed_stars(g.rgb)
            if speed==2:break
            if speed is None:self.advance(8)
            else:self.tap(1,'LEFT' if speed>2 else 'RIGHT',32)
        if visible_speed_stars(g.rgb)!=2:raise RuntimeError('Native hyper speed2 not verified')
        self.tap(1,'START',90);self.row('game',0);self.tap(1,'START',120)
        if mode=='pvp':self.tap(2,'START',128)
        for player in ((1,2) if mode=='pvp' else (1,)):
            if not self.cursor(player):raise RuntimeError(f'P{player} character cursor did not respond')
        self.cache_write(path,self.pack())

    def choose(self,player,name):
        target=CHARACTERS.index(name);g=self.game
        for _ in range(8):
            current=g.raw[f'p{player}_char']
            if current==target:return
            row,col=CELLS[current];tr,tc=CELLS[target]
            key=('DOWN' if tr>row else 'UP') if tr!=row else ('RIGHT' if tc>col else 'LEFT')
            self.tap(player,key)
            if g.raw[f'p{player}_char']==current:raise RuntimeError('Character cursor failed to move')
        raise RuntimeError('Character select route failed')

    def start(self,key):
        mode,stars,p1,p2=key;g=self.game
        path=self.cache/f'start_hyper_speed2_{mode}_{stars}_{p1}_{p2}.npz'
        if path.exists():return self.cache_read(path)
        try:
            self.menu(mode,stars);self.choose(1,p1)
            if mode=='pvp':self.choose(2,p2)
            self.stage=CHARACTERS.index(p2) if mode=='pve' else None
            for i in range(1800):
                self.advance(1,('A',) if i<8 else (),('START',) if mode=='pvp' and i<8 else ())
                r=g.raw
                if r['p1_hp']==r['p2_hp']==176 and r['timer_bcd']==153 and r['p1_wins']==r['p2_wins']==0:
                    self.stage=None;self.advance(8);break
            else:raise RuntimeError('Native first round not loaded')
            stable=0
            for _ in range(900):
                self.advance();timer=g.raw['timer_bcd']
                seconds=(timer>>4)*10+(timer&15)
                stable=stable+1 if (timer&15)<10 and 0<seconds<99 else 0
                if stable==3:break
            else:raise RuntimeError('Native countdown did not start')
            if g.raw['p1_char']!=CHARACTERS.index(p1) or g.raw['p2_char']!=CHARACTERS.index(p2):
                raise RuntimeError('Requested characters not loaded; no identity override permitted')
            # The course waits for three running-timer frames. A fast native
            # CPU can already connect in that same interval for some pairings.
            # Preserve the real health and timing; never heal or shift a pair
            # to a more favourable opening. First-round load above required176.
            if min(g.raw['p1_hp'],g.raw['p2_hp'])<0:raise RuntimeError('KO before the course live-start criterion')
            saved=self.pack();self.cache_write(path,saved);return saved
        except Exception:
            from PIL import Image
            Image.fromarray(g.rgb).save(self.cache/'setup_failure.png')
            raise
        finally:self.stage=None

_HEADER_B64 = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AB4+Px4eM4Aw3j4+AADMbLQwzGwAZsxsbAADMbDAwzGwAZsxsZAADMbDAwzHwAfsxsYAADMeDAwzHw"
    "AfsxseAADMYDAwzHwAZsxsYAADMYDAwzGwAZsxsZAADMYDAwzGwAZsxsbAAB48Hh4eOYA53j4+AAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)

_HEADER = np.unpackbits(np.frombuffer(base64.b64decode(_HEADER_B64), dtype=np.uint8))[:1632].reshape(16,102).astype(bool)

def _colour_mask(rgb):
    a = np.asarray(rgb, dtype=np.int16)
    return (a.max(axis=-1) >= 105) & (a.max(axis=-1)-a.min(axis=-1) >= 55)

def visible_options_stars(rgb):
    """Return 1..8 only when this exact OPTIONS header and star row are legible.

    This is limited to the verified 320x224 menu layout. Unknown layouts,
    fades, ambiguous star colours and different screens return None.
    """
    rgb = np.asarray(rgb)
    if rgb.shape != (224,320,3):
        return None
    mask = _colour_mask(rgb[12:28,76:178])
    union = int(np.count_nonzero(mask | _HEADER))
    score = np.count_nonzero(mask & _HEADER) / max(1, union)
    if score < 0.88:
        return None
    counts = [int(_colour_mask(rgb[48:57,136+8*i:144+8*i]).sum()) for i in range(8)]
    if any(5 < x < 12 for x in counts):
        return None
    active = [x >= 12 for x in counts]
    n = sum(active)
    if not 1 <= n <= 8 or active != [True]*n+[False]*(8-n):
        return None
    return n

MENU_TEXT_B64 = {'rules': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHz/GMN8PHzHAAAMZmGGZmGMZmAAAMZmPH5mGMZ2AAAMB+JH5mGMZ+AAAMZmfmZ8GMZuAAAMZmQmZgGMZmAAAHz/5+fwPHzjAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/nfHx8AAAQEBAGZmZmJmAAAQEBAGY8ZmBmAAD+/v4H4YZnh8AAB8fHwGYYfGBmAAA4ODgGYYYGJmAAB8fHwP888Pz2AABEREQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHx8fjx8x3wAAAAMZmmRjGZsYAAAAMZmGBjGdsAAAAAMZmGBjGfnwAAAAMZ8GBjGbgYAAAAMZgGBjGZsYAAAAHzwPDx843wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'game': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHwa03wAfH44fH4MYaZmIAxpkYZpkN48fmAAwBg9ZngMgkfngAfRhkfBgM5+ZuAABhh+ZhgNZCZmIAxtlCZhgH7n5/wAfD3n9nwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE4AfAAAfBh+fvBMQAxgAAZpiZmWBGwAwAAAZjwYeGBCgAfAAAfKQYGGBLgABgAAZn4a2GBFAwxjAAZkIYGGJBAwfDAA/Oc8PPzICAAIAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH18fO98AH2Yfn7MZmxmZmAGaYmZlN5mxmZmAGY8GBhNh9xuZmAPwkGBhM5mxm58AGZ+GppMZmxmZgAGZCGBhH72fDzwAPznfHzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'}

_MENU_MASKS = {k:np.unpackbits(np.frombuffer(base64.b64decode(v),dtype=np.uint8))[:48*84].reshape(48,84).astype(bool) for k,v in MENU_TEXT_B64.items()}

def menu_kind(rgb):
    if np.asarray(rgb).shape != (224,320,3):
        return None
    actual = _colour_mask(rgb[164:212,86:170])
    scores={k:np.count_nonzero(actual & v)/max(1,np.count_nonzero(actual | v)) for k,v in _MENU_MASKS.items()}
    best=max(scores,key=scores.get)
    # Exclude the HYPER star area for rule matching; speed is checked separately.
    rule_a=actual[:,:56];rule_t=_MENU_MASKS['rules'][:,:56]
    rule_score=np.count_nonzero(rule_a & rule_t)/max(1,np.count_nonzero(rule_a | rule_t))
    if rule_score>.82:
        return 'rules'
    return 'game' if scores['game']>.82 else None

def cursor_row(rgb):
    counts=[int(_colour_mask(rgb[164+16*i:180+16*i,67:84]).sum()) for i in range(3)]
    on=[i for i,n in enumerate(counts) if n>=12]
    return on[0] if len(on)==1 else None

def visible_speed_stars(rgb):
    if menu_kind(rgb)!='rules':
        return None
    counts=[int(_colour_mask(rgb[184:193,144+8*i:152+8*i]).sum()) for i in range(10)]
    if any(5<n<12 for n in counts):
        return None
    bits=[n>=12 for n in counts];n=sum(bits)
    return n if bits==[True]*n+[False]*(10-n) else None
