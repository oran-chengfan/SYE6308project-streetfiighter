"""Course-compatible Genesis setup and round termination."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from .native_setup import CHARACTERS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
GAME = "StreetFighterIISpecialChampionEdition-Genesis"


class NativeGame:
    def __init__(self):
        self.engine = None
        self.starts = {}
        self.raw = {}
        self.elapsed = 0
        self.key = None
        self.on_frame = None
        self.setup_frames = 0
        self.held = ((), ())
        for path in (ROOT / "assets/starts").glob("*.npz"):
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(data["metadata"].tobytes().decode())
                key = (
                    meta["mode"],
                    int(meta["stars"]),
                    meta.get("p1_character", "Ryu"),
                    meta["character"],
                )
                if key in self.starts:
                    raise ValueError("Duplicate native start")
                self.starts[key] = dict(
                    state=data["state"].tobytes(),
                    rgb=data["rgb"].copy(),
                    ram=data["ram"].copy(),
                    readings=meta["readings"],
                    definitions=meta["definitions"],
                    fps=meta["fps"],
                    buttons=meta["buttons"],
                )
        if not all(
            (mode, star, "Ryu", "Ryu") in self.starts
            for mode, star in [("pvp", 1), *[("pve", s) for s in range(1, 9)]]
        ):
            raise RuntimeError("Required Ryu starts are missing from assets/starts")

    def _create(self):
        import stable_retro as retro

        game_id = None
        for candidate in (GAME, GAME + "-v0"):
            try:
                if Path(retro.data.get_romfile_path(candidate)).is_file():
                    game_id = candidate
                    break
            except (FileNotFoundError, RuntimeError):
                pass
        if game_id is None:
            raise FileNotFoundError("Previously installed game is unavailable")
        self.work = tempfile.TemporaryDirectory(prefix="counter_native_")
        scenario = Path(self.work.name) / "scenario.json"
        scenario.write_text("{}")
        # The installed make/RetroEnv path reads the existing ROM directly;
        # data.groom_rom and data.verify_hash are separate import/check utilities.
        # A pre-call guard stops execution if a library unexpectedly enters them.
        old_profile = sys.getprofile()

        def prohibit_identity_calculations(frame, event, arg):
            if event == "call" and frame.f_code.co_name in (
                "groom_rom",
                "verify_hash",
                "get_known_hashes",
            ):
                raise RuntimeError(
                    "Unexpected identity-calculation path; stopped before entering it"
                )
            if event == "c_call" and getattr(arg, "__module__", "") in (
                "_hashlib",
                "_sha1",
                "_sha256",
                "_sha512",
                "_md5",
                "_blake2",
                "_sha3",
            ):
                raise RuntimeError(
                    "Unexpected digest constructor; stopped before calling it"
                )

        try:
            sys.setprofile(prohibit_identity_calculations)
            self.engine = retro.make(
                game=game_id,
                state=retro.State.NONE,
                players=2,
                use_restricted_actions=retro.Actions.ALL,
                scenario=str(scenario),
                render_mode="rgb_array",
            )
            self.engine.reset()
        finally:
            sys.setprofile(old_profile)
        self.buttons = list(self.engine.buttons)
        self.fps = float(self.engine.em.get_screen_rate())
        self.cold_state = bytes(self.engine.em.get_state())

    def _physical_restore(self, saved):
        engine = self.engine
        engine.em.set_state(saved["state"])
        self.held = tuple(tuple(k) for k in saved.get("held", ((), ())))
        for player in range(2):
            mask = np.zeros(engine.num_buttons, np.uint8)
            for name in self.held[player]:
                mask[self.buttons.index(name)] = 1
            engine.em.set_button_mask(mask, player)
        engine.data.reset()
        for name, spec in self.definitions.items():
            engine.data.set_variable("boot_" + name, dict(spec))
        engine.data.update_ram()
        if not np.array_equal(np.asarray(engine.get_ram(), np.uint8), saved["ram"]):
            raise RuntimeError("Restored RAM differs")
        self.raw = self.read()
        if self.raw != saved["readings"]:
            raise RuntimeError("Restored native readings differ")
        self.rgb = saved["rgb"].copy()
        engine.img = self.rgb.copy()
        engine.ram = None

    def open(self, mode="pvp", stars=1, character="Ryu", p1_character="Ryu"):
        if self.engine is None:
            self._create()
        # Legacy callers specify the P2 character. New callers explicitly name
        # both physical controller identities, independently of screen side.
        key = (mode, int(stars), p1_character, character)
        if (
            mode not in ("pvp", "pve")
            or not 1 <= int(stars) <= 8
            or any(c not in CHARACTERS for c in key[2:])
        ):
            raise ValueError("Invalid native match configuration")
        if key in self.starts:
            saved = self.starts[key]
        else:
            from .native_setup import NativeSetup

            self.definitions = self.starts[("pvp", 1, "Ryu", "Ryu")]["definitions"]
            saved = NativeSetup(self, OUT / "native_starts").start(key)
        self.definitions = saved["definitions"]
        if self.buttons != saved["buttons"] or abs(self.fps - saved["fps"]) > 1e-6:
            raise RuntimeError("Native controller/fps mismatch")
        self._physical_restore(saved)
        self.key = key
        self.elapsed = 0
        self.on_frame = None
        self.wins = (self.raw["p1_wins"], self.raw["p2_wins"])
        self._validate()

    def _validate(self):
        if any(
            self.raw[f"p{p}_char"] != CHARACTERS.index(self.key[p + 1]) for p in (1, 2)
        ):
            raise RuntimeError("Native character identity mismatch")
        if self.key[0] == "pve" and any(
            self.raw[k] != self.key[1] - 1
            for k in ("difficulty_menu", "difficulty_fight")
        ):
            raise RuntimeError("Native CPU difficulty mismatch")

    def read(self):
        return {
            name: int(self.engine.data.lookup_value("boot_" + name))
            for name in self.definitions
        }

    def step_keys(self, k1=(), k2=()):
        if self.key[0] == "pve" and k2:
            raise RuntimeError("Cannot control native CPU")
        action = np.zeros(2 * len(self.buttons), np.int8)
        for port, keys in enumerate((k1, k2)):
            for name in keys:
                action[port * len(self.buttons) + self.buttons.index(name)] = 1
        self.held = (tuple(k1), tuple(k2))
        before = self.raw
        self.rgb, *_ = self.engine.step(action)
        self.raw = self.read()
        self.elapsed += 1
        self._validate()
        h1, h2 = self.raw["p1_hp"], self.raw["p2_hp"]
        winner = reason = None
        if min(h1, h2) < 0:
            winner = "draw" if h1 < 0 and h2 < 0 else "p2" if h1 < 0 else "p1"
            reason = "signed_hp_ko"
        elif any(self.raw[f"p{p}_wins"] > self.wins[p - 1] for p in (1, 2)):
            a, b = [self.raw[f"p{p}_wins"] > self.wins[p - 1] for p in (1, 2)]
            winner = "draw" if a and b else "p1" if a else "p2"
            reason = "round_counter"
        elif self.raw["timer_bcd"] == 0:
            winner = "p1" if h1 > h2 else "p2" if h2 > h1 else "draw"
            reason = "native_time_up"
        elif self.elapsed >= 7200:
            reason = "frame_cap"
        if reason is None and any(
            self.raw[f"p{p}_hp"] > before[f"p{p}_hp"] for p in (1, 2)
        ):
            raise RuntimeError("Unexpected HP refill")
        if self.on_frame is not None:
            self.on_frame(self.rgb, self.raw, k1, k2, self.elapsed)
        return self.raw, winner, reason

    def snapshot(self):
        return dict(
            state=bytes(self.engine.em.get_state()),
            ram=np.asarray(self.engine.get_ram(), np.uint8).copy(),
            rgb=self.rgb.copy(),
            readings=dict(self.raw),
            elapsed=self.elapsed,
            key=self.key,
            wins=self.wins,
            held=self.held,
        )

    def restore(self, saved):
        if self.key != saved["key"]:
            mode, stars, p1, p2 = saved["key"]
            self.open(mode, stars, p2, p1)
        self._physical_restore(saved)
        self.elapsed = saved["elapsed"]
        self.wins = saved["wins"]

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
            self.work.cleanup()
