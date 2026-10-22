"""Copy an explicitly supplied local ROM into the installed game integration."""
import argparse
import json
import shutil
from pathlib import Path

GAME = "StreetFighterIISpecialChampionEdition-Genesis"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, type=Path,
                        help="Existing local raw .md/.bin/.gen ROM file; never a .state or archive")
    args = parser.parse_args()
    source = args.rom.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in (".md", ".bin", ".gen"):
        raise SystemExit("Supply the exact local raw Genesis ROM, not a state/archive/directory.")
    if source.stat().st_size != 3145728:
        raise SystemExit("The course raw ROM is 3145728 bytes; this file has a different size.")
    import stable_retro as retro
    for game_id in (GAME, GAME + "-v0"):
        metadata = retro.data.get_file_path(game_id, "data.json")
        if metadata:
            target = Path(metadata).parent / "rom.md"
            break
    else:
        raise SystemExit("Installed stable-retro lacks the course game integration.")
    if target.is_file():
        raise SystemExit("A local ROM already exists there; inspect it rather than overwriting it automatically.")
    shutil.copyfile(source, target)
    print(json.dumps(dict(installed_game=game_id, target=str(target), bytes=target.stat().st_size,
                          validation="Copy completed; validate compatibility by native state restoration."), indent=2))


if __name__ == "__main__":
    main()
