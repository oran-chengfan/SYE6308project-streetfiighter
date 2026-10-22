"""Report dependencies and hardware without inspecting file identities."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("numpy", "torch", "stable-retro", "stable-baselines3", "gymnasium",
            "cloudpickle", "opencv-python-headless", "imageio-ffmpeg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    versions, missing = {}, []
    for package in REQUIRED:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    report = dict(python=sys.version.split()[0], executable=sys.executable,
                  platform=platform.platform(), packages=versions,
                  missing=missing, start_files=len(list((ROOT / "assets/starts").glob("*.npz"))),
                  reference_present=(ROOT / "models/counter_ryu_final/agent.pt").is_file())
    if "torch" in versions:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing or not report["reference_present"] or report["start_files"] != 9:
        raise SystemExit("Environment is incomplete; see initial_readme_1022.md.")


if __name__ == "__main__":
    main()
