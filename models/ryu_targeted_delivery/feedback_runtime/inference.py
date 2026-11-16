"""Load standalone SB3 inference; this does not certify course Arena compatibility."""
from pathlib import Path
from stable_baselines3 import PPO

def load_model(device='cpu'):
    model=PPO.load(Path(__file__).resolve().parent/'policy.zip',device=device)
    model.policy.set_training_mode(False)
    return model
