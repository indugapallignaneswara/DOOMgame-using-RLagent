"""Application configuration for DOOM RL Web App."""

import os
import secrets

# Try to auto-detect ViZDoom scenarios path
try:
    import vizdoom
    SCENARIOS_PATH = os.path.join(os.path.dirname(vizdoom.__file__), "scenarios")
    if not os.path.isdir(SCENARIOS_PATH):
        SCENARIOS_PATH = os.environ.get("VIZDOOM_SCENARIOS_PATH", "scenarios")
except ImportError:
    SCENARIOS_PATH = os.environ.get("VIZDOOM_SCENARIOS_PATH", "scenarios")

# Directories
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "train")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# Flask
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(24))

# Ensure directories exist
for d in [MODELS_DIR, LOGS_DIR, TRAIN_DIR]:
    os.makedirs(d, exist_ok=True)
