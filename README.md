# DOOM RL Command Center

A web-based platform for training, evaluating, and watching Reinforcement Learning agents play DOOM. Built with Flask, SocketIO, Stable-Baselines3, and ViZDoom.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-SocketIO-green)
![RL](https://img.shields.io/badge/RL-PPO-orange)
![ViZDoom](https://img.shields.io/badge/ViZDoom-1.2+-red)

## Features

- **Training Dashboard** - Configure hyperparameters, reward shaping, and train PPO agents from the browser with real-time metrics
- **AI Demo** - Watch trained agents play DOOM with live game frame streaming
- **Model Management** - Save, compare, and manage trained models
- **Evaluation** - Run evaluation episodes and view performance statistics
- **5 DOOM Scenarios** - Basic, Deadly Corridor (3 difficulty levels), Defend the Center

## Screenshots

The UI features a DOOM-inspired dark theme with real-time Chart.js visualizations, live game streaming via WebSocket, and configurable reward shaping.

## Quick Start

```bash
# Clone
git clone https://github.com/indugapallignaneswara/DOOMgame-using-RLagent.git
cd DOOMgame-using-RLagent

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

Open `http://localhost:5000` in your browser.

## Project Structure

```
DOOMgame-using-RLagent/
├── app.py                    # Flask + SocketIO web application
├── config.py                 # App configuration and path resolution
├── requirements.txt
│
├── doom/                     # Core RL engine
│   ├── scenarios.py          # Scenario registry (5 DOOM environments)
│   ├── environment.py        # VizDoomGym (Gymnasium wrapper)
│   ├── rewards.py            # Configurable reward shaping wrapper
│   ├── callbacks.py          # SocketIO training callbacks
│   ├── trainer.py            # Training manager (background threads)
│   ├── player.py             # AI demo with frame streaming
│   └── evaluator.py          # Model evaluation engine
│
├── templates/                # Flask templates (dark DOOM theme)
│   ├── base.html             # Base layout with sidebar nav
│   ├── index.html            # Dashboard
│   ├── train.html            # Training configuration + live metrics
│   ├── play.html             # AI demo viewer (canvas frame display)
│   ├── models.html           # Model browser + comparison
│   └── evaluate.html         # Evaluation results + charts
│
├── static/
│   ├── css/styles.css        # DOOM-themed dark UI
│   └── js/
│       ├── app.js            # SocketIO setup, navigation, toasts
│       ├── training.js       # Training controls + real-time charts
│       ├── player.js         # Game frame rendering + controls
│       ├── charts.js         # Chart.js wrappers (dark theme)
│       └── models.js         # Model management UI
│
├── models/                   # Saved models + registry
│   └── registry.json
│
├── notebooks/                # Original Jupyter notebooks (reference)
│   ├── DOOM-RLbasedAGENT.ipynb
│   ├── DOOMcorridor-RLbasedAGENT.ipynb
│   ├── DOOMcorridor-level1-RLbasedAGENT-.ipynb
│   └── DOOMdefends-RLbasedAGENT.ipynb
│
├── train/                    # Training checkpoints (gitignored)
└── logs/                     # TensorBoard logs (gitignored)
```

## Scenarios

| Scenario | Actions | Description | Default Timesteps |
|----------|---------|-------------|-------------------|
| **Basic** | 3 | Shoot the monster at the end of the room | 100K |
| **Deadly Corridor S1** | 7 | Navigate enemy-filled corridor (easy) | 400K |
| **Deadly Corridor S2** | 7 | Navigate enemy-filled corridor (medium) | 400K |
| **Deadly Corridor S5** | 7 | Navigate enemy-filled corridor (hard) | 100K |
| **Defend the Center** | 3 | Defend against approaching enemy waves | 100K |

## How It Works

### Training
1. Select a scenario and configure hyperparameters in the web UI
2. Optionally set reward shaping (kill bonus, damage penalty, etc.)
3. Click Start - training runs in a background thread
4. Watch real-time reward curves and loss charts update via WebSocket
5. Model is automatically saved and registered on completion

### Reward Shaping
The reward shaping wrapper tracks game variable deltas per step:
- **Kill Reward**: Bonus per enemy killed (hitcount increase)
- **Damage Penalty**: Penalty per damage taken
- **Ammo Penalty**: Penalty for wasting ammo (shot without hit)
- **Step Penalty**: Small per-step cost to encourage efficiency

### AI Demo
Trained models play DOOM with frames streamed to the browser as base64 JPEG over WebSocket at ~20 FPS.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Algorithm | PPO (Stable-Baselines3) |
| Policy Network | CnnPolicy (CNN for image input) |
| Game Engine | ViZDoom |
| Environment Wrapper | Gymnasium |
| Web Framework | Flask + Flask-SocketIO |
| Real-time Communication | WebSocket (threading mode) |
| Frontend Charts | Chart.js |
| Frame Streaming | Base64 JPEG over SocketIO |

## Prerequisites

- Python 3.10+
- ViZDoom (installs DOOM scenarios automatically via pip)
- GPU recommended for training (CPU works but slower)

## License

MIT License - see [LICENSE](LICENSE)
