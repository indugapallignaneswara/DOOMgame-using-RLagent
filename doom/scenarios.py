"""Scenario registry for all supported ViZDoom environments."""

import os

try:
    import vizdoom
    _BASE_PATH = os.path.join(os.path.dirname(vizdoom.__file__), "scenarios")
except ImportError:
    _BASE_PATH = os.environ.get("VIZDOOM_SCENARIOS_PATH", "scenarios")


def _corridor(skill, description):
    """Helper to build corridor scenario variants with different skill levels."""
    return {
        "config_file": "deadly_corridor.cfg",
        "skill_level": skill,
        "num_actions": 7,
        "buttons": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD",
                     "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT"],
        "game_variables": ["HEALTH", "DAMAGE_TAKEN", "HITCOUNT", "SELECTED_WEAPON_AMMO"],
        "description": description,
        "default_hyperparams": {
            "learning_rate": 0.00001,
            "n_steps": 8192,
            "clip_range": 0.1,
            "gamma": 0.95,
            "gae_lambda": 0.9,
            "total_timesteps": 400000,
        },
        "reward_defaults": {
            "kill_reward": 200, "miss_penalty": 0, "step_penalty": 0,
            "damage_penalty": 10, "ammo_penalty": 5,
        },
    }


SCENARIOS = {
    "basic": {
        "config_file": "basic.cfg",
        "num_actions": 3,
        "buttons": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
        "game_variables": ["AMMO2"],
        "description": "Shoot the monster at the end of the room. Simple aiming task.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 100000,
        },
        "reward_defaults": {
            "kill_reward": 1.0, "miss_penalty": -0.1, "step_penalty": -0.01,
            "damage_penalty": -0.1, "ammo_penalty": -0.1,
        },
    },
    "deadly_corridor_s1": _corridor(1, "Navigate a corridor full of enemies. Skill level 1 (easiest)."),
    "deadly_corridor_s3": _corridor(3, "Navigate a corridor full of enemies. Skill level 3 (medium)."),
    "deadly_corridor_s5": _corridor(5, "Navigate a corridor full of enemies. Skill level 5 (hardest)."),
    "defend_the_center": {
        "config_file": "defend_the_center.cfg",
        "num_actions": 3,
        "buttons": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
        "game_variables": ["AMMO2", "HEALTH"],
        "description": "Stand in the center and defend against waves of approaching enemies.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 100000,
        },
        "reward_defaults": {
            "kill_reward": 1.5, "miss_penalty": -0.05, "step_penalty": -0.01,
            "damage_penalty": -0.5, "ammo_penalty": -0.1,
        },
    },
    "defend_the_line": {
        "config_file": "defend_the_line.cfg",
        "num_actions": 3,
        "buttons": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
        "game_variables": ["AMMO2", "HEALTH"],
        "description": "Defend your position on a line against approaching enemies.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 100000,
        },
        "reward_defaults": {
            "kill_reward": 1.5, "miss_penalty": -0.05, "step_penalty": -0.01,
            "damage_penalty": -0.5, "ammo_penalty": -0.1,
        },
    },
    "health_gathering": {
        "config_file": "health_gathering.cfg",
        "num_actions": 3,
        "buttons": ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"],
        "game_variables": ["HEALTH"],
        "description": "Navigate a toxic floor to collect health packs and survive as long as possible.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 100000,
        },
        "reward_defaults": {
            "kill_reward": 0, "miss_penalty": 0, "step_penalty": -0.01,
            "damage_penalty": -1.0, "ammo_penalty": 0,
        },
    },
    "my_way_home": {
        "config_file": "my_way_home.cfg",
        "num_actions": 3,
        "buttons": ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"],
        "game_variables": [],
        "description": "Navigate a maze to find the goal vest. Tests spatial navigation.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 200000,
        },
        "reward_defaults": {
            "kill_reward": 0, "miss_penalty": 0, "step_penalty": -0.005,
            "damage_penalty": 0, "ammo_penalty": 0,
        },
    },
    "take_cover": {
        "config_file": "take_cover.cfg",
        "num_actions": 2,
        "buttons": ["MOVE_LEFT", "MOVE_RIGHT"],
        "game_variables": ["HEALTH"],
        "description": "Dodge incoming fireballs by moving left and right. Survive as long as possible.",
        "default_hyperparams": {
            "learning_rate": 0.0001,
            "n_steps": 2048,
            "clip_range": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "total_timesteps": 100000,
        },
        "reward_defaults": {
            "kill_reward": 0, "miss_penalty": 0, "step_penalty": -0.01,
            "damage_penalty": -1.0, "ammo_penalty": 0,
        },
    },
}


def get_scenario(name):
    """Get scenario config by name. Resolves full config path."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    scenario = SCENARIOS[name].copy()
    scenario["config_path"] = os.path.join(_BASE_PATH, scenario["config_file"])
    return scenario


def list_scenarios():
    """Return list of scenario names."""
    return list(SCENARIOS.keys())
