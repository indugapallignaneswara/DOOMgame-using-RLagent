"""Unified VizDoom Gymnasium environment wrapper."""

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from doom.scenarios import get_scenario


class VizDoomGym(gym.Env):
    """Gymnasium-compatible wrapper for ViZDoom.

    Provides grayscale observation (100, 160, 1), discrete action space,
    and optional frame callback for streaming frames to the browser.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, scenario_key, render=False, frame_callback=None, frame_skip=4):
        super().__init__()

        self.scenario = get_scenario(scenario_key)
        self.frame_callback = frame_callback
        self.frame_skip = frame_skip
        self.render_mode = "human" if render else "rgb_array"

        # Initialize DoomGame
        self.game = DoomGame()
        self.game.load_config(self.scenario["config_path"])
        self.game.set_window_visible(render)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_mode(Mode.PLAYER)

        # Set skill level if specified (for corridor variants)
        if "skill_level" in self.scenario:
            self.game.set_doom_skill(self.scenario["skill_level"])

        self.game.init()

        # Spaces
        self.num_actions = self.scenario["num_actions"]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )

        # Build action table (one-hot encoded)
        self._actions = np.eye(self.num_actions, dtype=int).tolist()

        # Track game variables for reward shaping
        self.game_variables = {}

    def step(self, action):
        action_vec = self._actions[int(action)]
        reward = self.game.make_action(action_vec, self.frame_skip)
        done = self.game.is_episode_finished()

        state = self.game.get_state()

        if state is not None:
            screen = state.screen_buffer  # RGB (H, W, 3)
            obs = self._preprocess(screen)

            # Stream raw frame if callback set
            if self.frame_callback is not None:
                self.frame_callback(screen)

            # Extract game variables
            info = self._extract_variables(state)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {}

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()

        if state is not None:
            obs = self._preprocess(state.screen_buffer)
            info = self._extract_variables(state)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = {}

        self.game_variables = info.copy()
        return obs, info

    def _preprocess(self, screen):
        """Convert RGB screen to grayscale (100, 160, 1)."""
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
        return resized.reshape(100, 160, 1)

    def _extract_variables(self, state):
        """Extract game variables into a dict."""
        info = {}
        var_names = self.scenario.get("game_variables", [])
        if state.game_variables is not None:
            for i, name in enumerate(var_names):
                if i < len(state.game_variables):
                    info[name.lower()] = state.game_variables[i]
        return info

    def get_raw_frame(self):
        """Get current raw RGB frame (for streaming)."""
        state = self.game.get_state()
        if state is not None:
            return state.screen_buffer
        return None

    def close(self):
        if self.game is not None:
            self.game.close()

    def render(self):
        state = self.game.get_state()
        if state is not None:
            return state.screen_buffer
        return None
