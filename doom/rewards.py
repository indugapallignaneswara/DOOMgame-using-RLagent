"""Configurable reward shaping wrapper for ViZDoom environments."""

import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """Wraps VizDoomGym to apply configurable reward shaping.

    Tracks game variable deltas (health, ammo, damage, hitcount)
    and applies bonus/penalty based on configuration.
    """

    def __init__(self, env, kill_reward=0, miss_penalty=0, step_penalty=0,
                 damage_penalty=0, ammo_penalty=0):
        super().__init__(env)
        self.kill_reward = kill_reward
        self.miss_penalty = miss_penalty
        self.step_penalty = step_penalty
        self.damage_penalty = damage_penalty
        self.ammo_penalty = ammo_penalty

        # Previous game variable values
        self._prev_health = 0
        self._prev_ammo = 0
        self._prev_hitcount = 0
        self._prev_damage_taken = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_health = info.get("health", 100)
        self._prev_ammo = info.get("selected_weapon_ammo", info.get("ammo2", 0))
        self._prev_hitcount = info.get("hitcount", 0)
        self._prev_damage_taken = info.get("damage_taken", 0)
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = base_reward

        # Kill bonus (hitcount increased)
        hitcount = info.get("hitcount", self._prev_hitcount)
        hit_delta = hitcount - self._prev_hitcount
        if hit_delta > 0 and self.kill_reward > 0:
            shaped_reward += hit_delta * self.kill_reward

        # Damage penalty (damage_taken increased)
        damage = info.get("damage_taken", self._prev_damage_taken)
        damage_delta = damage - self._prev_damage_taken
        if damage_delta > 0 and self.damage_penalty > 0:
            shaped_reward -= damage_delta * self.damage_penalty

        # Ammo penalty (ammo decreased without hit)
        ammo = info.get("selected_weapon_ammo", info.get("ammo2", self._prev_ammo))
        ammo_delta = self._prev_ammo - ammo  # positive = ammo used
        if ammo_delta > 0 and hit_delta == 0 and self.ammo_penalty > 0:
            shaped_reward -= ammo_delta * self.ammo_penalty

        # Miss penalty (for backward compat with basic scenario)
        if ammo_delta > 0 and hit_delta == 0 and self.miss_penalty > 0:
            shaped_reward -= self.miss_penalty

        # Step penalty
        if self.step_penalty > 0:
            shaped_reward -= self.step_penalty

        # Update tracked values
        self._prev_health = info.get("health", self._prev_health)
        self._prev_ammo = ammo
        self._prev_hitcount = hitcount
        self._prev_damage_taken = damage

        return obs, shaped_reward, terminated, truncated, info
