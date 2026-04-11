"""Patch SB3 load_from_zip_file to work with PyTorch 2.6+.

Import this module BEFORE loading any SB3 models.
"""

import io
import json
import base64
import pickle
import zipfile
import torch as th
import stable_baselines3.common.save_util as save_util
import stable_baselines3.common.base_class as base_class
from stable_baselines3.common.utils import get_device
import gymnasium.spaces as gym_spaces
from stable_baselines3.common.vec_env import patch_gym as pg


def _patched_load_from_zip_file(load_path, load_data=True, custom_objects=None,
                                 device="auto", verbose=0, print_system_info=False):
    """Load SB3 model from zip, fixing PyTorch stream reader issues."""
    device = get_device(device)
    data = None
    pytorch_variables = None
    params = {}

    with zipfile.ZipFile(load_path, "r") as archive:
        namelist = archive.namelist()

        if load_data and "data" in namelist:
            json_data = archive.read("data").decode()
            data = json.loads(json_data)
            if custom_objects:
                for key, val in custom_objects.items():
                    if key in data:
                        data[key] = val

        for name in namelist:
            if name.endswith(".pth"):
                buf = io.BytesIO(archive.read(name))
                th_object = th.load(buf, map_location=device, weights_only=False)
                if name == "pytorch_variables.pth":
                    pytorch_variables = th_object
                else:
                    params[name.replace(".pth", "")] = th_object

    return data, params, pytorch_variables


def _safe_convert_space(space):
    """Handle dict-serialized gymnasium spaces from SB3 JSON data."""
    if isinstance(space, gym_spaces.Space):
        return space
    if isinstance(space, dict):
        # SB3 serializes spaces with :serialized: key containing pickled base64
        if ":serialized:" in space:
            try:
                raw = base64.b64decode(space[":serialized:"])
                return pickle.loads(raw)
            except Exception:
                pass
        # Fallback: reconstruct from dict fields
        if "dtype" in space and "shape" in space:
            import numpy as np
            return gym_spaces.Box(
                low=space.get("low", 0), high=space.get("high", 255),
                shape=tuple(space["shape"]), dtype=np.dtype(space["dtype"]))
        if "n" in space:
            return gym_spaces.Discrete(int(space["n"]))
    # Last resort: try original
    try:
        from stable_baselines3.common.vec_env.patch_gym import _convert_space
        return _convert_space(space)
    except Exception:
        return space


# Apply patches
save_util.load_from_zip_file = _patched_load_from_zip_file
base_class.load_from_zip_file = _patched_load_from_zip_file
pg._convert_space = _safe_convert_space
base_class._convert_space = _safe_convert_space
