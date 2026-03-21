from typing import Any

import cv2
import gymnasium as gym
import h5py
import numpy as np

import mani_skill.envs  # noqa: F401 — registers ManiSkill environments with gymnasium

from .base_env import BaseEnv


class ManiSkillEnv(BaseEnv):
    """Environment wrapper for ManiSkill tasks."""

    def __init__(
        self,
        env_id: str = "StackCube-v1",
        obs_mode: str = "rgb+state",
        control_mode: str = "pd_ee_delta_pos",
        render_size: tuple[int, int] = (128, 128),
        max_episode_steps: int = 200,
    ):
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_size = render_size

        self.env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
        )
        self._last_obs = None

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human", **args: Any) -> dict:
        obs = self._last_obs
        if obs is None:
            obs, _ = self.env.reset()
            self._last_obs = obs

        img_obs = {}
        sensor_data = obs["sensor_data"]
        for cam_name in sensor_data:
            if "rgb" in sensor_data[cam_name]:
                img = sensor_data[cam_name]["rgb"]
                # Handle batched envs: squeeze batch dim
                if img.ndim == 4:
                    img = img[0]
                # Convert torch tensor to numpy if needed
                if hasattr(img, "cpu"):
                    img = img.cpu().numpy()
                img = img.astype(np.uint8)
                h, w = self.render_size
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                img_obs[cam_name] = img
        return img_obs

    def compute_init_state(self, hdf5_file_path: str) -> dict:
        """Load initial environment state from a ManiSkill trajectory HDF5 file."""
        with h5py.File(hdf5_file_path, "r") as f:
            # Find the first trajectory
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            if not traj_keys:
                raise ValueError(f"No trajectories found in {hdf5_file_path}")
            traj_key = sorted(traj_keys, key=lambda x: int(x.split("_")[1]))[0]
            # ManiSkill env_states contain full simulation state
            env_states = {}
            state_group = f[traj_key]["env_states"]
            for key in state_group:
                env_states[key] = state_group[key][0]
        return env_states

    def reset(self, state: Any = None) -> None:
        if state is not None:
            # Set state dict for ManiSkill env
            self.env.unwrapped.set_state_dict(state)
            self._last_obs = self.env.unwrapped.get_obs()
        else:
            obs, _ = self.env.reset()
            self._last_obs = obs

    def get_state(self) -> dict:
        return self.env.unwrapped.get_state_dict()

    def get_observations(self) -> dict:
        if self._last_obs is None:
            obs, _ = self.env.reset()
            self._last_obs = obs
        return self._last_obs

    def get_render_size(self) -> tuple[int, int]:
        return self.render_size

    def get_curr_pos(self) -> np.ndarray:
        """Return the current end-effector position."""
        obs = self.get_observations()
        # Extract agent state (qpos) from observation
        if "agent" in obs:
            qpos = obs["agent"]["qpos"]
            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu().numpy()
            if qpos.ndim == 2:
                qpos = qpos[0]
            return qpos
        elif "state" in obs:
            state = obs["state"]
            if hasattr(state, "cpu"):
                state = state.cpu().numpy()
            if state.ndim == 2:
                state = state[0]
            return state
        return np.array([])

    def get_cam_intrinsic(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        obs = self.get_observations()
        if "sensor_param" in obs and name in obs["sensor_param"]:
            intrinsic = obs["sensor_param"][name]["intrinsic_cv"]
            if hasattr(intrinsic, "cpu"):
                intrinsic = intrinsic.cpu().numpy()
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[0]
            # Extract cx, cy, fx, fy from 3x3 intrinsic matrix
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            return np.array([cx, cy, fx, fy])
        return np.zeros(4)

    def get_cam_extrinsic(self, name: str) -> np.ndarray:
        obs = self.get_observations()
        if "sensor_param" in obs and name in obs["sensor_param"]:
            extrinsic = obs["sensor_param"][name]["extrinsic_cv"]
            if hasattr(extrinsic, "cpu"):
                extrinsic = extrinsic.cpu().numpy()
            if extrinsic.ndim == 3:
                extrinsic = extrinsic[0]
            # Convert 3x4 to 4x4
            ext_4x4 = np.eye(4)
            ext_4x4[:3, :] = extrinsic
            return ext_4x4
        return np.eye(4)
