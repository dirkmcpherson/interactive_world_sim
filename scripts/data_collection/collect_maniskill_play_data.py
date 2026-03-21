"""Collect play data from ManiSkill environments for world model training.

Generates diverse interaction data by combining random actions, noisy expert
replays, and exploratory behaviors. This provides the variety of physical
interactions needed to train a robust world model.

Usage:
    python scripts/data_collection/collect_maniskill_play_data.py \
        --env-id StackCube-v1 \
        --output-dir data/maniskill_play/StackCube-v1 \
        --num-episodes 10000 \
        --episode-length 200 \
        --demo-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state+rgb+depth.pd_ee_delta_pos.physx_cpu.h5 \
        --demo-json ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.state+rgb+depth.pd_ee_delta_pos.physx_cpu.json
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from tqdm import tqdm

import mani_skill.envs  # noqa: F401 — registers ManiSkill environments with gymnasium


def load_demo_actions(demo_path: str, demo_json_path: str) -> list[np.ndarray]:
    """Load expert demo actions for noisy replay."""
    with open(demo_json_path) as f:
        meta = json.load(f)
    n_episodes = len(meta["episodes"])

    actions_list = []
    with h5py.File(demo_path, "r") as f:
        for i in range(n_episodes):
            key = f"traj_{i}"
            if key in f:
                actions_list.append(f[key]["actions"][()])
    return actions_list


def random_policy(action_space: gym.spaces.Box, rng: np.random.Generator) -> np.ndarray:
    """Fully random actions within the action space."""
    return rng.uniform(action_space.low, action_space.high).astype(np.float32)


def noisy_demo_policy(
    demo_actions: np.ndarray,
    step: int,
    action_space: gym.spaces.Box,
    rng: np.random.Generator,
    noise_scale: float = 0.3,
) -> np.ndarray:
    """Replay a demo action with added noise for diversity."""
    if step < len(demo_actions):
        action = demo_actions[step].copy()
        noise = rng.normal(0, noise_scale, size=action.shape).astype(np.float32)
        action = action + noise
    else:
        # Past demo length, switch to random
        action = rng.uniform(action_space.low, action_space.high).astype(np.float32)
    return np.clip(action, action_space.low, action_space.high)


def smooth_random_policy(
    prev_action: np.ndarray,
    action_space: gym.spaces.Box,
    rng: np.random.Generator,
    smoothing: float = 0.7,
) -> np.ndarray:
    """Temporally correlated random actions for smooth exploratory motions."""
    random_action = rng.uniform(action_space.low, action_space.high).astype(np.float32)
    action = smoothing * prev_action + (1 - smoothing) * random_action
    return np.clip(action, action_space.low, action_space.high)


def collect_episode(
    env: gym.Env,
    policy_type: str,
    episode_length: int,
    rng: np.random.Generator,
    demo_actions: np.ndarray | None = None,
    noise_scale: float = 0.3,
) -> dict:
    """Collect a single episode of play data."""
    obs, info = env.reset()
    action_space = env.action_space

    actions = []
    rgb_base = []
    rgb_hand = []
    states = []

    # Extract initial observation
    base_img = obs["sensor_data"]["base_camera"]["rgb"]
    hand_img = obs["sensor_data"]["hand_camera"]["rgb"]
    if hasattr(base_img, "cpu"):
        base_img = base_img.cpu().numpy()
        hand_img = hand_img.cpu().numpy()
    if base_img.ndim == 4:
        base_img = base_img[0]
        hand_img = hand_img[0]

    rgb_base.append(base_img.astype(np.uint8))
    rgb_hand.append(hand_img.astype(np.uint8))
    if "state" in obs:
        s = obs["state"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        if s.ndim == 2:
            s = s[0]
        states.append(s)

    prev_action = np.zeros(action_space.shape, dtype=np.float32)

    for step in range(episode_length):
        if policy_type == "random":
            action = random_policy(action_space, rng)
        elif policy_type == "noisy_demo" and demo_actions is not None:
            action = noisy_demo_policy(
                demo_actions, step, action_space, rng, noise_scale
            )
        elif policy_type == "smooth_random":
            action = smooth_random_policy(prev_action, action_space, rng)
        else:
            action = random_policy(action_space, rng)

        obs, reward, terminated, truncated, info = env.step(action)
        prev_action = action

        # Extract observations
        base_img = obs["sensor_data"]["base_camera"]["rgb"]
        hand_img = obs["sensor_data"]["hand_camera"]["rgb"]
        if hasattr(base_img, "cpu"):
            base_img = base_img.cpu().numpy()
            hand_img = hand_img.cpu().numpy()
        if base_img.ndim == 4:
            base_img = base_img[0]
            hand_img = hand_img[0]

        actions.append(action.copy())
        rgb_base.append(base_img.astype(np.uint8))
        rgb_hand.append(hand_img.astype(np.uint8))
        if "state" in obs:
            s = obs["state"]
            if hasattr(s, "cpu"):
                s = s.cpu().numpy()
            if s.ndim == 2:
                s = s[0]
            states.append(s)

        if terminated or truncated:
            break

    result = {
        "actions": np.stack(actions),
        "base_camera_rgb": np.stack(rgb_base),
        "hand_camera_rgb": np.stack(rgb_hand),
    }
    if states:
        result["state"] = np.stack(states)
    return result


def save_episode_hdf5(filepath: str, episode_data: dict) -> None:
    """Save a single episode to HDF5 in the format expected by ManiSkillDataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("action", data=episode_data["actions"])
        obs_group = f.create_group("obs")
        img_group = obs_group.create_group("images")
        img_group.create_dataset(
            "base_camera_rgb", data=episode_data["base_camera_rgb"]
        )
        img_group.create_dataset(
            "hand_camera_rgb", data=episode_data["hand_camera_rgb"]
        )
        if "state" in episode_data:
            obs_group.create_dataset("state", data=episode_data["state"])


def save_as_maniskill_hdf5(
    output_path: str,
    all_episodes: list[dict],
    env_id: str,
    control_mode: str,
) -> None:
    """Save all episodes in a single ManiSkill-style HDF5 file with JSON metadata."""
    h5_path = output_path
    json_path = output_path.replace(".h5", ".json")

    episodes_meta = []
    with h5py.File(h5_path, "w") as f:
        for i, ep in enumerate(tqdm(all_episodes, desc="Saving episodes")):
            traj_group = f.create_group(f"traj_{i}")
            traj_group.create_dataset("actions", data=ep["actions"])

            n_actions = ep["actions"].shape[0]
            traj_group.create_dataset(
                "terminated", data=np.zeros(n_actions, dtype=bool)
            )
            traj_group.create_dataset(
                "truncated", data=np.zeros(n_actions, dtype=bool)
            )
            traj_group.create_dataset(
                "success", data=np.zeros(n_actions, dtype=bool)
            )
            traj_group.create_dataset(
                "rewards", data=np.zeros(n_actions, dtype=np.float32)
            )

            obs_group = traj_group.create_group("obs")
            sensor_data = obs_group.create_group("sensor_data")

            base_cam = sensor_data.create_group("base_camera")
            base_cam.create_dataset("rgb", data=ep["base_camera_rgb"])

            hand_cam = sensor_data.create_group("hand_camera")
            hand_cam.create_dataset("rgb", data=ep["hand_camera_rgb"])

            if "state" in ep:
                obs_group.create_dataset("state", data=ep["state"])

            episodes_meta.append({
                "episode_id": i,
                "episode_seed": i,
                "control_mode": control_mode,
                "elapsed_steps": n_actions,
                "reset_kwargs": {"seed": i},
                "success": False,
            })

    meta = {
        "env_info": {
            "env_id": env_id,
            "env_kwargs": {
                "obs_mode": "state+rgb+depth",
                "control_mode": control_mode,
            },
        },
        "episodes": episodes_meta,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(all_episodes)} episodes to {h5_path}")
    print(f"Metadata saved to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ManiSkill play data")
    parser.add_argument("--env-id", default="StackCube-v1")
    parser.add_argument("--output-dir", default="data/maniskill_play/StackCube-v1")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--obs-mode", default="rgb+state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo-path", default=None,
        help="Path to expert demo HDF5 for noisy replay episodes",
    )
    parser.add_argument(
        "--demo-json", default=None,
        help="Path to expert demo JSON metadata",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.3,
        help="Noise scale for noisy demo replay",
    )
    parser.add_argument(
        "--policy-mix",
        default="0.3,0.4,0.3",
        help="Comma-separated mix ratios for random,noisy_demo,smooth_random policies",
    )
    parser.add_argument(
        "--save-format", choices=["maniskill", "episodes"], default="maniskill",
        help="Save as single ManiSkill HDF5 or individual episode files",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse policy mix
    mix = [float(x) for x in args.policy_mix.split(",")]
    assert len(mix) == 3, "Policy mix must have 3 values: random,noisy_demo,smooth_random"
    mix = np.array(mix)
    mix = mix / mix.sum()

    # Load demos if available
    demo_actions_list = None
    if args.demo_path and args.demo_json:
        print(f"Loading demos from {args.demo_path}...")
        demo_actions_list = load_demo_actions(args.demo_path, args.demo_json)
        print(f"Loaded {len(demo_actions_list)} demo episodes")
    else:
        # No demos, redistribute noisy_demo weight to others
        print("No demo path provided, using only random and smooth_random policies")
        mix[0] += mix[1] / 2
        mix[2] += mix[1] / 2
        mix[1] = 0.0

    policy_types = ["random", "noisy_demo", "smooth_random"]

    print(f"Creating {args.env_id} environment...")
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        max_episode_steps=args.episode_length + 10,
    )

    print(f"Collecting {args.num_episodes} episodes...")
    print(f"Policy mix: random={mix[0]:.1%}, noisy_demo={mix[1]:.1%}, smooth_random={mix[2]:.1%}")

    h5_name = f"trajectory.{args.obs_mode}.{args.control_mode}.physx_cpu.h5"
    json_name = f"trajectory.{args.obs_mode}.{args.control_mode}.physx_cpu.json"
    h5_path = os.path.join(args.output_dir, h5_name)
    json_path = os.path.join(args.output_dir, json_name)

    episodes_meta = []
    with h5py.File(h5_path, "w") as h5f:
        for ep_idx in tqdm(range(args.num_episodes)):
            policy_type = rng.choice(policy_types, p=mix)

            demo_actions = None
            if policy_type == "noisy_demo" and demo_actions_list:
                demo_idx = rng.integers(len(demo_actions_list))
                demo_actions = demo_actions_list[demo_idx]

            ep = collect_episode(
                env=env,
                policy_type=policy_type,
                episode_length=args.episode_length,
                rng=rng,
                demo_actions=demo_actions,
                noise_scale=args.noise_scale,
            )

            # Write directly to HDF5 with gzip compression on images
            traj_group = h5f.create_group(f"traj_{ep_idx}")
            n_actions = ep["actions"].shape[0]
            traj_group.create_dataset("actions", data=ep["actions"])
            traj_group.create_dataset("terminated", data=np.zeros(n_actions, dtype=bool))
            traj_group.create_dataset("truncated", data=np.zeros(n_actions, dtype=bool))
            traj_group.create_dataset("success", data=np.zeros(n_actions, dtype=bool))
            traj_group.create_dataset("rewards", data=np.zeros(n_actions, dtype=np.float32))

            obs_group = traj_group.create_group("obs")
            sensor_data = obs_group.create_group("sensor_data")
            base_cam = sensor_data.create_group("base_camera")
            base_cam.create_dataset(
                "rgb", data=ep["base_camera_rgb"],
                compression="gzip", compression_opts=5, chunks=(1, 128, 128, 3),
            )
            hand_cam = sensor_data.create_group("hand_camera")
            hand_cam.create_dataset(
                "rgb", data=ep["hand_camera_rgb"],
                compression="gzip", compression_opts=5, chunks=(1, 128, 128, 3),
            )
            if "state" in ep:
                obs_group.create_dataset("state", data=ep["state"])

            episodes_meta.append({
                "episode_id": ep_idx,
                "episode_seed": ep_idx,
                "control_mode": args.control_mode,
                "elapsed_steps": n_actions,
                "reset_kwargs": {"seed": ep_idx},
                "success": False,
            })

            # Flush periodically to avoid buffering too much
            if (ep_idx + 1) % 100 == 0:
                h5f.flush()

    meta = {
        "env_info": {
            "env_id": args.env_id,
            "env_kwargs": {
                "obs_mode": args.obs_mode,
                "control_mode": args.control_mode,
            },
        },
        "episodes": episodes_meta,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    env.close()
    print(f"Saved {args.num_episodes} episodes to {h5_path}")
    print("Done!")


if __name__ == "__main__":
    main()
