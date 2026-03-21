"""PushT MuJoCo experiments: 4 configurations comparing real sim vs world model.

Usage:
    python scripts/experiments/run_pusht_experiments.py \
        --experiment {1,2,3,4,all} \
        --checkpoint <path_to_stage3_ckpt> \
        [--debug] [--output_dir data/pusht_experiments] [--episode_idx 0]

Experiments:
    1  Random actions in MuJoCo sim
    2  Random actions in world model
    3  Demo replay in MuJoCo sim
    4  Demo replay in world model (+ side-by-side with exp 3)
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from einops import rearrange

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)


# ── Model loading (adapted from teleoperate_keyboard.py:39-70) ────────────────
def load_model(ckpt_path: str) -> LatentWorldModel:
    """Load a trained world model from checkpoint."""
    from omegaconf import OmegaConf

    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location="cuda:0",
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


# ── Video writing helper ──────────────────────────────────────────────────────
def make_video_writer(path: str, fps: int, size: tuple[int, int]) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def tensor_to_bgr(img_tensor: torch.Tensor, size: int = 256) -> np.ndarray:
    """Convert (3, H, W) float tensor in [0,1] to BGR uint8 numpy array."""
    img = img_tensor.permute(1, 2, 0).detach().cpu().float().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Load data episode ────────────────────────────────────────────────────────
def load_episode(data_dir: str, episode_idx: int) -> dict:
    """Load an episode from HDF5."""
    path = Path(data_dir) / "train" / f"episode_{episode_idx}.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Episode not found: {path}")
    with h5py.File(path, "r") as f:
        return {
            "action": f["action"][()],  # (T, 4)
            "images": f["obs"]["images"]["top_pov"][()],  # (T, 128, 128, 3)
        }


# ── Experiment 1: Random actions in MuJoCo sim ──────────────────────────────
def exp1_random_mujoco(
    output_dir: str,
    data_dir: str,
    episode_idx: int = 0,
    num_steps: int = 200,
) -> str:
    """Run random actions in the real MuJoCo PushT environment."""
    from interactive_world_sim.environments.sim_aloha_pusht_env import SimAlohaPushTEnv

    print("=== Experiment 1: Random actions in MuJoCo sim ===")
    env = SimAlohaPushTEnv(task="transfer_cube", render_size=(128, 128))

    # Initialize from a data episode
    episode = load_episode(data_dir, episode_idx)
    hdf5_path = str(Path(data_dir) / "train" / f"episode_{episode_idx}.hdf5")
    init_state = env.compute_init_state(hdf5_path, t=0)
    env.reset(state=init_state)

    video_path = f"{output_dir}/exp1_random_mujoco.mp4"
    vis_size = 256
    writer = make_video_writer(video_path, 30, (vis_size, vis_size))

    frames = []
    for step in range(num_steps):
        action = np.random.uniform(-0.1, 0.1, size=(4,))
        # Get current position and add random delta
        curr_pos = env.get_curr_pos()
        abs_action = curr_pos + action
        env.step(abs_action)
        img_dict = env.render(mode="human")
        img = img_dict["top_pov"]  # (128, 128, 3) RGB
        img_vis = cv2.resize(img, (vis_size, vis_size), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)
        frames.append(img)
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    writer.release()
    print(f"  Saved: {video_path}")
    return video_path


# ── Experiment 2: Random actions in world model ─────────────────────────────
def exp2_random_worldmodel(
    checkpoint: str,
    output_dir: str,
    data_dir: str,
    episode_idx: int = 0,
    num_steps: int = 200,
) -> str:
    """Run random actions through the trained world model."""
    print("=== Experiment 2: Random actions in world model ===")
    model = load_model(checkpoint)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype
    resolution = 128

    # Load initial frame from data
    episode = load_episode(data_dir, episode_idx)
    init_img = episode["images"][0]  # (128, 128, 3)
    init_action = episode["action"][0]  # (4,)

    # Encode initial frame
    img_tensor = torch.from_numpy(init_img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    img_tensor = normalizer["top_pov"].normalize(img_tensor).to(device)
    with torch.no_grad():
        curr_latent = model.encoder_forward(img_tensor)[:, None]  # (1, 1, C, H, W)

    curr_action = torch.from_numpy(init_action).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    video_path = f"{output_dir}/exp2_random_worldmodel.mp4"
    vis_size = 256
    writer = make_video_writer(video_path, 30, (vis_size, vis_size))

    # Render initial frame
    with torch.no_grad():
        rendered = render_img_cm(model, curr_latent[:, -1], resolution, normalizer, num_views=1)
    writer.write(tensor_to_bgr(rendered[0], vis_size))

    hist_context = 10
    action_hist = []
    for step in range(num_steps):
        # Random action in normalized space
        delta = np.random.uniform(-0.02, 0.02, size=(4,))
        curr_action = curr_action + torch.from_numpy(delta).to(device).float()
        curr_action = torch.clamp(curr_action, -1.0, 1.0)

        action_chunk = curr_action.reshape(1, -1)  # (1, 4)
        action_hist.append(action_chunk)
        action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]  # (T, 4)
        action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)

        with torch.no_grad():
            latent_pred = model.dynamics_forward(curr_latent, action)
        curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
        curr_latent = curr_latent[:, -hist_context:]

        # Render
        with torch.no_grad():
            rendered = render_img_cm(model, curr_latent[:, -1], resolution, normalizer, num_views=1)
        writer.write(tensor_to_bgr(rendered[0], vis_size))

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    writer.release()
    print(f"  Saved: {video_path}")
    return video_path


# ── Experiment 3: Demo replay in MuJoCo sim ─────────────────────────────────
def exp3_demo_mujoco(
    output_dir: str,
    data_dir: str,
    episode_idx: int = 0,
    num_steps: int = 200,
) -> tuple[str, list[np.ndarray]]:
    """Replay recorded actions from a training episode in MuJoCo sim."""
    from interactive_world_sim.environments.sim_aloha_pusht_env import SimAlohaPushTEnv

    print("=== Experiment 3: Demo replay in MuJoCo sim ===")
    env = SimAlohaPushTEnv(task="transfer_cube", render_size=(128, 128))

    episode = load_episode(data_dir, episode_idx)
    actions = episode["action"]  # (T, 4) — absolute EEF XY positions
    num_steps = min(num_steps, len(actions))

    hdf5_path = str(Path(data_dir) / "train" / f"episode_{episode_idx}.hdf5")
    init_state = env.compute_init_state(hdf5_path, t=0)
    env.reset(state=init_state)

    video_path = f"{output_dir}/exp3_demo_mujoco.mp4"
    vis_size = 256
    writer = make_video_writer(video_path, 30, (vis_size, vis_size))

    frames = []
    for step in range(num_steps):
        env.step(actions[step])
        img_dict = env.render(mode="human")
        img = img_dict["top_pov"]  # (128, 128, 3) RGB
        img_vis = cv2.resize(img, (vis_size, vis_size), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)
        frames.append(img)
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    writer.release()
    print(f"  Saved: {video_path}")
    return video_path, frames


# ── Experiment 4: Demo replay in world model ────────────────────────────────
def exp4_demo_worldmodel(
    checkpoint: str,
    output_dir: str,
    data_dir: str,
    episode_idx: int = 0,
    num_steps: int = 200,
    mujoco_frames: list[np.ndarray] | None = None,
) -> str:
    """Replay recorded actions through the trained world model."""
    print("=== Experiment 4: Demo replay in world model ===")
    model = load_model(checkpoint)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype
    resolution = 128

    episode = load_episode(data_dir, episode_idx)
    actions = episode["action"]  # (T, 4)
    init_img = episode["images"][0]  # (128, 128, 3)
    num_steps = min(num_steps, len(actions))

    # Encode initial frame
    img_tensor = torch.from_numpy(init_img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    img_tensor = normalizer["top_pov"].normalize(img_tensor).to(device)
    with torch.no_grad():
        curr_latent = model.encoder_forward(img_tensor)[:, None]  # (1, 1, C, H, W)

    curr_action = torch.from_numpy(actions[0]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    vis_size = 256
    do_side_by_side = mujoco_frames is not None
    video_path = f"{output_dir}/exp4_demo_worldmodel.mp4"
    sbs_path = f"{output_dir}/exp4_side_by_side.mp4"

    writer = make_video_writer(video_path, 30, (vis_size, vis_size))
    sbs_writer = None
    if do_side_by_side:
        sbs_writer = make_video_writer(sbs_path, 30, (vis_size * 2, vis_size))

    # Render initial frame
    with torch.no_grad():
        rendered = render_img_cm(model, curr_latent[:, -1], resolution, normalizer, num_views=1)
    wm_bgr = tensor_to_bgr(rendered[0], vis_size)
    writer.write(wm_bgr)
    if sbs_writer and len(mujoco_frames) > 0:
        mj_vis = cv2.resize(mujoco_frames[0], (vis_size, vis_size), interpolation=cv2.INTER_AREA)
        mj_bgr = cv2.cvtColor(mj_vis, cv2.COLOR_RGB2BGR)
        # Add labels
        cv2.putText(mj_bgr, "MuJoCo", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(wm_bgr, "World Model", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        sbs_writer.write(np.concatenate([mj_bgr, wm_bgr], axis=1))

    hist_context = 10
    action_hist = []
    for step in range(1, num_steps):
        # Use the recorded action (normalized)
        next_action = torch.from_numpy(actions[step]).to(device).float()
        next_action_norm = normalizer["action"].normalize(next_action)
        curr_action = next_action_norm

        action_chunk = curr_action.reshape(1, -1)  # (1, 4)
        action_hist.append(action_chunk)
        action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]
        action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)

        with torch.no_grad():
            latent_pred = model.dynamics_forward(curr_latent, action)
        curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
        curr_latent = curr_latent[:, -hist_context:]

        # Render
        with torch.no_grad():
            rendered = render_img_cm(model, curr_latent[:, -1], resolution, normalizer, num_views=1)
        wm_bgr = tensor_to_bgr(rendered[0], vis_size)
        writer.write(wm_bgr)

        if sbs_writer and step < len(mujoco_frames):
            mj_vis = cv2.resize(mujoco_frames[step], (vis_size, vis_size), interpolation=cv2.INTER_AREA)
            mj_bgr = cv2.cvtColor(mj_vis, cv2.COLOR_RGB2BGR)
            cv2.putText(mj_bgr, "MuJoCo", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(wm_bgr, "World Model", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            sbs_writer.write(np.concatenate([mj_bgr, wm_bgr], axis=1))

        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}")

    writer.release()
    if sbs_writer:
        sbs_writer.release()
        print(f"  Saved side-by-side: {sbs_path}")
    print(f"  Saved: {video_path}")
    return video_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="PushT MuJoCo Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to stage 3 world model checkpoint (required for exp 2 & 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pusht_experiments",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/pusht_mujoco",
        help="Path to MuJoCo PushT dataset",
    )
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index to use")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of steps per experiment")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 10 steps only")
    args = parser.parse_args()

    if args.debug:
        args.num_steps = 10
        print("=== DEBUG MODE: 10 steps ===")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    needs_ckpt = args.experiment in ["2", "4", "all"]
    if needs_ckpt and args.checkpoint is None:
        parser.error("--checkpoint is required for experiments 2, 4, and all")

    exp = args.experiment
    mujoco_frames = None

    if exp in ["1", "all"]:
        exp1_random_mujoco(args.output_dir, args.data_dir, args.episode_idx, args.num_steps)

    if exp in ["2", "all"]:
        exp2_random_worldmodel(args.checkpoint, args.output_dir, args.data_dir, args.episode_idx, args.num_steps)

    if exp in ["3", "all"]:
        _, mujoco_frames = exp3_demo_mujoco(args.output_dir, args.data_dir, args.episode_idx, args.num_steps)

    if exp in ["4", "all"]:
        # If running exp 4 standalone, we don't have mujoco_frames for side-by-side
        exp4_demo_worldmodel(
            args.checkpoint, args.output_dir, args.data_dir, args.episode_idx, args.num_steps, mujoco_frames
        )

    print("\nAll requested experiments complete!")
    print(f"Videos saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
