import concurrent.futures
import copy
import gc
import json
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import psutil
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset

register_codecs()


def _convert_maniskill_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_path: str,
    json_path: str,
    val_ratio: float = 0.0,
    is_val: bool = False,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    """Convert ManiSkill HDF5 trajectory data to replay buffer format.

    ManiSkill stores all trajectories in a single HDF5 file with traj_0, traj_1, etc.
    This function reads the RGB observations and actions, converting them into the
    replay buffer format used by the Interactive World Simulator.
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    # Load JSON metadata to get episode list
    with open(json_path, "r") as f:
        meta = json.load(f)
    episodes_meta = meta["episodes"]
    n_total = len(episodes_meta)

    # Split into train/val
    if val_ratio > 0:
        n_val = max(1, int(n_total * val_ratio))
        if is_val:
            episode_indices = list(range(n_total - n_val, n_total))
        else:
            episode_indices = list(range(n_total - n_val))
    else:
        episode_indices = list(range(n_total))

    # Map from ManiSkill camera names to our obs keys
    # ManiSkill uses sensor_data/base_camera/rgb, sensor_data/hand_camera/rgb
    # We map these to the obs keys from shape_meta
    camera_key_map = {}
    for key in rgb_keys:
        if "base" in key:
            camera_key_map[key] = ("base_camera", "rgb")
        elif "hand" in key or "wrist" in key:
            camera_key_map[key] = ("hand_camera", "rgb")
    depth_camera_key_map = {}
    for key in depth_keys:
        if "base" in key:
            depth_camera_key_map[key] = ("base_camera", "depth")
        elif "hand" in key or "wrist" in key:
            depth_camera_key_map[key] = ("hand_camera", "depth")

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()

    with h5py.File(dataset_path, "r") as hdf5_file:
        for epi_idx in tqdm(episode_indices, desc="Loading ManiSkill episodes"):
            traj_key = f"traj_{epi_idx}"
            if traj_key not in hdf5_file:
                continue
            traj = hdf5_file[traj_key]

            # Actions: (T, action_dim)
            actions = traj["actions"][()]
            episode_length = actions.shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            if "action" not in lowdim_data_dict:
                lowdim_data_dict["action"] = list()
            lowdim_data_dict["action"].append(actions)

            # RGB observations: obs has T+1 frames, take first T to align with actions
            for obs_key, (cam_name, modality) in camera_key_map.items():
                if obs_key not in rgb_data_dict:
                    rgb_data_dict[obs_key] = list()
                imgs = traj["obs"]["sensor_data"][cam_name][modality][:episode_length]
                shape = tuple(shape_meta["obs"][obs_key]["shape"])
                c, h, w = shape
                # Resize if needed
                if imgs.shape[1] != h or imgs.shape[2] != w:
                    resized = [
                        cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                        for img in imgs
                    ]
                    imgs = np.stack(resized, axis=0)
                rgb_data_dict[obs_key].append(imgs)

            # Depth observations
            for obs_key, (cam_name, modality) in depth_camera_key_map.items():
                if obs_key not in depth_data_dict:
                    depth_data_dict[obs_key] = list()
                imgs = traj["obs"]["sensor_data"][cam_name][modality][:episode_length]
                shape = tuple(shape_meta["obs"][obs_key]["shape"])
                c, h, w = shape
                if imgs.shape[1] != h or imgs.shape[2] != w:
                    resized = [
                        cv2.resize(
                            img.squeeze(-1) if img.ndim == 3 else img,
                            (w, h),
                            interpolation=cv2.INTER_AREA,
                        )
                        for img in imgs
                    ]
                    imgs = np.stack(resized, axis=0)
                if imgs.ndim == 3:
                    imgs = imgs[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                depth_data_dict[obs_key].append(imgs)

            # Low-dim observations
            for key in lowdim_keys:
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                if "state" in traj["obs"]:
                    lowdim_data_dict[key].append(
                        traj["obs"]["state"][:episode_length]
                    )

    if not episode_ends:
        raise ValueError("No episodes found in the dataset!")

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def load_maniskill_replay_buffer(
    dataset_path: str,
    json_path: str,
    use_cache: bool,
    shape_meta: dict,
    cache_dir: str,
    val_ratio: float = 0.0,
    is_val: bool = False,
) -> ReplayBuffer:
    replay_buffer = None
    split = "val" if is_val else "train"
    if use_cache:
        cache_zarr_path = os.path.join(cache_dir, f"cache_maniskill_{split}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print(f"Cache does not exist. Creating for {split}!")
                    replay_buffer = _convert_maniskill_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_path=dataset_path,
                        json_path=json_path,
                        val_ratio=val_ratio,
                        is_val=is_val,
                    )
                    print("Saving cache to disk.")
                    os.makedirs(os.path.dirname(cache_zarr_path), exist_ok=True)
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    if os.path.exists(cache_zarr_path):
                        os.remove(cache_zarr_path)
                    raise e
            else:
                print(f"Loading cached ReplayBuffer ({split}) from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_maniskill_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            json_path=json_path,
            val_ratio=val_ratio,
            is_val=is_val,
        )
    return replay_buffer


class ManiSkillDataset(BaseImageDataset):
    """A dataset for ManiSkill demonstration data."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        shape_meta = cfg.shape_meta
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = cfg.aug_mode

        if cfg.aug_mode == "img_aug":
            from imgaug import augmenters as iaa

            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif cfg.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {cfg.aug_mode}")

        # ManiSkill uses a single HDF5 file with all trajectories
        dataset_path = cfg.dataset_path  # path to .h5 file
        json_path = cfg.json_path  # path to .json metadata
        cache_dir = cfg.get("cache_dir", os.path.dirname(dataset_path))
        val_ratio = cfg.get("val_ratio", 0.1)

        self.replay_buffer = load_maniskill_replay_buffer(
            dataset_path=dataset_path,
            json_path=json_path,
            use_cache=use_cache,
            shape_meta=shape_meta,
            cache_dir=cache_dir,
            val_ratio=val_ratio,
            is_val=False,
        )

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution
        self.dataset_path = dataset_path
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.val_ratio = cfg.get("val_ratio", 0.1)

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return self.replay_buffer.n_episodes // self.skip_idx
        else:
            return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.replay_buffer = load_maniskill_replay_buffer(
            dataset_path=self.dataset_path,
            json_path=self.json_path,
            use_cache=self.use_cache,
            shape_meta=self.shape_meta,
            cache_dir=self.cache_dir,
            val_ratio=self.val_ratio,
            is_val=True,
        )
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()

        apply_aug = np.random.random() < 0.2 if self.aug_mode == "img_aug" else False

        for key in self.rgb_keys:
            obs_images = sample[key].astype(np.uint8)
            final_images = sample[f"{key}_final"].astype(np.uint8)

            if apply_aug:
                aug_det = self.aug.to_deterministic()
                combined = [*obs_images, final_images]
                combined_aug = [aug_det.augment_image(img) for img in combined]
                obs_images = np.stack(combined_aug[:-1], axis=0)
                final_images = combined_aug[-1]

            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            final_dict[key] = (
                np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0
            )
            del sample[f"{key}_final"]
            del sample[key]

        for key in self.depth_keys:
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            final_dict[key] = (
                np.moveaxis(sample[f"{key}_final"], -1, 0).astype(np.float32) / 1000.0
            )
            del sample[f"{key}_final"]
            del sample[key]

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            val_horizon = self.val_horizon
            seq_end = min(epi_end, epi_start + val_horizon)
            sample = dict()
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:seq_end]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
                sample[f"{key}_final"] = sample[key][-1]
                sample["is_early_stop"] = False
                sample["rel_stop_idx"] = val_horizon - 1
        else:
            sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def test_maniskill_dataset() -> None:
    config_path = "configurations/dataset/maniskill_dataset.yaml"
    cfg = OmegaConf.load(config_path)
    dataset = ManiSkillDataset(cfg)
    print(f"Dataset length: {len(dataset)}")

    p = psutil.Process(os.getpid())

    def rss() -> float:
        return p.memory_info().rss / 1e9

    print(f"START RSS: {rss():.3f} GB")
    k = min(100, len(dataset))
    for i in range(k):
        data = dataset[i]
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    print(f"Sample obs keys: {list(data['obs'].keys())}")
    for key, val in data["obs"].items():
        print(f"  {key}: {val.shape} {val.dtype}")
    print(f"  action: {data['action'].shape} {data['action'].dtype}")

    val_dataset = dataset.get_validation_dataset()
    print(f"Validation dataset length: {len(val_dataset)}")
    for i in range(min(5, len(val_dataset))):
        data = val_dataset[i]
    print("Validation dataset success!")


if __name__ == "__main__":
    test_maniskill_dataset()
