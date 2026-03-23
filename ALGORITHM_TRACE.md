# Algorithm Trace: Latent World Model

End-to-end trace of data flow through the codebase, from raw HDF5 episodes to interactive inference.

---

## 1. Entry Point

**`main.py`**

Hydra composes configs from `configurations/` (algorithm, dataset, experiment), then:

```
run() → run_local()
  → build_experiment(cfg)          # exp_latent_dyn.py
    → instantiate(cfg.algorithm)   # → LatentWorldModel
    → _build_dataset("training")   # → SimAlohaDataset
    → _build_dataset("validation")
  → experiment.exec_task("training")
    → Trainer.fit(model, train_dl, val_dl)
```

**Experiment class:** `LatentDynExperiment` (`interactive_world_sim/experiments/exp_latent_dyn.py`) maps `latent_world_model` → `LatentWorldModel` and `sim_aloha_dataset` → `SimAlohaDataset`.

---

## 2. Dataset Pipeline

**`interactive_world_sim/datasets/latent_dynamics/sim_aloha_dataset.py`**

### 2.1 HDF5 → Zarr Conversion

`_convert_real_to_dp_replay()` (lines 42–230) runs once, cached at `dataset_dir/train/cache.zarr.zip`:

```
episode_*.hdf5
  ├─ obs/ee_pos  → action (T, 4)    # XY positions of both EEFs
  ├─ obs/images/top_pov → (T, 128, 128, 3) uint8, /255 → float32
  └─ ...
→ zarr store:
    data/action         (N_total, 4)     float32, uncompressed
    data/top_pov        (N_total, 128, 128, 3)  uint8, JPEG2k compressed
    meta/episode_ends   (num_episodes,)  int64
```

Key: actions come from `obs/ee_pos`, not from the `action` field in the HDF5.

### 2.2 SequenceSampler

**`interactive_world_sim/utils/sampler.py`**

`create_indices()` (numba JIT, lines 10–54) builds a table of valid sliding windows:

```
(buffer_start, buffer_end, sample_start, sample_end)
```

Each window is `horizon * skip_frame` steps long. Boundary windows are padded by replicating edge frames. `keys_to_keep_intermediate=["action"]` preserves full temporal resolution for actions even when `skip_frame > 1`.

### 2.3 Batch Assembly

`_sample_to_data()` (lines 424–490):

```
RGB:    (T, H, W, C) uint8 → augment → /255 → (T, C, H, W) float32 ∈ [0, 1]
Action: (T, 4) float32
```

### 2.4 Normalization

`LinearNormalizer` (`interactive_world_sim/utils/normalizer.py`):

| Key       | Method                  | Transform            |
|-----------|------------------------|----------------------|
| `top_pov` | `get_image_range_normalizer` | `x * 2 - 1` → [-1, 1] |
| `action`  | `get_range_normalizer_from_stat` | min-max → [-1, 1]  |

Applied in `training_step` before any forward pass.

---

## 3. Model Architecture

**`interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py`**

### 3.1 Encoder (lines 96–108)

Simple conv stack, no attention:

```
Input:  (B, 3, 128, 128)         # single-view RGB in [-1, 1]
Conv2d(3, 4, 3×3, pad=1)
× num_latent_downsample (=2):
  SiLU → Conv2d(4, 4, 3×3, pad=1) → SiLU → Conv2d(4, 4, 3×3, stride=2)
Output: (B, 4, 32, 32)
```

Post-processing (`encoder_forward`, lines 194–214): L2-normalize each view's channels independently.

### 3.2 Decoder: CMDecoder

**`interactive_world_sim/algorithms/latent_dynamics/cm_decoder.py`**

Two sub-networks:

- **CMControlNet**: Takes encoder latent `z (B, 4, 32, 32)`, upsamples to image resolution, produces 13-element control signal list (one per residual block).
- **CMControlledUnetModel**: Standard denoising U-Net (`in_channels=3`, model_channels=64, attention at resolutions [2, 4, 8]). Receives control signals as additive residuals.

```
Forward:
  controls = control_net(x_t, t, s, z)     # z conditions generation
  x_pred   = unet(x_t, t, s, controls)     # denoised toward level s
```

### 3.3 Dynamics: CMLatentDynamics

**`interactive_world_sim/algorithms/latent_dynamics/cm_latent_dynamics.py`** (lines 116–352)

3D U-Net operating on latent sequences. Wrapped with `EinopsWrapper` to rearrange `(T, B, C, H, W) → (B, C, T, H, W)`.

```
Input: (B, C=4, T, H=32, W=32)

Embeddings:
  noise level → sinusoidal + MLP → (B, 512)
  action      → 3-layer MLP(4 → 64 → 128 → 512) → (B, T, 512)

Backbone:
  Conv3d(4, 64, kernel=(1,7,7))
  TemporalAttention (causal, rotary PE)
  Down blocks (dim_mults=[1,2]):
    × 2: ResnetBlock(FiLM conditioning on noise+action) → SpatialAttn → TemporalAttn(causal)
  Mid block:
    ResnetBlock → SpatialAttn → TemporalAttn → ResnetBlock
  Up blocks (skip connections from down):
    × 2: ResnetBlock → SpatialAttn → TemporalAttn → ResnetBlock
  Conv3d → (B, 4, T, 32, 32)

Output: (B, C=4, T, H=32, W=32)
```

Causal masking in temporal attention ensures frame `t` only attends to frames `≤ t`.

---

## 4. Diffusion Framework: Consistency Trajectory Matching (CTM)

**`interactive_world_sim/utils/cm_utils.py`**

### 4.1 Core Idea

CTM trains a model to predict the **velocity** `v` at any noise level `t`. From `v`, the clean signal `x_0` is recovered, then the trajectory is interpolated to any lower noise level `s < t`:

```
v = model(x_t, t, s)
x_0 = predict_start_from_v(x_t, t, v)
x_s = x_t * (s/t) + x_0 * (1 - s/t)
```

This allows variable denoising steps at inference — 1 step (fast) or many (accurate).

### 4.2 Noise Schedule

Sigmoid beta schedule over 1000 timesteps:

```python
betas = sigmoid(linspace(-6, 6, 1000))
alphas_cumprod = cumprod(1 - betas)
```

Forward process: `x_t = sqrt(α_t) · x_0 + sqrt(1 - α_t) · ε`, where `ε ~ N(0, I)` clamped to ±6.

### 4.3 Training: Dual Noise Levels

`add_noise_to_t_s()` generates the same noise at two levels:

```python
noise = randn_like(x).clamp(-6, 6)
x_t = q_sample(x, t, noise)   # noisier
x_s = q_sample(x, s, noise)   # cleaner, s < t
```

The model learns to map `x_t → x_s` (trajectory matching).

### 4.4 Loss Weighting: Fused SNR

```
SNR(t) = α_t / (1 - α_t)
cum_snr(t) = 0.96 · cum_snr(t-1) + 0.04 · clamp(SNR(t), max=5)
weight(t) = clipped_fused_snr(t) / fused_snr(t)
```

High-noise timesteps (easy) get low weight; low-noise timesteps (hard, detail-critical) get high weight.

---

## 5. Training Stages

### Stage 1: Autoencoder

**Trains:** encoder + decoder. **Frozen:** nothing.

```
training_step (lines 515–588):
  xs = batch["obs"]                          # (B, T, 3, 128, 128)
  xs = rearrange(xs, "b t c h w → (b t) c h w")
  z  = encoder(xs)                           # (B·T, 4, 32, 32)

  t, s = sample_noise_levels()               # random t > s > 0
  x_t, x_s = add_noise_to_t_s(xs, t, s)     # same noise, two levels

  pred_s = decoder(x_t, t, s, z)             # denoise from t to s
  loss = weighted_mse(pred_s, x_s)           # match trajectory
```

Goal: learn a latent space and decoder that can reconstruct observations.

### Stage 2: Latent Dynamics

**Trains:** dynamics model. **Frozen:** encoder, decoder.

```
training_step (lines 589–659):
  with torch.no_grad():
    z = encoder_forward(xs)                  # (B·T, 4, 32, 32)
  z = rearrange(z, "(b t) c h w → t b c h w", t=T)
  a = rearrange(action, "b t a → t b a")

  t, s = sample_noise_levels()
  z_t, z_s = add_noise_to_t_s(z, t, s)

  pred_s = dynamics(z_t, t, s, action=a)     # predict latent trajectory
  loss = weighted_mse(pred_s, z_s)
```

Goal: learn `z_{t+1} = f(z_t, action_t)` in latent space.

### Stage 3: Decoder Finetuning

**Trains:** decoder (at 0.1× LR). **Frozen:** encoder, dynamics.

Same as Stage 1 but encoder output is perturbed with small Gaussian noise (`σ=0.02`) to make the decoder robust to dynamics prediction errors.

---

## 6. Inference: dynamics_forward

**`latent_world_model.py` lines 264–345**

Autoregressive rollout with sliding window:

```
Input:
  z_0:    (B, 1, 4, 32, 32)    # encoded initial observation
  action: (B, T_total, 4)       # full action sequence

Algorithm:
  z_pred = [z_0]
  for each future step:
    chunk = randn(1, B, 4, 32, 32).clamp(-6, 6)     # noise initialization
    z_pred.append(chunk)

    window = z_pred[-n_tokens:]                       # sliding window
    for step_i in range(dyn_infer_steps):             # typically 1
      t = linspace(999, 0, dyn_infer_steps+1)[step_i]
      s = linspace(999, 0, dyn_infer_steps+1)[step_i+1]
      window[-1] = dynamics(window, t, s, action_window)

    L2_normalize(z_pred[-1])                          # per-view normalization

Output: (B, T_total, 4, 32, 32)  # predicted latent trajectory
```

---

## 7. Inference: Image Rendering

**`interactive_world_sim/algorithms/common/diffusion_helper.py`** `render_img_cm()` (lines 67–127)

Decodes a latent back to an image via iterative CTM denoising:

```
Input:  latent (B, 4, 32, 32)

x = randn(B, 3, 128, 128)                           # pure noise image
for step_i in range(dec_infer_steps):                # typically 3
  t = linspace(999, 0, dec_infer_steps+1)[step_i]
  s = linspace(999, 0, dec_infer_steps+1)[step_i+1]
  x = decoder(x, t, s, external_cond=latent)         # denoise one step

x = unnormalize(x)                                   # [-1, 1] → [0, 1]
Output: (B, 3, 128, 128) RGB ∈ [0, 1]
```

---

## 8. Interactive Teleoperation

**`scripts/inference/teleoperate_keyboard.py`**

`load_model()` (lines 39–70): loads checkpoint, sets `n_frames=10`, `sampling_timesteps=10` for fast inference.

Loop:
```
1. Capture initial frame from camera or dataset
2. z = encoder(frame)                      # (1, 4, 32, 32)
3. While running:
   a. Read keyboard/controller → action (4,)
   b. action_norm = normalizer["action"].normalize(action)
   c. z_next = dynamics_forward(z_history, action_history)
   d. image = render_img_cm(z_next)        # predicted next frame
   e. Display image
   f. Append z_next to history (sliding window)
```

---

## 9. Tensor Shape Reference

| Location | Variable | Shape |
|----------|----------|-------|
| DataLoader output | `batch["obs"]["top_pov"]` | `(B, T, 3, 128, 128)` |
| DataLoader output | `batch["action"]` | `(B, T, 4)` |
| After normalization | obs | `[-1, 1]`, action `[-1, 1]` |
| Encoder input | `xs` | `(B·T, 3, 128, 128)` |
| Encoder output | `z` | `(B·T, 4, 32, 32)` |
| Dynamics input | `z` | `(T, B, 4, 32, 32)` or `(B, 4, T, 32, 32)` via EinopsWrapper |
| Dynamics input | `action` | `(T, B, 4)` |
| Dynamics output | `z_pred` | `(T, B, 4, 32, 32)` |
| Decoder input | `x_t` (noisy image) | `(B, 3, 128, 128)` |
| Decoder input | `z` (condition) | `(B, 4, 32, 32)` |
| Decoder output | `x_pred` | `(B, 3, 128, 128)` |
| render_img_cm output | image | `(B, 3, 128, 128)` ∈ `[0, 1]` |

---

## 10. Config Reference

Key hyperparameters from `configurations/algorithm/latent_world_model.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `training_stage` | 1 | Which stage to train (1/2/3) |
| `num_latent_channel` | 4 | Per view |
| `num_latent_downsample` | 2 | 128 → 32 spatial |
| `latent_dim` | 512 | Dynamics embedding dim |
| `action_dim` | 4 | EEF XY × 2 arms |
| `timesteps` | 1000 | Diffusion schedule length |
| `sampling_timesteps` | 50 | DDIM steps at inference |
| `dyn_infer_steps` | 1 | Dynamics denoising steps |
| `dec_infer_steps` | 3 | Decoder denoising steps |
| `loss_weighting` | `fused_snr` | CTM weight scheme |
| `load_ae` | null | Stage 1 ckpt path (for stage 2+) |
