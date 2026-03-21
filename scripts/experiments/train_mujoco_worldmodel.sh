#!/bin/bash
# Train a 3-stage latent world model on MuJoCo PushT data.
#
# Usage:
#   ./scripts/experiments/train_mujoco_worldmodel.sh [--debug] [--stage {1,2,3,all}]
#
# --debug: smoke-test on a 3060 (12GB). Tiny batch/steps, offline wandb.
# --stage: which stage to run (default: all, sequentially)
#
# After each stage, the script prints the checkpoint path for the next stage.
# For stages 2 and 3 you must supply the previous checkpoint via STAGE1_CKPT / STAGE2_CKPT
# environment variables (or the script will look for outputs/latest-run/checkpoints/best.ckpt).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── defaults ──────────────────────────────────────────────────────────────────
DEBUG=0
STAGE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=1; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── shared config ─────────────────────────────────────────────────────────────
DATASET=sim_aloha_dataset
if [[ $DEBUG -eq 1 ]]; then
    DATASET_DIR=data/pusht_mujoco_debug
else
    DATASET_DIR=data/pusht_mujoco
fi
OBS_KEYS="[top_pov]"
LATENT_DIM=512
ACTION_DIM=4

WANDB_ENTITY="${WANDB_ENTITY:-local}"

if [[ $DEBUG -eq 1 ]]; then
    echo "=== DEBUG MODE (smoke test) ==="
    S1_BATCH=1;  S1_STEPS=50;   S1_VAL_EVERY=25;  S1_LIMIT=1;   S1_VAL_BATCH=1;  S1_WORKERS=1; S1_CKPT_EVERY=25
    S2_BATCH=1;  S2_STEPS=50;   S2_VAL_EVERY=25;  S2_LIMIT=1;   S2_VAL_BATCH=1;  S2_WORKERS=1; S2_CKPT_EVERY=25; S2_VAL_HORIZON=10
    S3_BATCH=1;  S3_STEPS=50;   S3_VAL_EVERY=25;  S3_LIMIT=1;   S3_VAL_BATCH=1;  S3_WORKERS=1; S3_CKPT_EVERY=25; S3_VAL_HORIZON=10
    LOG_EVERY=10
    WANDB_MODE=offline
else
    S1_BATCH=1;  S1_STEPS=200005; S1_VAL_EVERY=6000;  S1_LIMIT=1.0; S1_VAL_BATCH=10; S1_WORKERS=0; S1_CKPT_EVERY=10000
    S2_BATCH=4;  S2_STEPS=200005; S2_VAL_EVERY=30000; S2_LIMIT=1.0; S2_VAL_BATCH=2;  S2_WORKERS=4; S2_CKPT_EVERY=10000; S2_VAL_HORIZON=200
    S3_BATCH=16; S3_STEPS=200005; S3_VAL_EVERY=30000; S3_LIMIT=1.0; S3_VAL_BATCH=2;  S3_WORKERS=4; S3_CKPT_EVERY=10000; S3_VAL_HORIZON=200
    LOG_EVERY=100
    WANDB_MODE=offline
fi

# ── helper ────────────────────────────────────────────────────────────────────
find_best_ckpt() {
    # Try best.ckpt first, then latest checkpoint by modification time
    local dir="$1"
    if [[ -f "$dir/best.ckpt" ]]; then
        echo "$dir/best.ckpt"
    else
        ls -t "$dir"/*.ckpt 2>/dev/null | head -1
    fi
}

# ── Stage 1: Autoencoder ─────────────────────────────────────────────────────
run_stage1() {
    echo ""
    echo "========================================"
    echo "  Stage 1: Autoencoder (encoder+decoder)"
    echo "========================================"
    python main.py +name=mujoco_pusht_stage1 \
        algorithm=latent_world_model \
        experiment=exp_latent_dyn \
        dataset=$DATASET \
        dataset.dataset_dir=$DATASET_DIR \
        dataset.horizon=1 dataset.val_horizon=1 \
        "dataset.obs_keys=$OBS_KEYS" \
        experiment.training.batch_size=$S1_BATCH \
        experiment.training.max_steps=$S1_STEPS \
        experiment.training.log_every_n_steps=$LOG_EVERY \
        experiment.validation.limit_batch=$S1_LIMIT \
        experiment.validation.batch_size=$S1_VAL_BATCH \
        experiment.validation.val_every_n_step=$S1_VAL_EVERY \
        algorithm.latent_dim=$LATENT_DIM algorithm.action_dim=$ACTION_DIM \
        experiment.training.checkpointing.every_n_train_steps=$S1_CKPT_EVERY \
        algorithm.training_stage=1 \
        wandb.mode=$WANDB_MODE \
        wandb.entity=$WANDB_ENTITY

    STAGE1_CKPT_DIR="$(readlink -f outputs/latest-run/checkpoints)"
    STAGE1_CKPT="$(find_best_ckpt "$STAGE1_CKPT_DIR")"
    echo "Stage 1 checkpoint: $STAGE1_CKPT"
    export STAGE1_CKPT
}

# ── Stage 2: Latent dynamics ─────────────────────────────────────────────────
run_stage2() {
    if [[ -z "${STAGE1_CKPT:-}" ]]; then
        STAGE1_CKPT="$(find_best_ckpt outputs/latest-run/checkpoints)"
    fi
    if [[ -z "$STAGE1_CKPT" || ! -f "$STAGE1_CKPT" ]]; then
        echo "ERROR: Stage 1 checkpoint not found. Set STAGE1_CKPT or run stage 1 first."
        exit 1
    fi
    echo ""
    echo "========================================"
    echo "  Stage 2: Latent dynamics model"
    echo "  Loading AE from: $STAGE1_CKPT"
    echo "========================================"
    python main.py +name=mujoco_pusht_stage2 \
        algorithm=latent_world_model \
        experiment=exp_latent_dyn \
        dataset=$DATASET \
        dataset.dataset_dir=$DATASET_DIR \
        dataset.horizon=10 dataset.val_horizon=$S2_VAL_HORIZON \
        "dataset.obs_keys=$OBS_KEYS" \
        experiment.training.batch_size=$S2_BATCH \
        experiment.training.max_steps=$S2_STEPS \
        experiment.training.log_every_n_steps=$LOG_EVERY \
        experiment.validation.limit_batch=$S2_LIMIT \
        experiment.validation.batch_size=$S2_VAL_BATCH \
        experiment.validation.val_every_n_step=$S2_VAL_EVERY \
        experiment.training.checkpointing.every_n_train_steps=$S2_CKPT_EVERY \
        experiment.training.data.num_workers=$S2_WORKERS \
        experiment.validation.data.num_workers=$S2_WORKERS \
        algorithm.latent_dim=$LATENT_DIM algorithm.action_dim=$ACTION_DIM \
        algorithm.noise_scheduler.loss_weighting=uniform \
        algorithm.sampling_strategy=terminal_only \
        "algorithm.load_ae='$STAGE1_CKPT'" \
        algorithm.training_stage=2 \
        wandb.mode=$WANDB_MODE \
        wandb.entity=$WANDB_ENTITY

    STAGE2_CKPT_DIR="$(readlink -f outputs/latest-run/checkpoints)"
    STAGE2_CKPT="$(find_best_ckpt "$STAGE2_CKPT_DIR")"
    echo "Stage 2 checkpoint: $STAGE2_CKPT"
    export STAGE2_CKPT
}

# ── Stage 3: Decoder finetuning ──────────────────────────────────────────────
run_stage3() {
    if [[ -z "${STAGE2_CKPT:-}" ]]; then
        STAGE2_CKPT="$(find_best_ckpt outputs/latest-run/checkpoints)"
    fi
    if [[ -z "$STAGE2_CKPT" || ! -f "$STAGE2_CKPT" ]]; then
        echo "ERROR: Stage 2 checkpoint not found. Set STAGE2_CKPT or run stage 2 first."
        exit 1
    fi
    echo ""
    echo "========================================"
    echo "  Stage 3: Decoder finetuning"
    echo "  Loading AE from: $STAGE2_CKPT"
    echo "========================================"
    python main.py +name=mujoco_pusht_stage3 \
        algorithm=latent_world_model \
        experiment=exp_latent_dyn \
        dataset=$DATASET \
        dataset.dataset_dir=$DATASET_DIR \
        dataset.horizon=1 dataset.val_horizon=$S3_VAL_HORIZON \
        "dataset.obs_keys=$OBS_KEYS" \
        experiment.training.batch_size=$S3_BATCH \
        experiment.training.max_steps=$S3_STEPS \
        experiment.training.log_every_n_steps=$LOG_EVERY \
        experiment.validation.limit_batch=$S3_LIMIT \
        experiment.validation.batch_size=$S3_VAL_BATCH \
        experiment.validation.val_every_n_step=$S3_VAL_EVERY \
        experiment.training.checkpointing.every_n_train_steps=$S3_CKPT_EVERY \
        experiment.training.data.num_workers=$S3_WORKERS \
        experiment.validation.data.num_workers=$S3_WORKERS \
        algorithm.latent_dim=$LATENT_DIM algorithm.action_dim=$ACTION_DIM \
        algorithm.noise_scheduler.loss_weighting=uniform \
        algorithm.sampling_strategy=terminal_only \
        "algorithm.load_ae='$STAGE2_CKPT'" \
        algorithm.training_stage=3 \
        wandb.mode=$WANDB_MODE \
        wandb.entity=$WANDB_ENTITY

    STAGE3_CKPT_DIR="$(readlink -f outputs/latest-run/checkpoints)"
    STAGE3_CKPT="$(find_best_ckpt "$STAGE3_CKPT_DIR")"
    echo ""
    echo "========================================"
    echo "  Training complete!"
    echo "  Final world model checkpoint: $STAGE3_CKPT"
    echo "========================================"
    export STAGE3_CKPT
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "$STAGE" in
    1)   run_stage1 ;;
    2)   run_stage2 ;;
    3)   run_stage3 ;;
    all) run_stage1; run_stage2; run_stage3 ;;
    *)   echo "Unknown stage: $STAGE"; exit 1 ;;
esac
