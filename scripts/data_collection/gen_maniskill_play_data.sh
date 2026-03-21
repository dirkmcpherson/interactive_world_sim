#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_ID="${ENV_ID:-StackCube-v1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH="${EPISODE_LENGTH:-200}"
CONTROL_MODE="${CONTROL_MODE:-pd_ee_delta_pos}"
OBS_MODE="${OBS_MODE:-rgb+state}"
SEED="${SEED:-42}"
NOISE_SCALE="${NOISE_SCALE:-0.3}"
POLICY_MIX="${POLICY_MIX:-0.3,0.4,0.3}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/maniskill_play/${ENV_ID}}"
DEMO_DIR="${DEMO_DIR:-${HOME}/.maniskill/demos/${ENV_ID}/motionplanning}"
DEMO_H5="${DEMO_DIR}/trajectory.state+rgb+depth.${CONTROL_MODE}.physx_cpu.h5"
DEMO_JSON="${DEMO_DIR}/trajectory.state+rgb+depth.${CONTROL_MODE}.physx_cpu.json"

echo "=== ManiSkill Play Data Generation ==="
echo "Environment:    ${ENV_ID}"
echo "Episodes:       ${NUM_EPISODES}"
echo "Episode length: ${EPISODE_LENGTH}"
echo "Control mode:   ${CONTROL_MODE}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Demo path:      ${DEMO_H5}"
echo ""

DEMO_ARGS=""
if [[ -f "$DEMO_H5" && -f "$DEMO_JSON" ]]; then
    echo "Found expert demos, will use for noisy replay."
    DEMO_ARGS="--demo-path ${DEMO_H5} --demo-json ${DEMO_JSON}"
else
    echo "No expert demos found at ${DEMO_DIR}, using random+smooth policies only."
fi

mkdir -p "$OUTPUT_DIR"

python "${SCRIPT_DIR}/collect_maniskill_play_data.py" \
    --env-id "$ENV_ID" \
    --output-dir "$OUTPUT_DIR" \
    --num-episodes "$NUM_EPISODES" \
    --episode-length "$EPISODE_LENGTH" \
    --control-mode "$CONTROL_MODE" \
    --obs-mode "$OBS_MODE" \
    --seed "$SEED" \
    --noise-scale "$NOISE_SCALE" \
    --policy-mix "$POLICY_MIX" \
    --save-format maniskill \
    $DEMO_ARGS

echo ""
echo "=== Done ==="
echo "Play data saved to: ${OUTPUT_DIR}"
echo ""
echo "To train the world model:"
echo "  python main.py dataset=maniskill_dataset \\"
echo "    dataset.dataset_path=${OUTPUT_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.h5 \\"
echo "    dataset.json_path=${OUTPUT_DIR}/trajectory.${OBS_MODE}.${CONTROL_MODE}.physx_cpu.json \\"
echo "    algorithm.training_stage=1 +name=${ENV_ID}_stage1"
