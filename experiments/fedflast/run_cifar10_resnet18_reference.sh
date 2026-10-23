#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

DENSITY="${1:-0.1}"
SEED="${2:-0}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT_PATH="$REPO_ROOT/weights/fedflast_cifar10_resnet18_density$(python -c 'import sys; print(f"{int(round(float(sys.argv[1])*100)):03d}")' "$DENSITY")_seed${SEED}.pt"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

bash run_fedflast_distributed_pytorch.sh \
  resnet18 cifar10 100 10 500 5 "$DENSITY" 0.001 \
  --batch_size 64 \
  --partition_alpha 0.5 \
  --delta_T 10 \
  --T_end 300 \
  --adjust_alpha 0.2 \
  --num_eval 128 \
  --frequency_of_the_test 10 \
  --client_optimizer adam \
  --wd 1e-3 \
  --seed "$SEED" \
  --save_final_checkpoint true \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --use_fedflast_perturbation true \
  --flast_rho 0.05 \
  --flast_momentum 0.05 \
  --flast_server_lr 1.0 \
  --use_flast_client_sam_prune true \
  --flast_local_rho 0.05 \
  --flast_candidate_multiplier 2.0 \
  --flast_local_sam_norm l2
