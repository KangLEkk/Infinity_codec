#!/usr/bin/env bash
set -e

# Stage-1 (Tokenizer): BSQ-VAE training with GM-BMSRQ + SRD
#
# NOTE:
#  - This is a template. Please edit DATA_ROOT / OUT_DIR / GPU settings.
#  - Use torchrun for multi-GPU.

DATA_ROOT=${DATA_ROOT:-"/path/to/images"}
OUT_DIR=${OUT_DIR:-"./checkpoints_arpc_vae"}

# model/tokens
CODEBOOK_DIM=${CODEBOOK_DIM:-32}   # 16 or 32
IMAGE_SIZE=${IMAGE_SIZE:-256}      # phase-1 resolution

# training
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
PHASE1_STEPS=${PHASE1_STEPS:-200000}
PHASE2_STEPS=${PHASE2_STEPS:-100000}
PHASE2_RESOS=${PHASE2_RESOS:-"256,512,1024"}

torchrun --nproc_per_node ${NPROC_PER_NODE:-8} scripts/train_vae_arpc.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --codebook_dim ${CODEBOOK_DIM} \
  --image_size ${IMAGE_SIZE} \
  --phase1_steps ${PHASE1_STEPS} \
  --phase2_steps ${PHASE2_STEPS} \
  --phase2_resos ${PHASE2_RESOS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --amp

echo "[OK] stage1 finished. ckpts are under ${OUT_DIR}"
