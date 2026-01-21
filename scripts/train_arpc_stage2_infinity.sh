#!/usr/bin/env bash
set -e

# Stage-2 (Probability model): train the bitwise Infinity VAR on BSQ-VAE bit labels.
#
# This follows Infinity's native trainer/CLI as closely as possible.
#
# NOTE:
#  - This is a template; adjust DATA_PATH / output / compute settings.
#  - For ARPC, you want: --use_bit_label 1 and a BSQ-VAE ckpt.

DATA_PATH=${DATA_PATH:-"/path/to/dataset.json"}
OUT_DIR=${OUT_DIR:-"./checkpoints_arpc_infinity"}
VAE_CKPT=${VAE_CKPT:-"./checkpoints_arpc_vae/ckpt_last.pt"}

# Resolution schedule (pn): 0.06M->256, 0.25M->512, 0.60M->768, 1M->1024
PN=${PN:-"1M"}
H_DIV_W_TEMPLATE=${H_DIV_W_TEMPLATE:-1.0}

# model size
MODEL_TYPE=${MODEL_TYPE:-"infinity_2b"}

torchrun --nproc_per_node ${NPROC_PER_NODE:-8} train.py \
  --data_path "${DATA_PATH}" \
  --exp_dir "${OUT_DIR}" \
  --model_type "${MODEL_TYPE}" \
  --pn "${PN}" \
  --h_div_w_template ${H_DIV_W_TEMPLATE} \
  --vae_path "${VAE_CKPT}" \
  --vae_type 32 \
  --use_bit_label 1 \
  --apply_spatial_patchify 0 \
  --use_flex_attn 0 \
  --rope2d_each_sa_layer 1 \
  --rope2d_normalized_by_hw 2 \
  --text_encoder_ckpt "google/flan-t5-xl" \
  --tlen 512 \
  --bs 64 \
  --lr 1e-4 \
  --epochs 50 \
  --bf16 1

echo "[OK] stage2 finished. ckpts are under ${OUT_DIR}"
