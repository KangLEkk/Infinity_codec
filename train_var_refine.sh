#!/usr/bin/env bash

set -x
export CUDA_VISIBLE_DEVICES=3
# set dist args
nproc_per_node=1


nnodes=1
node_rank=0
master_addr=127.0.0.1
master_port=12347

echo "[Using GPUs: $CUDA_VISIBLE_DEVICES, count: ${nproc_per_node}]"
# if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
#   echo "[single node alone] SINGLE=$SINGLE"
#   nnodes=1
#   node_rank=0
#   nproc_per_node=1
#   master_addr=127.0.0.1
#   master_port=12345
# else
#   MASTER_NODE_ID=0
#   nnodes=${ARNOLD_WORKER_NUM}
#   node_rank=${ARNOLD_ID}
#   master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
#   master_addr=${!master_addr}
#   master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
#   master_port=${!master_port}
#   ports=(`echo $master_port | tr ',' ' '`)
#   master_port=${ports[0]}
# fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CONDITION_TYPE=${CONDITION_TYPE:-depth}

swanlab online
exp_name=stage2_1024_125M_16vae_${CONDITION_TYPE}_multi
bed_path=checkpoints/${exp_name}/
local_out_path=$LOCAL_OUT/${exp_name}

# CONDITION_TYPE=depth 或 sam。SAM 在线 mask generation 明显更慢，建议先小 batch/少图 smoke test。
CONDITION_ARGS=(
  --enable_boundary_condition 1
  --condition_codec_type vae_token
  --condition_token_scales_min ${CONDITION_TOKEN_SCALES_MIN:-0}
  --condition_token_scales ${CONDITION_TOKEN_SCALES_MAX:-3}
  --condition_lr ${CONDITION_LR:-2e-4}
  --boundary_cond_rate_ratio 0
  --boundary_cond_recon_ratio 0
  --condition_adapter_init per_scale_zero
)
if [[ "${CONDITION_TYPE}" == "sam" || "${CONDITION_TYPE}" == "seg" || "${CONDITION_TYPE}" == "segmentation" ]]; then
  CONDITION_ARGS+=(
    --spatial_cond_type sam
    --seg_condition_source transformers
    --seg_model_name ${SEG_MODEL_NAME:-facebook/sam-vit-base}
    --seg_model_dtype ${SEG_MODEL_DTYPE:-fp32}
    --seg_max_masks ${SEG_MAX_MASKS:-8}
    --seg_points_per_batch ${SEG_POINTS_PER_BATCH:-16}
    --seg_output_mode ${SEG_OUTPUT_MODE:-region_boundary}
  )
else
  CONDITION_ARGS+=(
    --spatial_cond_type depth
    --depth_condition_source transformers
    --depth_model_name ${DEPTH_MODEL_NAME:-depth-anything/Depth-Anything-V2-Small-hf}
    --depth_model_dtype ${DEPTH_MODEL_DTYPE:-fp32}
  )
fi

# ====== 改成你的 processed dataset 路径 ======
proc_data_path='/datasets/pixelprose/embedding_mmap'
# 例如只训 256 分辨率就写 256；多尺度可写 256,320,384
proc_res_list='1024'

rm -rf ${bed_path}
# rm -rf ${local_out_path}

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train_stage2_var_entropy_patched.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--exp_name=${exp_name} \
--tblr=6e-5 \
--pn 1M \
--model=layer12c4 \
--lbs=16 \
--workers=1 \
--Ct5=2048 \
--vae_type 16 \
--vae_ckpt=/workspace/CKPT/Infinity/infinity_vae_d16.pth \
--rush_resume=/workspace/CKPT/Infinity/infinity_125M_256x256.pth \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 5000 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
\
--online_t5=0 \
--use_streaming_dataset=0 \
--dataset_backend='proc_memmap' \
--proc_data_path=${proc_data_path} \
--proc_res_list=${proc_res_list} \
--proc_memmap_cache_size=2 \
\
--enable_student_entropy=0 \
--student_start_step=50000 \
--student_hidden_dim=256 \
--student_depth=4 \
--student_dropout=0.0 \
--student_lr=2e-4 \
--student_wd=1e-4 \
--student_grad_clip=1.0 \
--student_start_scale=1 \
--student_kd_ratio=1.0 \
--student_gt_ratio=1.0 \
\
"${CONDITION_ARGS[@]}"
