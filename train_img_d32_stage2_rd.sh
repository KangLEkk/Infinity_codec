WORKER_GPU=2
# NODE_NUM=1
NUM_WORKERS=0

if [[ "$*" == *"--debug"* ]]; then
    WORKER_GPU=1
    # NODE_NUM=1
    NUM_WORKERS=0
fi

# RD fine-tuning: add entropy-model rate term + coarse-prefix reconstruction
# Tips:
# - Start with a small rate_lambda (e.g., 1e-4 ~ 5e-4) and ramp up.
# - Use power weights to penalize fine scales more (push info to coarse scales).
# - Keep coarse_prefix_weight modest to avoid over-smoothing.

PROC_DIR=${PROC_DIR:-"/datasets/pixelprose/embedding_mmap"}
PROC_VAL_DIR=${PROC_VAL_DIR:-"/datasets/pixelprose/div2k_test_embedding_mmap"}
OUTDIR=${OUTDIR:-"outputs/bitvae_tok_stage1_dino0.1_8bs_dynamic"}

# --nnodes=$NODE_NUM \
# export MASTER_PORT=$((29500 + RANDOM % 100)) 
# --master_addr=$WORKER_0_HOST \
# --node_rank=$NODE_ID --master_port=$PORT \
#  --multiscale_training
# strict_entropy soft_nll
CUDA_VISIBLE_DEVICES=1,3 torchrun \
    --nproc_per_node=$WORKER_GPU  --master_port 29500\
    train_tokenizer.py --num_workers $NUM_WORKERS \
    --patch_size 16 \
    --base_ch 128 --encoder_ch_mult 1 2 4 4 4 --decoder_ch_mult 1 2 4 4 4 \
    --codebook_dim 32 \
    --optim_type AdamW --lr 1e-4 --disable_sch --max_steps 600000 \
    --resolution 1024 1024 --batch_size 4 --dataset_list proc_memmap --proc_dir ${PROC_DIR}  --proc_val_dir ${PROC_VAL_DIR} --dataaug "resizecrop" --multiscale_training \
    --disc_layers 3 --discriminator_iter_start 0 \
    --l1_weight 1 --perceptual_weight 1 --image_disc_weight 1 --image_gan_weight 0.3 --gan_feat_weight 0 --lfq_weight 4 \
    --codebook_size 4294967296 --entropy_loss_weight 0.1 --diversity_gamma 1 \
    --default_root_dir ${OUTDIR} --log_every 100 --ckpt_every 10000 --visu_every 5000 \
    --new_quant --lr_drop 45000 \
    --remove_residual_detach --use_lecam_reg_zero --base_ch_disc 128 --dis_lr_multiplier 2.0 --use_checkpoint \
    --schedule_mode "dynamic" --use_stochastic_depth --drop_rate 0.5 --keep_last_quant --tokenizer 'flux' --quantizer_type 'MultiScaleBSQ' \
    --pretrained "/workspace/CKPT/Infinity/infinity_vae_d32reg.pth" --not_load_optimizer \
    --rate_lambda 5.0 --rate_scale_mode uniform --rate_scale_alpha 2.0 --entropy_hidden 256 --entropy_resblocks 4 --entropy_heads 8 --entropy_mlp_ratio 4.0 --entropy_use_pos2d --prior_entropy_weight 0 --prior_entropy_start_scale 7 \
    --entropy_cond prev_sum  \
    --predict_objective soft_nll \
    --dino_weight 0.1 \
    --dino_max_scale 6 \
    --dino_every 1 \
    --dino_model dinov2_vits14 \
    --dino_feat_dim 768 \
    --dino_input_size 224 \
    --dino_amp \
    --coarse_prefix_sample \
    --coarse_prefix_scales 2 4 6 8 13 \
    --coarse_prefix_full_prob 0.5 \
    --dino_scales 2 4 6 8 13 \
    --dino_scale_decay 0.7
    $@

