WORKER_GPU=2
NUM_WORKERS=0

if [[ "$*" == *"--debug"* ]]; then
    WORKER_GPU=1
    NUM_WORKERS=0
fi

PROC_DIR=${PROC_DIR:-"/datasets/pixelprose/embedding_mmap"}
PROC_VAL_DIR=${PROC_VAL_DIR:-"/datasets/pixelprose/div2k_test_embedding_mmap"}
OUTDIR=${OUTDIR:-"outputs/bitvae_tok_stage1_rdvq_group_bs8_swanlab"}

CUDA_VISIBLE_DEVICES=1,3 torchrun \
    --nproc_per_node=$WORKER_GPU --master_port 29500 \
    train_tokenizer_swanlab.py --num_workers $NUM_WORKERS \
    --patch_size 16 \
    --base_ch 128 --encoder_ch_mult 1 2 4 4 4 --decoder_ch_mult 1 2 4 4 4 \
    --codebook_dim 32 \
    --optim_type AdamW \
    --lr 1e-4 \
    --base_lr 1e-4 \
    --base_batch_size 8 \
    --lr_scale_mode linear \
    --disable_sch \
    --max_steps 600000 \
    --resolution 1024 1024 \
    --batch_size 4 \
    --dataset_list proc_memmap \
    --proc_dir ${PROC_DIR} \
    --proc_val_dir ${PROC_VAL_DIR} \
    --dataaug "resizecrop" \
    --multiscale_training \
    --disc_layers 3 --discriminator_iter_start 0 \
    --l1_weight 1 --perceptual_weight 1 \
    --image_disc_weight 1 --image_gan_weight 0.3 --gan_feat_weight 0 --lfq_weight 4 \
    --codebook_size 4294967296 --entropy_loss_weight 0.1 --diversity_gamma 1 \
    --default_root_dir ${OUTDIR} --log_every 100 --ckpt_every 10000 --visu_every 5000 \
    --new_quant --lr_drop 45000 \
    --remove_residual_detach --use_lecam_reg_zero --base_ch_disc 128 --dis_lr_multiplier 2.0 --use_checkpoint \
    --schedule_mode "dynamic" --use_stochastic_depth --drop_rate 0.5 --keep_last_quant \
    --tokenizer 'flux' --quantizer_type 'MultiScaleBSQ' \
    --pretrained "/workspace/CKPT/Infinity/infinity_vae_d32reg.pth" --not_load_optimizer \
    \
    --use_swanlab \
    --swanlab_project "Tokenizer" \
    --swanlab_experiment_name "bitvae_tok_stage1_rdvq_group_bs8" \
    --swanlab_mode cloud \
    \
    --rate_lambda 1.0 \
    --entropy_hidden 256 --entropy_resblocks 4 --entropy_heads 8 --entropy_mlp_ratio 4.0 \
    --entropy_use_pos2d \
    \
    --use_group_rate \
    --group_size 4 \
    --prior_base_slices 2 \
    --prior_max_slices 8 \
    \
    --dino_weight 0.1 \
    --dino_max_scale 6 \
    --dino_every 1 \
    --dino_backend hf \
    --dino_model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --dino_feat_dim 768 \
    --dino_input_size 256 \
    --dino_amp \
    \
    --coarse_prefix_sample \
    --coarse_prefix_scales 2 4 6 8 13 \
    --coarse_prefix_full_prob 0.5 \
    --dino_scales 2 4 6 8 13 \
    --dino_scale_decay 0.7 \
    $@
