# ARPC (Autoregressive-based Progressive Coding) in this repo

This project adds a **paper-style ARPC codec** on top of Infinity's bitwise VAR model.

- `codec/arpc_codec.py` : main codec (compress / decompress)
- `codec/arpc_bitstream.py` : `.arpc` container (v2) storing header + range-coded payload
- `scripts/arpc_cli.py` : CLI wrapper
- `scripts/arpc_eval_entropy_mask.py` : evaluation (bpp + PSNR/SSIM/MS-SSIM)

## 1. What is implemented

### Baseline ARPC (paper-aligned)
- Encode image into multi-scale BSQ **bitwise tokens**.
- For the first `k_transmit` scales, entropy-code active bits using Infinity's VAR probabilities.
- For later scales, reconstruct by autoregressive inference, while **forcing** the transmitted scales.

### Optional: no-training entropy masking (3 variants)
This is a drop-in replacement for handcrafted GM-BMSRQ grouping/masking in Stage-1.

All variants:
- **use VAR as entropy estimator** (no training)
- **mask low-entropy parts** (easy to predict)
- **only entropy-code unmasked tokens**
- **fill masked tokens** with deterministic argmax prediction

Available strategies:
- `none` : transmit all active bits (baseline)
- `entropy_channel` : keep top bit-channels (bit-planes) by entropy
- `entropy_scale` : derive per-scale keep ratio from mean entropy, then do channel masking
- `entropy_spatial` : keep top spatial positions by entropy (transmit all active bits there)

## 2. CLI

### Compress
```bash
python scripts/arpc_cli.py compress \
  --image input.png \
  --out out.arpc \
  --prompt "a clear photo" \
  --k_transmit 5 \
  --pn 1M \
  --vae_ckpt /path/to/vae.ckpt \
  --model_ckpt /path/to/infinity_ar.ckpt \
  --mask_strategy entropy_channel --keep_ratio 0.5
```

### Decompress
```bash
python scripts/arpc_cli.py decompress \
  --stream out.arpc \
  --out recon.png \
  --vae_ckpt /path/to/vae.ckpt \
  --model_ckpt /path/to/infinity_ar.ckpt
```

## 3. Evaluation

```bash
python scripts/arpc_eval_entropy_mask.py \
  --data_dir /path/to/kodak \
  --vae_ckpt /path/to/vae.ckpt \
  --model_ckpt /path/to/infinity_ar.ckpt \
  --strategies none,entropy_channel,entropy_spatial \
  --keep_ratios 0.25,0.5,0.75 \
  --k_list 1,3,5 \
  --pn 1M \
  --out_csv results.csv \
  --save_recon_dir recon
```

The output CSV contains per-image entries and can be grouped later for RD curves.

## 4. Notes
- Current implementation assumes **B=1**.
- Mask decisions are **deterministic** and do not require transmitting extra side information.
- `.arpc` is versioned; we use **v2** to store mask strategy/params.
