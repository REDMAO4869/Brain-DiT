# Stage-3 Raw Downstream

Downstream training/evaluation for raw-signal Stage-1 checkpoints.

## Supports

- `linear_probe`
- `lora`
- `full_finetune`
- unified `general` runner

## Core idea

For each sample, add DDPM noise at selected timestep(s), extract hidden tokens from selected DiT blocks, aggregate, then feed task head.

## Run

```bash
python -m downstream.run --mode general --config downstream/configs/stage3_raw_general.yaml
```

Modes in config:

- `mode: linear_probe`
- `mode: lora`
- `mode: full_finetune`

## Notes

- `embedding.capture_layers` supports negative indexing.
- `embedding.t_list` supports multi-timestep aggregation.
- `ckpt.stage1_ckpt` should point to a checkpoint produced by `train/train_stage1_raw.py`.
