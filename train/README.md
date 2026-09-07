# Stage-1 Raw Diffusion

Raw-signal diffusion pretraining on fMRI windows `(T, ROI)`.

## Input/shape

- Loader output: `(B, T, N)`
- DiT input: `x2d = x.permute(0,2,1).unsqueeze(1)` -> `(B, 1, N, T)`

## Run

```bash
python train/train_stage1_raw.py --config train/configs/stage1_raw_dit_hcp.yaml
```

Optional DDP:

```bash
torchrun --nproc_per_node=2 train/train_stage1_raw.py --config train/configs/stage1_raw_dit_hcp.yaml --ddp
```

## Outputs

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `logs/metrics.csv`
- `logs/dataset_summary.txt`
- `config_used.yaml`

## Notes

Shared components are now under `core/` (`core/data`, `core/diffusion`).
