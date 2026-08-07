# Minimal VT-Bench adapter

This adapter evaluates **Full MMPFN** on VT-Bench's already-exported Adoption
or Breast Cancer splits.  It leaves `VT-Bench-main/` unchanged: the benchmark
remains the authority for data inclusion, preprocessing, and train/validation/
test partitioning.

## Required files

Pass the directory containing the files named by the corresponding VT-Bench
dataset configuration:

| Dataset | Split files |
| --- | --- |
| Adoption | `features_{train,valid,test}.csv`, `labels_{train,valid,test}.pt`, `paths_{train,valid,test}.pt`, `tabular_lengths.pt` |
| Breast Cancer | `{train,val,test}_features.csv`, `{train,val,test}_labels.pt`, `{train,val,test}_paths.pt`, `tabular_lengths.pt` |

`paths*.pt` must point to existing `.npy` images.  Adoption arrays are expected
to be in `[0, 1]`; Breast arrays are expected to be ImageNet-normalized, exactly
as written by the supplied VT-Bench preprocessing notebook.

The original MMPFN checkpoint must be placed at
`mmpfn/parameters/tabpfn-v2-classifier.ckpt`.  The DINOv2 ViT-B/14 checkpoint
is supplied explicitly.  Neither checkpoint is included in this repository.

The exported CSV files intentionally have no header.  By default the adapter
lets MMPFN infer categorical columns from the encoded values, avoiding an
assumption that the feature order matches `tabular_lengths.pt`.  If inspection
of a prepared dataset establishes the exact order, pass it explicitly with
`--categorical-indices 0 4 ...`.

## Run one fixed VT-Bench split

From `MultiModalPFN-main` after the environment and checkpoints are available:

```powershell
python -m mmpfn.run_vtbench `
  --dataset adoption `
  --data-root <VT_BENCH_ADOPTION_FEATURE_DIR> `
  --dino-checkpoint .\mmpfn\parameters\dinov2_vitb14_pretrain.pth
```

Replace `adoption` and `--data-root` with `breast` and its feature directory for
Breast Cancer.  The command writes only under
`mmpfn/checkpoints/vtbench/<dataset>/seed_<seed>/`, including cached DINOv2
embeddings, the fine-tuned model, and `metrics.json`.

This is intentionally only the Full-MMPFN bridge.  Image-only, tabular-only,
and MCR ablations remain separate evaluation runs so their definitions can be
kept exactly aligned with VT-Bench.
