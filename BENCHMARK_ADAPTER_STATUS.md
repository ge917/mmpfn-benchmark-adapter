# Benchmark adapter status

This file distinguishes **adapter availability** from a completed MMPFN
experiment. A dataset is only marked as `run completed` after all three modes
(`full`, `image_only`, and `tabular_only`) have produced metrics on the target
server. Do not treat an adapter-only entry as a reported benchmark result.

Last updated: 2026-08-08.

## VT-Bench

| Dataset | Adapter status | Experiment status | Notes |
|---|---|---|---|
| Breast Cancer | implemented | run completed | Public CBIS-DDSM source was converted to the legacy VT-Bench layout. |
| Skin Cancer | not implemented | not run | Not present on server 09: expected `/mnt/hdd/jiazy/skin-cancer` with `metadata.csv` and `imgs/`. |
| Infarction | implemented | run completed | Uses the read-only UK Biobank source and user-owned prepared files. |
| Pneumonia | implemented | run completed | Uses user-owned prepared MIMIC-CXR image inputs. |
| Length of Stay | implemented | not run | User-owned MIMIC-IV v2.2 adapter prepared 64,044 aligned image-tabular rows (train 51,782 / val 6,188 / test 6,074). Three CXR JPEGs are unavailable upstream (404) and were excluded before preprocessing. |
| Respiratory Rate | implemented | not run | MIMIC source was prepared, but no final three-mode run has completed. |
| Adoption | implemented | not run | Adapter exists; no final three-mode result is recorded. |
| DVM-Car | implemented | not run | 286-way vehicle-type adapter exported and image-validated under the user-owned DVM-Car feature directory. |
| CelebA | not implemented | not run | Not present on server 09: expected `/mnt/hdd/jiazy/CelebA` with attributes CSV and images. |
| Pawpularity | implemented | run completed | Public Kaggle source was converted to the legacy VT-Bench layout. |
| Anime | not implemented | not run | Not present on server 09: expected `/data1/jiazy/anime` or an equivalent raw export. |

## MulTaBench (`Text=0`, image--tabular)

The generic MulTaBench adapter is implemented in `mmpfn.prepare_multabench`.
Every dataset below is registered and can be attempted through
`mmpfn.run_benchmark_suite`; `not run` means no completed three-mode experiment
has been recorded yet.

| Dataset key | Dataset | Task | Adapter status | Experiment status |
|---|---|---|---|---|
| `mt_cbis_ddsm` | CBIS-DDSM | classification | generic adapter | run completed (fold 0) |
| `mt_celeb_attractiveness` | Celeb Attractiveness | classification | generic adapter | not run |
| `mt_chexpert` | CheXpert | classification | generic adapter | not run |
| `mt_glaucoma_smdg` | Glaucoma SMDG | classification | generic adapter | not run |
| `mt_hateful_meme` | Hateful Meme | classification | generic adapter | not run |
| `mt_justin_instagram` | Justin Instagram | classification | generic adapter | not run |
| `mt_mammography_cmmd` | Mammography CMMD | classification | generic adapter | not run |
| `mt_zooscan_zooplankton` | Zooscan Plankton | classification | generic adapter | not run |
| `mt_amazon_bestseller` | Amazon Bestseller | regression | generic adapter | not run |
| `mt_mango_mass` | Mango Mass | regression | generic adapter | run completed (fold 0) |
| `mt_mkphoto_bots` | MkPhoto Bots | regression | generic adapter | not run |

## Next adapter-only work

1. Locate and inspect raw sources for Skin Cancer, Length of Stay, CelebA, and
   Anime before adding their adapters.
2. Do not launch a formal run until a GPU is available and the requested
   evaluation protocol is confirmed.
