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
| Skin Cancer | implemented | not run | Public PAD-UFES-20 source was downloaded into user-owned storage and exported as a stratified 6-class 80/10/10 split (1,838 / 230 / 230). |
| Infarction | implemented | run completed | Uses the read-only UK Biobank source and user-owned prepared files. |
| Pneumonia | implemented | run completed | Uses user-owned prepared MIMIC-CXR image inputs. |
| Length of Stay | implemented | not run | User-owned MIMIC-IV v2.2 adapter prepared 64,044 aligned image-tabular rows (train 51,782 / val 6,188 / test 6,074). Three CXR JPEGs are unavailable upstream (404) and were excluded before preprocessing. |
| Respiratory Rate | implemented | not run | MIMIC source was prepared, but no final three-mode run has completed. |
| Adoption | implemented | not run | Adapter exists; no final three-mode result is recorded. |
| DVM-Car | implemented | not run | 286-way vehicle-type adapter exported and image-validated under the user-owned DVM-Car feature directory. |
| CelebA | implemented | not run | Public CelebA source was downloaded into user-owned storage and exported as a stratified 80/10/10 split (162,079 / 20,260 / 20,260) for `Attractive` prediction from the other 39 attributes. |
| Pawpularity | implemented | run completed | Public Kaggle source was converted to the legacy VT-Bench layout. |
| Anime | implemented, source not downloaded | not run | Public Kaggle source is `dbdmobile/myanimelist-dataset`; the adapter preserves VT-Bench's image-URL eligibility, feature processing and 80/10/10 random split. |

## MulTaBench (`Text=0`, image--tabular)

The generic MulTaBench adapter is implemented in `mmpfn.prepare_multabench`.
Every dataset below is registered and can be attempted through
`mmpfn.run_benchmark_suite`; `not run` means no completed three-mode experiment
has been recorded yet.

| Dataset key | Dataset | Task | Adapter status | Experiment status |
|---|---|---|---|---|
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

## Original MMPFN text--tabular datasets

| Dataset | Adapter status | Experiment status | Notes |
|---|---|---|---|
| Airbnb | implemented, source required | not run | Uses the original MMPFN field selection and text concatenation; source location is supplied explicitly. |
| Salary | implemented, source required | not run | Uses the original MMPFN field selection and text concatenation; source location is supplied explicitly. |
| Cloth | implemented, source required | not run | Uses the original MMPFN field selection and text concatenation; source location is supplied explicitly. |

The original-paper PetFinder-I, PetFinder-t and PetFinder-A tasks are
intentionally **not** added in this adapter extension.

## Encoder comparison interface

The image encoder probe and MMPFN runner support frozen DINOv2-B, DINOv3-B,
CLIP ViT-L/14, ResNet-50 and supervised ViT-B/16.  MMPFN retains its released
TabPFN-v2 backbone; `mmpfn.run_tabpfn_v3` is a separate TabPFN-3 tabular-only
baseline.  The visual adapters are implemented but no formal multi-encoder
comparison has been completed yet.

## Next adapter-only work

1. Download the public Anime source/images and generate its user-owned feature export.
2. Obtain the original source folders for Airbnb, Salary and Cloth before preparing them.
3. Do not launch a formal run until a GPU is available and the requested
   evaluation protocol is confirmed.
