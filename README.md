# HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction

Accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) 2026

[Project Page](https://xiliu8006.github.io/HAD-Project-website/) |
[Paper](https://arxiv.org/abs/2605.16873)

This repository contains the HAD training and evaluation code for sparse-view
3D reconstruction with hallucination-aware diffusion priors.

## Abstract

Diffusion priors can improve sparse-view 3D reconstruction by augmenting
training views at novel viewpoints, but they may also introduce hallucinated
content that is inconsistent with the input images. HAD estimates pixel-wise
hallucination score maps for diffusion-augmented views and uses these scores to
mask unreliable pixels during 3D Gaussian Splatting optimization. The method
also samples multiple diffusion refinements conditioned on different input views
and fuses the most reliable pixels into the final augmented view.

## Method Overview

<img src="https://xiliu8006.github.io/HAD-Project-website/static/images/method_overview.png" alt="HAD method overview" width="100%">

HAD combines:

- Diffusion-based novel-view refinement from 3DGS-rendered views.
- A hallucination scoring network that predicts pixel-wise reliability maps.
- Multi-sampling fusion that selects reliable pixels across multiple generated
  versions before supervising 3DGS.

## Layout

```text
configs/                     Evaluation scene lists.
examples/gsplat/             Training code, COLMAP dataset loader, and gsplat helpers.
examples/gsplat/pycolmap/    Lightweight COLMAP binary parser used by the dataset loader.
LVSM/                        Hallucination scoring network built on LVSM codebase
scripts/                     Result summarization utilities.
slurm/                       Single-GPU shard launchers.
src/                         DiFix pipeline wrapper.
```

Generated data, logs, checkpoints, and evaluation outputs are intentionally not
part of this repository.

## Dependencies

Install PyTorch for your CUDA version first, then install the Python
dependencies:

```bash
pip install -r requirements.txt
```

The DiFix weights are loaded through the Hugging Face model id
`nvidia/difix_ref`.

## Checkpoint

The hallucination scoring network runtime code is bundled with this repository.
Download the hallucination scoring checkpoint before running HAD:

[ckpt_0000000000010000.pt](https://drive.google.com/file/d/1004LRTsUr0k1D42Ivcg2Q2Zbzjx2mVCg/view?usp=sharing)

```bash
mkdir -p checkpoints/LVSM_decoder_only_conf_Resi_unet_512
# Download ckpt_0000000000010000.pt from the link above, then place it at:
# checkpoints/LVSM_decoder_only_conf_Resi_unet_512/ckpt_0000000000010000.pt
```

Then pass that checkpoint directory to the launcher, or set `LVSM_CKPT_PATH` to
the downloaded `.pt` file. SHA256:
`dd272986eaded1eee8058af4cc619ec94c0dab320942e2549db6d1ea84a2acce`.

## Dataset Format

This codebase supports two dataset layouts:

- DL3DV evaluation scenes, selected with `DATASET=dl3dv`. This is the
  default when `DATASET` is not set.
- Mip-NeRF 360 scenes, selected with `DATASET=mipnerf360`.

### DL3DV

Each DL3DV scene should contain a Nerfstudio/COLMAP-style directory:

```text
<DATA_ROOT>/<scene_id>/nerfstudio/
  images/
  images_4/
  colmap/sparse/0/
```

The default launcher expects scenes listed in `configs/dl3dv_eval_scenes.txt`.
Override the data root with `DATA_ROOT=/path/to/DL3DV-10K-Benchmark` when your
dataset is stored elsewhere.

### Mip-NeRF 360

Each Mip-NeRF 360 scene should be stored directly under the Mip-NeRF 360 data
root:

```text
<MIPNERF_DATA_ROOT>/<scene>/
  images/
  images_4/
  sparse/0/
  train_test_split_<sparse_view>.json
```

The Mip-NeRF 360 training-view splits used by this code come from
[Reconfusion](https://drive.google.com/drive/folders/10oT2_OQ9Sjh5wlfJQoGx2y7ZKYwpgNg5).
Place the corresponding `train_test_split_<sparse_view>.json` file inside each
scene directory before training.

The Mip-NeRF 360 launcher uses scenes listed in `configs/mipnerf360_scenes.txt`.
Set `DATASET=mipnerf360` to select this dataset. Override the data root with
`MIPNERF_DATA_ROOT=/path/to/MipNeRF360` when needed.

## Run Evaluation

Run DL3DV evaluation:

```bash
./run_had_eval_dataset.sh 24 9 /path/to/hallucination_scoring/checkpoint_dir
```

Run Mip-NeRF 360 evaluation:

```bash
DATASET=mipnerf360 \
./run_had_eval_dataset.sh 9 9 /path/to/hallucination_scoring/checkpoint_dir
```

The launcher skips scenes that already have `stats/val_step19999.json`.

## Run One Scene

Run one DL3DV scene:

```bash
./run_train_scene.sh 093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334 9 20000
```

Run one Mip-NeRF 360 scene:

```bash
DATASET=mipnerf360 \
./run_train_scene.sh garden 9 20000
```

## Run Hallucination Scoring Only

```bash
/home/xi9/miniconda3/envs/difix3D/bin/python scripts/run_hallucination_scoring.py \
  --input /path/to/hallucination_scoring_input.npz \
  --output-dir /path/to/hallucination_scoring_outputs
```

The `.npz` should contain `ref_images`, `ref_c2w`, `ref_intrinsics`,
`target_c2w`, `target_intrinsics`, and `hallucinated_images`. Optional
`pred_images` defaults to `hallucinated_images`. Images can be `NHWC` or `NCHW`;
intrinsics can be `fx,fy,cx,cy` or 3x3 `K`; poses are camera-to-world matrices
in the same convention used by the training COLMAP loader.

## Summarize Results

```bash
python scripts/summarize_had_eval_results.py \
  --root /path/to/outputs/dreamaware3d_view9_fusion3 \
  --scene-list configs/dl3dv_eval_scenes.txt \
  --step 19999
```

## Acknowledgements

This codebase is largely built on two excellent open-source projects:
[Difix3D](https://github.com/nv-tlabs/Difix3D) and
[LVSM](https://github.com/haian-jin/LVSM). We thank the authors for releasing
their code. Our hallucination scoring network is based on modifications to the
LVSM architecture.

## Citation

If you find this codebase useful, please cite:

```bibtex
@inproceedings{liu2026had,
  title={HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction},
  author={Liu, Xi and Sun, Weiwei and Ren, Zhou and Broaddus, Chris and Huang, Siyu and Guigues, Laurent},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  pages={29781--29791},
  year={2026}
}
```
