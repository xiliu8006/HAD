# HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction

This is our codebase for HAD training and evaluation.
It keeps only the code needed to run the COLMAP/gsplat training pipeline with
DiFix refinement and hallucination scoring network guidance.

## News

We paper was accepted by CVPR 2026!

## Layout

```text
configs/                     Evaluation scene lists.
examples/gsplat/             Training code, COLMAP dataset loader, and gsplat helpers.
examples/gsplat/pycolmap/    Lightweight COLMAP binary parser used by the dataset loader.
LVSM/                        Hallucination scoring network build on LVSM codebase
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

The hallucination scoring network runtime code is bundled with this repository.
Put the released hallucination scoring checkpoint files in a checkpoint
directory and pass that directory to the launcher.

The DiFix weights are loaded through the Hugging Face model id
`nvidia/difix_ref`.

## Dataset Format

Each scene should contain a Nerfstudio/COLMAP-style directory:

```text
<DATA_ROOT>/<scene_id>/nerfstudio/
  images/
  images_4/
  colmap/sparse/0/
```

The default launcher expects scenes listed in `configs/had_eval_scenes.txt`.

## Run Evaluation

```bash
./run_had_eval_dataset.sh 24 9 /path/to/hallucination_scoring/checkpoint_dir
```

The launcher skips scenes that already have `stats/val_step19999.json`.

## Run One Scene

```bash
./run_train_scene.sh 093ef327b4e4f9d4ee52c02a354a53558a8652157fb0d58f3b4a708734afb334 9 20000
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
  --scene-list configs/had_eval_scenes.txt \
  --step 19999
```

## Citation

If you find this codebase useful, please cite:

```bibtex
@inproceedings{liu2026had,
  title={HAD: Hallucination-Aware Diffusion Priors for 3D Reconstruction},
  author={Liu, Xi and Sun, Weiwei and Ren, Zhou and Broaddus, Chris and Huang, Siyu and Guigues, Laurent},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

This codebase is largely built on two excellent open-source projects:
[Difix3D](https://github.com/nv-tlabs/Difix3D) and
[LVSM](https://github.com/haian-jin/LVSM). We thank the authors for releasing
their code. Our hallucination scoring network is based on modifications to the
LVSM architecture.
