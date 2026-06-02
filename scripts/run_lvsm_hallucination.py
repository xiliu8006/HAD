#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from omegaconf import OmegaConf


PROJECT_ROOT = Path("/home/xi9/code/DreamAware3D_open_source")
DEFAULT_LVSM_ROOT = PROJECT_ROOT / "LVSM"
DEFAULT_LVSM_CKPT = Path("/home/xi9/code/LVSM/experiments/checkpoints/LVSM_decoder_only_conf_Resi_unet_512")


def image_array_to_tensor(array, height, width):
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[None]
    if array.shape[-1] == 3:
        array = np.transpose(array, (0, 3, 1, 2))
    tensor = torch.from_numpy(array).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.clamp(0.0, 1.0)


def intrinsic_to_fxfycxcy(array):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[-2:] == (3, 3):
        return np.stack([array[..., 0, 0], array[..., 1, 1], array[..., 0, 2], array[..., 1, 2]], axis=-1)
    if array.shape[-1] == 4:
        return array
    raise ValueError("Intrinsics must be fx,fy,cx,cy or 3x3 K matrices.")


def scale_intrinsics(fxfycxcy, old_h, old_w, new_h, new_w):
    fxfycxcy = np.asarray(fxfycxcy, dtype=np.float32).copy()
    fxfycxcy[..., 0] *= new_w / old_w
    fxfycxcy[..., 2] *= new_w / old_w
    fxfycxcy[..., 1] *= new_h / old_h
    fxfycxcy[..., 3] *= new_h / old_h
    return fxfycxcy


def compute_rays(c2w, fxfycxcy, height, width, device):
    b, v = c2w.shape[:2]
    c2w = c2w.reshape(b * v, 4, 4)
    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    x = x[None].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)
    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=height, w=width, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=height, w=width, c=3)
    return ray_o, ray_d


def preprocess_poses(c2w, scene_scale_factor=1.35):
    center = c2w[:, :3, 3].mean(0)
    avg_forward = F.normalize(c2w[:, :3, 2].mean(0), dim=-1)
    avg_down = c2w[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=c2w.device)
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    c2w = torch.linalg.inv(avg_pose) @ c2w
    scene_scale = scene_scale_factor * torch.max(torch.abs(c2w[:, :3, 3]))
    c2w[:, :3, 3] /= scene_scale
    return c2w


def load_lvsm(lvsm_root, checkpoint, device):
    lvsm_root = Path(lvsm_root)
    parent = str(lvsm_root.parent)
    project = str(PROJECT_ROOT)
    for path in (project, parent):
        if path not in sys.path:
            sys.path.append(path)

    config = edict(OmegaConf.load(lvsm_root / "configs/LVSM_scene_decoder_only_conf_512.yaml"))
    config.training.checkpoint_dir = str(checkpoint)
    module, class_name = config.model.class_name.rsplit(".", 1)
    model_cls = importlib.import_module(f"LVSM.{module}").__dict__[class_name]
    model = model_cls(config).to(device).eval()
    model.load_ckpt(config.training.checkpoint_dir)
    return model


def save_rgb(path, tensor):
    image = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    imageio.imwrite(path, (image * 255).astype(np.uint8))


def save_gray(path, tensor):
    image = tensor.detach().cpu().clamp(0, 1).numpy()
    imageio.imwrite(path, (image * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with ref_images, ref_c2w, ref_intrinsics, target_c2w, target_intrinsics, hallucinated_images.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lvsm-root", default=str(DEFAULT_LVSM_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_LVSM_CKPT))
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poses-preprocessed", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    data = np.load(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_images_np = np.asarray(data["ref_images"])
    hallucinated_np = np.asarray(data["hallucinated_images"])
    if hallucinated_np.ndim == 3:
        hallucinated_np = hallucinated_np[None]

    ref_old_h, ref_old_w = ref_images_np.shape[-3], ref_images_np.shape[-2]
    if ref_images_np.shape[-1] != 3:
        ref_old_h, ref_old_w = ref_images_np.shape[-2], ref_images_np.shape[-1]
    target_old_h, target_old_w = hallucinated_np.shape[-3], hallucinated_np.shape[-2]
    if hallucinated_np.shape[-1] != 3:
        target_old_h, target_old_w = hallucinated_np.shape[-2], hallucinated_np.shape[-1]

    ref_images = image_array_to_tensor(ref_images_np, args.height, args.width).unsqueeze(0).to(device)
    hallucinated = image_array_to_tensor(hallucinated_np, args.height, args.width).unsqueeze(0).to(device)
    pred_source = data["pred_images"] if "pred_images" in data.files else hallucinated_np
    pred_images = image_array_to_tensor(pred_source, args.height, args.width).unsqueeze(0).to(device)

    ref_c2w = torch.from_numpy(np.asarray(data["ref_c2w"], dtype=np.float32)).to(device)
    target_c2w = np.asarray(data["target_c2w"], dtype=np.float32)
    if target_c2w.ndim == 2:
        target_c2w = target_c2w[None]
    target_c2w = torch.from_numpy(target_c2w).to(device)

    if not args.poses_preprocessed:
        all_c2w = preprocess_poses(torch.cat([ref_c2w, target_c2w], dim=0))
        ref_c2w = all_c2w[: ref_c2w.shape[0]]
        target_c2w = all_c2w[ref_c2w.shape[0] :]

    ref_intr = intrinsic_to_fxfycxcy(data["ref_intrinsics"])
    if ref_intr.ndim == 1:
        ref_intr = np.repeat(ref_intr[None], ref_images.shape[1], axis=0)
    ref_intr = scale_intrinsics(ref_intr, ref_old_h, ref_old_w, args.height, args.width)
    target_intr = intrinsic_to_fxfycxcy(data["target_intrinsics"])
    if target_intr.ndim == 1:
        target_intr = np.repeat(target_intr[None], hallucinated.shape[1], axis=0)
    target_intr = scale_intrinsics(target_intr, target_old_h, target_old_w, args.height, args.width)
    if pred_images.shape[1] == 1 and hallucinated.shape[1] > 1:
        pred_images = pred_images.expand(-1, hallucinated.shape[1], -1, -1, -1)

    ref_c2w = ref_c2w.unsqueeze(0)
    target_c2w = target_c2w.unsqueeze(0)
    ref_intr = torch.from_numpy(ref_intr).float().unsqueeze(0).to(device)
    target_intr = torch.from_numpy(target_intr).float().unsqueeze(0).to(device)

    ref_rays_o, ref_rays_d = compute_rays(ref_c2w, ref_intr, args.height, args.width, device)
    target_rays_o, target_rays_d = compute_rays(target_c2w, target_intr, args.height, args.width, device)

    input_dict = edict(
        image=ref_images,
        c2w=ref_c2w,
        fxfycxcy=ref_intr,
        ray_o=ref_rays_o.float(),
        ray_d=ref_rays_d.float(),
        image_h_w=[args.height, args.width],
        difix3D_image=hallucinated,
        pred_image=pred_images,
    )
    target_dict = edict(
        c2w=target_c2w,
        fxfycxcy=target_intr,
        ray_o=target_rays_o.float(),
        ray_d=target_rays_d.float(),
        image_h_w=[args.height, args.width],
        difix3D_image=hallucinated,
        pred_image=pred_images,
    )

    model = load_lvsm(args.lvsm_root, args.checkpoint, device)
    with torch.no_grad():
        result = model.forward_direct(input_dict, target_dict, has_target_image=False)

    render = result["render"][0]
    predicted = result["difix3D_render"][0]
    confidence = result["difix3D_conf"][0, :, 0]
    lvsm_confidence = result["conf"][0, :, 0]

    for i in range(predicted.shape[0]):
        save_rgb(output_dir / f"lvsm_render_{i:04d}.png", render[i])
        save_rgb(output_dir / f"predicted_hallucination_{i:04d}.png", predicted[i])
        save_gray(output_dir / f"confidence_{i:04d}.png", confidence[i])
        save_gray(output_dir / f"lvsm_confidence_{i:04d}.png", lvsm_confidence[i])
        np.save(output_dir / f"confidence_{i:04d}.npy", confidence[i].detach().cpu().numpy())
        np.save(output_dir / f"lvsm_confidence_{i:04d}.npy", lvsm_confidence[i].detach().cpu().numpy())


if __name__ == "__main__":
    main()
