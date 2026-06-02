# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio
import torch.nn.functional as F

def cal_target_visibility_batch(
    input_depths, K_input, Tcw_input,
    target_depths, K_target, Tcw_target,
    tau_rel=0.3, tau_abs=1e-3
):
    """
    Optimized version with better memory usage for large batches.
    
    Args:
        K_input, K_target: (B, N, 4) - camera intrinsics [fx, fy, cx, cy]
    """
    device = target_depths.device
    B, N_target, H, W = target_depths.shape
    _, N_input, _, _ = input_depths.shape

    # Create pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(xs, dtype=torch.float32)
    pix_coords = torch.stack([xs.float(), ys.float()], dim=0).reshape(2, -1)  # (2, H*W)

    all_visibility = torch.zeros(B, N_input, N_target, H, W, dtype=torch.bool, device=device)
    
    for b in range(B):
        for j in range(N_target):  # For each target view
            # Get target view data
            depth_tgt = target_depths[b, j]  # (H, W)
            K_tgt = K_target[b, j]  # (4,) [fx, fy, cx, cy]
            Tcw_tgt = Tcw_target[b, j]  # (4, 4)
            
            # Back-project all target pixels to world coordinates
            z_tgt = depth_tgt.reshape(-1)  # (H*W,)
            valid_z_tgt = (z_tgt > 0) & torch.isfinite(z_tgt)
            
            # Unproject using target intrinsics
            fx_tgt, fy_tgt, cx_tgt, cy_tgt = K_tgt[0], K_tgt[1], K_tgt[2], K_tgt[3]
            u_tgt, v_tgt = pix_coords[0], pix_coords[1]  # (H*W,)
            
            x_tgt = (u_tgt - cx_tgt) * z_tgt / fx_tgt
            y_tgt = (v_tgt - cy_tgt) * z_tgt / fy_tgt
            X_tgt = torch.stack([x_tgt, y_tgt, z_tgt], dim=0)  # (3, H*W)
            
            X_tgt_h = torch.cat([X_tgt, torch.ones(1, H*W, device=device)], dim=0)  # (4, H*W)
            X_world = Tcw_tgt @ X_tgt_h  # (4, H*W)
            
            # Check visibility from each input view
            for i in range(N_input):
                depth_src = input_depths[b, i]  # (H, W)
                K_src = K_input[b, i]  # (4,) [fx, fy, cx, cy]
                Tcw_src = Tcw_input[b, i]  # (4, 4)
                
                # Transform to input camera coordinates
                Twc_src = torch.inverse(Tcw_src)
                X_src = Twc_src @ X_world  # (4, H*W)
                X_src = X_src[:3]  # (3, H*W)
                
                x_src, y_src, z_src = X_src[0], X_src[1], X_src[2]
                
                # Project to input pixels
                fx_src, fy_src, cx_src, cy_src = K_src[0], K_src[1], K_src[2], K_src[3]
                
                u_src = fx_src * (x_src / z_src) + cx_src
                v_src = fy_src * (y_src / z_src) + cy_src
                
                # Check constraints
                in_front = z_src > 0
                in_bounds = (u_src >= 0) & (u_src <= (W-1)) & (v_src >= 0) & (v_src <= (H-1))
                ok = valid_z_tgt & in_front & in_bounds
                
                # Sample input depth
                uu = (u_src / (W-1)) * 2 - 1
                vv = (v_src / (H-1)) * 2 - 1
                grid = torch.stack([uu, vv], dim=-1).reshape(1, 1, -1, 2)
                
                d_src_sampled = F.grid_sample(
                    depth_src.reshape(1, 1, H, W),
                    grid, mode='bilinear', padding_mode='zeros', align_corners=True
                ).reshape(-1)
                
                # Z-buffer consistency
                thr = torch.maximum(torch.full_like(z_src, tau_abs), tau_rel * z_src)
                depth_ok = torch.abs(z_src - d_src_sampled) <= thr
                
                mask = (ok & depth_ok).reshape(H, W)
                all_visibility[b, i, j] = mask
    
    # Calculate multi-view visibility
    multi_view_visibility = all_visibility.float().sum(dim=1)  # (B, N_target, H, W)
    
    return multi_view_visibility

def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)



class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """
        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False) 
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.randint(2, 5)

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views
        num_target_views, num_views, bs = self.config.training.num_target_views, data_batch["c2w"].size(1), data_batch["image"].size(0)
        assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."
        # Decide the target view indices
        if target_has_input: 
            # Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device) # [b, num_target_views]
        else:
            assert (
                self.config.training.num_input_views + num_target_views <= self.config.training.num_views
            ), f"We have {self.config.training.num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."
            
            index = torch.tensor([
                [self.config.training.num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device)
            index = torch.sort(index, dim=1).values # [b, num_target_views]


        for key, value in data_batch.items():
            if key == "scene_name":
                input_dict[key] = value
                target_dict[key] = value
                continue
            if key == "difix3D_image":
                input_dict[key] = value
                target_dict[key] = value
                continue
            if key == "pred_image":
                input_dict[key] = value
                target_dict[key] = value
                continue
            if key == "depth":
                input_dict[key] = value
                target_dict[key] = value
                continue
            # if key == "depth":
            #     target_dict[key] = value
            # if key == 'image':
            #     input_dict[key] = value[:, :self.config.training.num_input_views, ...]
            #     target_dict[key] = value[:, self.config.training.num_input_views:, ...]
            #     continue
            
            # if key == 'c2w':
            #     input_dict[key] = value[:, :self.config.training.num_input_views, ...]
            #     target_dict[key] = value[:, self.config.training.num_input_views:, ...]
            #     continue
            
            # if key == 'fxfycxcy':
            #     input_dict[key] = value[:, :self.config.training.num_input_views, ...]
            #     target_dict[key] = value[:, self.config.training.num_input_views:, ...]
            #     continue

            input_dict[key] = value[:, :self.config.training.num_input_views, ...]

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)

            if key == "image" and not has_target_image:                
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)
            
        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)
        return input_dict, target_dict


    
    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input)

        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]
                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d
        # v_input = input_dict.image.shape[1]
        # visibility = cal_target_visibility_batch(input_dict.depth[:, :v_input] / 1000, input_dict.fxfycxcy, input_dict.c2w, \
        #                                          target_dict.depth[:, v_input:] / 1000, target_dict.fxfycxcy, target_dict.c2w)

        # # visibility = cal_target_visibility_batch(input_dict.depth[:, :v_input], input_dict.fxfycxcy, input_dict.c2w, \
        # #                                         input_dict.depth[:, :v_input], input_dict.fxfycxcy, input_dict.c2w)
        # # visibility = cal_target_visibility_batch(target_dict.depth[:, v_input:], target_dict.fxfycxcy, target_dict.c2w, input_dict.depth[:, :v_input], input_dict.fxfycxcy, input_dict.c2w)
        # # target_dict["visibility"] = visibility.unsqueeze(2)
        # visibility = torch.clamp(visibility.unsqueeze(2), 0, 1)
        # target_dict["visibility"] = visibility
                # visibilty = cal_visibility_mask(c2w, fxfycxcy)

        return input_dict, target_dict

      
      