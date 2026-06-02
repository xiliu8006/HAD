import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from typing import List, Tuple

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

# This code refer to: https://github.com/DaLi-Jack/G4Splat/blob/main/2d-gaussian-splatting/guidance/vis_grid.py

class VisibilityGrid:
    """
    A 3D visibility grid that tracks which regions of space are visible from input views.
    
    The grid divides a 3D bounding box into voxels and marks each voxel as:
    - 1: visible (grid center is visible from at least one input camera)
    - 0: invisible (grid center is not visible from any input camera)
    """
    
    def __init__(
        self, 
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor, 
        resolution: int,
        input_intrinsics: List[torch.Tensor],  # List of [3, 3] or [4, 4] intrinsic matrices
        input_extrinsics: List[torch.Tensor],  # List of [4, 4] camera2world matrices
        input_depths: List[torch.Tensor],      # List of depth maps [H, W]
        device: str = "cuda"
    ):
        """
        Initialize the visibility grid.
        
        Args:
            bbox_min: Minimum corner of 3D bounding box, shape (3,)
            bbox_max: Maximum corner of 3D bounding box, shape (3,)
            resolution: Grid resolution
            input_intrinsics: List of camera intrinsic matrices [3, 3] or [4, 4]
            input_extrinsics: List of camera2world transformation matrices [4, 4]
            input_depths: List of depth maps corresponding to input cameras [H, W]
            device: Device to run computations on
        """
        self.device = device
        self.bbox_min = bbox_min.to(device)
        self.bbox_max = bbox_max.to(device)
        self.resolution = resolution
        
        # Store camera parameters
        self.input_intrinsics = [K.to(device) for K in input_intrinsics]
        self.input_extrinsics = [E.to(device) for E in input_extrinsics]
        self.input_depths = [d.to(device) for d in input_depths]
        
        # Calculate grid properties
        self.grid_size = (self.bbox_max - self.bbox_min) / resolution
        self.min_grid_size = self.grid_size.min().item()
        
        # Initialize visibility grid, default to invisible (0)
        self.visibility_grid = torch.zeros(
            (resolution, resolution, resolution), 
            dtype=torch.float32, 
            device=device
        )
        
        # Build the grid
        self._build_grid()
    
    def _unproject_depth(self, depth: torch.Tensor, intrinsic: torch.Tensor, 
                        extrinsic: torch.Tensor) -> torch.Tensor:
        """
        Unproject depth map to 3D points in world coordinates.
        
        Args:
            depth: Depth map [H, W]
            intrinsic: Camera intrinsic matrix [3, 3] or [4, 4]
            extrinsic: Camera2world transformation [4, 4]
            
        Returns:
            points_3d: 3D points in world coordinates [H, W, 3]
        """
        H, W = depth.shape
        
        # Extract intrinsic parameters
        if intrinsic.shape[0] == 4:
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        else:  # 3x3
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Create pixel coordinates
        u = torch.arange(W, device=self.device, dtype=torch.float32)
        v = torch.arange(H, device=self.device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        
        # Unproject to camera coordinates
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        # Stack to [H, W, 3]
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        
        # Transform to world coordinates
        # Add homogeneous coordinate
        ones = torch.ones_like(points_cam[..., :1])
        points_cam_hom = torch.cat([points_cam, ones], dim=-1)  # [H, W, 4]
        
        # Apply camera2world transformation
        points_world_hom = torch.matmul(points_cam_hom, extrinsic.T)  # [H, W, 4]
        points_world = points_world_hom[..., :3]  # [H, W, 3]
        
        return points_world
    
    def _project_points(self, points_3d: torch.Tensor, intrinsic: torch.Tensor, 
                       extrinsic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to image plane.
        
        Args:
            points_3d: 3D points in world coordinates [..., 3]
            intrinsic: Camera intrinsic matrix [3, 3] or [4, 4]
            extrinsic: Camera2world transformation [4, 4]
            
        Returns:
            uv: Pixel coordinates [..., 2]
            depth: Depth values [...]
        """
        original_shape = points_3d.shape[:-1]
        points_flat = points_3d.reshape(-1, 3)
        
        # Transform to camera coordinates (world2cam)
        world2cam = torch.inverse(extrinsic)
        
        # Add homogeneous coordinate
        ones = torch.ones((points_flat.shape[0], 1), device=self.device)
        points_hom = torch.cat([points_flat, ones], dim=-1)  # [N, 4]
        
        # Transform to camera space
        points_cam_hom = torch.matmul(points_hom, world2cam.T)  # [N, 4]
        points_cam = points_cam_hom[:, :3]  # [N, 3]
        
        # Get depth
        depth = points_cam[:, 2]  # [N]
        
        # Project to image plane
        if intrinsic.shape[0] == 4:
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        else:  # 3x3
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        u = fx * points_cam[:, 0] / (points_cam[:, 2] + 1e-8) + cx
        v = fy * points_cam[:, 1] / (points_cam[:, 2] + 1e-8) + cy
        
        uv = torch.stack([u, v], dim=-1)  # [N, 2]
        
        # Reshape back
        uv = uv.reshape(*original_shape, 2)
        depth = depth.reshape(*original_shape)
        
        return uv, depth
    
    def _check_point_visible_from_camera(self, points_3d: torch.Tensor, 
                                        intrinsic: torch.Tensor,
                                        extrinsic: torch.Tensor,
                                        depth_map: torch.Tensor,
                                        depth_threshold: float = 0.01) -> torch.Tensor:
        """
        Check if 3D points are visible from a camera.
        
        Args:
            points_3d: 3D points in world coordinates [N, 3]
            intrinsic: Camera intrinsic matrix
            extrinsic: Camera2world transformation
            depth_map: Depth map [H, W]
            depth_threshold: Threshold for depth comparison
            
        Returns:
            visible_mask: Boolean mask [N]
        """
        H, W = depth_map.shape
        
        # Project points to image plane
        uv, point_depths = self._project_points(points_3d, intrinsic, extrinsic)
        
        # Check if points are within image bounds
        u, v = uv[..., 0], uv[..., 1]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (point_depths > 0)
        
        # Sample depth values at projected locations
        u_idx = torch.clamp(u.long(), 0, W - 1)
        v_idx = torch.clamp(v.long(), 0, H - 1)
        
        sampled_depths = depth_map[v_idx, u_idx]
        
        # Check if point is in front of surface (visible)
        depth_check = point_depths < (sampled_depths + depth_threshold)
        
        visible_mask = in_bounds & depth_check
        
        return visible_mask
    
    def _check_valid_camera_center_by_depth(self, points_3d: torch.Tensor, depth_threshold) -> torch.Tensor:
        """
        Check if points are visible from any input camera.
        
        Args:
            points_3d: 3D points [N, 3]
            
        Returns:
            valid_mask: Boolean mask [N], True if visible from at least one camera
        """
        N = points_3d.shape[0]
        valid_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        # Check visibility from each input camera
        for intrinsic, extrinsic, depth_map in zip(
            self.input_intrinsics, self.input_extrinsics, self.input_depths
        ):
            visible_from_cam = self._check_point_visible_from_camera(
                points_3d, intrinsic, extrinsic, depth_map, depth_threshold
            )
            valid_mask = valid_mask | visible_from_cam
        
        return valid_mask
    
    def _build_grid(self):
        """Build the initial visibility grid based on input camera visibility."""
        print(f"Building visibility grid with resolution {self.resolution}")
        
        # Generate all grid center points
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        # Use meshgrid to generate all grid indices efficiently
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        # Create meshgrid for all grid positions
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Convert indices to world coordinates (grid centers)
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)  # Shape: (nx, ny, nz, 3)
        
        # Flatten for batch processing
        grid_centers_flat = grid_centers.reshape(-1, 3)
        
        print(f"Checking visibility for {grid_centers_flat.shape[0]} grid points")
        
        # Check visibility
        a = max(grid_size)
        valid_mask = self._check_valid_camera_center_by_depth(grid_centers_flat)
        
        # Reshape back to grid shape and set visibility values
        valid_mask_grid = valid_mask.reshape(nx, ny, nz)
        
        # Set visibility: 1 for visible, 0 for invisible
        self.visibility_grid[valid_mask_grid] = 1.0
        self.visibility_grid[~valid_mask_grid] = 0.0
        
        visible_count = valid_mask.sum().item()
        total_count = grid_centers_flat.shape[0]
        print(f"Grid initialized: {visible_count}/{total_count} "
              f"({100*visible_count/total_count:.1f}%) voxels are visible")
    
    def _sample_points_along_ray(self, depth_map: torch.Tensor, 
                                 intrinsic: torch.Tensor,
                                 extrinsic: torch.Tensor,
                                 num_samples: int = None) -> torch.Tensor:
        """
        Sample points along rays from camera center to depth surface.
        
        Args:
            depth_map: Depth map [H, W]
            intrinsic: Camera intrinsic matrix
            extrinsic: Camera2world transformation
            num_samples: Number of samples along each ray (auto-computed if None)
            
        Returns:
            sample_points: Sampled 3D points [H, W, num_samples, 3]
        """
        H, W = depth_map.shape
        
        # Auto-compute number of samples based on max depth and grid size
        if num_samples is None:
            max_depth = depth_map.max().item()
            num_samples = int(max_depth / self.min_grid_size) + 1
        
        # Get camera center in world coordinates
        cam_center_world = extrinsic[:3, 3]  # [3]
        
        # Unproject depth to get surface points
        surface_points = self._unproject_depth(depth_map, intrinsic, extrinsic)  # [H, W, 3]
        
        # Sample along rays
        t_values = torch.linspace(0, 1, num_samples, device=self.device)  # [num_samples]
        t_values = t_values[None, None, :, None]  # [1, 1, num_samples, 1]
        
        # Interpolate between camera center and surface points
        cam_center_expanded = cam_center_world[None, None, None, :]  # [1, 1, 1, 3]
        surface_points_expanded = surface_points[:, :, None, :]  # [H, W, 1, 3]
        
        sample_points = cam_center_expanded + t_values * (
            surface_points_expanded - cam_center_expanded
        )  # [H, W, num_samples, 3]
        
        return sample_points
    
    def _world_to_grid_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to grid indices.
        
        Args:
            points: World coordinates, shape (..., 3)
            
        Returns:
            Grid indices, shape (..., 3), values in [0, resolution-1]
        """
        # Normalize to [0, 1] within bbox
        normalized = (points - self.bbox_min) / (self.bbox_max - self.bbox_min)
        
        # Convert to grid indices
        indices = normalized * self.resolution
        
        # Clamp to valid range
        indices = torch.clamp(indices, 0, self.resolution - 1)
        
        return indices.long()
    
    def _sample_visibility_at_points(self, points: torch.Tensor, 
                                    max_batch_point_num: int = 100000) -> torch.Tensor:
        """
        Sample visibility values at given 3D points using nearest neighbor interpolation.
        
        Args:
            points: World coordinates, shape (..., 3)
            max_batch_point_num: Maximum number of points to process in a single batch
            
        Returns:
            Visibility values, shape (...,)
        """
        original_shape = points.shape[:-1]
        points_flat = points.reshape(-1, 3)
        
        # If number of points is less than batch size, process directly
        if points_flat.shape[0] <= max_batch_point_num:
            grid_indices = self._world_to_grid_indices(points_flat)
            visibility_values = self.visibility_grid[
                grid_indices[:, 0], 
                grid_indices[:, 1], 
                grid_indices[:, 2]
            ]
        else:
            # Process in batches
            visibility_values_list = []
            num_points = points_flat.shape[0]
            num_batches = (num_points + max_batch_point_num - 1) // max_batch_point_num
            
            for i in range(num_batches):
                start_idx = i * max_batch_point_num
                end_idx = min((i + 1) * max_batch_point_num, num_points)
                
                batch_points = points_flat[start_idx:end_idx]
                grid_indices = self._world_to_grid_indices(batch_points)
                
                batch_visibility_values = self.visibility_grid[
                    grid_indices[:, 0], 
                    grid_indices[:, 1], 
                    grid_indices[:, 2]
                ]
                
                visibility_values_list.append(batch_visibility_values)
            
            visibility_values = torch.cat(visibility_values_list, dim=0)
        
        return visibility_values.reshape(original_shape)

    def check_valid_camera_center(self, cam_centers: torch.Tensor) -> torch.Tensor:
        """
        Check camera center is visible or not.

        Args:
            cam_centers: World coordinates, shape (N, 3)
            
        Returns:
            valid_mask: 1 -> visible, 0 -> invisible, shape (N,)
        """
        valid_mask = self._sample_visibility_at_points(cam_centers)
        valid_mask = valid_mask > 0.5
        return valid_mask

    def render_visibility_map(
        self, 
        novel_intrinsics: List[torch.Tensor],
        novel_extrinsics: List[torch.Tensor],
        novel_depths: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Render visibility maps for novel cameras.
        
        Args:
            novel_intrinsics: List of novel camera intrinsic matrices
            novel_extrinsics: List of novel camera2world transformations
            novel_depths: List of depth maps for novel cameras
            
        Returns:
            List of visibility maps, each with shape (H, W). 1 for visible, 0 for occluded
        """
        visibility_maps = []
        
        for cam_idx, (intrinsic, extrinsic, depth_map) in enumerate(
            zip(novel_intrinsics, novel_extrinsics, novel_depths)
        ):
            print(f"Rendering visibility map for camera {cam_idx+1}/{len(novel_depths)}")
            
            # Ensure depth map is on correct device
            if isinstance(depth_map, np.ndarray):
                depth_map = torch.from_numpy(depth_map).to(self.device)
            else:
                depth_map = depth_map.to(self.device)
            
            H, W = depth_map.shape
            
            # Handle invalid depths
            invalid_depth_mask = depth_map <= 1e-6
            depth_map_processed = depth_map.clone()
            depth_map_processed[invalid_depth_mask] = 1e-3
            
            # Sample points along rays
            max_samples = int(depth_map_processed.max().item() / self.min_grid_size) + 1
            sample_points = self._sample_points_along_ray(
                depth_map_processed, intrinsic, extrinsic, max_samples
            )  # [H, W, max_samples, 3]
            
            # Remove last few points to avoid surface boundary issues
            sample_points = sample_points[:, :, :-10, :]
            max_samples = sample_points.shape[2]
            
            # Flatten for visibility sampling
            sample_points_flat = sample_points.reshape(-1, 3)
            
            # Sample visibility values
            visibility_values_flat = self._sample_visibility_at_points(sample_points_flat)
            visibility_values = visibility_values_flat.reshape(H, W, max_samples)
            
            # Check if any point along each ray is invisible (occluded)
            occlusion_map = (visibility_values < 0.5).any(dim=-1).float()
            
            # Handle invalid depths, set them as occluded
            occlusion_map[invalid_depth_mask] = 1.0
            visibility_map = 1 - occlusion_map
            
            visibility_maps.append(visibility_map)
        
        return visibility_maps
    
    def get_visible_boundary(self):
        """Get visible boundary of the grid."""
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        visible_mask = self.visibility_grid > 0.5
        
        if not visible_mask.any():
            print("No visible voxels found.")
            return None
        
        # Generate grid center coordinates
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)
        
        visible_points = grid_centers[visible_mask]
        x_min = visible_points[:, 0].min()
        y_min = visible_points[:, 1].min()
        z_min = visible_points[:, 2].min()
        x_max = visible_points[:, 0].max()
        y_max = visible_points[:, 1].max()
        z_max = visible_points[:, 2].max()
        
        return x_min, y_min, z_min, x_max, y_max, z_max
    
    def get_all_visible_pnts(self):
        """Get all visible points of the grid."""
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        visible_mask = self.visibility_grid > 0.5
        
        if not visible_mask.any():
            print("No visible voxels found.")
            return None
        
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)
        
        visible_points = grid_centers[visible_mask]
        
        return visible_points

    def vis_invisible_pnts(self, save_path: str):
        """Visualize invisible grid centers as points and save to file."""
        nx, ny, nz = self.resolution, self.resolution, self.resolution
        
        invisible_mask = self.visibility_grid < 0.5
        
        if not invisible_mask.any():
            print("No invisible voxels found.")
            return
        
        x_indices = torch.arange(nx, device=self.device)
        y_indices = torch.arange(ny, device=self.device)
        z_indices = torch.arange(nz, device=self.device)
        
        X, Y, Z = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        grid_centers = torch.stack([
            self.bbox_min[0] + (X + 0.5) * self.grid_size[0],
            self.bbox_min[1] + (Y + 0.5) * self.grid_size[1], 
            self.bbox_min[2] + (Z + 0.5) * self.grid_size[2]
        ], dim=-1)
        
        invisible_points = grid_centers[invisible_mask]
        
        points_np = invisible_points.detach().cpu().numpy()
        pm = trimesh.PointCloud(points_np)
        pm.export(save_path)
        print(f"Invisible points saved to {save_path}")