import json
import math
import os
import time
import shutil
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import random
from PIL import Image
from copy import deepcopy
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from examples.utils import CameraPoseInterpolator
from src.pipeline_difix import DifixPipeline

from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import importlib

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LVSM_ROOT = os.environ.get("LVSM_ROOT", os.path.join(PROJECT_ROOT, "LVSM"))
LVSM_PARENT = os.path.dirname(LVSM_ROOT)
for _path in (PROJECT_ROOT, LVSM_PARENT):
    if _path not in sys.path:
        sys.path.append(_path)
from torchvision import transforms
from einops import rearrange, repeat

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True # ! turn off viser
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 20_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [1_0000, 2_0000, 3_0000, 4_5000, 6_0000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_0000, 6_0000])
    # Steps to fix the artifacts
    fix_steps: List[int] = field(default_factory=lambda: [3_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000, 22_000, 24_000, 26_000, 28_000, 30_000, 32_000, 34_000, 36_000, 38_000, 40_000, 42_000, 44_000, 46_000, 48_000, 50_000, 52_000, 54_000, 56_000, 58_000])

    num_sparse_view: int = 9
    target_sample_step: int = 2
    view_fusion: int = 1
    split_json: Optional[str] = None

    use_eval: bool = True
    use_pefect_conf: bool = True
    use_conf: bool = True
    use_lvsm: bool = True
    lvsm_mode: bool = False
    partial_setting: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for iterative 3d update
    novel_data_lambda: float = 0.3

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    frame_names: Optional[list] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        if frame_names is not None: 
            points_indices = set()
            for frame_name in frame_names:
                points_indices.update(parser.point_indices[frame_name])
            points_indices = list(points_indices)
            points = torch.from_numpy(parser.points[points_indices]).float()
            rgbs = torch.from_numpy(parser.points_rgb[points_indices] / 255.0).float()

        else:
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 / 2 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 5))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 5))
        # params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 50))
        # params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 50))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.input_view_num = 3
        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )

        # num_sparse_view = 9
        num_sparse_view = self.cfg.num_sparse_view
        split_json = cfg.split_json
        if split_json is not None:
            split_json = os.path.abspath(split_json)
            if not os.path.isfile(split_json):
                raise FileNotFoundError(f"Split json does not exist: {split_json}")
            print(f"Using split json: {split_json}")

        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            sparse_views=num_sparse_view,
            partial_setting=cfg.partial_setting,
            json_file=split_json,
        )
        self.valset = Dataset(
            self.parser,
            split="val",
            sparse_views=num_sparse_view,
            partial_setting=cfg.partial_setting,
            json_file=split_json,
        )
        self.target = Dataset(self.parser, split="target", sparse_views=num_sparse_view, \
                              target_sample_step=self.cfg.target_sample_step, partial_setting=cfg.partial_setting, json_file=split_json)

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        frame_names = [ self.parser.image_names[indice] for indice in self.trainset.indices]
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            frame_names = frame_names,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
            
        # Fixer trajectory 
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)

        self.current_novel_poses = self.parser.camtoworlds[self.trainset.indices]
        self.current_parser = self.parser

        self.novelloaders = []
        self.novelloaders_iter = []
        
        # Diffusion fixer
        self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.difix.set_progress_bar_config(disable=True)
        self.difix.to("cuda")

        # config = OmegaConf.load(os.path.join(LVSM_ROOT, "configs/LVSM_scene_decoder_only_SSIM.yaml"))
        config = OmegaConf.load(os.path.join(LVSM_ROOT, "configs/LVSM_scene_decoder_only_conf_512.yaml"))
        config = edict(config)
        lvsm_ckpt_path = os.environ.get("LVSM_CKPT_PATH") or os.environ.get("LVSM_CKPT_DIR")
        if lvsm_ckpt_path:
            if not os.path.isabs(lvsm_ckpt_path):
                lvsm_ckpt_path = os.path.abspath(lvsm_ckpt_path)
            config.training.checkpoint_dir = lvsm_ckpt_path
        elif not os.path.isabs(config.training.checkpoint_dir):
            config.training.checkpoint_dir = os.path.join(LVSM_ROOT, config.training.checkpoint_dir)

        module, class_name = config.model.class_name.rsplit(".", 1)
        module = f"LVSM.{module}"
        LVSM = importlib.import_module(module).__dict__[class_name]
        self.model_lvsm = LVSM(config).to(self.device)
        self.model_lvsm.load_ckpt(config.training.checkpoint_dir)


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self, step=0):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            if len(self.novelloaders) == 0 or random.random() < 0.7:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                is_novel_data = False
            else:
                try:
                    data = next(self.novelloaders_iter[-1])
                except StopIteration:
                    self.novelloaders_iter[-1] = iter(self.novelloaders[-1])
                    data = next(self.novelloaders_iter[-1])        
                is_novel_data = True

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            alpha_masks = data["alpha_mask"].to(device) if "alpha_mask" in data else None  # [1, H, W, 1]
            uncertainty_masks = data["uncertainty_mask"].to(device) if "uncertainty_mask" in data else None  # [1, H, W, 1]

            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if is_novel_data and alpha_masks is not None:
                colors = colors * (alpha_masks > 0.5).float()
                pixels = pixels * (alpha_masks > 0.5).float()

            threshold = 0.9
            if is_novel_data and uncertainty_masks is not None:
                colors = colors * (uncertainty_masks > threshold).float()
                pixels = pixels * (uncertainty_masks > threshold).float()

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            if is_novel_data:
                loss = loss * cfg.novel_data_lambda
            else:
                loss = loss * 1.5
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            # run fixer
            if step in [i - 1 for i in cfg.fix_steps]:
                is_last = False
                if step == cfg.max_steps - 1:
                    is_last = True
                # self.fix(step, is_last)
                self.fix(step, is_last, cfg.use_eval, cfg.use_conf, cfg.use_pefect_conf, cfg.use_lvsm, cfg.lvsm_mode)
            
            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
    
    @torch.no_grad()
    def find_nearest_values(self, target):
        smaller = None
        larger = None
        
        min_idx = min(self.trainset.indices)
        max_idx = max(self.trainset.indices)
        
        for idx in self.trainset.indices:
            if idx < target:
                if smaller is None or idx > smaller:
                    smaller = idx
            elif idx > target:
                if larger is None or idx < larger:
                    larger = idx
        
        # If no smaller value found, use the smallest available
        if smaller is None:
            smaller = min_idx
            
        # If no larger value found, use the largest available
        if larger is None:
            larger = max_idx
                        
        return smaller, larger
        
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
    
    @torch.no_grad()
    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws
    
    def merge_by_confidence(self, images, confidence_maps):
        # Convert list of images to tensor [N, C, H, W]
        images = torch.stack(images, dim=0)
        # Convert list of confidence maps to tensor [N, 1, H, W]
        # confidence_maps = torch.stack(confidence_maps, dim=0)
        
        # Get both max values and indices
        merged_conf, max_conf_indices = torch.max(confidence_maps, dim=0)  # returns (values, indices)
        
        # Expand indices to match image dimensions
        max_conf_indices = max_conf_indices.expand(1, images.shape[1], -1, -1)
        
        # Gather from the correct indices
        merged_image = torch.gather(images, 0, max_conf_indices)
        
        return merged_image.squeeze(0), merged_conf.unsqueeze(0)
    
    def merge_by_confidence_weighted_avg(self, images, confidence_maps):
        # Convert list of images to tensor [N, C, H, W]
        images = torch.stack(images, dim=0)
        
        # Normalize confidence maps to create weights that sum to 1
        # confidence_maps shape: [N, 1, H, W]
        confidence_weights = confidence_maps / (confidence_maps.sum(dim=0, keepdim=True) + 1e-8)
        
        # Compute weighted mean of images
        # confidence_weights: [N, 1, H, W], images: [N, C, H, W]
        merged_image = (images * confidence_weights).sum(dim=0)
        
        # Merged confidence (mean or sum of confidences)
        merged_conf = confidence_maps.mean(dim=0)
        
        return merged_image, merged_conf
    
    def merge_by_confidence_mean(self, images, confidence_maps):
        # Convert list of images to tensor [N, C, H, W]
        images = torch.stack(images, dim=0)
        
        merged_image = images.mean(dim=0)
        merged_conf = confidence_maps.mean(dim=0)
        return merged_image, merged_conf

    # Input shapes:
    # images: list of tensors, each with shape [C, H, W]
    # confidence_maps: list of tensors, each with shape [1, H, W]

    @torch.no_grad()
    def fix(self, step: int, is_last=False, using_eval=True, use_conf=False, use_pefect_conf=True, use_lvsm=True, lvsm_mode=False, image_level=False):
        print("Running fixer...")
        if len(self.cfg.fix_steps) == 1:
            novel_poses = self.parser.camtoworlds[self.target.indices]
        else:
            novel_poses = self.interpolator.shift_poses(self.current_novel_poses, self.parser.camtoworlds[self.target.indices], distance=0.5)
        
        gt_paths = [self.parser.image_paths[i] for i in self.target.indices]

        top_N_ref_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses, top=self.input_view_num)
        top_N_ref_indices = [top_N_ref_indices[i].tolist() for i in range(1, len(top_N_ref_indices), 2)]

        top_N_ref_index = [[i for i in np.array(self.trainset.indices)[indices]] for indices in top_N_ref_indices]

        self.render_traj(step, novel_poses)
        image_paths = [f"{self.render_dir}/novel/{step}/Pred/{i:04d}.png" for i in range(len(novel_poses))]
        
        if len(self.novelloaders) == 0:
            ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
            ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]
        else:
            ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
            ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]

        assert len(image_paths) == len(ref_image_paths) == len(novel_poses)

        for i in tqdm.trange(0, len(novel_poses), desc="Fixing artifacts..."):
            image = Image.open(image_paths[i]).convert("RGB")
            gt_image = Image.open(gt_paths[i]).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
            ref_image = Image.open(ref_image_paths[i]).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)

            if use_lvsm:
                TARGET_HEIGHT = 536  # or whatever resolution you need
                TARGET_WIDTH = 960

                # TARGET_HEIGHT = 360  # or whatever resolution you need
                # TARGET_WIDTH = 640

                target_index = self.target.indices[i]
                target_image = gt_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

                # prev_index, next_index = self.find_nearest_values(target_index)
                # prev_image = Image.open(self.parser.image_paths[prev_index]).convert("RGB")
                # next_image = Image.open(self.parser.image_paths[next_index]).convert("RGB")

                # orig_height = prev_image.height
                # orig_width = prev_image.width

                ref_indices = top_N_ref_index[i]  # Get reference indices for current novel view
                ref_tensors = []
                for ref_idx in ref_indices:
                    ref_image = Image.open(self.parser.image_paths[ref_idx]).convert("RGB")
                    orig_height = ref_image.height
                    orig_width = ref_image.width

                    ref_image = ref_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    ref_tensor = transforms.ToTensor()(ref_image)  # [3,H,W]
                    ref_tensors.append(ref_tensor)
                input_images = torch.stack(ref_tensors).float().unsqueeze(0).to(self.device)  # [1, N, 3, H, W]
                
                if image_level:
                    cur_conf = 0.0
                    for ref_idx in ref_indices:
                        ref_image = Image.open(self.parser.image_paths[ref_idx]).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
                        output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
                        output_image = output_image.resize(image.size, Image.LANCZOS)

                        target_images = transforms.ToTensor()(target_image).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [3,H,W]
                        difix3d_images = transforms.ToTensor()(output_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        pred_images = transforms.ToTensor()(image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [3,H,W]


                        input_c2ws = torch.stack([
                            torch.from_numpy(self.parser.camtoworlds[prev_index]), 
                            torch.from_numpy(self.parser.camtoworlds[next_index])
                        ]).unsqueeze(0).to(self.device) # [1, 2, 4, 4]
                    
                        target_c2w = torch.from_numpy(novel_poses[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                        all_c2ws = torch.cat([input_c2ws, target_c2w], dim=1).squeeze(0)
                        all_c2ws = self.preprocess_poses(all_c2ws.float())

                        input_c2ws = all_c2ws[:self.input_view_num].unsqueeze(0).float()
                        target_c2w = all_c2ws[self.input_view_num:].unsqueeze(0).float()

                        height_scale = TARGET_HEIGHT / orig_height
                        width_scale = TARGET_WIDTH / orig_width

                        K = self.parser.Ks_dict[1]
                        fxfycxcy_single = [
                            K[0,0] * width_scale,   # fx
                            K[1,1] * height_scale,  # fy
                            K[0,2] * width_scale,   # cx
                            K[1,2] * height_scale   # cy
                        ]

                        # Repeat for all reference views
                        num_refs = input_images.shape[1]  # Number of reference views
                        fxfycxcy = torch.tensor([fxfycxcy_single] * num_refs)[None].float().to(self.device)  # [1, N, 4]

                        fxfycxcy_target = [
                            K[0,0] * width_scale,   # fx
                            K[1,1] * height_scale,  # fy
                            K[0,2] * width_scale,   # cx
                            K[1,2] * height_scale   # cy
                        ]

                        # fxfycxcy = torch.tensor([fxfycxcy_prev, fxfycxcy_next])[None].float().to(self.device)  # [1, 2, 4]
                        fxfycxcy_target = torch.tensor([fxfycxcy_target])[None].float().to(self.device)  # [1, 1, 4]

                        image_height = input_images.shape[-2]
                        image_width = input_images.shape[-1]


                        ray_o, ray_d = self.compute_rays(input_c2ws, fxfycxcy, image_height, image_width, device=self.device)
                        target_ray_o, target_ray_d = self.compute_rays(target_c2w, fxfycxcy_target, image_height, image_width, device=self.device)


                        input_dict = dict()
                        target_dict = dict()

                        input_dict["image"] = input_images
                        input_dict["c2w"] = input_c2ws
                        input_dict["fxfycxcy"] = fxfycxcy
                        input_dict["ray_o"] = ray_o.float()
                        input_dict["ray_d"] = ray_d.float()
                        input_dict['image_h_w'] = [image_height, image_width]
                        input_dict['difix3D_image'] = difix3d_images
                        input_dict['pred_image'] = pred_images

                        target_dict["image"] = target_images
                        target_dict["c2w"] = target_c2w
                        target_dict["fxfycxcy"] = fxfycxcy_target
                        target_dict["ray_o"] = target_ray_o.float()
                        target_dict["ray_d"] = target_ray_d.float()
                        target_dict['image_h_w'] = [image_height, image_width]
                        target_dict['difix3D_image'] = difix3d_images
                        target_dict['pred_image'] = pred_images

                        input_dict = edict(input_dict)
                        target_dict = edict(target_dict)
                        lvsm_result = self.model_lvsm.forward_direct(input_dict, target_dict)
                        if (lvsm_result["difix3D_conf"] >0.9).sum() > cur_conf:
                            cur_conf = (lvsm_result["difix3D_conf"] >0.9).sum()

                            conf = lvsm_result["difix3D_conf"].squeeze()
                            conf = conf.cpu().numpy()
                            conf_pil = Image.fromarray(conf.astype(np.float32), mode='F')  # 'F' mode for float32
                            conf_resized = conf_pil.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
                            conf = np.array(conf_resized)

                            os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
                            output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")

                            os.makedirs(f"{self.render_dir}/novel/{step}/Mask", exist_ok=True)
                            Image.fromarray((conf * 255).astype(np.uint8), mode='L').save(f"{self.render_dir}/novel/{step}/Mask/{i:04d}.png")
                else:
                    output_images = []
                    confs = []
                    difix_outputs = []
                    for ref_idx in ref_indices[:self.cfg.view_fusion]:
                        ref_image = Image.open(self.parser.image_paths[ref_idx]).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
                        output_image = self.difix(
                            prompt="remove degradation", 
                            image=image, 
                            ref_image=ref_image, 
                            num_inference_steps=1, 
                            timesteps=[199], 
                            guidance_scale=0.0
                        ).images[0]
                        output_image = output_image.resize(image.size, Image.LANCZOS)
                        output_images.append(transforms.ToTensor()(output_image))
                        difix_outputs.append(output_image)
                    
                    # Step 2: Prepare batch data for LVSM
                    batch_input_dicts = []
                    batch_target_dicts = []

                    # Prepare camera parameters once (shared across batch)
                    ref_c2ws = []
                    for idx in ref_indices:
                        c2w = torch.from_numpy(self.parser.camtoworlds[idx])
                        ref_c2ws.append(c2w)
                    input_c2ws_base = torch.stack(ref_c2ws).unsqueeze(0).to(self.device)  # [1, N, 4, 4]

                    target_c2w_base = torch.from_numpy(novel_poses[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                    all_c2ws = torch.cat([input_c2ws_base, target_c2w_base], dim=1).squeeze(0)
                    all_c2ws = self.preprocess_poses(all_c2ws.float())

                    input_c2ws = all_c2ws[:self.input_view_num].unsqueeze(0).float()
                    target_c2w = all_c2ws[self.input_view_num:].unsqueeze(0).float()

                    # Compute intrinsics
                    height_scale = TARGET_HEIGHT / orig_height
                    width_scale = TARGET_WIDTH / orig_width
                    K = self.parser.Ks_dict[1]

                    fxfycxcy_single = [
                        K[0,0] * width_scale,
                        K[1,1] * height_scale,
                        K[0,2] * width_scale,
                        K[1,2] * height_scale
                    ]

                    num_refs = input_images.shape[1]
                    fxfycxcy = torch.tensor([fxfycxcy_single] * num_refs)[None].float().to(self.device)
                    fxfycxcy_target = torch.tensor([fxfycxcy_single])[None].float().to(self.device)

                    image_height = input_images.shape[-2]
                    image_width = input_images.shape[-1]

                    # Compute rays once
                    ray_o, ray_d = self.compute_rays(input_c2ws, fxfycxcy, image_height, image_width, device=self.device)
                    target_ray_o, target_ray_d = self.compute_rays(target_c2w, fxfycxcy_target, image_height, image_width, device=self.device)

                    # Step 3: Create batched tensors for LVSM
                    target_images_tensor = transforms.ToTensor()(target_image).float().unsqueeze(0).to(self.device)
                    pred_images_tensor = transforms.ToTensor()(image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)).float().unsqueeze(0).to(self.device)

                    # Stack all difix outputs into a batch
                    difix3d_images_list = []
                    for output_image in difix_outputs:
                        difix3d_img = transforms.ToTensor()(output_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)).float().to(self.device)
                        difix3d_images_list.append(difix3d_img)

                    difix3d_images_batch = torch.stack(difix3d_images_list, dim=0)  # [B, 3, H, W]

                    # Expand other tensors to match batch size
                    batch_size = len(difix_outputs)
                    target_images_batch = target_images_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [B, 1, 3, H, W]
                    pred_images_batch = pred_images_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [B, 1, 3, H, W]
                    difix3d_images_batch = difix3d_images_batch.unsqueeze(1)  # [B, 1, 3, H, W]

                    # Expand camera parameters
                    input_images_batch = input_images.expand(batch_size, -1, -1, -1, -1)  # [B, N, 3, H, W]
                    input_c2ws_batch = input_c2ws.expand(batch_size, -1, -1, -1)  # [B, N, 4, 4]
                    target_c2w_batch = target_c2w.expand(batch_size, -1, -1, -1)  # [B, 1, 4, 4]
                    fxfycxcy_batch = fxfycxcy.expand(batch_size, -1, -1)  # [B, N, 4]
                    fxfycxcy_target_batch = fxfycxcy_target.expand(batch_size, -1, -1)  # [B, 1, 4]
                    ray_o_batch = ray_o.expand(batch_size, -1, -1, -1, -1)  # [B, N, H, W, 3]
                    ray_d_batch = ray_d.expand(batch_size, -1, -1, -1, -1)  # [B, N, H, W, 3]
                    target_ray_o_batch = target_ray_o.expand(batch_size, -1, -1, -1, -1)  # [B, 1, H, W, 3]
                    target_ray_d_batch = target_ray_d.expand(batch_size, -1, -1, -1, -1)  # [B, 1, H, W, 3]

                    # Create batched input and target dictionaries
                    input_dict = edict({
                        "image": input_images_batch,
                        "c2w": input_c2ws_batch,
                        "fxfycxcy": fxfycxcy_batch,
                        "ray_o": ray_o_batch.float(),
                        "ray_d": ray_d_batch.float(),
                        'image_h_w': [image_height, image_width],
                        'difix3D_image': difix3d_images_batch,
                        'pred_image': pred_images_batch
                    })

                    target_dict = edict({
                        "image": target_images_batch,
                        "c2w": target_c2w_batch,
                        "fxfycxcy": fxfycxcy_target_batch,
                        "ray_o": target_ray_o_batch.float(),
                        "ray_d": target_ray_d_batch.float(),
                        'image_h_w': [image_height, image_width],
                        'difix3D_image': difix3d_images_batch,
                        'pred_image': pred_images_batch
                    })

                    # Step 4: Single batched LVSM inference
                    lvsm_result = self.model_lvsm.forward_direct(input_dict, target_dict)

                    # Step 5: Process batched results
                    conf_batch = lvsm_result["difix3D_conf"].squeeze(1) 

                    conf_resized = F.interpolate(conf_batch, size=(orig_height, orig_width), mode='bilinear', align_corners=False)
                    merged_difix3D_image, merged_conf = self.merge_by_confidence(output_images, conf_resized.cpu())

                    output_image = merged_difix3D_image.squeeze()
                    output_image = output_image.cpu().permute(1, 2, 0).numpy()
                    output_image = Image.fromarray((output_image * 255).astype(np.uint8)).resize((orig_width, orig_height), Image.Resampling.LANCZOS)

                    os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
                    output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")

                    conf = merged_conf.squeeze().cpu().numpy()
                    # conf_pil = Image.fromarray(conf.astype(np.float32), mode='F')  # 'F' mode for float32
                    # conf_resized = conf_pil.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
                    # conf = np.array(conf_pil)
                    os.makedirs(f"{self.render_dir}/novel/{step}/Mask", exist_ok=True)
                    Image.fromarray((conf * 255).astype(np.uint8), mode='L').save(f"{self.render_dir}/novel/{step}/Mask/{i:04d}.png")
                    

            elif use_pefect_conf:
                output_image = \
                    self.difix(prompt="remove degradation", image=image, ref_image=ref_image, num_inference_steps=1,
                            timesteps=[199], guidance_scale=0.0).images[0]

                output_image = output_image.resize(gt_image.size, Image.LANCZOS)      
                img_np = np.array(output_image)
                gt_np = np.array(gt_image) 
                # Convert to torch tensors and reshape to [B, C, H, W]
                img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                gt_tensor = torch.from_numpy(gt_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

                # If using GPU
                img_tensor = img_tensor.cuda()
                gt_tensor = gt_tensor.cuda()

                # Now compute pixel-wise loss
                pixel_wise_loss = 1.0 - F.l1_loss(img_tensor, gt_tensor, reduction='none')
                pixel_wise_loss = pixel_wise_loss.mean(dim=1, keepdim=True)
                diff_image = pixel_wise_loss.squeeze().detach().cpu().numpy()

                os.makedirs(f"{self.render_dir}/novel/{step}/Mask", exist_ok=True)
                Image.fromarray((diff_image * 255).astype(np.uint8), mode='L').save(f"{self.render_dir}/novel/{step}/Mask/{i:04d}.png")
            else:
                output_image = self.difix(prompt="remove degradation", image=image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
                output_image = output_image.resize(image.size, Image.LANCZOS)
                os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
                output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")
            
        parser = deepcopy(self.parser)
        if use_conf:
            parser.uncertainty_mask_paths = [f"{self.render_dir}/novel/{step}/Mask/{i:04d}.png" for i in range(len(novel_poses))]

        parser.test_every = 0
        parser.image_paths = [f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png" for i in range(len(novel_poses))]
        parser.image_names = [os.path.basename(p) for p in parser.image_paths]
        parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.png" for i in range(len(novel_poses))]
        parser.camtoworlds = novel_poses
        parser.camera_ids = [parser.camera_ids[0]] * len(novel_poses)
        
        print(f"Adding {len(parser.image_paths)} fixed images to novel dataset...")
        dataset = Dataset(parser, split="train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        self.novelloaders.append(dataloader)
        self.novelloaders_iter.append(iter(dataloader))

        self.current_novel_poses = novel_poses

        if is_last:
            novel_dir = os.path.join(self.render_dir, "novel")
            steps = [d for d in os.listdir(novel_dir) if os.path.isdir(os.path.join(novel_dir, d))]
            steps = sorted([int(s) for s in steps])

            for step in steps[:-1]:  # all steps except the last one
                step_dir = os.path.join(novel_dir, str(step))
                shutil.rmtree(step_dir)
            
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm.tqdm(valloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                pixels_path = f"{self.render_dir}/val/{step}/GT/{i:04d}.png"
                os.makedirs(os.path.dirname(pixels_path), exist_ok=True)
                pixels_canvas = pixels.squeeze(0).cpu().numpy()
                pixels_canvas = (pixels_canvas * 255).astype(np.uint8)
                imageio.imwrite(pixels_path, pixels_canvas)

                colors_path = f"{self.render_dir}/val/{step}/Pred/{i:04d}.png"
                os.makedirs(os.path.dirname(colors_path), exist_ok=True)
                colors_canvas = colors.squeeze(0).cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)
                
                alphas_path = f"{self.render_dir}/val/{step}/Alpha/{i:04d}.png"
                os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
                alphas_canvas = (alphas < 0.5).squeeze(0).cpu().numpy()
                alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
                Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int, camtoworlds_all=None, batch_size=8, tag="novel"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if camtoworlds_all is None:
            camtoworlds_all = self.parser.camtoworlds[5:-5]
            if cfg.render_traj_path == "interp":
                camtoworlds_all = generate_interpolated_path(
                    camtoworlds_all, 1
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "ellipse":
                height = camtoworlds_all[:, 2, 3].mean()
                camtoworlds_all = generate_ellipse_path_z(
                    camtoworlds_all, height=height
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "spiral":
                camtoworlds_all = generate_spiral_path(
                    camtoworlds_all,
                    bounds=self.parser.bounds * self.scene_scale,
                    spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                )
            else:
                raise ValueError(
                    f"Render trajectory type not supported: {cfg.render_traj_path}"
                )

            camtoworlds_all = np.concatenate(
                [
                    camtoworlds_all,
                    np.repeat(
                        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                    ),
                ],
                axis=1,
            )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(0, len(camtoworlds_all), batch_size, desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + batch_size]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [B, H, W, 4]

            for j in range(renders.shape[0]):
                colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
                depths = renders[j, ..., 3:4]  # [H, W, 1]
                depths = (depths - depths.min()) / (depths.max() - depths.min())
                
                idx = i + j
                colors_path = f"{self.render_dir}/{tag}/{step}/Pred/{idx:04d}.png"
                os.makedirs(os.path.dirname(colors_path), exist_ok=True)
                colors_canvas = colors.cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)
                
                alphas_path = f"{self.render_dir}/{tag}/{step}/Alpha/{idx:04d}.png"
                os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
                alphas_canvas = alphas[j].float().cpu().numpy()
                alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
                Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.train(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(cap_max=2_000_000, verbose=True),
                # strategy=MCMCStrategy(cap_max=2_000_000, verbose=True, refine_stop_iter=45000),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
