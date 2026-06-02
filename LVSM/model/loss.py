# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import lpips
import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path
from fused_ssim import fused_ssim
from torchmetrics.functional.image import structural_similarity_index_measure

from typing import Tuple, Union
from piq.functional import gaussian_filter

# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()
        
    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        
        return model.to(self.device).eval()
    
    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        
        if torch.distributed.get_rank() == 0:
            # Download weights if needed
            if not weight_file.exists():
                os.system(f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')
        torch.distributed.barrier()
        
        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        
        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                
                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        
        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())
        
        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
    
    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean
    
    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    
    def forward(self, pred_img, target_img):
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)
        
        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)
        
        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)
        
        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5
        
        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
        
        return total_loss

class LossComputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.ssim = StructuralSimilarityIndexMeasure(
        #     data_range=1.0,          # Range of the input images (1.0 for normalized)
        #     return_full_image=True,  # Whether to return the full SSIM image
        #     reduction=None
        # )
        if self.config.training.lpips_loss_weight > 0.0:
            # avoid multiple GPUs from downloading the same LPIPS model multiple times
            if torch.distributed.get_rank() == 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def check_and_fix_inf_nan(self, input_tensor, loss_name="default", hard_max=100):
        """
        Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
        
        Args:
            input_tensor (torch.Tensor): The loss tensor to check and fix.
            loss_name (str): Name of the loss (for diagnostic prints).
            hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                    [-hard_max, hard_max] will be clamped. If None, 
                                    no clamping is performed. Defaults to 100.
        """
        if input_tensor is None:
            return input_tensor
        
        # Check for inf/nan values
        has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
        if has_inf_nan:
            logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
            input_tensor = torch.where(
                torch.isnan(input_tensor) | torch.isinf(input_tensor),
                torch.zeros_like(input_tensor),
                input_tensor
            )

        # Apply hard clamping if specified
        if hard_max is not None:
            input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

        return input_tensor
    
    def ssim_per_channel(self, x: torch.Tensor, y: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
            kernel: 2D Gaussian kernel.
            data_range: Maximum value range of images (usually 1.0 or 255).
            k1: Algorithm parameter, K1 (small constant, see [1]).
            k2: Algorithm parameter, K2 (small constant, see [1]).
                Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

        Returns:
            Full Value of Structural Similarity (SSIM) index.
        """
        kernel = gaussian_filter(11, 1.5, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
        if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
            raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                            f'Input size: {x.size()}. Kernel size: {kernel.size()}')

        # Calculate padding
        pad_h = (kernel.size(-2) - 1) // 2
        pad_w = (kernel.size(-1) - 1) // 2

        c1 = k1 ** 2
        c2 = k2 ** 2
        n_channels = x.size(1)
        
        # Add padding to maintain shape
        mu_x = F.conv2d(x, weight=kernel, stride=1, padding=(pad_h, pad_w), groups=n_channels)
        mu_y = F.conv2d(y, weight=kernel, stride=1, padding=(pad_h, pad_w), groups=n_channels)

        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=(pad_h, pad_w), groups=n_channels) - mu_xx
        sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=(pad_h, pad_w), groups=n_channels) - mu_yy
        sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=(pad_h, pad_w), groups=n_channels) - mu_xy

        cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
        ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

        # Now ss and cs maintain the spatial dimensions
        return ss, cs

    def forward(
        self,
        rendering,
        target,
    ):
        """
        Calculate various losses between rendering and target images.
        
        Args:
            rendering: [b, v, 3, h, w], value range [0, 1]
            target: [b, v, 3, h, w], value range [0, 1]
        
        Returns:
            Dictionary of loss metrics
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)
        
        # Handle alpha channel if present
        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)

        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss_map = F.mse_loss(rendering, target, reduction='none')
        
        l2_loss = l2_loss_map.mean()

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
        )


        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss, 
            norm_lpips_loss=lpips_loss / l2_loss
        )
        return loss_metrics

    def forward_withconf(
        self,
        rendering,
        target,
        difix3d_image,
        conf,
        difix_conf,
        difix_render,
        visibility=None,
        gamma=1.0,
        alpha=0.1
        ):
        """
        Calculate various losses between rendering and target images.
        
        Args:
            rendering: [b, v, 4, h, w], value range [0, 1]
            target: [b, v, 3, h, w], value range [0, 1]
        
        Returns:
            Dictionary of loss metrics
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)
        difix3d_image = difix3d_image.reshape(b * v, -1, h, w)
        difix_render = difix_render.reshape(b * v, -1, h, w)
        if visibility is not None:
            visibility = visibility.reshape(b * v, -1, h, w)

        conf = conf.reshape(b * v, -1, h, w)
        difix_conf = difix_conf.reshape(b * v, -1, h, w)
        # Handle alpha channel if present
        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)
        
        if rendering.size(1) == 4:
            rendering, conf = rendering.split([3, 1], dim=1)



        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss_map = F.mse_loss(rendering, target, reduction='none')
            l2_difix_loss_map = F.mse_loss(difix_render, target, reduction='none')

            if self.config.training.use_ssim_loss:
                # _, ssim_map = structural_similarity_index_measure(rendering, target, return_full_image=True) 
                # ssim_map = torch.clamp(ssim_map.mean(dim=1, keepdim=True), -1, 1)
                # loss_conf = F.l1_loss(conf, ssim_map.detach(), reduction='none')


                # _, difix_ssim_map = structural_similarity_index_measure(difix3d_image, target, return_full_image=True) 
                # difix_ssim_map = torch.clamp(difix_ssim_map.mean(dim=1, keepdim=True), -1, 1)
                # loss_difix_conf = F.l1_loss(difix_conf, difix_ssim_map.detach(), reduction='none')

                ssim_map = 1.0 - F.l1_loss(rendering, target.detach(), reduction='none')
                ssim_map = ssim_map.mean(dim=1, keepdim=True)
                # ssim_map = torch.clamp(ssim_map.mean(dim=1, keepdim=True), -1, 1)
                loss_conf = F.l1_loss(conf, ssim_map.detach(), reduction='none')


                difix_ssim_map =  1.0 - F.l1_loss(difix3d_image, target, reduction='none') 
                difix_ssim_map = difix_ssim_map.mean(dim=1, keepdim=True)
                # difix_ssim_map = torch.clamp(difix_ssim_map.mean(dim=1, keepdim=True), -1, 1)
                loss_difix_conf = F.l1_loss(difix_conf[:, 0:1, :, :], difix_ssim_map.detach(), reduction='none')
                if visibility is not None:
                    loss_visibility = F.l1_loss(difix_conf[:, 1:, :, :], visibility.detach(), reduction='none')

            else:
                loss_reg = torch.norm(target - rendering, dim=1, keepdim=True) / 1.732
                loss_reg = self.check_and_fix_inf_nan(loss_reg, "loss_reg")
                # loss_conf = F.mse_loss(conf, loss_reg, reduction='none')
                if self.config.training.use_l1_norm_loss:
                    loss_conf = F.mse_loss(conf, loss_reg.detach(), reduction='none')
                    loss_conf = self.check_and_fix_inf_nan(loss_conf, "loss_conf")
                else:
                    loss_conf = gamma * loss_reg * conf - alpha * torch.log(conf)
                    loss_conf = self.check_and_fix_inf_nan(loss_conf, "loss_conf")
        # ssim_map = self.ssim(rendering, target)


        l2_loss = l2_loss_map.mean()
        l2_difix_loss = l2_difix_loss_map.mean()

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        if visibility is not None:
            loss = (
                self.config.training.l2_loss_weight * l2_loss
                + self.config.training.l2_loss_weight * l2_difix_loss
                + self.config.training.lpips_loss_weight * lpips_loss
                + self.config.training.perceptual_loss_weight * perceptual_loss
                + self.config.training.conf_loss_weight * loss_conf.mean()
                + self.config.training.conf_loss_weight * loss_difix_conf.mean()
                + self.config.training.conf_loss_weight * loss_visibility.mean()
            )
        else:
            loss = (
                self.config.training.l2_loss_weight * l2_loss
                + self.config.training.l2_loss_weight * l2_difix_loss
                + self.config.training.lpips_loss_weight * lpips_loss
                + self.config.training.perceptual_loss_weight * perceptual_loss
                + self.config.training.conf_loss_weight * loss_conf.mean()
                + self.config.training.conf_loss_weight * loss_difix_conf.mean()
            )


        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            l2_difix_loss=l2_difix_loss,
            conf_loss=loss_conf.mean(),
            difix_conf_loss=loss_difix_conf.mean(),
            # visibility_loss = loss_visibility.mean(),
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss, 
            norm_lpips_loss=lpips_loss / l2_loss
        )
        return loss_metrics

    def forward_visibility_only(
        self,
        rendering,
        target,
        difix3d_image,
        conf,
        difix_conf,
        # difix_render,
        visibility=None,
        gamma=1.0,
        alpha=0.1
        ):
        """
        Calculate various losses between rendering and target images.
        
        Args:
            rendering: [b, v, 4, h, w], value range [0, 1]
            target: [b, v, 3, h, w], value range [0, 1]
        
        Returns:
            Dictionary of loss metrics
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)
        difix3d_image = difix3d_image.reshape(b * v, -1, h, w)
        # difix_render = difix_render.reshape(b * v, -1, h, w)
        if visibility is not None:
            visibility = visibility.reshape(b * v, -1, h, w)

        conf = conf.reshape(b * v, -1, h, w)
        difix_conf = difix_conf.reshape(b * v, -1, h, w)
        # Handle alpha channel if present
        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)
        
        if rendering.size(1) == 4:
            rendering, conf = rendering.split([3, 1], dim=1)



        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss_map = F.mse_loss(rendering, target, reduction='none')
            # l2_difix_loss_map = F.mse_loss(difix_render, target, reduction='none')

            ssim_map = 1.0 - F.l1_loss(rendering, target.detach(), reduction='none')
            ssim_map = ssim_map.mean(dim=1, keepdim=True)
            loss_conf = F.l1_loss(conf, ssim_map.detach(), reduction='none')

            if self.config.training.use_ssim_loss:

                visibility_classes = (visibility).long()  # Convert 1,2 to 0,1
                
                seg_loss = F.cross_entropy(
                    difix_conf,  # [B, 2, H, W] - ch0: visible, ch1: invisible  
                    visibility_classes.detach().squeeze(),
                    reduction='none'
                )               
            
            # else:
            #     loss_reg = torch.norm(target - rendering, dim=1, keepdim=True) / 1.732
            #     loss_reg = self.check_and_fix_inf_nan(loss_reg, "loss_reg")
            #     if self.config.training.use_l1_norm_loss:
            #         loss_conf = F.mse_loss(conf, loss_reg.detach(), reduction='none')
            #         loss_conf = self.check_and_fix_inf_nan(loss_conf, "loss_conf")
            #     else:
            #         loss_conf = gamma * loss_reg * conf - alpha * torch.log(conf)
            #         loss_conf = self.check_and_fix_inf_nan(loss_conf, "loss_conf")
        # ssim_map = self.ssim(rendering, target)


        l2_loss = l2_loss_map.mean()
        # l2_difix_loss = l2_difix_loss_map.mean()

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            # + self.config.training.l2_loss_weight * l2_difix_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
            + self.config.training.conf_loss_weight * seg_loss.mean()
            + self.config.training.conf_loss_weight * loss_conf.mean()
        )

        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            # l2_difix_loss=l2_difix_loss,
            conf_loss=loss_conf.mean(),
            # difix_conf_loss=loss_difix_conf.mean(),
            visibility_loss = seg_loss.mean(),
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss, 
            norm_lpips_loss=lpips_loss / l2_loss
        )
        return loss_metrics
