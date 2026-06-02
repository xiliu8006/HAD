# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).


import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
try:
    from LVSM.utils import camera_utils, data_utils
except ImportError:
    try:
        from utils import camera_utils, data_utils
    except ImportError:
        pass
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer

class AttentionConfDecoder(nn.Module):
    def __init__(self, d_model, patch_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model * 3, bias=False)
        self.attention = nn.MultiheadAttention(d_model * 3, 8, batch_first=True)
        self.linear1 = nn.Linear(d_model * 3, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.linear2 = nn.Linear(d_model, patch_size**2, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self-attention
        out = self.norm1(x)
        out, _ = self.attention(out, out, out)
        out = self.dropout(out)
        
        # FFN
        out = self.linear1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        return torch.tanh(out)

class CrossAttentionConfDecoder(nn.Module):
    def __init__(self, d_model, patch_size, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Cross-attention layers
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        
        # Final prediction
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size**2),
            nn.Tanh()
        )

        self.img_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, (patch_size**2)*3),
            nn.Sigmoid()
        )
        
        
    def forward(self, difix_tokens, target_tokens, pred_tokens):
        # First cross attention: difix -> target
        q1 = self.norm1(difix_tokens)
        k1 = v1 = self.norm1(target_tokens)
        attn1, _ = self.cross_attn1(q1, k1, v1)
        attn1 = difix_tokens + attn1
        
        # First self attention
        attn1_norm = self.norm2(attn1)
        self_attn1, _ = self.self_attn1(attn1_norm, attn1_norm, attn1_norm)
        attn1 = attn1 + self_attn1
        
        # Second cross attention: result -> pred
        q2 = self.norm3(attn1)
        k2 = v2 = self.norm3(pred_tokens)
        attn2, _ = self.cross_attn2(q2, k2, v2)
        attn2 = attn1 + attn2
        
        # Second self attention
        attn2_norm = self.norm4(attn2)
        self_attn2, _ = self.self_attn2(attn2_norm, attn2_norm, attn2_norm)
        
        # Final prediction
        out = self.norm5(attn2)
        return self.pred_head(out), self.img_head(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        return self.relu(out)

class UNetWithConfidence(nn.Module):
    def __init__(self, in_channels=12):  # 2 images * 3 channels + 6 = 12
        super().__init__()
        
        # Encoder (downsampling)
        self.conv1 = DoubleConv(in_channels, 64)
        # self.pool1 = nn.MaxPool2d(2)
        self.pool1 = nn.AvgPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)  # 512 because of skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)  # 256 because of skip connection
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)   # 128 because of skip connection
        
        # Output layers
        # self.output_image = nn.Sequential(
        #     nn.Conv2d(64, 3, kernel_size=1),
        #     nn.Sigmoid()  # assuming output should be in [0, 1]
        # )
        
        # self.output_confidence = nn.Sequential(
        #     nn.Conv2d(64, 2, kernel_size=1),
        #     nn.Sigmoid()  # confidence in [-1, 1]
        # )

        self.output_image = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.output_confidence = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder
        up3 = self.upconv3(bottleneck)
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.conv_up3(up3)
        
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.conv_up2(up2)
        
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.conv_up1(up1)
        
        # Output
        image_out = self.output_image(up1)
        conf_out = self.output_confidence(up1)
        return image_out, conf_out

class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )

        self.multi_feature_adpator = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 6,
                bias=False,
            )
        )

        # For confidence (without sigmoid)
        self.image_conf_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 1,  # 1 for confidence
                bias=False,
            ),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.difix3d_conf_decoder = UNetWithConfidence()


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.n_layer)
        ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()


    
    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(self.transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
            


    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    
    
    def forward(self, data_batch, has_target_image=True):
        
        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)

        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )

        posed_difix3D_images = self.get_posed_input(
            images=input.difix3D_image, ray_o=target.ray_o, ray_d=target.ray_d
        )

        posed_pred_images = self.get_posed_input(
            images=input.pred_image, ray_o=target.ray_o, ray_d=target.ray_d
        )

        b, v_input, c, h, w = posed_input_images.size()

        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        difix_img_tokens = self.image_tokenizer(posed_difix3D_images)
        pred_img_tokens = self.image_tokenizer(posed_pred_images)

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        
     
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]

        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_patches, n_patches], dim=1
        ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

        rendered_images = self.image_token_decoder(target_image_tokens)
        rendered_confs = self.image_conf_decoder(target_image_tokens)
        multi_view_features = self.multi_feature_adpator(target_image_tokens)

        height, width = target.image_h_w


        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )

        rendered_confs = rearrange(
            rendered_confs, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=1
        )

        multi_view_features = rearrange(
            multi_view_features, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=6
        )

        

        rendered_images = rendered_images.reshape(b * v_target, 3, height, width)
        difix3D_images=(input.difix3D_image).reshape(b * v_target, 3, height, width)
        pred_images=(input.pred_image).reshape(b * v_target, 3, height, width)
        multi_view_features = multi_view_features.reshape(b * v_target, 6, height, width)
        
        concat_images = torch.cat([multi_view_features, difix3D_images, pred_images], dim=1)
        
        difix3d_renders, difix3d_confs = self.difix3d_conf_decoder(concat_images)
        difix3d_renders = difix3d_renders.reshape(b, v_target, -1, height, width)
        difix3d_confs = difix3d_confs.reshape(b, v_target, -1, height, width)
        rendered_images = rendered_images.reshape(b, v_target, -1, height, width)
        # difix3d_confs = rearrange(
        #     difix3d_confs, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
        #     v=v_target,
        #     h=height // patch_size, 
        #     w=width // patch_size, 
        #     p1=patch_size, 
        #     p2=patch_size, 
        #     c=1
        # )

        # difix3d_renders =  rearrange(
        #     difix3d_renders, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
        #     v=v_target,
        #     h=height // patch_size, 
        #     w=width // patch_size, 
        #     p1=patch_size, 
        #     p2=patch_size, 
        #     c=3
        # )
        if has_target_image:
            loss_metrics = self.loss_computer.forward_withconf(
                rendered_images,
                target.image,
                input.difix3D_image,
                rendered_confs,
                difix3d_confs,
                difix3d_renders,
                # target.visibility
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            conf=rendered_confs,
            difix3D_conf=difix3d_confs,
            difix3D_render=difix3d_renders,        
            )
        
        return result
    
    def forward_direct(self, input, target, has_target_image=True):

        # input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)

        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )

        posed_difix3D_images = self.get_posed_input(
            images=input.difix3D_image, ray_o=target.ray_o, ray_d=target.ray_d
        )

        b, v_input, c, h, w = posed_input_images.size()

        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        difix_img_tokens = self.image_tokenizer(posed_difix3D_images)

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        
     
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]

        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(concat_img_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_patches, n_patches], dim=1
        ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

        rendered_images = self.image_token_decoder(target_image_tokens)
        rendered_confs = self.image_conf_decoder(target_image_tokens)
        multi_view_features = self.multi_feature_adpator(target_image_tokens)

        height, width = target.image_h_w
        patch_size = self.config.model.target_pose_tokenizer.patch_size
        
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )

        rendered_confs = rearrange(
            rendered_confs, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=1
        )

        multi_view_features = rearrange(
            multi_view_features, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=6
        )

        rendered_images = rendered_images.reshape(b * v_target, 3, height, width)
        difix3D_images=(input.difix3D_image).reshape(b * v_target, 3, height, width)
        pred_images=(input.pred_image).reshape(b * v_target, 3, height, width)
        multi_view_features = multi_view_features.reshape(b * v_target, 6, height, width)
        # concat_images = torch.cat([rendered_images, difix3D_images, pred_images], dim=1)
        concat_images = torch.cat([multi_view_features, difix3D_images, pred_images], dim=1)

        difix3d_renders, difix3d_confs = self.difix3d_conf_decoder(concat_images)
        difix3d_renders = difix3d_renders.reshape(b, v_target, -1, height, width)
        difix3d_confs = difix3d_confs.reshape(b, v_target, -1, height, width)
        rendered_images = rendered_images.reshape(b, v_target, -1, height, width)

        # difix3d_confs = rearrange(
        #     difix3d_confs, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
        #     v=v_target,
        #     h=height // patch_size, 
        #     w=width // patch_size, 
        #     p1=patch_size, 
        #     p2=patch_size, 
        #     c=1
        # )

        if has_target_image:
            loss_metrics = self.loss_computer.forward_withconf(
                rendered_images,
                target.image,
                input.difix3D_image,
                rendered_confs,
                difix3d_confs,
                difix3d_renders,
            )
        else:
            loss_metrics = None
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            conf=rendered_confs,
            difix3D_conf=difix3d_confs[:, :, 0:1],
            difix3D_render=difix3d_renders        
            )
        
        return result




    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4

        video_rendering_list = []
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=n_patches* v_input)

            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches            
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) # [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = self.transformer_input_layernorm(
                cur_concat_input_tokens
            )

            transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)

            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_patches, n_patches], dim=1
            ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

            height, width = target.image_h_w

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            # [b, v_target*n_patches, p*p*3]
            video_rendering = self.image_token_decoder(pred_target_image_tokens)
            # conf = self.image_conf_decoder(pred_target_image_tokens)
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()
            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering


        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, reverse=True)
            preferred = [name for name in ckpt_names if name.startswith("ckpt_")]
            ckpt_names = preferred or ckpt_names
            if not ckpt_names:
                raise FileNotFoundError(f"No .pt LVSM checkpoint found in {load_path}")
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]

        ckpt_path = ckpt_paths[0]
        print(f"Loading LVSM checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.load_state_dict(state_dict, strict=False)
        return 0
