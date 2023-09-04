# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from .unet_2d import UNet2DModel
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["/home/tkhurana/freespaceForecasting/ff3d-private/lib/dvr/dvr.cpp", "/home/tkhurana/freespaceForecasting/ff3d-private/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])


@dataclass
class UNetStatic3DVoxelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: Dict


class UNetStatic3DVoxelModel(ModelMixin, ConfigMixin):
    r"""
    UNetStatic3DVoxelModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 5,
        out_channels: int = 5,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        pc_range: List[float] = [-5.0, -5.0, -2.5, 5.0, 5.0, 2.5],
        voxel_size: float = 0.2,
        n_input: int = 5,
        n_output: int = 5,
        masking_strategy="none"
    ):
        super().__init__()


        self.model_type = "dynamic"
        self.loss_type = "l1"
        self.masking_strategy = masking_strategy

        self.n_input = n_input
        self.n_output = n_output

        self.n_height = int(round((pc_range[5] - pc_range[2]) / voxel_size))
        self.n_length = int(round((pc_range[4] - pc_range[1]) / voxel_size))
        self.n_width = int(round((pc_range[3] - pc_range[0]) / voxel_size))

        self.dynamic_grid = [self.n_input + self.n_output, self.n_height, self.n_length, self.n_width]
        self.input_output_grid = [1, self.n_height, self.n_length, self.n_width]
        print("input+output grid:", self.input_output_grid)

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )
        self.norm_min = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.norm_max = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[3:])[None, None, :], requires_grad=False
        )

        _in_channels = self.n_height

        _out_channels = self.n_height

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds
        )


    def forward(
        self,
        noisy_points: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        input_origin_orig: torch.FloatTensor,
        input_points_orig: torch.FloatTensor,
        input_tindex: torch.FloatTensor,
        output_origin_orig: torch.FloatTensor,
        output_points_orig: torch.FloatTensor,
        output_tindex: torch.FloatTensor,
        output_labels: Optional[torch.Tensor] = None,
        loss: Optional[str] = None,
        mode: Optional[str] = "training",
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        num_masks = None
    ) -> Union[UNetStatic3DVoxelOutput, Tuple]:
        r"""
        Args:
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        if loss == None:
            loss = self.loss_type

        # preprocess input/output points
        input_origin = ((input_origin_orig - self.offset) / self.scaler).float()
        input_points = ((input_points_orig - self.offset) / self.scaler).float()
        output_origin = ((output_origin_orig - self.offset) / self.scaler).float()
        output_points = ((output_points_orig - self.offset) / self.scaler).float()

        # concatenate all points and normalize them to [-1, 1] to add noise to them later
        all_tindex = torch.cat([input_tindex, output_tindex + self.n_input], dim=1)
        all_origin = torch.cat([input_origin, output_origin], dim=1)
        all_points = torch.cat([input_points, output_points], dim=1)

        # -1: freespace, 0: unknown, 1: occupied
        # N x T1 x H x L x W
        dynamic_occupancy = dvr.init(all_points, all_tindex, self.dynamic_grid)
        total_occupancy = dvr.init(all_points, all_tindex, self.input_output_grid)

        # double check
        N, T_total, H, L, W = total_occupancy.shape
        assert T_total == 1 and H == self.n_height

        clean_4dvoxel = total_occupancy

        # Initialize a noisy voxel grid
        noisy_4dvoxel = dvr.init(noisy_points, all_tindex, self.input_output_grid)

        # Mask the clean voxel grid according to the masking strategy
        # Create a mask according to the masking strategy
        # Prepare the inputs to the UNet

        if self.masking_strategy == "none":
            # merge the timesteps and height dimension of all voxel grids
            noisy_4dvoxel = noisy_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel = clean_4dvoxel.reshape(N, -1, L, W)
            model_inputs = torch.cat([noisy_4dvoxel, clean_4dvoxel], dim=1)

        elif self.masking_strategy == "all":
            # merge the timesteps and height dimension of all voxel grids
            noisy_4dvoxel = noisy_4dvoxel.reshape(N, -1, L, W)
            model_inputs = noisy_4dvoxel

        elif self.masking_strategy == "random":
            # first pick a random number of frames to mask
            # then pick above number of frame IDs to mask
            frame_indices = torch.from_numpy(
                    np.random.choice(self.n_input + self.n_output, size=(num_masks.numpy()[0], ), replace=False)
                )
            mask_4dvoxel = torch.zeros_like(dynamic_occupancy)
            mask_4dvoxel[:, frame_indices, ...] = torch.ones_like(dynamic_occupancy[:, frame_indices, ...])
            clean_4dvoxel_masked = dynamic_occupancy * (1 - mask_4dvoxel)

            # reduce the channel dimension to 1 for static occupancy grid
            clean_4dvoxel_masked = torch.sum(clean_4dvoxel_masked, 1)[:, None, ...]
            clean_4dvoxel_masked = torch.where(clean_4dvoxel_masked > 0, 1, 0)
            mask_4dvoxel = torch.sum(mask_4dvoxel, 1)[:, None, ...]
            mask_4dvoxel = torch.where(mask_4dvoxel > 0, 1, 0)

            # merge the timesteps and height dimension of all voxel grids
            noisy_4dvoxel = noisy_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel = clean_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel_masked = clean_4dvoxel_masked.reshape(N, -1, L, W)
            mask_4dvoxel = mask_4dvoxel.reshape(N, -1, L, W)
            model_inputs = torch.cat([noisy_4dvoxel, clean_4dvoxel_masked, mask_4dvoxel], dim=1)

        elif self.masking_strategy == "half":
            # mask the last n_output frames
            mask_4dvoxel = torch.zeros_like(dynamic_occupancy)
            mask_4dvoxel[:, self.n_input:, ...] = torch.ones_like(dynamic_occupancy[:, self.n_input:, ...])
            clean_4dvoxel_masked = dynamic_occupancy * (1 - mask_4dvoxel)

            # reduce the channel dimension to 1 for static occupancy grid
            clean_4dvoxel_masked = torch.sum(clean_4dvoxel_masked, 1)[:, None, ...]
            clean_4dvoxel_masked = torch.where(clean_4dvoxel_masked > 0, 1, 0)
            mask_4dvoxel = torch.sum(mask_4dvoxel, 1)[:, None, ...]
            mask_4dvoxel = torch.where(mask_4dvoxel > 0, 1, 0)

            # merge the timesteps and height dimension of all voxel grids
            noisy_4dvoxel = noisy_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel = clean_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel_masked = clean_4dvoxel_masked.reshape(N, -1, L, W)
            mask_4dvoxel = mask_4dvoxel.reshape(N, -1, L, W)
            model_inputs = torch.cat([noisy_4dvoxel, clean_4dvoxel_masked, mask_4dvoxel], dim=1)

        elif self.masking_strategy == "random-half":
            # choose any number of n_input frames to mask
            frame_indices = torch.from_numpy(np.random.choice(self.n_input + self.n_output, size=(self.n_output, ), replace=False))
            mask_4dvoxel = torch.zeros_like(dynamic_occupancy)
            mask_4dvoxel[:, frame_indices, ...] = torch.ones_like(dynamic_occupancy[:, frame_indices, ...])
            clean_4dvoxel_masked = dynamic_occupancy * (1 - mask_4dvoxel)

            # reduce the channel dimension to 1 for static occupancy grid
            clean_4dvoxel_masked = torch.sum(clean_4dvoxel_masked, 1)[:, None, ...]
            clean_4dvoxel_masked = torch.where(clean_4dvoxel_masked > 0, 1, 0)
            mask_4dvoxel = torch.sum(mask_4dvoxel, 1)[:, None, ...]
            mask_4dvoxel = torch.where(mask_4dvoxel > 0, 1, 0)

            # merge the timesteps and height dimension of all voxel grids
            noisy_4dvoxel = noisy_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel = clean_4dvoxel.reshape(N, -1, L, W)
            clean_4dvoxel_masked = clean_4dvoxel_masked.reshape(N, -1, L, W)
            mask_4dvoxel = mask_4dvoxel.reshape(N, -1, L, W)
            model_inputs = torch.cat([noisy_4dvoxel, clean_4dvoxel_masked, mask_4dvoxel], dim=1)

        # Predict the noise residual or the denoised sample itself
        output = self.unet(model_inputs, timestep, class_labels, return_dict).sample

        #
        output = output.reshape(N, -1, H, L, W)

        ret_dict = {}

        if mode == "training":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)

                if sigma.requires_grad:
                    pred_dist, gt_dist, grad_sigma = dvr.render(
                        sigma,
                        all_origin,
                        all_points,
                        all_tindex,
                        loss
                    )
                    # take care of nans and infs if any
                    invalid = torch.isnan(grad_sigma)
                    grad_sigma[invalid] = 0.0
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    invalid = torch.isinf(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0
                    # sigma.backward(grad_sigma)
                    ret_dict["sigma"] = sigma
                    ret_dict["grad_sigma"] = grad_sigma
                else:
                    pred_dist, gt_dist = dvr.render_forward(
                        sigma,
                        all_origin,
                        all_points,
                        all_tindex,
                        self.input_output_grid,
                        "train"
                    )
                    # take care of nans if any
                    invalid = torch.isnan(pred_dist)
                    pred_dist[invalid] = 0.0
                    gt_dist[invalid] = 0.0

                pred_dist *= self.voxel_size
                gt_dist *= self.voxel_size

                # compute training losses
                valid = gt_dist >= 0
                count = valid.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                # record training losses
                if count == 0:
                    count = 1
                ret_dict["l1_loss"] = l1_loss[valid].sum() / count
                ret_dict["l2_loss"] = l2_loss[valid].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[valid].sum() / count

            else:
                raise RuntimeError(f"Unknown loss type: {loss}")

        elif mode in ["testing", "plotting"]:

            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pred_dist, gt_dist = dvr.render_forward(
                    sigma, all_origin, all_points, all_tindex, self.input_output_grid, "test")
                pog = 1 - torch.exp(-sigma)

                pred_dist = pred_dist.detach()
                gt_dist = gt_dist.detach()

            #
            pred_dist *= self.voxel_size
            gt_dist *= self.voxel_size

            if mode == "testing":
                # L1 distance and friends
                mask = gt_dist > 0
                count = mask.sum()
                l1_loss = torch.abs(gt_dist - pred_dist)
                l2_loss = ((gt_dist - pred_dist) ** 2) / 2
                absrel_loss = torch.abs(gt_dist - pred_dist) / gt_dist

                ret_dict["l1_loss"] = l1_loss[mask].sum() / count
                ret_dict["l2_loss"] = l2_loss[mask].sum() / count
                ret_dict["absrel_loss"] = absrel_loss[mask].sum() / count

                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict['pog'] = pog.detach()
                ret_dict["sigma"] = sigma.detach()

            if mode == "plotting":
                ret_dict["gt_dist"] = gt_dist
                ret_dict["pred_dist"] = pred_dist
                ret_dict["pog"] = pog

        elif mode == "dumping":
            if loss in ["l1", "l2", "absrel"]:
                sigma = F.relu(output, inplace=True)
                pog = 1 - torch.exp(-sigma)

            pog_max, _ = pog.max(dim=1)
            ret_dict["pog_max"] = pog_max

        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        return UNetStatic3DVoxelOutput(sample=ret_dict)
