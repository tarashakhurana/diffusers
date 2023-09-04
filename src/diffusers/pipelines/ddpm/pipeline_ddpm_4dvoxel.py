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


from typing import List, Optional, Tuple, Union, Dict

import torch

from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from PIL import Image
import numpy as np


def get_grid_mask(points, pc_range):
    points = points.T
    mask1 = np.logical_and(pc_range[0] <= points[0], points[0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= points[1], points[1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= points[2], points[2] <= pc_range[5])

    mask = mask1 & mask2 & mask3

    # print("shape of mask being returned", mask.shape)
    return mask


def save_bev(filename, points, resolution, pc_range, voxel_size):
    bev = np.ones((resolution, resolution)) * 255
    mask = get_grid_mask(points.cpu().numpy(), pc_range)
    points = points[mask]
    points_normalized = (points.cpu().numpy() - np.array(pc_range[:3])[None, ...] ) / (voxel_size)

    bev[points_normalized.astype(int)[:, 1], points_normalized.astype(int)[:, 0]] = 0
    img = Image.fromarray(bev.astype(np.uint8))
    img.save(filename)


def get_points(origin, tindex, points, depths):
    # we have to copy the groundtruth (sorry!) because it contains info
    # about the padded points, and we need that info before entering the UNet
    new_points = torch.clone(points)
    for b in range(origin.shape[0]):
        for t in range(origin.shape[1]):
            _origin = origin[b, t:t+1, ...]
            _points = points[b, tindex[b] == t, ...]
            _magnitude = torch.sqrt(torch.sum(torch.square(_points - _origin), dim=-1))
            print("origin points magnitude shape", _origin.shape, _points.shape, _magnitude.shape)
            _unitdir = _points / _magnitude[..., None]
            _depths = depths[b, tindex[b] == t, ...]
            _new_points = _origin + _unitdir # * _depths
            new_points[b, tindex[b] == t, ...] = _new_points

    return new_points


def get_rendered_pcds(origin, points, tindex, pred_dist):
    pred_points = torch.zeros_like(points)
    for b in range(points.shape[0]):
        for t in range(origin.shape[1]):
            mask = tindex[b] == t
            # skip the ones with no data
            if not mask.any():
                continue
            _pts = points[b, mask, :]
            # use ground truth lidar points for the raycasting direction
            v = _pts - origin[b, t][None, :]
            d = v / torch.sqrt((v ** 2).sum(dim=-1, keepdims=True))
            pred_pts = origin[b, t][None, :] + d * pred_dist[b, mask][..., None]
            pred_points[b, mask, :] = pred_pts
    return pred_points


class DDPM4DVoxelInpaintingPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        input_origin: torch.cuda.FloatTensor,
        input_points: torch.cuda.FloatTensor,
        input_tindex: torch.cuda.FloatTensor,
        output_origin: torch.cuda.FloatTensor,
        output_points: torch.cuda.FloatTensor,
        output_tindex: torch.cuda.FloatTensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        return_all: bool = True,
    ) -> Union[Tuple, Dict]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            inpainting_image:
                The image to use as context for the inpainted output. Currently implemented for depth video forecasting.
                This image should have the first dimension as the batch dimension. for video forecasting this shape should
                be B x F x H x W.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        # note the change to out channels

        input_origin = input_origin.to(self.device)
        input_points = input_points.to(self.device)
        input_tindex = input_tindex.to(self.device)
        output_origin = output_origin.to(self.device)
        output_points = output_points.to(self.device)
        output_tindex = output_tindex.to(self.device)

        all_tindex = torch.cat([input_tindex, output_tindex + self.unet.config.n_input], dim=1)
        all_origin = torch.cat([input_origin, output_origin], dim=1)
        all_points = torch.cat([input_points, output_points], dim=1)
        points_shape = all_points.shape
        norm_min = torch.tensor(self.unet.config.pc_range, device=self.device)[:3][None, None, :]
        norm_max = torch.tensor(self.unet.config.pc_range, device=self.device)[3:][None, None, :]

        if "noise_strategy" not in self.unet.config:
            self.unet.config.noise_strategy = "points"

        if self.unet.config.noise_strategy == "points":
            points_normalized = randn_tensor(points_shape, generator=generator, device=self.device)
        elif self.unet.config.noise_strategy == "raydepth":
            all_raydepths = randn_tensor((*points_shape[:-1], 1), generator=generator, device=self.device)
            all_points_normalized = (all_points - norm_min.to(all_points.device)) / (norm_max.to(all_points.device) - norm_min.to(all_points.device))
            all_origin_normalized = (all_origin - norm_min.to(all_origin.device)) / (norm_max.to(all_origin.device) - norm_min.to(all_origin.device))
            points_normalized = get_points(all_origin_normalized, all_tindex, all_points_normalized, all_raydepths)

        points = (points_normalized * (norm_max - norm_min)) + norm_min
        """
        save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/starting.png',
                    points[0][all_tindex[0] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
        """

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        num_masks = torch.tensor([11]) # torch.randint(0, self.unet.config.n_input + self.unet.config.n_output, (1,)) + 1

        for t in self.progress_bar(self.scheduler.timesteps):

            """
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/first_{number:06}.png'.format(number=1000-t),
                    points[0][all_tindex[0] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/second_{number:06}.png'.format(number=1000-t),
                    points[1][all_tindex[1] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/third_{number:06}.png'.format(number=1000-t),
                    points[2][all_tindex[2] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            """

            # 1. predict noise model_output
            ret_dict = self.unet(
                    points,
                    t,
                    input_origin,
                    input_points,
                    input_tindex,
                    output_origin,
                    output_points,
                    output_tindex,
                    mode="testing",
                    num_masks=num_masks).sample

            assert ret_dict["pred_dist"].shape[1] == all_points.shape[1]
            model_output = get_rendered_pcds(all_origin, all_points, all_tindex, ret_dict["pred_dist"])

            """
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/modelout_first_{number:06}.png'.format(number=1000-t),
                    model_output[0][all_tindex[0] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/modelout_second_{number:06}.png'.format(number=1000-t),
                    model_output[1][all_tindex[1] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            save_bev(
                    '/data3/tkhurana/diffusers/logs/test_visuals/modelout_third_{number:06}.png'.format(number=1000-t),
                    model_output[2][all_tindex[2] == 6],
                    self.unet.config.sample_size,
                    self.unet.config.pc_range,
                    self.unet.config.voxel_size
            )
            """

            model_output_normalized = (model_output - norm_min) / (norm_max - norm_min)

            # 2. compute previous image: x_t -> x_t-1
            points_normalized = self.scheduler.step(model_output_normalized, t, points_normalized).prev_sample
            points = (points_normalized * (norm_max - norm_min)) + norm_min

        points = points.cpu().numpy()

        if not return_all:
            return (points,)

        return {'points': points,
                'ret_dict': ret_dict}
