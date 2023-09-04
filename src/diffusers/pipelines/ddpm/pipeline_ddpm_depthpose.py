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


from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMInpaintingPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, kwargs):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.n_input = kwargs["n_input"]
        self.n_output = kwargs["n_output"]
        self.masking_strategy = kwargs["masking_strategy"]
        self.train_with_plucker_coords = kwargs["train_with_plucker_coords"]

    @torch.no_grad()
    def __call__(
        self,
        inpainting_image: torch.cuda.FloatTensor,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        # note the change to out channels
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.out_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.out_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 0. prepare the model inputs
            # if plucker coords == True then the inpainting image is a 5D tensor
            # with the third dimension containing the plucker coords + depth map
            # otherwise the inpainting image is a 4D tensor with the second dimension
            # being the number of timesteps.
            clean_images = inpainting_image

            if not self.train_with_plucker_coords:
                clean_images = clean_images[:, :, None, :, :]

            B, T, C, H, W = clean_images.shape
            clean_depths = clean_images[:, :, 0, :, :]

            if self.train_with_plucker_coords:
                noisy_images = torch.cat([image[:, :, None, :, :], clean_images[:, :, 1:, :, :]], dim=2)
            else:
                noisy_images = image

            # choose the type of masking strategy
            # remember that both the clean and noisy images are 5D tensors now
            if self.masking_strategy == "none":
                # merge the timesteps and height dimension of all voxel grids
                model_inputs = torch.cat([
                    noisy_images.reshape(B, T*C, H, W),
                    clean_images.reshape(B, T*C, H, W)], dim=1)

            elif self.masking_strategy == "all":
                # merge the timesteps and height dimension of all voxel grids
                model_inputs = noisy_images.reshape(B, T*C, H, W)

            elif self.masking_strategy == "random" or self.masking_strategy == "random-half" or self.masking_strategy == "half":
                # first pick a random number of frames to mask
                # then pick above number of frame IDs to mask
                num_masks = torch.randint(0, self.n_input + self.n_output, (1,)) + 1
                num_masks = num_masks.numpy()[0]
                if self.masking_strategy == "random-half":
                    num_masks = self.n_input
                time_indices = torch.from_numpy(
                    np.random.choice(self.n_input + self.n_output, size=(num_masks, ), replace=False)
                )
                if self.masking_strategy == "half":
                    time_indices = torch.arange(self.n_input, self.n_input + self.n_output, 1)
                mask_images = torch.zeros_like(clean_images[:, :, 0, :, :])  # 4d
                mask_images[:, time_indices, ...] = torch.ones_like(clean_images[:, time_indices, 0, ...])
                clean_depths_masked = clean_depths * (1 - mask_images)
                if self.train_with_plucker_coords:
                    clean_images_masked = torch.cat([clean_depths_masked[:, :, None, :, :], clean_images[:, :, 1:, :, :]], dim=2)
                else:
                    clean_images_masked = clean_depths_masked
                model_inputs = torch.cat([
                    noisy_images.reshape(B, T*C, H, W),
                    clean_images_masked.reshape(B, T*C, H, W),
                    mask_images.reshape(B, T, H, W)], dim=1)

            # 1. predict noise model_output
            model_output = self.unet(model_inputs, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample  # , generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
