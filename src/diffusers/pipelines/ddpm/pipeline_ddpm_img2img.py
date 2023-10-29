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


class DDPMImg2ImgPipeline(DiffusionPipeline):
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
        self.register_to_config(kwargs=kwargs)
        self.n_input = kwargs["n_input"]
        self.n_output = kwargs["n_output"]
        self.masking_strategy = kwargs["masking_strategy"]
        self.train_with_plucker_coords = kwargs["train_with_plucker_coords"]
        self.use_rendering = kwargs["use_rendering"]
        self.data_format = kwargs["data_format"]

    @torch.no_grad()
    def __call__(
        self,
        inpainting_image: Union[torch.cuda.FloatTensor, Tuple],
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        user_num_masks=None,
        user_time_indices=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            inpainting_image:
                The image to use as context for the inpainted output. Currently implemented for depth video forecasting.
                This image should have the first dimension as the batch dimension. for video forecasting this shape should
                be B x F x H x W. This shape has been changed to B T C H W for this pipeline.
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
            if not self.use_rendering:
                image_shape = (
                    batch_size,
                    self.unet.config.out_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (
                    batch_size,
                    self.unet.config.out_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )

        else:
            if self.use_rendering:
                image_shape = (batch_size, self.unet.config.out_channels, *self.unet.config.sample_size)
            else:
                image_shape = (batch_size, 1, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)
            # image = image.repeat(batch_size, 1, 1, 1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # last timestep in clean images is what we have to predict, so remove that
        clean_images = inpainting_image[0][:, :-1, :, :, :]

        if self.use_rendering:
            assert self.train_with_plucker_coords
            rendering_poses = inpainting_image[1]

        B, T, C, H, W = clean_images.shape
        clean_images = clean_images.reshape(B, -1, H, W)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 0. prepare the model inputs
            # choose the type of masking strategy
            # remember that both the clean and noisy images are 5D tensors now
            model_inputs = torch.cat([image, clean_images], axis=1)

            # 1. Pass the inputs through the model
            if self.use_rendering:
                model_output = self.unet(
                            model_inputs,
                            t,
                            encoder_hidden_states=rendering_poses,
                            input_indices=None,
                            output_indices=None).sample
            else:
                model_output = self.unet(model_inputs, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample  # , generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=(image, [1]))
