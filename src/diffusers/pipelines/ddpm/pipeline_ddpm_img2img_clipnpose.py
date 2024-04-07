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
import PIL
from matplotlib import pyplot as plt

from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMImg2ImgCLIPPosePipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, feature_extractor, image_encoder, kwargs):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder)
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
        guidance: float = 1.0,
        mix_guidance: float = 1.0,
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

        do_classifier_free_guidance = False
        if guidance > 1.0:
            do_classifier_free_guidance = True

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # last timestep in clean images is what we have to predict, so remove that
        clean_images = inpainting_image[0][:, :-1, :, :, :]
        B, T, C, H, W = clean_images.shape

        # TODO: needs to be manually adjusted for grayscale vs rgb

        ###################### grayscale ###########################################################################
        if self.data_format == "rgbd":
            image_preprocessor_input = clean_images[:, :T, ...].reshape(B*T*C, 1, H, W)
        else:
            image_preprocessor_input = clean_images[:, :T, ...].reshape(B*T, C, H, W)
        image_preprocessor_input = image_preprocessor_input / 2 + 0.5
        device = image_preprocessor_input.device
        image_preprocessor_input = image_preprocessor_input.repeat(1, 3, 1, 1).float().cpu().numpy()
        ############################################################################################################

        ########################3 colored ##########################################################################
        """
        if "d" in self.data_format:
            image_preprocessor_input_depth = clean_images[:, :, -1:, ...].reshape(B, T, 1, 1, H, W).repeat(1, 1, 1, 3, 1, 1)
        if "rgb" in self.data_format:
            image_preprocessor_input_rgb   = clean_images[:, :, :3, ...].reshape(B, T, 1, 3, H, W)
        if self.data_format == "d":
            image_preprocessor_input = image_preprocessor_input_depth.reshape(-1, 3, H, W)
        elif self.data_format == "rgb":
            image_preprocessor_input = image_preprocessor_input_rgb.reshape(-1, 3, H, W)
        else:
            image_preprocessor_input = torch.cat([image_preprocessor_input_depth, image_preprocessor_input_rgb], dim=-4).reshape(-1, 3, H, W)
        image_preprocessor_input = image_preprocessor_input / 2 + 0.5
        device = image_preprocessor_input.device
        image_preprocessor_input = image_preprocessor_input.float().cpu().numpy()
        """
        ############################################################################################################

        image_resized = [self.feature_extractor.resize(image=image, size={"shortest_edge": 224}, resample=PIL.Image.BICUBIC, input_data_format="channels_first") for image in image_preprocessor_input]
        image_resized = np.stack(image_resized, axis=0)
        image_preprocessed = (image_resized - 0.5) / 0.5
        image_preprocessed = torch.from_numpy(image_preprocessed).to(device)
        print("image preprocessed shape", image_preprocessed.shape)

        dtype = next(self.image_encoder.parameters()).dtype
        image_preprocessed = image_preprocessed.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image_preprocessed).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)  # N x 1 x 768
        image_embeddings = image_embeddings.reshape(B, T, C, -1).reshape(B, T, -1)
        print("final shape of image embeddings", image_embeddings.shape)

        if self.use_rendering:
            assert self.train_with_plucker_coords
            rendering_poses = inpainting_image[1]

        clean_images = clean_images.reshape(B, -1, H, W)
        i = 0

        for t in self.progress_bar(self.scheduler.timesteps):
            # 0. prepare the model inputs
            # choose the type of masking strategy
            # remember that both the clean and noisy images are 5D tensors now
            if batch_size != B:
                image = torch.cat([image] * 2)

            model_inputs = torch.cat([image, clean_images], axis=1)

            if do_classifier_free_guidance:
                model_inputs = torch.cat([model_inputs] * 2)

            # 1. Pass the inputs through the model
            if self.use_rendering:
                model_output = self.unet(
                            model_inputs,
                            t,
                            encoder_hidden_states=(image_embeddings, rendering_poses),
                            guidance=guidance,
                            input_indices=None,
                            output_indices=None).sample
            else:
                model_output = self.unet(model_inputs, t).sample

             # perform guidance
            if do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + guidance * (model_output_cond - model_output_uncond)

            if batch_size != B:
                jump, autoregressive = model_output.chunk(2)
                model_output = autoregressive + mix_guidance * (jump - autoregressive)
                image, _ = image.chunk(2)

            # 2. compute previous image: x_t -> x_t-1
            scheduler_output = self.scheduler.step(model_output, t, image)  # , generator=generator).prev_sample
            image = scheduler_output.prev_sample
            im = scheduler_output.pred_original_sample

            # im = (im / 2 + 0.5).clamp(0, 1)
            # im = im.cpu().permute(0, 2, 3, 1).numpy()
            # plt.imsave(f'/data3/tkhurana/diffusers/logs/x{i}_firstiter.png', im[0, :, :, 0])

            i += 1


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=(image, [1]))
