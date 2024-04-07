import torch
import requests
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from data import TAOForecastingDataset
from torchvision import transforms
from diffusers import StableDiffusionDepth2ImgPipeline
import numpy as np

dataset = TAOForecastingDataset(
        instance_data_root="/data/tkhurana/TAO-depth/zoe/frames/train/",
        size=64,
        center_crop=False,
        num_images=3,
        autoregressive=True,
        num_autoregressive_frames=10,
        fps=30,
        horizon=1,
        offset=int(30 / 2),
        split="val",
        load_rgb=True,
        rgb_data_root="/data3/chengyeh/TAO/frames/train/",
        normalization_factor=20480.0,
        visualize=False
    )

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(512),
            ]
        )

depth_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]
    )


url = "/data3/chengyeh/TAO/frames/train/LaSOT/bicycle-4/00002754.jpg"
init_image = image_transforms(Image.open(url))
prompt = "a real-world human girl standing on a bike while biking in an indoor hall, photorealistic image, natural environment"
n_propmt = "ugly looking, bad quality, cartoonish"


depth_map_path = "/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_autoregressive_suppconsecutive/LaSOT/bicycle-4_2_depth/00001_3.png"
depth_map = ImageOps.grayscale(Image.open(depth_map_path))
depth_map = depth_transforms(depth_map)
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.2, guidance_scale=5.0, depth_map=depth_map).images[0]
plt.imsave("/data3/tkhurana/1.png", np.array(image))


depth_map_path = "/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_autoregressive_suppconsecutive/LaSOT/bicycle-4_2_depth/00000_3.png"
depth_map = ImageOps.grayscale(Image.open(depth_map_path))
depth_map = depth_transforms(depth_map)
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.2, guidance_scale=5.0, depth_map=depth_map).images[0]
plt.imsave("/data3/tkhurana/2.png", np.array(image))


depth_map_path = "/data3/tkhurana/step3.png"
depth_map = ImageOps.grayscale(Image.open(depth_map_path))
depth_map = depth_transforms(depth_map)
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.2, guidance_scale=5.0, depth_map=depth_map).images[0]
plt.imsave("/data3/tkhurana/3.png", np.array(image))


depth_map_path = "/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_autoregressive_suppconsecutive/LaSOT/bicycle-4_2_depth/00005_3.png"
depth_map = ImageOps.grayscale(Image.open(depth_map_path))
depth_map = depth_transforms(depth_map)
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.2, guidance_scale=5.0, depth_map=depth_map).images[0]
plt.imsave("/data3/tkhurana/4.png", np.array(image))
