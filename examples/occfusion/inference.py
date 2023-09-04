import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionDepth2DepthPipeline as occfusion
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel


def preprocess_depth(depth_map):
    depth_map = np.array(depth_curr) / (256.0 * 80.0)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = Image.fromarray(depth_map.astype("uint8"))
    image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    depth_map = image_transforms(depth_map)
    return depth_map


if __name__ == "__main__":

    unet = UNet2DConditionModel.from_pretrained("./logs")
    pipe = occfusion.from_pretrained("stabilityai/stable-diffusion-2-depth")
    pipe.unet = unet
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    depth_map = np.array(Image.open("/compute/trinity-1-38/tkhurana/TAO-depth/zoe/frames/test/LaSOT/skateboard-14/00000999.png")) / 256.0
    depth_map = preprocess_depth(depth_map)

    image = pipe("", depth_map=depth_map, strength=0.5, num_inference_steps=50).images[0]

    image = ((image + 1) / 2) * 255.0
    image = image.astype("uint8")
    image = Image.fromarray(image)
    image.save("sample_result.png")

