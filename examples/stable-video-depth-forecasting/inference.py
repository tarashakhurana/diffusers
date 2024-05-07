import torch
from PIL import Image
import numpy as np

from diffusers import StableVideoDiffusionPipeline, UNetSpatioTemporalConditionModel
from diffusers.utils import load_image, export_to_video
from data import TAOMAEDataset

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
pipe.unet = UNetSpatioTemporalConditionModel.from_pretrained("outputs/checkpoint-23000", subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float16,)
pipe.enable_model_cpu_offload()

# Load the conditioning image
eval_dataset = TAOMAEDataset(
        instance_data_root="/compute/trinity-2-25/tkhurana/datasets/data/data/tkhurana/TAO-depth/zoe/frames/train/",
        size=320,
        center_crop=False,
        num_images=14,
        fps=15,
        horizon=2,
        load_rgb=False,
        rgb_data_root="/data3/chengyeh/TAO/frames/train/",
        split="val",
        normalization_factor=20480,
)

generator = torch.manual_seed(42)

for i in range(len(eval_dataset)):
    image = eval_dataset[i]["pixel_values"][0].permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255.0).astype(np.uint8)
    image = Image.fromarray(image)
    frames = pipe(image, decode_chunk_size=8, generator=generator, width=320, height=320).frames[0]

    export_to_video(frames, f"/data3/tkhurana/diffusers/svd/logs/generated_{i}.mp4", fps=7)
