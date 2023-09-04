import torch
import inspect
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DiffusionPipeline, DDPMPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, DDPMScheduler, DDPMConditioningScheduler

from totalrecondataset import utils_mini as totalrecon_utils

def preprocess_depth(depth_map):
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = Image.fromarray(depth_map.astype("uint8"))

    image_transforms = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    depth_map = image_transforms(depth_map)

    return depth_map


def make_gif(frames, save_path, duration):
    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames,
        save_all=True, duration=duration, loop=0)

if __name__ == "__main__":

    # Initialize the UNet2D
    folder_name = "/compute/trinity-1-38/tkhurana/diffusers-runs/logs/totalrecon-dddpm-ema-unconditional-depthvideo-64-8frames-secondrun"
    # folder_name = "/compute/trinity-1-38/tkhurana/diffusers-runs/logs/ddpm-ema-inpainting-depthvideo-64-8frames-concat-imagemaskedimagemask/"
    duration = 800
    cond_duration = 400
    reconstruction_guidance = True
    inpainting = False
    prediction_type = "epsilon"
    Scheduler = DDPMScheduler # DDPMConditioningScheduler
    # num = "4_16frames_8framecond"
    num = "4_reconguidance_4framecond"
    # unet = UNet2DModel.from_pretrained("./ddpm-ema-depthvideo-512/checkpoint-34500/unet_ema/")
    # unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-142500/unet_ema")
    unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-3500/unet_ema")
    # unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-27000/unet_ema")
    # unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-323000/unet_ema")

    # Load conditioning
    """
    """
    frame0 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/catamelie-dualrig-fgbg002/depths/00315.depth', (256, 192)))
    frame1 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/catamelie-dualrig-fgbg002/depths/00330.depth', (256, 192)))
    frame2 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/catamelie-dualrig-fgbg002/depths/00345.depth', (256, 192)))
    frame3 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/catamelie-dualrig-fgbg002/depths/00360.depth', (256, 192)))
    """
    frame0 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00315.depth', (256, 192)))
    frame1 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00330.depth', (256, 192)))
    frame2 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00345.depth', (256, 192)))
    frame3 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00360.depth', (256, 192)))
    frame0 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/humanhouse-dualrig-fgbg000/depths/00315.depth', (256, 192)))
    frame1 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/humanhouse-dualrig-fgbg000/depths/00330.depth', (256, 192)))
    frame2 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/humanhouse-dualrig-fgbg000/depths/00345.depth', (256, 192)))
    frame3 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/humanhouse-dualrig-fgbg000/depths/00360.depth', (256, 192)))

    frame0 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00315.depth', (256, 192)))
    frame1 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00330.depth', (256, 192)))
    frame2 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00345.depth', (256, 192)))
    frame3 = preprocess_depth(totalrecon_utils.read_depth('/compute/trinity-1-38/tkhurana/totalrecondataset/leftcam/dog-dualrig-fgbg000/depths/00360.depth', (256, 192)))
    """
    # conditioning = torch.concatenate([frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7], dim=0)
    # conditioning_plot = torch.concatenate([frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7], dim=0)
    conditioning = torch.concatenate([frame0, frame1, frame2, frame3], dim=0)
    conditioning_plot = torch.concatenate([frame0, frame1, frame2, frame3], dim=0)

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(Scheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        print("was able to specify the prediction type")
        noise_scheduler = Scheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type=prediction_type,
        )
    else:
        noise_scheduler = Scheduler(num_train_timesteps=1000, beta_schedule="linear")

    # noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear")
    if reconstruction_guidance:
        pipeline = DDPMReconstructionPipeline(unet=unet, scheduler=noise_scheduler)
    elif inpainting:
        pipeline = DDPMInpaintingPipeline(unet=unet, scheduler=noise_scheduler)
    else:
        pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)

    if reconstruction_guidance:
        video = pipeline(
                # generator=generator,
                batch_size=1,
                num_inference_steps=50,
                # cond_inds=[0,1,2,3,4,5,6,7],
                cond_inds=[0,1,2,3],
                recon_scale=10,
                conditioning=conditioning,
                output_type="numpy").images[0]
    elif inpainting:
        video = pipeline(
                inpainting_image=conditioning[None, ...],
                # generator=generator,
                batch_size=1,
                num_inference_steps=50,
                output_type="numpy").images[0]
    else:
        video = pipeline(
                # generator=generator,
                batch_size=1,
                num_inference_steps=50,
                output_type="numpy").images[0]

    if reconstruction_guidance or inpainting:
        images = []
        print("shape of video", video.shape)
        for i in range(video.shape[2]):
            frame = video[..., i]
            frame = ((frame + 1.0) / 2.0) * 255.0
            frame = np.dstack([frame, frame, frame])
            img = Image.fromarray(frame.astype("uint8"))
            # img.save(f"frame{num}_{i}.png")
            images.append(img)
        make_gif(images, f"{folder_name}/results/frame{num}.gif", duration)

        images = []
        for i in range(conditioning_plot.shape[0]):
            frame = conditioning_plot[i, ...]
            frame = ((frame + 1.0) / 2.0) * 255.0
            frame = np.dstack([frame, frame, frame])
            img = Image.fromarray(frame.astype("uint8"))
            images.append(img)
            # img.save(f"conditioning{num}_{i}.png")
        make_gif(images, f"{folder_name}/results/conditioning{num}.gif", cond_duration)

    else:

        images = []
        for i in range(video.shape[2]):
            frame = video[..., i]
            frame = ((frame + 1.0) / 2.0) * 255.0
            frame = np.dstack([frame, frame, frame])
            img = Image.fromarray(frame.astype("uint8"))
            # img.save(f"uncond_frame{num}_{i}.png")
            images.append(img)
        make_gif(images, f"{folder_name}/results/uncond_frame{num}.gif", duration)
