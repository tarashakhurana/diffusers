import os
import torch
import inspect
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMDepthPoseInpaintingPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, DDPMScheduler, DDPMConditioningScheduler
from matplotlib import pyplot as plt

from data import OccfusionDataset
from utils import write_pointcloud


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3))
    a_01 = torch.sum(mask * prediction, (1, 2, 3))
    a_11 = torch.sum(mask, (1, 2, 3))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3))
    b_1 = torch.sum(mask * target, (1, 2, 3))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1


def collate_fn(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    ray_origin = torch.stack([example["ray_origin"] for example in examples])
    ray_origin = ray_origin.to(memory_format=torch.contiguous_format).float()

    ray_direction = torch.stack([example["ray_direction"] for example in examples])
    ray_direction = ray_direction.to(memory_format=torch.contiguous_format).float()

    return {
        "input": inputs,
        "ray_origin": ray_origin,
        "ray_direction": ray_direction
    }


def make_gif(frames, save_path, duration):
    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames, save_all=True, duration=duration, loop=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Path to saved checkpoints"
        ),
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help=(
            "Path to the eval data directory"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="The batch size to use for evaluation.",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Output channels in the UNet.",
    )
    parser.add_argument(
        "--checkpoint_number",
        type=int,
        default=2500,
        help="The iteration number of the checkpoint to load from the model_dir.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Input channels in the UNet.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="Number of frames in the depth video.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=15,
        help="Number of frames in the original video after which a frame should be picked.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--normalization_factor",
        type=float,
        default=20400.0,
        help=(
            "Factor to divide the input depth images by"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--visualize_2d",
        default=False,
        action="store_true",
        help=(
            "If you want the 2D visualization of the depth video."
        ),
    )
    parser.add_argument(
        "--visualize_3d",
        default=False,
        action="store_true",
        help=(
            "If you want the 3D visualization of the depth video."
        ),
    )
    parser.add_argument(
        "--train_with_plucker_coords",
        default=False,
        action="store_true",
        help=(
            "Whether to train with Plucker coordinates or not."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="random",
        choices=["all", "none", "random", "random-half", "random"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="depthpose",
        choices=["reconstruction", "inpainting", "depthpose"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    # Initialize the UNet2D
    Scheduler = DDPMScheduler  # DDPMConditioningScheduler
    unet = UNet2DModel.from_pretrained(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/unet_ema")

    dataset = OccfusionDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        offset=args.offset,
        normalization_factor=args.normalization_factor,
        plucker_coords=args.train_with_plucker_coords,
        use_harmonic=False,
        visualize=False
    )

    assert args.eval_batch_size == 1, "eval batch size must be 1"

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(Scheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        print("was able to specify the prediction type")
        noise_scheduler = Scheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = Scheduler(num_train_timesteps=1000, beta_schedule="linear")

    # noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear")
    if args.model_type == "reconstruction":
        pipeline = DDPMReconstructionPipeline(unet=unet, scheduler=noise_scheduler)
    elif args.model_type == "inpainting":
        pipeline = DDPMInpaintingPipeline(unet=unet, scheduler=noise_scheduler)
    elif args.model_type == "depthpose":
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        pipeline = DDPMDepthPoseInpaintingPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)

    count = 0

    for b, batch in enumerate(eval_dataloader):
        data_point = batch["input"]
        ray_origin = batch["ray_origin"]
        ray_direction = batch["ray_direction"]

        total_frames = data_point.shape[1]
        past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 5)
        future_frames = torch.stack([data_point[0, int(total_frames / 2):, ...]] * 5)

        if args.model_type == "reconstruction":
            prediction = pipeline(
                    batch_size=5,
                    num_inference_steps=40,
                    cond_inds=torch.arange(int(total_frames / 2)),
                    recon_scale=10,
                    conditioning=past_frames,
                    output_type="numpy").images
        elif args.model_type == "inpainting":
            prediction = pipeline(
                    inpainting_image=past_frames,
                    batch_size=5,
                    num_inference_steps=40,
                    output_type="numpy").images
        elif args.model_type == "depthpose":
            prediction, unmasked_indices = pipeline(
                    inpainting_image=past_frames,
                    batch_size=1,
                    num_inference_steps=40,
                    output_type="numpy",
                    num_masks=5).images

        masked_indices = np.array([i for i in range(args.n_input + args.n_output) if i not in unmasked_indices.tolist()])
        prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)

        count += 1

        colored_images = []
        if args.visualize_2d:
            for framenumber in range(prediction.shape[1]):
                if framenumber in masked_indices:
                    cmap = plt.cmap('inferno')
                else:
                    cmap = plt.cmap('gray')
                colored_image = cmap(np.array(prediction[0, framenumber, ...]) / 255.0)
                colored_images.append((colored_image * 255).astype(np.uint8))
            os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals", exist_ok=True)
            make_gif(colored_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals/2d_{count}.gif", duration=100)


if __name__ == "__main__":
    args = parse_args()
    main(args)
