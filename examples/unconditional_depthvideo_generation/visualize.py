import os
import torch
import inspect
import argparse
import numpy as np
from PIL import Image
from prettytable import PrettyTable
from torchvision import transforms
from diffusers import DiffusionPipeline, DDPMPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, DDPMScheduler, DDPMConditioningScheduler

from data import TAODepthDataset


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

    return {
        "input": inputs,
    }

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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Output channels in the UNet.",
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
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
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
    folder_name = "/compute/trinity-1-38/tkhurana/diffusers-runs/logs/ddpm-ema-unconditional-depthvideo-64-8frames"
    # folder_name = "/compute/trinity-1-38/tkhurana/diffusers-runs/logs/ddpm-ema-inpainting-depthvideo-64-8frames-concat-imagemaskedimagemask/"
    reconstruction_guidance = True
    inpainting = False
    prediction_type = "epsilon"
    Scheduler = DDPMScheduler # DDPMConditioningScheduler
    # unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-142500/unet_ema")
    unet = UNet2DModel.from_pretrained(f"{folder_name}/checkpoint-31500/unet_ema")

    dataset = TAODepthDataset(
        instance_data_root=args.train_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        offset=args.offset
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
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
            prediction_type=prediction_type,
        )
    else:
        noise_scheduler = Scheduler(num_train_timesteps=1000, beta_schedule="linear")

    # noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear")
    if reconstruction_guidance:
        pipeline = DDPMReconstructionPipeline(unet=unet, scheduler=noise_scheduler)
    elif inpainting:
        pipeline = DDPMInpaintingPipeline(unet=unet, scheduler=noise_scheduler)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)

    top1_metric, top3_metric, top5_metric = 0, 0, 0
    top1_inv_metric, top3_inv_metric, top5_inv_metric = 0, 0, 0
    count = 0

    headers = ['Top1', 'Top1 (inv)', 'Top3', 'Top3 (inv)', 'Top5', 'Top5 (inv)']

    for b, batch in enumerate(train_dataloader):
        data_point = batch["input"]
        total_frames = data_point.shape[1]
        past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 5)
        future_frames = torch.stack([data_point[0, int(total_frames / 2):, ...]] * 5)

        if reconstruction_guidance:
            prediction = pipeline(
                    batch_size=5,
                    num_inference_steps=40,
                    cond_inds=torch.arange(int(total_frames / 2)),
                    recon_scale=10,
                    conditioning=past_frames,
                    output_type="numpy").images
        elif inpainting:
            prediction = pipeline(
                    inpainting_image=past_frames,
                    # generator=generator,
                    batch_size=5,
                    num_inference_steps=40,
                    output_type="numpy").images

        # both methods output all the past and future frames, so separate these
        prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)
        prediction = prediction[:, int(total_frames / 2):, ...]

        count += 1

        # instead of computing the metrics here, we want to visualize the outputs



if __name__ == "__main__":
    args = parse_args()
    main(args)
