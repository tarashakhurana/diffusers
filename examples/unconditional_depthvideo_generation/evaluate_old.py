import os

from numpy.ma import masked

import torch
import inspect
import argparse
import numpy as np
from prettytable import PrettyTable
from diffusers import DDPMDepthPoseInpaintingPipeline, DDPMReconstructionPipeline
from diffusers import UNet2DModel, DDPMScheduler

import utils
from data import OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting


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
        default="/data3/tkhurana/diffusers/logs/PointOdyssey-depth_train_6s_randomhalf_masking_resolution64_with_plucker/",
        help=(
            "Path to saved checkpoints"
        ),
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="/data/tkhurana/datasets/pointodyssey/val/",
        help=(
            "Path to the eval data directory"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="The batch size to use for evaluation.",
    )
    parser.add_argument(
        "--n_input",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--n_output",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=12,
        help="Output channels in the UNet.",
    )
    parser.add_argument(
        "--checkpoint_number",
        type=int,
        default=1000,
        help="The iteration number of the checkpoint to load from the model_dir.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=180,
        help="Input channels in the UNet.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=12,
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
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--normalization_factor",
        type=float,
        default=65.535,
        help=(
            "Factor to divide the input depth images by"
        ),
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=10.0,
        help=(
            "Factor to multiply the output by in order to get the predictions in a metric space"
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
        "--visualize_3d",
        default=False,
        action="store_true",
        help=(
            "If you want the 3D visualization of the depth video."
        ),
    )
    parser.add_argument(
        "--visualize_spiral",
        default=False,
        action="store_true",
        help=(
            "If you want to render additional poses around a single timestep in a spiral."
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
        "--fair_comparison",
        default=False,
        action="store_true",
        help=(
            "Whether to fix the number of masks and the masked indices for all experiments."
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
        "--num_masks",
        type=int,
        default=6,
        help=(
            "The number of masks to use if the masking strategy is \"random\" ."
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
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.n_input + args.n_output == args.num_images, "n_input + n_output must equal num_images"

    if args.dataset_name is None and args.eval_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    # Initialize the UNet2D
    Scheduler = DDPMScheduler  # DDPMConditioningScheduler
    unet = UNet2DModel.from_pretrained(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/unet")
    unet = unet.to("cuda:0")


    dataset = OccfusionDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        offset=args.offset,
        normalization_factor=args.normalization_factor,
        plucker_coords=args.train_with_plucker_coords,
        use_harmonic=False,
        visualize=False,
        spiral_poses=args.visualize_spiral,
    )

    assert args.eval_batch_size == 1, "eval batch size must be 1"

    if args.model_type == "inpainting" or args.model_type == "reconstruction":
        collate_fn = collate_fn_inpainting
    elif args.model_type == "depthpose":
        collate_fn = collate_fn_depthpose

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
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        pipeline = DDPMDepthPoseInpaintingPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)
        # pipeline = DDPMInpaintingPipeline(unet=unet, scheduler=noise_scheduler)
    elif args.model_type == "depthpose":
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        pipeline = DDPMDepthPoseInpaintingPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)

    # run pipeline in inference (sample random noise and denoise)
    top1_metric, top3_metric = 0, 0
    top1_inv_metric, top3_inv_metric = 0, 0
    count = 0

    headers = ['Top1', 'Top1 (inv)', 'Top3', 'Top3 (inv)']

    for b, batch in enumerate(tqdm(eval_dataloader)):

        dp = batch["input"]
        data_point = torch.stack([dp[0]] * 3).to("cuda:0")
        B, T, C, H, W = data_point.shape
        total_frames = data_point.shape[1]
        past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 3)
        batch_size = 3

        if args.fair_comparison:
            assert args.masking_strategy == "random", "fair comparison only works with random masking"
            time_indices = [1, 3, 5, 7, 9]
            args.num_masks = 5

        if args.model_type == "reconstruction":
            prediction = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    cond_inds=torch.arange(int(total_frames / 2)),
                    recon_scale=10,
                    conditioning=past_frames,
                    output_type="numpy").images
        elif args.model_type == "inpainting":
            prediction, unmasked_indices = pipeline(
                    inpainting_image=data_point,
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    user_num_masks=args.num_masks,
                    user_time_indices=time_indices).images
            masked_indices = [i for i in range(total_frames) if i not in unmasked_indices]
        elif args.model_type == "depthpose":
            prediction, unmasked_indices = pipeline(
                    inpainting_image=data_point,
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    user_num_masks=args.num_masks,
                    user_time_indices=time_indices,
                ).images
            masked_indices = [i for i in range(total_frames) if i not in unmasked_indices]

        prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)
        B, T, H, W = prediction.shape
        count += 1

        groundtruth = data_point[:, :, 0, ...].cpu()

        # shapes of prediction are batch x frames x height x width
        # shapes of groundtruth is the same
        # but now we need to find top 1 and top 3 numbers
        # we will compute both the normal and scale/shift invariant loss
        top1_prediction = prediction[:1, masked_indices, ...]
        top1_groundtruth = groundtruth[:1, masked_indices, ...]
        top1_metric += utils.topk_l1_error(top1_prediction, top1_groundtruth)
        top1_inv_metric += utils.topk_scaleshift_inv_l1_error(top1_prediction, top1_groundtruth)

        top3_prediction = prediction[:3, masked_indices, ...]
        top3_groundtruth = groundtruth[:3, masked_indices, ...]
        top3_metric += utils.topk_l1_error(top3_prediction, top3_groundtruth)
        top3_inv_metric += utils.topk_scaleshift_inv_l1_error(top3_prediction, top3_groundtruth)

        all_metrics = np.array([top1_metric, top1_inv_metric, top3_metric, top3_inv_metric])
        all_metrics /= count

        tab = PrettyTable(headers)
        tab.add_rows([list(all_metrics)])
        print(tab)


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
