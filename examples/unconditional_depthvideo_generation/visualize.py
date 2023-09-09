import os

from matplotlib.cbook import time
import torch
import inspect
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMDepthPoseInpaintingPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, DDPMScheduler, DDPMConditioningScheduler
import matplotlib.cm

from data import OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting
from utils import render_path_spiral, write_pointcloud


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

    if args.dataset_name is None and args.eval_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    # Initialize the UNet2D
    Scheduler = DDPMScheduler  # DDPMConditioningScheduler
    unet = UNet2DModel.from_pretrained(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/unet")

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

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)

    count = 0

    for b, batch in enumerate(eval_dataloader):

        # if b != 3:
        #     continue

        data_point = batch["input"]
        if args.train_with_plucker_coords:
            ray_origin = batch["ray_origin"]
            # ray_direction = batch["ray_direction"]
            # cam_coords = batch["cam_coords"]

        total_frames = data_point.shape[1]
        past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 1)
        future_frames = torch.stack([data_point[0, int(total_frames / 2):, ...]] * 1)

        if args.visualize_spiral:
            assert data_point.shape[0] == 1, "only works for batch size 1"
            assert args.masking_strategy == "random", "only works for random masking strategy"
            # we always only visualize the 6th timestep in a spiral
            time_indices = [6]
            args.num_masks = 1
            batch_size = 12
            data_point = data_point[0]
            print(data_point[0, 6, 1, 0, :], data_point[1, 6, 1, 0, :])
        else:
            time_indices = [6]
            args.num_masks = 1
            batch_size = 1

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
            masked_indices = np.array([i for i in range(args.n_input + args.n_output) if i not in unmasked_indices.tolist()])
        elif args.model_type == "depthpose":
            print("getting predictions using args.num_masks", args.num_masks)
            prediction, unmasked_indices = pipeline(
                    inpainting_image=data_point,
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    user_num_masks=args.num_masks,
                    user_time_indices=time_indices,
                ).images
            masked_indices = np.array([i for i in range(args.n_input + args.n_output) if i not in unmasked_indices.tolist()])

        prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)
        B, T, H, W = prediction.shape

        colored_images = []

        if args.visualize_3d:
            all_points = []
            all_colors = []
            all_edges = []
            i = 0

        for b in range(prediction.shape[0]):
            count += 1
            spiral = []
            for framenumber in range(prediction.shape[1]):
                prediction_minmax = (prediction[b, framenumber] - prediction[b, framenumber].min()) / (prediction[b, framenumber].max() - prediction[b, framenumber].min())
                if framenumber in masked_indices:
                    cmap = matplotlib.cm.get_cmap('inferno')
                else:
                    cmap = matplotlib.cm.get_cmap('gray')
                colored_image = cmap(prediction_minmax)
                colored_images.append(Image.fromarray((colored_image * 255).astype(np.uint8)))

                if args.visualize_3d:
                    ro = ray_origin[b, framenumber, ...].numpy().reshape(1, 3)
                    rd = ray_direction[b, framenumber, ...].numpy().reshape(H, W, 3)
                    scene_coords = rd * np.array(prediction[b, framenumber, ..., None]) * args.scale_factor
                    prediction_minmax = (prediction[b, framenumber] - prediction[b, framenumber].min()) / (prediction[b, framenumber].max() - prediction[b, framenumber].min())
                    corners = np.array([rd[0, 0], rd[0, W-1], rd[H-1, 0], rd[H-1, W-1]])

                    # append all the predicted points in 3D
                    all_points.append(scene_coords.reshape(-1, 3))
                    all_colors.append(colored_image[..., :3].reshape(-1, 3))
                    i += all_points[-1].shape[0]

                    # print("1", all_points[-1].shape, all_colors[-1].shape)

                    # append all the predicted points in 2D as a depth map
                    all_points.append(rd.reshape(-1, 3))
                    all_colors.append(colored_image[..., :3].reshape(-1, 3))
                    i += all_points[-1].shape[0]
                    # print("2", all_points[-1].shape, all_colors[-1].shape)

                    # append all the image corners
                    all_points.append(corners)
                    all_colors.append(np.ones_like(all_points[-1]) * 255)
                    i += all_points[-1].shape[0]
                    # print("3", all_points[-1].shape, all_colors[-1].shape)

                    # append the camera center
                    all_points.append(ro)
                    color = np.ones_like(all_points[-1]) * 255
                    if framenumber in masked_indices:
                        color[:, :2] = 0
                    else:
                        color[:, 1:] = 0
                    all_colors.append(color)
                    i += all_points[-1].shape[0]
                    # print("4", all_points[-1].shape, all_colors[-1].shape)

                    # append the 4 edges between the camera center and the corners of the image
                    all_edges.append(np.array([i-1, i-2]))
                    all_edges.append(np.array([i-1, i-3]))
                    all_edges.append(np.array([i-1, i-4]))
                    all_edges.append(np.array([i-1, i-5]))


            if args.visualize_3d:
                all_points = np.concatenate(all_points, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                all_edges = np.stack(all_edges, axis=0)
                write_pointcloud(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals/3d_{count}.ply", all_points, all_colors, edges=all_edges)

            os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals", exist_ok=True)
            make_gif(colored_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals/2d_{count}.gif", duration=800)

            spiral.append(colored_images[6])

        if args.visualize_spiral:
            make_gif(spiral, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals/spiral_{count}.gif", duration=800)


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
