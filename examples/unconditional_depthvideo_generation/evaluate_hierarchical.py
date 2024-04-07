import os

from matplotlib.cbook import time
import torch
import inspect
import argparse
import numpy as np
from prettytable import PrettyTable
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMDepthPoseInpaintingPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline, DDPMImg2ImgPipeline, DDPMImg2ImgCLIPPosePipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, UNet2DConditionRenderModel, UNet2DConditionSpacetimeRenderModel, DDPMScheduler, DDIMScheduler, DDPMConditioningScheduler
import matplotlib.cm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import utils
from data import TAOMAEDataset, TAOForecastingDataset, OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting, TAOTemporalSuperResolutionDataset, collate_fn_temporalsuperres
from utils import render_path_spiral, write_pointcloud

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


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
        "--num_autoregressive_frames",
        type=int,
        default=60,
        help="Number of frames to generate autoregressively.",
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
        "--output_fps",
        type=int,
        default=15,
        help="Output FPS for the autoregressive video",
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
        "--guidance",
        type=float,
        default=1.0,
        help=(
            "Amount of guidance at inference for classifier free guidance"
        ),
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=100.0,
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
        "--train_with_plucker_coords",
        default=False,
        action="store_true",
        help=(
            "Whether to train with Plucker coordinates or not."
        ),
    )
    parser.add_argument(
        "--use_rendering",
        default=False,
        action="store_true",
        help=(
            "Whether to use rendering at the end of the network"
        ),
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="d",
        choices=["d", "rgb", "rgbd"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument(
        "--eval_rgb_data_dir",
        type=str,
        default="/data/tkhurana/datasets/pointodyssey/val/",
        help=(
            "Path to the eval data directory"
        ),
    )
    parser.add_argument(
        "--use_harmonic",
        default=False,
        action="store_true",
        help=(
            "Whether to use rendering at the end of the network"
        ),
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "autoregressive", "direct", "aidedautoregressive"],
        help="What kind of sampling strategy to use at inference.",
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
        choices=["all", "none", "random", "random-half", "half", "custom"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="depthpose",
        choices=["reconstruction", "inpainting", "depthpose", "img2img", "clippose"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--shuffle_video", action="store_true")
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
    timesteps = torch.Tensor([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    hierarchical_timesteps = [[0, 1, 2, 12],
                            [1, 2, 12, 7],
                            [1, 2, 7, 5],
                            [5, 7, 12, 10],
                            [0, 1, 2, 3],
                            [2, 3, 5, 4],
                            [5, 7, 10, 8],
                            [7, 8, 10, 9],
                            [4, 5, 7, 6],
                            [9, 10, 12, 11]]

    autoregressive_timesteps = [[0, 1, 2, 3],
                                [1, 2, 3, 4],
                                [2, 3, 4, 5],
                                [3, 4, 5, 6],
                                [4, 5, 6, 7],
                                [5, 6, 7, 8],
                                [6, 7, 8, 9],
                                [7, 8, 9, 10],
                                [8, 9, 10, 11],
                                [9, 10, 11, 12]]

    direct_timesteps = [[0, 1, 2, 3],
                        [0, 1, 2, 4],
                        [0, 1, 2, 5],
                        [0, 1, 2, 6],
                        [0, 1, 2, 7],
                        [0, 1, 2, 8],
                        [0, 1, 2, 9],
                        [0, 1, 2, 10],
                        [0, 1, 2, 11],
                        [0, 1, 2, 12]]

    aided_autoregressive_timesteps = [[[0, 1, 2, 3], [0, 1, 2, 3]],
                                    [[0, 1, 2, 4], [1, 2, 3, 4]],
                                    [[0, 1, 2, 5], [2, 3, 4, 5]],
                                    [[0, 1, 2, 6], [3, 4, 5, 6]],
                                    [[0, 1, 2, 7], [4, 5, 6, 7]],
                                    [[0, 1, 2, 8], [5, 6, 7, 8]],
                                    [[0, 1, 2, 9], [6, 7, 8, 9]],
                                    [[0, 1, 2, 10], [7, 8, 9, 10]],
                                    [[0, 1, 2, 11], [8, 9, 10, 11]],
                                    [[0, 1, 2, 12], [9, 10, 11, 12]]]
    """

    timesteps = torch.Tensor([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    hierarchical_timesteps = [[0, 1, 2, 7],
                            [0, 1, 2, 3],
                            [2, 3, 7, 5],
                            [2, 3, 5, 4],
                            [4, 5, 7, 6]]

    autoregressive_timesteps = [[0, 1, 2, 3],
                                [1, 2, 3, 4],
                                [2, 3, 4, 5],
                                [3, 4, 5, 6],
                                [4, 5, 6, 7]]

    direct_timesteps = [[0, 1, 2, 3],
                        [0, 1, 2, 4],
                        [0, 1, 2, 5],
                        [0, 1, 2, 6],
                        [0, 1, 2, 7]]

    aided_autoregressive_timesteps = [[[0, 1, 2, 3], [0, 1, 2, 3]],
                                    [[0, 1, 2, 4], [1, 2, 3, 4]],
                                    [[0, 1, 2, 5], [2, 3, 4, 5]],
                                    [[0, 1, 2, 6], [3, 4, 5, 6]],
                                    [[0, 1, 2, 7], [4, 5, 6, 7]]]
    """
    sampling_timesteps = None
    if args.sampling_strategy == "hierarchical":
        sampling_timesteps = hierarchical_timesteps
        mixing_guidance = 1.0
    elif args.sampling_strategy == "autoregressive":
        sampling_timesteps = autoregressive_timesteps
        mixing_guidance = 1.0
    elif args.sampling_strategy == "direct":
        sampling_timesteps = direct_timesteps
        mixing_guidance = 1.0
    elif args.sampling_strategy == "aidedautoregressive":
        sampling_timesteps = aided_autoregressive_timesteps
        mixing_guidance = 2.0

    load_rgb = False
    rgb_data_root = None

    if "rgb" in args.data_format:
        load_rgb = True
        rgb_data_root = args.eval_rgb_data_dir

    # Initialize the UNet2D
    Scheduler = DDPMScheduler  # DDPMConditioningScheduler
    if args.use_rendering:
        assert args.train_with_plucker_coords
        # assert args.prediction_type == "sample"
        if not args.model_type == "clippose":
            UNetModel = UNet2DConditionRenderModel
        else:
            UNetModel = UNet2DConditionSpacetimeRenderModel
    else:
        UNetModel = UNet2DModel
    unet = UNetModel.from_pretrained(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/unet")
    unet = unet.to("cuda:0")

    dataset = TAOForecastingDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        autoregressive=True,
        num_autoregressive_frames=args.num_autoregressive_frames,
        fps=30,
        horizon=11,
        offset=15,
        split="val",
        load_rgb=load_rgb,
        rgb_data_root=rgb_data_root,
        normalization_factor=args.normalization_factor,
        visualize=False
    )

    assert args.eval_batch_size == 1, "eval batch size must be 1"

    if args.model_type == "inpainting" or args.model_type == "reconstruction":
        collate_fn = collate_fn_inpainting
    elif args.model_type == "depthpose" or args.model_type == "img2img" or args.model_type == "clippose":
        collate_fn = collate_fn_depthpose

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
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
    if args.model_type == "clippose":
        feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", do_rescale=False, do_normalize=False)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        image_encoder.to("cuda:0")
        image_encoder.eval()
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        kwargs["use_rendering"] = args.use_rendering
        kwargs["data_format"] = args.data_format
        pipeline = DDPMImg2ImgCLIPPosePipeline(unet=unet, scheduler=noise_scheduler, feature_extractor=feature_extractor, image_encoder=image_encoder, kwargs=kwargs)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)

    count = 0
    colored_images = []
    colored_rgb_images = []
    cmap = matplotlib.cm.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.6, 1.0)

    suffix = "_nogt"

    image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # for framenumber in range(6, min(len(dataset), 6 + args.num_autoregressive_frames)):
    batch_index_counter = 0
    count = 0

    top1_metric, top3_metric, top5_metric = 0, 0, 0
    top1_inv_metric, top3_inv_metric, top5_inv_metric = 0, 0, 0
    top1_psnr, top3_psnr, top5_psnr = 0, 0, 0

    if load_rgb:
        headers_rgb = ['Top1 PSNR', 'Top3 PSNR', 'Top5 PSNR']

    headers = ['Top1', 'Top1 (inv)', 'Top3', 'Top3 (inv)', 'Top5', 'Top5 (inv)']

    # open the results file
    resultsfile = open(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/results_sampling-{args.sampling_strategy}-2.0-bugfix_guidance{args.guidance}.txt", "w")

    for b, batch in enumerate(eval_dataloader):

        print(f"Doing {b}/{len(eval_dataloader)}")

        # while batch_index_counter < len(dataset):
        last_output = None
        colored_images = []
        colored_rgb_images = []
        all_frames = torch.zeros(5, 13, 1, 64, 64).to("cuda:0")
        gt_frames = torch.zeros(5, 13, 1, 64, 64).to("cuda:0")

        filename = batch["filenames"][0][2]
        seq = str(filename)[len(str(args.eval_data_dir)):str(filename).rfind("/")]

        data_point_ = batch["input"].to("cuda:0")[:, :, None, :, :]
        rgb_images_ = batch["rgb_input"].to("cuda:0")[:, :, None, :, :]  # shape: B T C H W
        if args.data_format == "d":
            data_point_ = data_point_
        elif args.data_format == "rgb":
            data_point_ = rgb_images_
        elif args.data_format == "rgbd":
            data_point_ = torch.cat([rgb_images_, data_point_], dim=2)

        print("shape of data points", data_point_.shape)

        data_point_ = torch.cat([data_point_] * 5, axis=0)
        print("shapes", all_frames.shape, data_point_.shape, all_frames[:, 0].shape, data_point_[:, 0].shape)
        all_frames[:, 0] = data_point_[:, 0]
        all_frames[:, 1] = data_point_[:, 1]
        all_frames[:, 2] = data_point_[:, 2]
        gt_frames[:, 0] = data_point_[:, 0]
        gt_frames[:, 1] = data_point_[:, 1]
        gt_frames[:, 2] = data_point_[:, 2]

        for framenumber in range(args.num_autoregressive_frames):

            if args.sampling_strategy == "aidedautoregressive":
                indices = [y for x in sampling_timesteps[framenumber] for y in x]
                frame_index = sampling_timesteps[framenumber][-1][-1]
                direct, autoregressive = all_frames[:, indices].reshape(5, 4, 2, 1, 64, 64).permute(2, 0, 1, 3, 4, 5)
                data_point = torch.cat([direct, autoregressive])
                direct, autoregressive = timesteps[indices].to("cuda:0").reshape(2, -1, 1)
                direct = torch.stack([direct] * 5)
                autoregressive = torch.stack([autoregressive] * 5)
                plucker = torch.cat([direct, autoregressive], axis=0)
            else:
                indices = sampling_timesteps[framenumber]
                frame_index = sampling_timesteps[framenumber][-1]
                data_point = all_frames[:, indices]
                plucker = timesteps[sampling_timesteps[framenumber]].to("cuda:0")[None, :, None]
                plucker = torch.cat([plucker] * 5, axis=0)

            plucker = plucker - plucker[:, 2:3, :]
            gt_frames[:, frame_index] = data_point_[:, frame_index]

            B, T, C, H, W = data_point.shape

            time_indices = torch.Tensor([3]).int()
            args.num_masks = 1  # 11
            batch_size = 5

            if args.model_type == "clippose":
                prediction, _ = pipeline(
                    inpainting_image=(data_point, plucker),
                    batch_size=batch_size,
                    guidance=args.guidance,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    mix_guidance=mixing_guidance,
                    user_num_masks=args.num_masks,
                    user_time_indices=time_indices,
                ).images

            prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)

            if not args.model_type in ["img2img", "clippose"]:
                prediction = prediction[:, :, None, :, :]
            else:
                prediction = prediction.reshape(batch_size, 1, C, H, W)

            all_frames[:, frame_index] =  2 * prediction[:, 0] - 1.0

        prediction = all_frames[:, 3:, :, ...].cpu() / 2 + 0.5
        groundtruth = gt_frames[:, 3:, :, ...].cpu() / 2 + 0.5

        print("min max of prediction", prediction.min(), prediction.max())
        print("min max of groundtruth", groundtruth.min(), groundtruth.max())

        B, T, C, H, W = prediction.shape
        count += 1

        if "d" in args.data_format:
            top1_prediction = prediction[:1, :, -1, ...]
            top1_groundtruth = groundtruth[:1, :, -1, ...]
            top1_metric += utils.topk_l1_error(top1_prediction, top1_groundtruth)
            top1_inv_metric += utils.topk_scaleshift_inv_l1_error(top1_prediction, top1_groundtruth)

            top3_prediction = prediction[:3, :, -1, ...]
            top3_groundtruth = groundtruth[:3, :, -1, ...]
            top3_metric += utils.topk_l1_error(top3_prediction, top3_groundtruth)
            top3_inv_metric += utils.topk_scaleshift_inv_l1_error(top3_prediction, top3_groundtruth)

            top5_prediction = prediction[:5, :, -1, ...]
            top5_groundtruth = groundtruth[:5, :, -1, ...]
            top5_metric += utils.topk_l1_error(top5_prediction, top5_groundtruth)
            top5_inv_metric += utils.topk_scaleshift_inv_l1_error(top5_prediction, top5_groundtruth)

            all_metrics = np.array([top1_metric, top1_inv_metric, top3_metric, top3_inv_metric, top5_metric, top5_inv_metric])

            all_metrics /= count

            tab = PrettyTable(headers)
            tab.add_rows([list(all_metrics)])
            print(tab)

        if "rgb" in args.data_format:
            if args.data_format == "rgb":
                end_index = prediction.shape[2]
            else:
                end_index = -1
            top1_prediction = prediction[:1, :, :end_index, ...]
            top1_groundtruth = groundtruth[:1, :, :end_index, ...]
            top1_psnr += utils.topk_psnr(top1_prediction, top1_groundtruth)

            top3_prediction = prediction[:3, :, :end_index, ...]
            top3_groundtruth = groundtruth[:3, :, :end_index, ...]
            top3_psnr += utils.topk_psnr(top3_prediction, top3_groundtruth)

            top5_prediction = prediction[:5, :, :end_index, ...]
            top5_groundtruth = groundtruth[:5, :, :end_index, ...]
            top5_psnr += utils.topk_psnr(top5_prediction, top5_groundtruth)

            all_psnr = np.array([top1_psnr, top3_psnr, top5_psnr])

            all_psnr /= count

            tab = PrettyTable(headers_rgb)
            tab.add_rows([list(all_psnr)])
            print(tab)


    if "d" in args.data_format:
        resultsfile.write(",".join(headers) + "\n")
        resultsfile.write(",".join([str(p) for p in all_metrics]) + "\n")
        resultsfile.close()

    if "rgb" in args.data_format:
        resultsfile.write(",".join(headers_rgb) + "\n")
        resultsfile.write(",".join([str(p) for p in all_psnr]) + "\n")
        resultsfile.close()



if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
