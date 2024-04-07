import os

from matplotlib.cbook import time
import torch
import random
import inspect
import argparse
import numpy as np
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
        "--subfolder_name",
        type=str,
        default="run_1",
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
        "--use_groundtruth",
        default=False,
        action="store_true",
        help=(
            "Whether to use future groundtruth frames as input for the autoregressive output."
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
        offset=int(30 / 2),
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
    gray_cmap = matplotlib.cm.get_cmap('binary')
    cmap = matplotlib.cm.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.6, 1.0)

    if args.use_groundtruth:
        suffix = ""
    else:
        suffix = "_nogt"

    last_input = None
    image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    # for framenumber in range(6, min(len(dataset), 6 + args.num_autoregressive_frames)):
    batch_index_counter = 0
    if args.use_groundtruth:
        iterable_array = range(int(len(dataset) / args.num_autoregressive_frames))
    else:
        iterable_array = range(len(dataset))

    vid_done = []

    print("Saving outputs for run name", args.subfolder_name)

    for batchidx in iterable_array:

        # while batch_index_counter < len(dataset):
        last_output = None
        last_input = None
        colored_images = []
        colored_rgb_images = []
        all_frames = torch.zeros(13, 64, 64).to("cuda:0")

        for framenumber in range(args.num_autoregressive_frames):

            if args.use_groundtruth:
                batch_index = batch_index_counter
                batch_index_counter += 1
            else:
                batch_index = batchidx
            # print(batch_index, "batch index counter is", framenumber, args.num_autoregressive_frames)

            batch = dataset[batch_index]
            filename = batch["filenames"][2]
            seq = str(filename)[len(str(args.eval_data_dir)):str(filename).rfind("/")]

            # if "AVA/" in seq:
            #     continue

            # selected_vids = ['AVA/WKqbLbU68wU_scene_4_18239-19159_2_depth_nogt', 'AVA/XOe9GeojzCs_scene_10_51401-53477_2_depth_nogt', 'AVA/XOe9GeojzCs_scene_13_62523-64711_2_depth_nogt', 'AVA/XOe9GeojzCs_scene_27_108850-110834_2_depth_nogt', 'AVA/XV_FF3WC7kA_scene_5_72980-74069_2_depth_nogt', 'AVA/YAAUPjq-L-Q_scene_1_83217-84762_2_depth_nogt', 'AVA/uwW0ejeosmk_scene_3_50442-52200_2_depth_nogt', 'AVA/z-fsLpGHq6o_scene_2_40193-41361_2_depth_nogt', 'ArgoVerse/4518c79d-10fb-300e-83bb-6174d5b24a45_2_depth_nogt', 'ArgoVerse/5ab2697b-6e3e-3454-a36a-aba2c6f27818_2_depth_nogt', 'BDD/b231a630-c4522992_2_depth_nogt', 'BDD/b27127df-eac9b95e_2_depth_nogt', 'BDD/b3079ec6-df7b2d92_2_depth_nogt', 'Charades/0LHWF_2_depth_nogt', 'Charades/1410C_2_depth_nogt', 'Charades/35LUV_2_depth_nogt', 'Charades/G40U3_2_depth_nogt', 'Charades/O87OF_2_depth_nogt', 'Charades/PJUM0_2_depth_nogt', 'HACS/Croquet_v_vrWYdPeIUqw_scene_0_0-1779_2_depth_nogt', 'HACS/Dodgeball_v_IS3OtsJFP7Y_scene_0_2835-4590_2_depth_nogt', 'HACS/Doing_step_aerobics_v_8QyDjT0ZsHE_scene_0_0-3823_2_depth_nogt', 'HACS/Doing_step_aerobics_v_h3-lxgAoXwI_scene_0_0-2516_2_depth_nogt', 'HACS/Getting_a_haircut_v_aboKQqtoowA_scene_0_0-4543_2_depth_nogt', 'HACS/Horseback_riding_v_YD5C5LX7j4k_scene_0_962-4591_2_depth_nogt', 'HACS/Longboarding_v_dYAA0JLFrTc_scene_0_861-2179_2_depth_nogt', 'HACS/Making_a_lemonade_v_Zyo70ZiXmYY_scene_0_1938-3070_2_depth_nogt', 'HACS/Painting_furniture_v_xNxxM-OOMfw_scene_0_0-1910_2_depth_nogt', 'HACS/Tai_chi_v_87ksUt7mO3o_scene_0_0-1298_2_depth_nogt', 'HACS/Washing_dishes_v_25eIK85JWi4_scene_0_183-3069_2_depth_nogt', 'LaSOT/basketball-11_2_depth_nogt', 'LaSOT/basketball-6_2_depth_nogt', 'LaSOT/book-11_2_depth_nogt', 'LaSOT/car-17_2_depth_nogt', 'LaSOT/microphone-12_2_depth_nogt', 'LaSOT/monkey-16_2_depth_nogt', 'LaSOT/swing-12_2_depth_nogt', 'YFCC100M/v_39e0a6e2fd23c9796a7ac04ed257c461_2_depth_nogt', 'YFCC100M/v_8794cde67aad8552ecfc59bb6fbe72d_2_depth_nogt', 'YFCC100M/v_8b6283255797fc7e94f3a93947a2803_2_depth_nogt', 'YFCC100M/v_9018a07931e02026b3bd6e82489d4624_2_depth_nogt', 'YFCC100M/v_c1182f41d79bb9cb7917331393ec7e5_2_depth_nogt', 'YFCC100M/v_c132c89de7fa33cbbdbe9669d114e33_2_depth_nogt', 'YFCC100M/v_d4fa85cf4d613518a6e9e7948102452_2_depth_nogt', 'YFCC100M/v_f729d4f362aea24236153ffc589adac_2_depth_nogt']
            selected_vids = ['AVA/WKqbLbU68wU_scene_4_18239-19159_2_depth_nogt', 'AVA/uwW0ejeosmk_scene_3_50442-52200_2_depth_nogt', 'AVA/z-fsLpGHq6o_scene_2_40193-41361_2_depth_nogt', 'ArgoVerse/4518c79d-10fb-300e-83bb-6174d5b24a45_2_depth_nogt', 'ArgoVerse/5ab2697b-6e3e-3454-a36a-aba2c6f27818_2_depth_nogt', 'BDD/b231a630-c4522992_2_depth_nogt', 'Charades/1410C_2_depth_nogt', 'Charades/35LUV_2_depth_nogt', 'HACS/Dodgeball_v_IS3OtsJFP7Y_scene_0_2835-4590_2_depth_nogt', 'HACS/Doing_step_aerobics_v_8QyDjT0ZsHE_scene_0_0-3823_2_depth_nogt', 'HACS/Painting_furniture_v_xNxxM-OOMfw_scene_0_0-1910_2_depth_nogt', 'HACS/Washing_dishes_v_25eIK85JWi4_scene_0_183-3069_2_depth_nogt', 'LaSOT/basketball-11_2_depth_nogt', 'LaSOT/swing-12_2_depth_nogt', 'YFCC100M/v_d4fa85cf4d613518a6e9e7948102452_2_depth_nogt', 'YFCC100M/v_f729d4f362aea24236153ffc589adac_2_depth_nogt']
            selected_vids = [s[:-13] for s in selected_vids[:5]]

            if seq not in selected_vids or seq in vid_done:
                continue

            if framenumber + 1 == args.num_autoregressive_frames:
                vid_done.append(seq)

            print("Doing sequence", seq)

            if args.sampling_strategy == "aidedautoregressive":
                indices = [y for x in sampling_timesteps[framenumber] for y in x]
                frame_index = sampling_timesteps[framenumber][-1][-1]
            else:
                indices = sampling_timesteps[framenumber]
                frame_index = sampling_timesteps[framenumber][-1]

            data_point, rgb_images, plucker = None, None, None
            if last_input == None or args.use_groundtruth:
                data_point = batch["input"].to("cuda:0")
                os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_sampling-{args.sampling_strategy}/{seq}_{args.output_fps}_depth{suffix}/{args.subfolder_name}", exist_ok=True)
                all_frames[0] = data_point[0]
                all_frames[1] = data_point[1]
                all_frames[2] = data_point[2]
                for l in range(3):
                    inp = all_frames[l].cpu().numpy()
                    inp = (inp - inp.min()) / (inp.max() - inp.min())
                    inp = gray_cmap(inp)
                    inp = Image.fromarray((inp * 255).astype(np.uint8))
                    path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_sampling-{args.sampling_strategy}/{seq}_{args.output_fps}_depth{suffix}/{args.subfolder_name}/{l:0>5}.png"
                    inp.save(path)
                data_point = data_point[None, :, None, :, :]
                if args.sampling_strategy == "aidedautoregressive":
                    # make batch size 2 for data_point
                    data_point = torch.cat([data_point, data_point], axis=0)
                if args.sampling_strategy == "aidedautoregressive":
                    plucker = timesteps[indices].to("cuda:0")
                    plucker = plucker.reshape(2, -1, 1)
                else:
                    plucker = timesteps[sampling_timesteps[framenumber]].to("cuda:0")[None, :, None]
                plucker = plucker - plucker[:, 2:3, :]
                print("plucker is", plucker.shape, plucker)
                print("data point is", data_point.shape)
                if load_rgb:
                    rgb_images = batch["rgb_input"].to("cuda:0")[None, ...]  # shape: B T C H W
                last_input = (data_point, rgb_images, plucker)

            else:
                if args.sampling_strategy == "aidedautoregressive":
                    data_point = all_frames[indices].reshape(2, 4, 1, 64, 64)
                    plucker = timesteps[indices].to("cuda:0").reshape(2, -1, 1)
                else:
                    data_point = all_frames[indices][None, :, None, :, :]
                    plucker = timesteps[sampling_timesteps[framenumber]].to("cuda:0")[None, :, None]
                plucker = plucker - plucker[:, 2:3, :]
                print("plucker is", plucker.shape, plucker)
                print("data point is", data_point.shape)

            if args.data_format == "d":
                data_point = data_point
            elif args.data_format == "rgb":
                data_point = rgb_images
            elif args.data_format == "rgbd":
                data_point = torch.cat([rgb_images, data_point], dim=2)

            B, T, C, H, W = data_point.shape

            time_indices = torch.Tensor([3]).int()
            args.num_masks = 1  # 11
            batch_size = 1

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
                prediction_to_save = prediction.reshape(batch_size, 1, C, H, W)
                prediction = prediction.reshape(batch_size, 1, C, H, W)
                data_point, _, _ = last_input
                data_point_minmax = data_point[:1, :args.num_images, :, :, :].cpu()
                data_point_minmax = torch.clamp((data_point_minmax / 2 + 0.5), 0, 1)
                prediction = torch.cat([data_point_minmax, prediction], axis=1)

            B, T, C, H, W = prediction.shape

            colored_image = None
            colored_rgb_image = None

            save_index = framenumber + 3

            for b in range(prediction.shape[0]):
                count += 1
                prediction_minmax_depth = prediction[b, :, -1]  # (prediction[b, :, -1] - prediction[b, :, -1].min()) / (prediction[b, :, -1].max() - prediction[b, :, -1].min())
                for fn in range(prediction.shape[1]):

                    if "d" in args.data_format:
                        colored_image = prediction_minmax_depth[fn].numpy().squeeze()
                        plot_image = (colored_image - colored_image.min()) / (colored_image.max() - colored_image.min())
                        plot_image = cmap(plot_image)
                        plot_image = Image.fromarray((plot_image * 255).astype(np.uint8))
                        if fn == args.num_images:
                            path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_sampling-{args.sampling_strategy}/{seq}_{args.output_fps}_depth{suffix}/{args.subfolder_name}/{save_index:0>5}.png"
                            plot_image.save(path)
                            print("shape of image just saved", colored_image.shape)
                            colored_images.append(colored_image)

                        #     groundtruth = data_point[b, fn, -1].cpu().numpy()
                        #     groundtruth = (groundtruth - groundtruth.min()) / (groundtruth.max() - groundtruth.min())
                        #     groundtruth = cmap(groundtruth)
                        #     groundtruth = Image.fromarray((groundtruth * 255).astype(np.uint8))
                        #     path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive_multifps/{seq}_{args.output_fps}_depth{suffix}/groundtruth_{frame_index:0>5}_3.png"
                        #     # groundtruth.save(path)

                        # else:
                        #     groundtruth = data_point[b, fn, -1].cpu().numpy()
                        #     groundtruth = (groundtruth - groundtruth.min()) / (groundtruth.max() - groundtruth.min())
                        #     groundtruth = cmap(groundtruth)
                        #     groundtruth = Image.fromarray((groundtruth * 255).astype(np.uint8))
                        #     path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive_multifps/{seq}_{args.output_fps}_depth{suffix}/input_{frame_index:0>5}_{fn}.png"
                        #     if framenumber == 0:
                        #         groundtruth.save(path)

                    if "rgb" in args.data_format:
                        colored_rgb_image = prediction[b, fn, :3].permute(1, 2, 0).numpy().squeeze()[..., ::-1]
                        plot_image = (colored_rgb_image - colored_rgb_image.min()) / (colored_rgb_image.max() - colored_rgb_image.min())
                        plot_image = Image.fromarray((plot_image * 255).astype(np.uint8))
                        os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive/{seq}/{args.output_fps}/rgb{suffix}", exist_ok=True)
                        if fn == args.num_images:
                            path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive/{seq}/{args.output_fps}/rgb{suffix}/{count:0>5}_3.jpg"
                            plot_image.save(path)
                            colored_rgb_images.append(colored_rgb_image)
                        else:
                            path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive/{seq}/{args.output_fps}/rgb{suffix}/input_{count:0>5}_3.jpg"
                            plot_image.save(path)

            all_frames[frame_index] = prediction_to_save[0, 0, 0]

        """
        if count > 0:
            if "d" in args.data_format:
                print(len(colored_images), count)
                mini = [np.min(c) for c in colored_images]
                maxi = [np.max(c) for c in colored_images]
                colored_images = [cmap((c - mini[i]) / (maxi[i] - mini[i])) for i, c in enumerate(colored_images)]
                print(colored_images[0])
                colored_images = [Image.fromarray((c * 255).astype(np.uint8)) for c in colored_images]
                # utils.make_gif(colored_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive_newvis/{seq}/{args.output_fps}/depth{suffix}_3.gif", duration=800)

            if "rgb" in args.data_format:
                print(len(colored_rgb_images), count)
                mini = np.min(colored_rgb_images)
                maxi = np.max(colored_rgb_images)
                colored_rgb_images = [(c - mini) / (maxi - mini) for c in colored_rgb_images]
                colored_rgb_images = [Image.fromarray((c * 255).astype(np.uint8)) for c in colored_rgb_images]
                utils.make_gif(colored_rgb_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_autoregressive/{seq}/{args.output_fps}/rgb{suffix}_3.gif", duration=800)

        """


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main(args)
