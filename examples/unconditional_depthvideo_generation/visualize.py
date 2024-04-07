import os

from matplotlib.cbook import time
import torch
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
from data import TAOMAEDataset, CO3DEvalDataset, TAOForecastingDataset, OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting, TAOTemporalSuperResolutionDataset, collate_fn_temporalsuperres
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
        "--n_input",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--co3d_annotations_root",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--co3d_object_crop", action="store_true")
    parser.add_argument("--visualize_interpolation", action="store_true")
    parser.add_argument(
        "--co3d_rgb_data_root",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
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

    """
    dataset = OccfusionDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        offset=args.offset,
        normalization_factor=args.normalization_factor,
        plucker_coords=args.train_with_plucker_coords,
        plucker_resolution=[64, 32, 16] if args.use_rendering else [64],
        shuffle_video=args.shuffle_video,
        use_harmonic=args.use_harmonic,
        visualize=True,
        spiral_poses=args.visualize_spiral,
    )
    dataset = TAOMAEDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        split="val",
        normalization_factor=args.normalization_factor,
        visualize=False
    )
    """
    if "trainco3d" in args.eval_data_dir:
        dataset = CO3DEvalDataset(
            instance_data_root=args.eval_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
            num_images=args.num_images,
            split="val",
            load_rgb=load_rgb,
            co3d_object_crop=args.co3d_object_crop,
            co3d_annotations_root=args.co3d_annotations_root,
            co3d_rgb_data_root=args.co3d_rgb_data_root,
            normalization_factor=args.normalization_factor,
        )
    else:
        dataset = TAOForecastingDataset(
            instance_data_root=args.eval_data_dir,
            size=args.resolution,
            center_crop=args.center_crop,
            num_images=args.num_images,
            fps=30,
            horizon=1,
            offset=15,
            split="val",
            load_rgb=load_rgb,
            rgb_data_root=rgb_data_root,
            normalization_factor=args.normalization_factor,
            visualize=False,
            interpolation_baseline=args.visualize_interpolation
        )
    """
    dataset = TAOTemporalSuperResolutionDataset(
        instance_data_root=args.eval_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        batch_size=args.eval_batch_size,
        split="train",
        normalization_factor=args.normalization_factor,
        visualize=False
    )
    """

    assert args.eval_batch_size == 1, "eval batch size must be 1"

    if args.model_type == "inpainting" or args.model_type == "reconstruction":
        collate_fn = collate_fn_inpainting
    elif args.model_type == "depthpose" or args.model_type == "img2img" or args.model_type == "clippose":
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
        kwargs["use_rendering"] = args.use_rendering
        pipeline = DDPMDepthPoseInpaintingPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)
        # pipeline = DDPMInpaintingPipeline(unet=unet, scheduler=noise_scheduler)
    elif args.model_type == "depthpose":
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        kwargs["use_rendering"] = args.use_rendering
        pipeline = DDPMDepthPoseInpaintingPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)
    elif args.model_type == "img2img":
        kwargs = {}
        kwargs["n_input"] = args.n_input
        kwargs["n_output"] = args.n_output
        kwargs["masking_strategy"] = args.masking_strategy
        kwargs["train_with_plucker_coords"] = args.train_with_plucker_coords
        kwargs["use_rendering"] = args.use_rendering
        kwargs["data_format"] = args.data_format
        pipeline = DDPMImg2ImgPipeline(unet=unet, scheduler=noise_scheduler, kwargs=kwargs)
    elif args.model_type == "clippose":
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

    for b, batch in enumerate(eval_dataloader):

        # if b != 3:
        #     continue

        data_point = batch["input"].to("cuda:0")
        filename = batch["filenames"][0][2]
        seq = str(filename)[len(str(args.eval_data_dir)):str(filename).rfind("/")]
        data_point = data_point[:, :, None, :, :]

        if args.train_with_plucker_coords:
            plucker = batch["plucker_coords"].to("cuda:0")
            # print("found frame idS to be", data_point.shape, plucker.shape)
            # plucker = [pc.to("cuda:0") for pc in batch["plucker_coords"]]
            # ray_origin = batch["ray_origin"]
            # ray_direction = batch["ray_direction"]
            # cam_coords = batch["cam_coords"]

        if load_rgb:
            rgb_images = batch["rgb_input"].to("cuda:0")  # shape: B T C H W

        if args.data_format == "d":
            data_point = data_point
        elif args.data_format == "rgb":
            data_point = rgb_images
        elif args.data_format == "rgbd":
            data_point = torch.cat([rgb_images, data_point], dim=2)

        B, T, C, H, W = data_point.shape

        total_frames = data_point.shape[1]
        past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 1)
        future_frames = torch.stack([data_point[0, int(total_frames / 2):, ...]] * 1)
        data_point = torch.stack([data_point[0, ...]] * 12)
        plucker = torch.stack([plucker[0, ...]] * 12)
        # plucker[:, 3:4, :] = plucker[:, 3:4, :] + (torch.arange(4)[:, None, None].to("cuda:0") - 2)
        batch_size = 12
        print("data point and plucker shape", data_point.shape, plucker.shape)

        if args.visualize_spiral:
            assert data_point.shape[0] == 1, "only works for batch size 1"
            assert args.masking_strategy == "random", "only works for random masking strategy"
            # we always only visualize the 6th timestep in a spiral
            time_indices = torch.Tensor([6]).int()
            args.num_masks = 1
            batch_size = 12
            print(data_point.shape, plucker.shape)
            data_point = data_point[0]
            plucker = plucker[0]
        else:
            time_indices = torch.Tensor([3]).int() # [0,1,2,3,4,5,7,8,9,10,11]
            args.num_masks = 1  # 11

        if not args.visualize_interpolation:
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
                        inpainting_image=(data_point, plucker),
                        batch_size=batch_size,
                        num_inference_steps=args.num_inference_steps,
                        output_type="numpy",
                        user_num_masks=args.num_masks,
                        user_time_indices=time_indices,
                    ).images
                masked_indices = np.array([i for i in range(args.n_input + args.n_output) if i not in unmasked_indices.tolist()])
            elif args.model_type == "img2img":
                print("getting predictions using args.num_masks", args.num_masks)
                prediction, _ = pipeline(
                        inpainting_image=(data_point, plucker),
                        batch_size=batch_size,
                        num_inference_steps=args.num_inference_steps,
                        output_type="numpy",
                        user_num_masks=args.num_masks,
                        user_time_indices=time_indices,
                    ).images
            elif args.model_type == "clippose":
                print("getting predictions using args.num_masks", args.num_masks)
                prediction, _ = pipeline(
                        inpainting_image=(data_point, plucker),
                        batch_size=batch_size,
                        guidance=args.guidance,
                        num_inference_steps=args.num_inference_steps,
                        output_type="numpy",
                        user_num_masks=args.num_masks,
                        user_time_indices=time_indices,
                    ).images
        else:
            prediction = batch["interp_depth"].numpy()
            masked_indices = np.array([0])

        prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)

        if not args.model_type in ["img2img", "clippose"]:
            prediction = prediction[:, :, None, :, :]
        else:
            prediction = prediction.reshape(batch_size, 1, C, H, W)
            data_point_minmax = data_point[:, :args.num_images, :, :, :].cpu()
            data_point_minmax = torch.clamp((data_point_minmax / 2 + 0.5), 0, 1)
            prediction = torch.cat([data_point_minmax, prediction], axis=1)
            masked_indices = [args.num_images]

        B, T, C, H, W = prediction.shape

        if args.visualize_3d:
            all_points = []
            all_colors = []
            all_edges = []
            i = 0

        batch_colored_images = {0: [], 1: [], 2: [], 3: []}
        batch_colored_rgb_images = {0: [], 1: [], 2: [], 3: []}

        for b in range(prediction.shape[0]):
            count += 1
            spiral = []
            colored_images = []
            colored_rgb_images = []
            for framenumber in range(prediction.shape[1]):

                if "d" in args.data_format:
                    prediction_minmax = (prediction[b, framenumber, -1] - prediction[b, framenumber, -1].min()) / (prediction[b, framenumber, -1].max() - prediction[b, framenumber, -1].min())
                    if framenumber in masked_indices:
                        cmap = matplotlib.cm.get_cmap('jet_r')
                        cmap = truncate_colormap(cmap, 0.6, 1.0)
                    else:
                        cmap = matplotlib.cm.get_cmap('gray_r')
                        # cmap = matplotlib.cm.get_cmap('jet_r')
                        # cmap = truncate_colormap(cmap, 0.6, 1.0)
                    colored_image = cmap(prediction_minmax)
                    colored_images.append(Image.fromarray((colored_image * 255).astype(np.uint8)))
                    if framenumber in masked_indices:
                        groundtruth = data_point[b, framenumber, -1].cpu().numpy() / 2 + 0.5
                        groundtruth = cmap(groundtruth)
                        os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm", exist_ok=True)
                        os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}", exist_ok=True)
                        path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}/depth_3_{b}.png"
                        gtpath = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}/depth_gt.png"
                        npypath = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}/depth.npy"
                        np.save(npypath, prediction[b, framenumber, -1])
                        Image.fromarray((colored_image * 255).astype(np.uint8)).save(path)
                        Image.fromarray((groundtruth * 255).astype(np.uint8)).save(gtpath)

                    else:
                        os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}", exist_ok=True)
                        path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{seq}/input_{framenumber}_depth.png"
                        Image.fromarray((colored_image * 255).astype(np.uint8)).save(path)

                    batch_colored_images[b].append(Image.fromarray((colored_image * 255).astype(np.uint8)))

                if "rgb" in args.data_format:
                    prediction_minmax = prediction[b, framenumber, :3].permute(1, 2, 0).numpy().squeeze()[:, :, ::-1]
                    colored_rgb_image = Image.fromarray((prediction_minmax * 255).astype(np.uint8))
                    # if framenumber not in masked_indices:
                    #     colored_rgb_image = ImageOps.grayscale(colored_rgb_image)
                    if framenumber in masked_indices:
                        os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm", exist_ok=True)
                        path = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/{count}_rgb.png"
                        colored_rgb_image.save(path)
                    colored_rgb_images.append(colored_rgb_image)
                    batch_colored_rgb_images[b].append(colored_rgb_image)

                """
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
                """

            if args.visualize_3d:
                all_points = np.concatenate(all_points, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                all_edges = np.stack(all_edges, axis=0)
                write_pointcloud(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals_ddpm/3d_{count}.ply", all_points, all_colors, edges=all_edges)

            if "d" in args.data_format:
                os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm", exist_ok=True)
                # utils.make_gif(colored_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/2d_{count}.gif", duration=800)

            if "rgb" in args.data_format:
                os.makedirs(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_rgb_ddpm", exist_ok=True)
                utils.make_gif(colored_rgb_images, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_rgb_ddpm/2d_{count}.gif", duration=800)

        """
        if "d" in args.data_format:
            batch_colored_images_toplot = []
            widths, heights = zip(*(i.size for i in batch_colored_images[0]))
            total_width = sum(widths[:4])
            max_height = max(heights)

            for i in range(len(batch_colored_images[0])):
                x_offset = 0
                new_im = Image.new('RGB', (total_width, max_height))
                for b in range(prediction.shape[0]):
                    new_im.paste(batch_colored_images[b][i], (x_offset,0))
                    x_offset += batch_colored_images[b][i].size[0]
                batch_colored_images_toplot.append(new_im)

            utils.make_gif(batch_colored_images_toplot, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_ddpm/batch_2d_{count}.gif", duration=800)

        if "rgb" in args.data_format:
            batch_colored_rgb_images_toplot = []
            widths, heights = zip(*(i.size for i in batch_colored_rgb_images[0]))
            total_width = sum(widths[:4])
            max_height = max(heights)

            for i in range(len(batch_colored_rgb_images[0])):
                x_offset = 0
                new_im = Image.new('L', (total_width, max_height))
                for b in range(prediction.shape[0]):
                    new_im.paste(batch_colored_rgb_images[b][i], (x_offset,0))
                    x_offset += batch_colored_rgb_images[b][i].size[0]
                batch_colored_rgb_images_toplot.append(new_im)

            utils.make_gif(batch_colored_rgb_images_toplot, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_visuals_rgb_ddpm/batch_2d_{count}.gif", duration=800)

        if args.visualize_spiral:
            spiral.append(colored_images[6])
            utils.make_gif(spiral, f"{args.model_dir}/checkpoint-{args.checkpoint_number}/visuals_ddpm/spiral_{count}.gif", duration=800)

        """


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
