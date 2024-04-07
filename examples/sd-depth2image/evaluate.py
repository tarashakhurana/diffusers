import os

from matplotlib.cbook import time
import torch
import inspect
import argparse
import numpy as np
from prettytable import PrettyTable
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMDepthPoseInpaintingPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline, DDPMImg2ImgPipeline, DDPMImg2ImgCLIPPosePipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, UNet2DConditionRenderModel, UNet2DConditionSpacetimeRenderModel, DDPMScheduler, DDIMScheduler, DDPMConditioningScheduler, EulerDiscreteScheduler
import matplotlib.cm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import utils
from data import TAOMAEDataset, CO3DEvalDataset, TAOForecastingDataset, OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting, TAOTemporalSuperResolutionDataset, collate_fn_temporalsuperres


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
        "--eval_rgb_data_dir",
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
            "amount of guidance to use for classifier free guidance"
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
        "--evaluate_interpolation",
        default=False,
        action="store_true",
        help="whether to evaluate the interpolation baseline given by the dataloader",
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
        "--data_format",
        type=str,
        default="d",
        choices=["d", "rgb", "rgbd"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="random",
        choices=["all", "none", "random", "random-half", "half", "custom"],
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "dpm", "euler"],
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
    parser.add_argument("--co3d_object_crop", action="store_true")
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


def return_luminance(pred, gt):
    weights = torch.Tensor([0.1140, 0.5870, 0.2989])
    print(pred.shape, gt.shape, weights.shape)
    pred = torch.einsum("b t c h w, c -> b t h w", pred, weights)
    gt = torch.einsum("b t c h w, c -> b t h w", gt, weights)
    pred = pred[:, :, None, :, :]
    gt = gt[:, :, None, :, :]
    return pred, gt


def main(args):
    load_rgb = False
    rgb_data_root = None

    if "rgb" in args.data_format:
        load_rgb = True
        rgb_data_root = args.eval_rgb_data_dir

    # Initialize the UNet2D
    sampler_dict = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler, "dpm": DPMSolverMultistepScheduler, "euler": EulerDiscreteScheduler}
    Scheduler = sampler_dict[args.sampler]  # DDPMConditioningScheduler
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
        load_rgb=load_rgb,
        rgb_data_root=rgb_data_root,
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
            horizon=11,
            offset=15,
            split="val",
            load_rgb=load_rgb,
            rgb_data_root=rgb_data_root,
            interpolation_baseline=args.evaluate_interpolation,
            normalization_factor=args.normalization_factor,
            visualize=False
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

    # load the depth models to get depth maps for generated images on the fly
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
    repo = "isl-org/ZoeDepth"
    # Zoe_N
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    DEVICE = "cuda"
    zoe = model_zoe_n.to(DEVICE)

    top1_metric, top3_metric, top5_metric = 0, 0, 0
    top1_inv_metric, top3_inv_metric, top5_inv_metric = 0, 0, 0
    top1_zoe_metric, top3_zoe_metric, top5_zoe_metric = 0, 0, 0
    top1_zoe_inv_metric, top3_zoe_inv_metric, top5_zoe_inv_metric = 0, 0, 0

    top1_psnr, top3_psnr, top5_psnr = 0, 0, 0

    count = 0

    if not args.evaluate_interpolation:
        headers = ['Top1', 'Top1 (inv)', 'Top3', 'Top3 (inv)', 'Top5', 'Top5 (inv)']
    else:
        headers = ['Top1', 'Top1 (inv)']

    if load_rgb:
        headers_rgb = ['Top1 PSNR', 'Top3 PSNR', 'Top5 PSNR']

    # open the results file
    resultsfile = open(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/results_{args.sampler}_guidance{args.guidance}.txt", "w")

    for b, batch in enumerate(eval_dataloader):

        print(f"Doing {b}/{len(eval_dataloader)}")

        data_point = batch["input"].to("cuda:0")  # shape: B T H W
        data_point = data_point[:, :, None, :, :]  # shape: B T C H W

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

        if not args.evaluate_interpolation:
            total_frames = data_point.shape[1]
            past_frames = torch.stack([data_point[0, :int(total_frames / 2), ...]] * 1)
            future_frames = torch.stack([data_point[0, int(total_frames / 2):, ...]] * 1)
            data_point = torch.stack([data_point[0, ...]] * 5)
            plucker = torch.stack([plucker[0, ...]] * 5)
            B, T, C, H, W = data_point.shape

            time_indices = torch.Tensor([3]).int() # [0,1,2,3,4,5,7,8,9,10,11]
            args.num_masks = 1  # 11
            batch_size = 5

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
                        inpainting_image=data_point.squeeze(),
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

                prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)
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

                prediction = torch.from_numpy(prediction).permute(0, 3, 1, 2)

        else:
            prediction = batch["interp_depth"]
            masked_indices = np.array([0])

        if not args.model_type in ["img2img", "clippose"]:
            prediction = prediction[:, :, None, :, :]
        else:
            prediction = prediction.reshape(B, 1, C, H, W)
            prediction = torch.cat([data_point[:, :args.num_images, :, :, :].cpu(), prediction], axis=1)
            masked_indices = [args.num_images]

        B, T, C, H, W = prediction.shape
        count += 1

        if args.model_type in ["img2img", "clippose"]:
            assert data_point.ndim == 5
        else:
            assert data_point.ndim == 4
        groundtruth = data_point.cpu()

        # prediction = prediction / 2 + 0.5
        groundtruth = groundtruth / 2 + 0.5

        # shapes of prediction are batch x frames x height x width
        # shapes of groundtruth is the same
        # but now we need to find top 1 and top 3 numbers
        # we will compute both the normal and scale/shift invariant loss
        if "d" in args.data_format:
            top1_prediction = prediction[:1, masked_indices, -1, ...]
            top1_groundtruth = groundtruth[:1, masked_indices, -1, ...]

            top1_metric += utils.topk_l1_error(top1_prediction, top1_groundtruth)
            top1_inv_metric += utils.topk_scaleshift_inv_l1_error(top1_prediction, top1_groundtruth)

            all_metrics = np.array([top1_metric, top1_inv_metric])

            if not args.evaluate_interpolation:
                top3_prediction = prediction[:3, masked_indices, -1, ...]
                top3_groundtruth = groundtruth[:3, masked_indices, -1, ...]
                top3_metric += utils.topk_l1_error(top3_prediction, top3_groundtruth)
                top3_inv_metric += utils.topk_scaleshift_inv_l1_error(top3_prediction, top3_groundtruth)

                top5_prediction = prediction[:5, masked_indices, -1, ...]
                top5_groundtruth = groundtruth[:5, masked_indices, -1, ...]
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
            top1_prediction = prediction[:1, masked_indices, :end_index, ...]
            top1_groundtruth = groundtruth[:1, masked_indices, :end_index, ...]
            top1_prediction_l, top1_groundtruth_l = return_luminance(top1_prediction, top1_groundtruth)
            top1_psnr += utils.topk_psnr(top1_prediction_l, top1_groundtruth_l)

            top3_prediction = prediction[:3, masked_indices, :end_index, ...]
            top3_groundtruth = groundtruth[:3, masked_indices, :end_index, ...]
            top3_prediction_l, top3_groundtruth_l = return_luminance(top3_prediction, top3_groundtruth)
            top3_psnr += utils.topk_psnr(top3_prediction_l, top3_groundtruth_l)

            top5_prediction = prediction[:5, masked_indices, :end_index, ...]
            top5_groundtruth = groundtruth[:5, masked_indices, :end_index, ...]
            top5_prediction_l, top5_groundtruth_l = return_luminance(top5_prediction, top5_groundtruth)
            top5_psnr += utils.topk_psnr(top5_prediction_l, top5_groundtruth_l)

            all_psnr = np.array([top1_psnr[0], top3_psnr[0], top5_psnr[0]])

            all_psnr /= count

            if top5_prediction.shape[2] == 1:
                top5_prediction = torch.cat([top5_prediction] * 3, axis=2)[:, 0, ...]
                top5_groundtruth = torch.cat([top5_groundtruth] * 3, axis=2)[:, 0, ...]
            else:
                top5_prediction = top5_prediction[:, 0, ...]

            top5_prediction_depth = zoe.infer(top5_prediction.to(DEVICE)).cpu().detach()
            top5_groundtruth_depth = zoe.infer(top5_groundtruth.to(DEVICE)).cpu().detach()
            top1_prediction_depth = top5_prediction_depth[:1, ...]
            top1_groundtruth_depth = top5_groundtruth_depth[:1, ...]
            top3_prediction_depth = top5_prediction_depth[:3, ...]
            top3_groundtruth_depth = top5_groundtruth_depth[:3, ...]

            top1_zoe_metric += utils.topk_l1_error(top1_prediction_depth, top1_groundtruth_depth)
            top1_zoe_inv_metric += utils.topk_scaleshift_inv_l1_error(top1_prediction_depth, top1_groundtruth_depth)

            top3_zoe_metric += utils.topk_l1_error(top3_prediction_depth, top3_groundtruth_depth)
            top3_zoe_inv_metric += utils.topk_scaleshift_inv_l1_error(top3_prediction_depth, top3_groundtruth_depth)

            top5_zoe_metric += utils.topk_l1_error(top5_prediction_depth, top5_groundtruth_depth)
            top5_zoe_inv_metric += utils.topk_scaleshift_inv_l1_error(top5_prediction_depth, top5_groundtruth_depth)

            all_zoe_metrics = np.array([top1_zoe_metric, top1_zoe_inv_metric, top3_zoe_metric, top3_zoe_inv_metric, top5_zoe_metric, top5_zoe_inv_metric])

            all_zoe_metrics /= count

            tab = PrettyTable(headers_rgb)
            tab.add_rows([list(all_psnr)])
            print(tab)

            tab = PrettyTable(headers)
            tab.add_rows([list(all_zoe_metrics)])
            print(tab)

    if "d" in args.data_format:
        resultsfile.write(",".join(headers) + "\n")
        resultsfile.write(",".join([str(p) for p in all_metrics]) + "\n")

    if "rgb" in args.data_format:
        resultsfile.write(",".join(headers_rgb) + "\n")
        resultsfile.write(",".join([str(p) for p in all_psnr]) + "\n")

    resultsfile.close()


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
