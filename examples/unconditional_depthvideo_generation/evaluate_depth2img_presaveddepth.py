import os
import requests
from matplotlib.cbook import time
import torch
import inspect
import argparse
import cv2
import glob
import numpy as np
from prettytable import PrettyTable
from PIL import Image
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMDepthPoseInpaintingPipeline, DDPMInpaintingPipeline, DDPMReconstructionPipeline, DDPMImg2ImgPipeline, DDPMImg2ImgCLIPPosePipeline
from diffusers import DPMSolverMultistepScheduler, UNet2DModel, UNet2DConditionRenderModel, UNet2DConditionSpacetimeRenderModel, DDPMScheduler, DDIMScheduler, DDPMConditioningScheduler, EulerDiscreteScheduler, StableDiffusionDepth2ImgPipeline
import matplotlib.cm
from matplotlib import pyplot as plt
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, AutoProcessor, LlavaForConditionalGeneration
import utils
from data import TAOMAEDataset, CO3DEvalDataset, TAOForecastingDataset, OccfusionDataset, collate_fn_depthpose, collate_fn_inpainting, TAOTemporalSuperResolutionDataset, collate_fn_temporalsuperres
from utils import render_path_spiral, write_pointcloud

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


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
        "--sampling_strategy",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "autoregressive", "direct", "aidedautoregressive"],
        help="What kind of sampling strategy to use at inference.",
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
            autoregressive=True,
            num_autoregressive_frames=10,
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

    llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava_prompt = "<image>\nUSER: Caption the image in one long sentence.\nASSISTANT:"
    assert args.data_format == "rgbd"

    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(512)
            ]
        )
    image_transforms_reverse = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )
    depth_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]
    )

    n_propmt = "ugly looking, bad quality, cartoonish"

    top1_psnr, top3_psnr, top5_psnr = 0, 0, 0

    count = 0

    if not args.evaluate_interpolation:
        headers = ['Top1', 'Top1 (inv)', 'Top3', 'Top3 (inv)', 'Top5', 'Top5 (inv)']
    else:
        headers = ['Top1', 'Top1 (inv)']

    if load_rgb:
        headers_rgb = ['Top1 PSNR', 'Top3 PSNR', 'Top5 PSNR']

    # open the results file
    resultsfile = open(f"{args.model_dir}/checkpoint-{args.checkpoint_number}/depth2img_results_{args.sampler}_guidance{args.guidance}.txt", "w")
    saveddepths_dir = f"{args.model_dir}/checkpoint-{args.checkpoint_number}/train_ddpm_sampling-{args.sampling_strategy}/"

    for b, batch in enumerate(eval_dataloader):

        print(f"Doing {b}/{len(eval_dataloader)}")

        data_point = batch["input"].to("cuda:0")  # shape: B T H W
        data_point = data_point[:, :, None, :, :]  # shape: B T C H W

        filename = batch["filenames"][0][2]
        seq = str(filename)[len(str(args.eval_data_dir)):str(filename).rfind("/")]

        if args.train_with_plucker_coords:
            plucker = batch["plucker_coords"].to("cuda:0")
            # print("found frame idS to be", data_point.shape, plucker.shape)
            # plucker = [pc.to("cuda:0") for pc in batch["plucker_coords"]]
            # ray_origin = batch["ray_origin"]
            # ray_direction = batch["ray_direction"]
            # cam_coords = batch["cam_coords"]

        if load_rgb:
            rgb_images = batch["rgb_input"].to("cuda:0")  # shape: B T C H W
            caption_image = rgb_images[:, 2, ...]

        data_point = data_point

        B, T, C, H, W = data_point.shape

        prediction = torch.zeros(5, T-3, C, H, W)
        for b in range(5):
            files = sorted(glob.glob(saveddepths_dir + f"{seq}_1_depth_nogt/run_{b+1}/00*.png"))
            for fnum, f in enumerate(files[3:]):
                # should be a torch tensor with values between 0 to 1
                d = rgb2gray(cv2.imread(f))
                d = d / 255.0
                if fnum == 0:
                    print("min max values in d", d.min(), d.max())
                prediction[b, fnum, 0] = torch.from_numpy(d)

        if not args.evaluate_interpolation:
            data_point = torch.stack([data_point[0, ...]] * 5)
            plucker = torch.stack([plucker[0, ...]] * 5)
            B, T, C, H, W = data_point.shape
            batch_size = 5

        else:
            prediction = batch["interp_depth"]
            masked_indices = np.array([0])

        prediction = prediction.reshape(B, 10, C, H, W)
        prediction = torch.cat([data_point[:, :args.num_images, :, :, :].cpu(), prediction], axis=1)
        masked_indices = [k for k in range(args.num_images, T)]

        B, T, C, H, W = prediction.shape
        count += 1

        groundtruth = rgb_images[:, 3:, ...].cpu()
        groundtruth = torch.stack([groundtruth[0]] * 5)
        # prediction = prediction / 2 + 0.5
        groundtruth = groundtruth / 2 + 0.5

        # shapes of prediction are batch x frames x height x width
        # shapes of groundtruth is the same
        # but now we need to find top 1 and top 3 numbers
        # we will compute both the normal and scale/shift invariant loss
        caption_image = caption_image / 2 + 0.5
        caption_image = (caption_image.cpu().numpy() * 255).astype(np.uint8)[:, ::-1, ...]
        llava_inputs = llava_processor(text=[llava_prompt], images=caption_image, return_tensors="pt")
        input_ids = llava_inputs["input_ids"].cuda()
        attention_mask = llava_inputs["attention_mask"].cuda()
        pixel_values = llava_inputs["pixel_values"].cuda()
        llava_model = llava_model.cuda()
        generate_ids = llava_model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_length=500)
        prompt = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(llava_prompt)-7:]

        # get new predictions from stable diffusion depth2img model
        depth_maps = (prediction[:, masked_indices, -1, ...].numpy() * 255).astype(np.uint8).squeeze()
        depth_map = [Image.fromarray(255 - d) for depth_map in depth_maps for d in depth_map]
        depth_map = [depth_transforms(d).squeeze() for d in depth_map]
        init_images = np.concatenate([caption_image] * 50, axis=0).astype(np.uint8)
        init_images = [image_transforms(Image.fromarray(d.transpose(1, 2, 0)[:, :, ::-1])) for d in init_images]
        print(prompt)
        images = []
        for j in range(5):
            image = pipe(prompt=[prompt] * 10, image=init_images[j*10: (j+1)*10], negative_prompt=[n_propmt] * 10, strength=0.3, guidance_scale=5.0, depth_map=torch.stack(depth_map[j*10:(j+1)*10])).images
            images += image

        print("len of all images found", len(images))


        prediction = [image_transforms_reverse(im) for im in images]
        prediction = torch.stack([torch.from_numpy(np.array(im)) for im in prediction]).permute(0, 3, 1, 2) / 255.0
        prediction = prediction.reshape(5, 10, 3, 64, 64)
        # prediction = prediction / 2 + 0.5
        print("min max of images after", prediction[0].min(), prediction[0].max())


        # pred_to_save = (prediction[0, 0] * 255).numpy().astype(np.uint8)[::-1]
        # gt_to_save = (groundtruth[0, 0] * 255).numpy().astype(np.uint8)[::-1]

        # plt.imsave("/data3/tkhurana/d2i_gt.png", gt_to_save.transpose(1, 2, 0))
        # plt.imsave("/data3/tkhurana/d2i_pd.png", pred_to_save.transpose(1, 2, 0))

        # gt_new = groundtruth.numpy()[:, ::-1, :, :].copy()
        # groundtruth = torch.from_numpy(gt_new)

        if "rgb" in args.data_format:
            top1_prediction = prediction[:1, ...]
            top1_groundtruth = groundtruth[:1, ...]
            top1_psnr += utils.topk_psnr(top1_prediction, top1_groundtruth)

            top3_prediction = prediction[:3, ...]
            top3_groundtruth = groundtruth[:3, ...]
            top3_psnr += utils.topk_psnr(top3_prediction, top3_groundtruth)

            top5_prediction = prediction[:5, ...]
            top5_groundtruth = groundtruth[:5, ...]
            top5_psnr += utils.topk_psnr(top5_prediction, top5_groundtruth)

            all_psnr = np.array([top1_psnr[0], top3_psnr[0], top5_psnr[0]])

            all_psnr /= count

            tab = PrettyTable(headers_rgb)
            tab.add_rows([list(all_psnr)])
            print(tab)

    resultsfile.write(",".join(headers_rgb) + "\n")
    resultsfile.write(",".join([str(p) for p in all_psnr]) + "\n")

    resultsfile.close()


if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
