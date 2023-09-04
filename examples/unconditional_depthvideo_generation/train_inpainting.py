import argparse
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import torch
from copy import deepcopy
from einops import repeat
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils import HarmonicEmbedding, write_pointcloud

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    arr = arr.to(timesteps.device)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


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
        "--masking_strategy",
        type=str,
        default="all",
        choices=["all", "none", "half", "random", "random-half"]
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
        default=12,
        help="Output channels in the UNet.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=36,
        help="Input channels in the UNet.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=12,
        help="Number of frames in the depth video.",
    )
    parser.add_argument(
        "--n_input",
        type=int,
        default=6,
        help="Number of input frames in the depth video.",
    )
    parser.add_argument(
        "--n_output",
        type=int,
        default=6,
        help="Number of output frames in the depth video.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=15,
        help="Number of frames in the original video after which a frame should be picked.",
    )
    parser.add_argument(
        "--normalization_factor",
        type=float,
        default=20480,
        help="What to divide the loaded depth with in order to bring it between -1 to 1",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
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
        default=128,
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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10000000000000000, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
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
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--train_with_plucker_coords", action="store_true", help="Whether or not to append plucker coord maps to every depth map"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    assert args.n_input + args.n_output == args.num_images

    return args


class OccfusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        size=256,
        num_images=3,
        offset=15,
        ext=["png"],
        center_crop=True,
        normalization_factor=20480.0,
        plucker_coords=False,
        visualize=False
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.visualize = visualize
        self.normalization_factor = normalization_factor
        self.plucker_coords = plucker_coords
        self.sequence = num_images > 1

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        if self.plucker_coords:
            self.harmonic_embedding = HarmonicEmbedding()

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        self.extrinsics = []
        self.intrinsics = []
        all_frames = []
        seq_to_frames = {}
        seq_to_ann = {}

        if "pointodyssey" in str(self.instance_data_root):
            for e in ext:
                all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*/depths/depth_*.{e}"))))
        else:
            for e in ext:
                all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))

        all_frames = list(all_frames)

        self.paths = all_frames

        if self.sequence:
            for frame in all_frames:
                seq = str(frame)[len(str(self.instance_data_root)) + 1:str(frame).rfind("/")]
                if seq not in seq_to_frames:
                    seq_to_frames[seq] = []
                seq_to_frames[seq].append(frame)
                if self.plucker_coords:
                    seq_to_ann[seq] = os.path.join(self.instance_data_root, seq, "../annotations.npz")

            for seq in seq_to_frames:
                frames = seq_to_frames[seq]
                start_index = len(self.filenames)
                if self.plucker_coords:
                    extrinsics = np.load(seq_to_ann[seq])["extrinsics"].astype(np.float32)
                    intrinsics = np.load(seq_to_ann[seq])["intrinsics"].astype(np.float32)

                for idx in range(0, len(frames), self.offset):
                    frame_path = frames[idx]
                    self.sequences.append(seq)
                    self.filenames.append(frame_path)
                    if self.plucker_coords:
                        R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                        R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                        extrinsic = R2 @ extrinsics[idx] @ R1
                        self.extrinsics.append(torch.from_numpy(extrinsic.astype(np.float32)))
                        self.intrinsics.append(torch.from_numpy(intrinsics[idx]))

                #
                end_index = len(self.filenames)
                #
                valid_start_index = start_index + self.num_images # (self.n_input // 10)
                valid_end_index = end_index
                self.valid_indices += list(range(valid_start_index, valid_end_index))
        self.num_instance_images = len(self.valid_indices)
        self._length = self.num_instance_images

        print("found length to be", self._length)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def encode_plucker(self, ray_origins, ray_dirs, use_harmonic=False):
        """
        ray to plucker w/ pos encoding
        """
        plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
        if use_harmonic:
            plucker = self.harmonic_embedding(plucker)
        return plucker

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        depth_frames = []

        if self.visualize:
            all_points = []
            all_edges = []
            all_colors = []
            offset = 0

        for i in range(self.num_images):
            depth = Image.open(self.filenames[ref_index - i])
            # print("min and max values in depth", np.array(depth).min(), np.array(depth).max(), np.array(depth).dtype)
            depth = np.array(depth).astype(np.float32) / self.normalization_factor

            if "pointodyssey" in str(self.filenames[ref_index - i]):
                depth = np.where(depth > 10.0, 10.0, depth)
                depth = depth / 10.0

            # just to be sure, clip away all values beyond the range of 0-1
            depth = np.clip(depth, 0, 1)
            depth = depth * 255.0
            depth_preprocessed = self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze()
            scale = depth_preprocessed.shape[0] / depth.shape[0]

            if self.plucker_coords:
                K = self.intrinsics[ref_index - i]

                # account for the image resizing operation
                K[0] *= scale
                K[1] *= scale

                Rt = self.extrinsics[ref_index - i]
                Rt_inv = torch.linalg.inv(Rt)
                v, u = torch.meshgrid(torch.arange(int(depth.shape[0] * scale)), torch.arange(int(depth.shape[1] * scale)), indexing="ij")
                # v, u = torch.meshgrid(torch.arange(depth.shape[0]), torch.arange(depth.shape[1]), indexing='ij')
                u = int(depth.shape[1] * scale) - u - 1
                uv_homo = torch.stack([u, v, torch.ones_like(v)]).float()
                C, H, W = uv_homo.shape
                # print("uv_homo.shape", uv_homo.shape)
                cam_coords = torch.linalg.inv(K) @ uv_homo.reshape(3, -1)
                cam_coords_homo = torch.cat([torch.ones((1, cam_coords.shape[1])), cam_coords], dim=0)

                # cam_coords /= cam_coords[:, :, 2:]
                # cam_ray_dirs = F.normalize(cam_coords, dim=0)
                ray_dirs = Rt_inv @ cam_coords_homo
                ray_dirs = ray_dirs[:3, :]
                ray_origins = repeat(Rt_inv[:3, 3], 'c -> c n', n=ray_dirs.shape[1])
                plucker_map = self.encode_plucker(ray_origins.T, ray_dirs.T, use_harmonic=False).reshape(H, W, -1).permute(2, 0, 1)
                plucker_map = transforms.CenterCrop(self.size)(plucker_map)
                depth_with_plucker_map = torch.cat([depth_preprocessed[None, ...], plucker_map], dim=0)
                depth_frames.append(depth_with_plucker_map)


                if self.visualize:

                    center_cam = torch.Tensor([[[0, 0, 0]]])
                    center_world = Rt_inv[:3, 3:].T[None, ...]
                    center_world_wrong = Rt[:3, 3:].T[None, ...]

                    icenter_cam = cam_coords[:, int(ray_dirs.shape[1] / 2)][None, None, :]
                    icenter_world = ray_dirs[:, int(ray_dirs.shape[1] / 2)][None, None, :]

                    all_points.append(np.concatenate([Rt_inv[:3, 3:].T, ray_dirs.numpy().T], axis=0))
                    # all_points.append(np.concatenate([np.array([[0, 0, 0]]), cam_coords.numpy().T], axis=0))
                    edges = np.arange(all_points[-1].shape[0]) + 1 + offset
                    edges = np.hstack([np.zeros((all_points[-1].shape[0], 1)) + offset, edges[:, None]])
                    all_edges.append(edges.astype(int))
                    all_colors.append(np.zeros_like(all_points[-1]) + 19 * (i+1))
                    offset += all_points[-1].shape[0]

            else:
                depth_frames.append(depth_preprocessed)

        depth_video = torch.stack(depth_frames, axis=0)

        if self.visualize:
            all_points = np.concatenate(all_points, axis=0)
            all_edges = np.concatenate(all_edges, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            write_pointcloud(f"/data/tkhurana/visualizations/plucker/{ref_index}.ply", all_points, rgb_points=all_colors)

        example["input"] = depth_video # video is of shape T x H x W or T x C x H x W
        return example


def collate_fn(examples):
    # inputs has an extra first dimension of B in addition to the dimenions returned from getitem
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    return {
        "input": inputs,
    }



def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(
            sample_size=args.resolution,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = OccfusionDataset(
        instance_data_root=args.train_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        num_images=args.num_images,
        offset=args.offset,
        normalization_factor=args.normalization_factor,
        plucker_coords=args.train_with_plucker_coords,
        visualize=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["input"]

            if not args.train_with_plucker_coords:
                clean_images = clean_images[:, :, None, :, :]

            B, T, C, H, W = clean_images.shape
            clean_depths = clean_images[:, :, 0, :, :]
            noise = torch.randn(clean_depths.shape).to(clean_depths.device)
            bsz = clean_depths.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_depths.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_depths = noise_scheduler.add_noise(clean_depths, noise, timesteps)

            if args.train_with_plucker_coords:
                noisy_images = torch.cat([noisy_depths[:, :, None, :, :], clean_images[:, :, 1:, :, :]], dim=2)
            else:
                noisy_images = noisy_depths[:, :, None, :, :]


            # choose the type of masking strategy
            # remember that both the clean and noisy images are 5D tensors now
            if args.masking_strategy == "none":
                # merge the timesteps and height dimension of all voxel grids
                model_inputs = torch.cat([
                    noisy_images.reshape(B, T*C, H, W),
                    clean_images.reshape(B, T*C, H, W)], dim=1)

            elif args.masking_strategy == "all":
                # merge the timesteps and height dimension of all voxel grids
                model_inputs = noisy_images.reshape(B, T*C, H, W)

            elif args.masking_strategy == "random" or args.masking_strategy == "random-half" or args.masking_strategy == "half":
                # first pick a random number of frames to mask
                # then pick above number of frame IDs to mask
                num_masks = torch.randint(0, args.n_input + args.n_output, (1,)) + 1
                num_masks = num_masks.numpy()[0]
                if args.masking_strategy == "random-half":
                    num_masks = args.n_input
                time_indices = torch.from_numpy(
                    np.random.choice(args.n_input + args.n_output, size=(num_masks, ), replace=False)
                )
                if args.masking_strategy == "half":
                    time_indices = torch.arange(self.n_input, self.n_input + self.n_output, 1)
                mask_images = torch.zeros_like(clean_images[:, :, 0, :, :]) # 4d
                mask_images[:, time_indices, ...] = torch.ones_like(clean_images[:, time_indices, 0, ...])
                clean_depths_masked = clean_depths * (1 - mask_images)
                if args.train_with_plucker_coords:
                    clean_images_masked = torch.cat([clean_depths_masked[:, :, None, :, :], clean_images[:, :, 1:, :, :]], dim=2)
                else:
                    clean_images_masked = clean_depths_masked
                model_inputs = torch.cat([
                    noisy_images.reshape(B, T*C, H, W),
                    clean_images_masked.reshape(B, T*C, H, W),
                    mask_images.reshape(B, T, H, W)], dim=1)


            with accelerator.accumulate(model):
                # Predict the noise residual
                # timesteps_unet = torch.zeros((bsz,), device=clean_images.device).long()
                model_output = model(model_inputs, timesteps).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise)  # this could have different weights!
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_depths.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_depths, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1) and epoch != 0:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
