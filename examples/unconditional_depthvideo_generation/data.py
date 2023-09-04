from torch.utils.data import Dataset
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
from einops import repeat
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path
from totalrecondataset import utils_mini as totalrecon_utils

from utils import HarmonicEmbedding, write_pointcloud

class TAODepthDataset(Dataset):
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
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.sequence = num_images > 1

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        all_frames = []
        seq_to_frames = {}

        for e in ext:
            all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))

        all_frames = list(all_frames)

        self.paths = all_frames

        if self.sequence:
            for frame in all_frames:
                seq = str(frame)[len(str(self.instance_data_root)):str(frame).rfind("/")]
                if seq not in seq_to_frames:
                    seq_to_frames[seq] = []
                seq_to_frames[seq].append(frame)

            for seq in seq_to_frames:
                frames = seq_to_frames[seq]
                start_index = len(self.filenames)

                for idx in range(0, len(frames), self.offset):
                    frame_path = frames[idx]
                    self.sequences.append(seq)
                    self.filenames.append(frame_path)

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

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        depth_frames = []

        for i in range(self.num_images):
            depth = Image.open(self.filenames[ref_index - i])
            depth = np.array(depth) / (256.0 * 80.0)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_frames.append(self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze())

        depth_video = torch.stack(depth_frames, axis=0)

        example["input"] = depth_video
        return example


class TotalReconDataset(Dataset):
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
        frame_ext=["depth"],
        pose_ext=["txt"],
        center_crop=True,
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.sequence = num_images > 1

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.poses = []
        self.valid_indices = []
        all_frames, all_poses = [], []
        seq_to_frames, seq_to_poses = {}, {}

        for e in frame_ext:
            all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))
        all_frames = list(all_frames)
        self.paths = all_frames

        for e in pose_ext:
            all_poses.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))
        all_poses = list(all_poses)
        # all_poses = [p for p in all_poses if '/camera_rtks/' in p]
        self.posepaths = all_poses

        print(len(self.paths), len(self.posepaths))

        # assert len(self.paths) == len(self.posepaths)

        if self.sequence:
            for i, frame in enumerate(all_frames):
                seq = str(frame)[len(str(self.instance_data_root)):str(frame).rfind("/")]
                if seq not in seq_to_frames:
                    seq_to_frames[seq] = []
                    seq_to_poses[seq] = []
                seq_to_frames[seq].append(frame)
                seq_to_poses[seq].append(all_poses[i])

            for seq in seq_to_frames:
                frames = seq_to_frames[seq]
                poses = seq_to_poses[seq]
                start_index = len(self.filenames)

                for idx in range(0, len(frames), self.offset):
                    frame_path = frames[idx]
                    pose_path = poses[idx]
                    self.sequences.append(seq)
                    self.filenames.append(frame_path)
                    self.poses.append(pose_path)

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

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        world2ref, ref_K = totalrecon_utils.read_rtk(self.poses[ref_index])
        depth_frames, poses = [], []

        for i in range(self.num_images):
            depth = totalrecon_utils.read_depth(self.filenames[ref_index - i], (256, 192))
            world2cam, _ = totalrecon_utils.read_rtk(self.poses[ref_index - i])
            # depth = np.array(depth) / (256.0 * 80.0)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_frames.append(self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze())
            cam2ref = torch.from_numpy(world2ref) @ torch.linalg.inv(torch.from_numpy(world2cam))
            poses.append(cam2ref)

        depth_video = torch.stack(depth_frames, axis=0)
        poses = torch.stack(poses, axis=0)

        example["input"] = depth_video
        example["intrinsics"] = torch.from_numpy(ref_K)
        example["extrinsics"] = poses
        return example


class PointOdysseyDepthDataset(Dataset):
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
        normalization_factor=20480.0
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.normalization_factor = normalization_factor
        self.sequence = num_images > 1

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        all_frames = []
        seq_to_frames = {}

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
                seq = str(frame)[len(str(self.instance_data_root)):str(frame).rfind("/")]
                if seq not in seq_to_frames:
                    seq_to_frames[seq] = []
                seq_to_frames[seq].append(frame)

            for seq in seq_to_frames:
                frames = seq_to_frames[seq]
                start_index = len(self.filenames)

                for idx in range(0, len(frames), self.offset):
                    frame_path = frames[idx]
                    self.sequences.append(seq)
                    self.filenames.append(frame_path)

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

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        depth_frames = []

        for i in range(self.num_images):
            depth = Image.open(self.filenames[ref_index - i])
            depth = np.array(depth).astype(np.float32) / self.normalization_factor

            if "pointodyssey" in str(self.filenames[ref_index - i]):
                depth = np.where(depth > 10.0, 10.0, depth)
                depth = depth / 10.0

            # just to be sure, clip away all values beyond the range of 0-1
            depth = np.clip(depth, 0, 1)
            depth = depth * 255.0
            depth_frames.append(self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze())

        depth_video = torch.stack(depth_frames, axis=0)

        example["input"] = depth_video
        return example


def collate_fn(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    return {
        "input": inputs,
    }


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
        use_harmonic=False,
        visualize=False
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.visualize = visualize
        self.normalization_factor = normalization_factor
        self.plucker_coords = plucker_coords
        self.use_harmonic = use_harmonic
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
                    # some intrinsics matrices are invalid
                    if np.sum(intrinsics[0].astype(int)) == 1: #  or int(intrinsics[0][2][2]) != 1 or int(intrinsics[0][0][0]) == 0 or int(intrinsics[0][1][1]) == 0:
                        continue

                for idx in range(0, len(frames), self.offset):
                    frame_path = frames[idx]
                    self.sequences.append(seq)
                    self.filenames.append(frame_path)
                    if self.plucker_coords:
                        R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                        R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                        extrinsic = R2 @ extrinsics[idx] @ R1
                        self.extrinsics.append(torch.from_numpy(extrinsic.astype(np.float32)))
                        self.intrinsics.append(torch.from_numpy(intrinsics[0]))

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
        all_ray_origin = []
        all_ray_dirs_unnormalized = []

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
            assert torch.sum(torch.isinf(depth_preprocessed)) == 0, print(torch.unique(depth_preprocessed))
            assert torch.sum(torch.isnan(depth_preprocessed)) == 0, print(torch.unique(depth_preprocessed))
            scale = depth_preprocessed.shape[0] / depth.shape[0]

            if self.plucker_coords:
                K_beforescale = self.intrinsics[ref_index - i]
                K = K_beforescale.clone()
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
                ray_dirs_unnormalized = Rt_inv @ cam_coords_homo
                ray_dirs = ray_dirs_unnormalized[:3, :].T / torch.norm(ray_dirs_unnormalized[:3, :].T, dim=-1, keepdim=True)
                ray_origins = repeat(Rt_inv[:3, 3], 'c -> c n', n=ray_dirs.shape[0]).T
                assert torch.sum(torch.isnan(ray_origins)) == 0, print(torch.unique(ray_origins))
                assert torch.sum(torch.isinf(ray_origins)) == 0, print(torch.unique(ray_origins))
                assert torch.sum(torch.isnan(ray_dirs)) == 0, print(torch.unique(ray_dirs), K_beforescale, K, Rt_inv, torch.unique(cam_coords_homo), torch.sum(K.int()), int(K[0][0]), int(K[1][1]))
                assert torch.sum(torch.isinf(ray_dirs)) == 0, print(torch.unique(ray_dirs))
                plucker_rays = self.encode_plucker(ray_origins, ray_dirs, use_harmonic=self.use_harmonic)
                assert torch.sum(torch.isnan(plucker_rays)) == 0, print(torch.unique(plucker_rays))
                assert torch.sum(torch.isinf(plucker_rays)) == 0, print(torch.unique(plucker_rays))
                if False:
                    plucker_rays = plucker_rays / torch.norm(plucker_rays, dim=-1,keepdim=True)
                plucker_map = plucker_rays.reshape(H, W, -1).permute(2, 0, 1)
                assert torch.sum(torch.isnan(plucker_map)) == 0, print("after norm and reshape", torch.unique(plucker_map))
                assert torch.sum(torch.isinf(plucker_map)) == 0, print("after norm and reshape", torch.unique(plucker_map))
                plucker_map = transforms.CenterCrop(self.size)(plucker_map)
                assert torch.sum(torch.isnan(plucker_map)) == 0, print("after transforms", torch.unique(plucker_map))
                assert torch.sum(torch.isinf(plucker_map)) == 0, print("after transforms", torch.unique(plucker_map))
                depth_with_plucker_map = torch.cat([depth_preprocessed[None, ...], plucker_map], dim=0)
                assert torch.sum(torch.isnan(depth_with_plucker_map)) == 0, print(torch.unique(plucker_map), torch.unique(depth_preprocessed))
                assert torch.sum(torch.isinf(depth_with_plucker_map)) == 0, print(torch.unique(plucker_map), torch.unique(depth_preprocessed))

                depth_frames.append(depth_with_plucker_map)
                all_ray_origin.append(ray_origins[0][None, :])
                all_ray_dirs_unnormalized.append(ray_dirs_unnormalized[:3, :].T)

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
        all_ray_origin = torch.stack(all_ray_origin, axis=0)
        all_ray_dirs_unnormalized = torch.stack(all_ray_dirs_unnormalized, axis=0)

        if self.visualize:
            all_points = np.concatenate(all_points, axis=0)
            all_edges = np.concatenate(all_edges, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            write_pointcloud(f"/data/tkhurana/visualizations/plucker/{ref_index}.ply", all_points, rgb_points=all_colors)

        example["input"] = depth_video  # video is of shape T x H x W or T x C x H x W
        example["ray_origin"] = all_ray_origin
        example["ray_direction"] = all_ray_dirs_unnormalized
        return example

