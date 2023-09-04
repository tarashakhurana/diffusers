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



