from torch.utils.data import Dataset
import argparse
import inspect
import logging
import math
import os
import cv2
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
from matplotlib import pyplot as plt

import utils

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
                depth = np.where(depth > 100.0, 100.0, depth)
                depth = depth / 100.0

            # just to be sure, clip away all values beyond the range of 0-1
            depth = np.clip(depth, 0, 1)
            depth = depth * 255.0
            depth_frames.append(self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze())

        depth_video = torch.stack(depth_frames, axis=0)

        example["input"] = depth_video
        return example


def collate_fn_depthpose(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    ray_origin = torch.stack([example["ray_origin"] for example in examples])
    ray_origin = ray_origin.to(memory_format=torch.contiguous_format).float()

    image_plane_in_cam = torch.stack([example["image_plane_in_cam"] for example in examples])
    image_plane_in_cam = image_plane_in_cam.to(memory_format=torch.contiguous_format).float()

    Rt = torch.stack([example["Rt"] for example in examples])
    Rt = Rt.to(memory_format=torch.contiguous_format).float()

    # ray_direction = torch.stack([example["ray_direction"] for example in examples])
    # ray_direction = ray_direction.to(memory_format=torch.contiguous_format).float()

    # cam_coords = torch.stack([example["cam_coords"] for example in examples])
    # cam_coords = cam_coords.to(memory_format=torch.contiguous_format).float()

    filenames = [example["filenames"] for example in examples]

    return {
        "input": inputs,
        "ray_origin": ray_origin,
        "image_plane_in_cam": image_plane_in_cam,
        "Rt": Rt,
        "filenames": filenames
    }


def collate_fn_inpainting(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    return {
        "input": inputs
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
        visualize=False,
        spiral_poses=False
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.visualize = visualize
        self.normalization_factor = normalization_factor
        self.plucker_coords = plucker_coords
        self.use_harmonic = use_harmonic
        self.spiral_poses = spiral_poses
        self.sequence = num_images > 1

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        if self.plucker_coords:
            self.harmonic_embedding = utils.HarmonicEmbedding()

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

                    assert len(frames) == extrinsics.shape[0] == intrinsics.shape[0]

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
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                # brings the training data between -1 and 1
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def encode_plucker(self, ray_origins, ray_dirs):
        """
        ray to plucker w/ pos encoding
        """
        plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
        if self.use_harmonic:
            plucker = self.harmonic_embedding(plucker)
        return plucker

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        ref_seq = self.sequences[ref_index]
        if self.plucker_coords:
            ref_from_global = self.extrinsics[ref_index]
        depth_frames = []
        all_filenames = []
        all_ray_origin = []
        all_ray_dirs_unnormalized = []
        all_cam_coords = []
        all_rt = []
        all_image_plane_in_cam = []
        all_k = []

        if self.visualize:
            all_points = []
            all_edges = []
            all_colors = []
            offset = 0

        for i in range(self.num_images):
            all_filenames.append(self.filenames[ref_index - i])

            curr_seq = self.sequences[ref_index - i]
            assert ref_seq == curr_seq

            depth = cv2.imread(str(self.filenames[ref_index - i]), cv2.IMREAD_ANYDEPTH)

            depth = depth.astype(np.float32) / self.normalization_factor

            if "pointodyssey" in str(self.filenames[ref_index - i]):
                depth = np.where(depth > 100.0, 100.0, depth)
                depth = depth / 100.0
                # just to be sure, clip away all values beyond the range of 0-1
                depth_clipped = np.clip(depth, 0, 1)
                depth = depth_clipped * 255.0
                depth_to_visualize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST)(torch.from_numpy(depth_clipped[None, ...]))
            else:
                # in TAO, we predicted per frame relative depth
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth_preprocessed = self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze()
            scale = depth_preprocessed.shape[0] / depth.shape[0]
            H, W = depth.shape[0] * scale, depth.shape[1] * scale
            H, W = int(H), int(W)

            if self.plucker_coords:
                K_beforescale = self.intrinsics[ref_index - i]
                K = K_beforescale.clone()
                # account for the image resizing operation
                K[0] *= scale
                K[1] *= scale

                # Rt = self.extrinsics[ref_index - i]
                # Rt_inv = torch.linalg.inv(Rt)

                curr_from_global = self.extrinsics[ref_index - i]
                global_from_curr = torch.linalg.inv(curr_from_global)

                ref_from_curr = ref_from_global @ global_from_curr
                Rt_inv = ref_from_curr
                Rt = torch.linalg.inv(ref_from_curr)  # curr_from_ref

                ray_origins, ray_dirs, image_rays_in_cam = utils.get_plucker(K, Rt, H, W, return_image_plane=True)
                plucker_rays = utils.encode_plucker(ray_origins, ray_dirs)
                plucker_map = transforms.CenterCrop(self.size)(plucker_rays.reshape(H, W, -1).permute(2, 0, 1))
                # image_plane_in_cam = image_rays_in_cam.reshape(-1, H, W)
                image_plane_in_cam = transforms.CenterCrop(self.size)(image_rays_in_cam.reshape(-1, H, W))
                depth_with_plucker_map = torch.cat([depth_preprocessed[None, ...], plucker_map], dim=0)

                depth_frames.append(depth_with_plucker_map)
                # ray_dirs_preprocess = transforms.CenterCrop(self.size)(ray_dirs_unnormalized[:3, :].reshape(3, H, W)).permute(1, 2, 0).reshape(-1, 3)
                all_ray_origin.append(ray_origins[0][None, :])
                all_image_plane_in_cam.append(image_plane_in_cam)
                # all_ray_dirs_unnormalized.append(ray_dirs_preprocess)
                # all_cam_coords.append(cam_coords.T)
                all_rt.append(Rt)
                all_k.append(K)

                """
                if self.visualize:
                    # depth_to_visualize = transforms.CenterCrop(self.size)(depth_to_visualize)
                    depth_to_visualize = (depth_preprocessed / 2 + 0.5).clamp(0, 1)
                    points_in_3d = utils.get_points_given_imageplane(image_plane_in_cam, depth_to_visualize, Rt, 100.0)
                    all_points.append(points_in_3d.numpy().T)
                    edges = np.arange(all_points[-1].shape[0]) + 1 + offset
                    edges = np.hstack([np.zeros((all_points[-1].shape[0], 1)) + offset, edges[:, None]])
                    all_edges.append(edges.astype(int))
                    all_colors.append(np.zeros_like(all_points[-1]) + 19 * (i+1))
                    offset += all_points[-1].shape[0]
                """

            else:
                depth_frames.append(depth_preprocessed)

        depth_video = torch.stack(depth_frames, axis=0)

        if self.plucker_coords:
            all_ray_origin = torch.stack(all_ray_origin, axis=0)
            all_image_plane_in_cam = torch.stack(all_image_plane_in_cam, axis=0)
            all_rt = torch.stack(all_rt, axis=0)
            # all_ray_dirs_unnormalized = torch.stack(all_ray_dirs_unnormalized, axis=0)
            # all_cam_coords = torch.stack(all_cam_coords, axis=0)

            if self.spiral_poses:
                middle_pose = np.zeros((1, 3, 5))
                middle_K = all_k[6]
                mp = torch.linalg.inv(all_rt[6])
                middle_pose[0, :3, :4] = mp[:3, :4]
                middle_pose[0, :, 4] = np.array([H, W, middle_K[0,0]])
                render_poses = np.stack(utils.render_path_spiral(middle_pose, 3.0, N = 12))

                # render poses should be of shape 12 x 3 x 5
                # visualize them if the user wants to
                if self.visualize:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    for p in range(12):
                        vis_c2w = render_poses[p][:4, :4]
                        utils.draw_wireframe_camera(ax, vis_c2w, scale=1.0, color='g')
                    for p in range(12):
                        vis_c2w = torch.linalg.inv(all_rt[p])
                        utils.draw_wireframe_camera(ax, vis_c2w.numpy(), scale=1.0, color='b')
                    utils.draw_wireframe_camera(ax, mp.numpy(), scale=1.0, color='r')
                    plt.show()

                render_poses = torch.from_numpy(render_poses[:, :3, :4])
                depth_video = depth_video.unsqueeze(0).repeat(12, 1, 1, 1, 1)
                for p in range(12):
                    pose = render_poses[p]
                    pose_homo = torch.cat([pose, torch.Tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0).float()
                    render_ray_origins, render_ray_dirs = utils.get_plucker(middle_K, torch.linalg.inv(pose_homo), H, W)
                    render_plucker_rays = utils.encode_plucker(render_ray_origins, render_ray_dirs)
                    render_plucker_map = transforms.CenterCrop(self.size)(render_plucker_rays.reshape(H, W, -1).permute(2, 0, 1))
                    depth_video[p, 6, 1:, ...] = render_plucker_map

        """
        if self.visualize:
            all_points = np.concatenate(all_points, axis=0)
            all_edges = np.concatenate(all_edges, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            utils.write_pointcloud(f"/data/tkhurana/visualizations/plucker/{ref_index}.ply", all_points, rgb_points=all_colors)
        """

        example["input"] = depth_video  # video is of shape T x H x W or T x C x H x W
        example["filenames"] = all_filenames
        if self.plucker_coords:
            example["ray_origin"] = all_ray_origin
            example["image_plane_in_cam"] = all_image_plane_in_cam
            example["Rt"] = all_rt
            # example["ray_direction"] = all_ray_dirs_unnormalized
            # example["cam_coords"] = all_cam_coords
        return example
