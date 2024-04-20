from torch.utils.data import Dataset
import argparse
import inspect
import logging
import math
import os
import cv2
import json
import gzip
from pathlib import Path
from typing import Optional
import random
import accelerate
import datasets
import torch
from einops import repeat
from collections import defaultdict
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
from scipy.interpolate import interpn

import utils


def collate_fn_depthpose(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    """
    num_res = len(examples[0]["plucker_coords"])
    plucker_coords = [torch.stack([example["plucker_coords"][res] for example in examples]) for res in range(num_res)]
    plucker_coords = [pc.to(memory_format=torch.contiguous_format).float() for pc in plucker_coords]
    """

    plucker_coords = torch.stack([example["plucker_coords"] for example in examples])
    plucker_coords = plucker_coords.to(memory_format=torch.contiguous_format).float()

    # ray_origin = torch.stack([example["ray_origin"] for example in examples])
    # ray_origin = ray_origin.to(memory_format=torch.contiguous_format).float()

    # image_plane_in_cam = torch.stack([example["image_plane_in_cam"] for example in examples])
    # image_plane_in_cam = image_plane_in_cam.to(memory_format=torch.contiguous_format).float()

    # Rt = torch.stack([example["Rt"] for example in examples])
    # Rt = Rt.to(memory_format=torch.contiguous_format).float()

    # ray_direction = torch.stack([example["ray_direction"] for example in examples])
    # ray_direction = ray_direction.to(memory_format=torch.contiguous_format).float()

    # cam_coords = torch.stack([example["cam_coords"] for example in examples])
    # cam_coords = cam_coords.to(memory_format=torch.contiguous_format).float()

    filenames = [example["filenames"] for example in examples]

    if "interp_depth" in examples[0]:
        interp_depth = torch.stack([example["interp_depth"] for example in examples])
        interp_depth = interp_depth.to(memory_format=torch.contiguous_format).float()
        return {
            "input": inputs,
            "plucker_coords": plucker_coords,
            "interp_depth": interp_depth,
            "filenames": filenames
        }

    # interp depth should never be asked for alongside rgb input so its okay to
    # return with only rgb input here
    if "rgb_input" in examples[0]:
        rgb_input = torch.stack([example["rgb_input"] for example in examples])
        rgb_input = rgb_input.to(memory_format=torch.contiguous_format).float()
        return {
            "input": inputs,
            "plucker_coords": plucker_coords,
            "rgb_input": rgb_input,
            "filenames": filenames
        }

    return {
        "input": inputs,
        "plucker_coords": plucker_coords,
        # "ray_origin": ray_origin,
        # "image_plane_in_cam": image_plane_in_cam,
        # "Rt": Rt,
        "filenames": filenames
    }


class TAOMAEDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        load_rgb=False,
        rgb_data_root=None,
        co3d_annotations_root=None,
        co3d_rgb_data_root=None,
        co3d_object_crop=False,
        co3d_fps=30,
        size=256,
        num_images=14,
        split="train",
        ext=["png"],
        horizon=1,
        center_crop=True,
        window_horizon=0.5,
        fps=30,
        interpolation_baseline=False,
        normalization_factor=20480.0,
        visualize=False,
    ):
        if load_rgb:
            assert rgb_data_root is not None
        self.size = size
        self.center_crop = center_crop
        self.num_images = num_images
        self.split = split

        self.load_co3d_annotations = False
        load_co3d_files = False
        if "co3d" in instance_data_root:
            self.load_co3d_annotations = True
            assert co3d_annotations_root is not None
            assert co3d_rgb_data_root is not None
            self.co3d_annotations_root = Path(co3d_annotations_root)
            self.co3d_rgb_data_root = Path(co3d_rgb_data_root)
            self.co3d_fps = co3d_fps
            self.co3d_object_crop = co3d_object_crop
            self.co3d_annotations = []
            self.co3d_annotations.extend(sorted(list(self.co3d_annotations_root.rglob(f"*.jgz"))))
            self.frame_to_timestamps = {}
            load_co3d_files = True
            if os.path.exists("./co3d_train_frame_to_metadata.json"):
                load_co3d_files = False
                self.frame_to_timestamps = json.load(open("./co3d_train_frame_to_metadata.json"))

        self.fps = fps
        self.load_rgb = load_rgb
        self.rgb_data_root = rgb_data_root
        self.horizon = horizon
        self.window_horizon = window_horizon
        self.visualize = visualize
        self.interpolation_baseline = interpolation_baseline
        self.normalization_factor = normalization_factor

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        all_frames = []
        seq_to_frames = {}
        self.seq_to_startend = {}
        if self.load_rgb:
            self.rgb_filenames = []

        for e in ext:
            all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))

        all_frames = list(all_frames)

        self.paths = all_frames
        for frame in all_frames:
            seq = str(frame)[len(str(self.instance_data_root)) + 1:str(frame).rfind("/")]
            if seq not in seq_to_frames:
                seq_to_frames[seq] = []
            seq_to_frames[seq].append(frame)

        for seq in seq_to_frames.keys():

            if "co3d" in seq and load_co3d_files:
                _, category, subseq = seq.split("/")
                annotations_root = os.path.join(str(self.co3d_annotations_root), category, "frame_annotations.jgz")
                with gzip.open(annotations_root) as ann:
                    ann = json.load(ann)
                    if category not in self.frame_to_timestamps:
                        self.frame_to_timestamps[category] = {}
                    if subseq not in self.frame_to_timestamps[category]:
                        self.frame_to_timestamps[category][subseq] = {}
                    self.frame_to_timestamps[category][subseq] = {f["image"]["path"]: {"timestamp": f["frame_timestamp"], "mask":f["mask"]["path"]} for f in ann if f["sequence_name"] == subseq}

            if "co3d" in seq:
                valid_fps = self.co3d_fps
            else:
                valid_fps = self.fps

            frames = seq_to_frames[seq]
            start_index = len(self.filenames)

            for idx in range(0, len(frames)):
                frame_path = frames[idx]
                self.sequences.append(seq)
                self.filenames.append(frame_path)
                if self.load_rgb:
                    if "co3d" in seq:
                        rgb_path = str(self.co3d_rgb_data_root) + str(frame_path)[len(str(self.instance_data_root)):-4] + ".jpg"
                    else:
                        rgb_path = str(self.rgb_data_root) + str(frame_path)[len(str(self.instance_data_root)):-4] + ".jpg"
                    self.rgb_filenames.append(rgb_path)

            #
            end_index = len(self.filenames)
            #
            if self.split == "train":
                valid_start_index = start_index + self.horizon * valid_fps
                valid_end_index = end_index - self.horizon * valid_fps
            elif self.split == "val":
                valid_start_index = np.random.choice(
                        np.arange(
                            start_index + self.horizon * valid_fps,
                            end_index - self.horizon * valid_fps),
                        size=(1,),
                        replace=False
                    )[0]
                valid_end_index = valid_start_index + 1

            self.seq_to_startend[seq] = (valid_start_index, valid_end_index)
            self.valid_indices += list(range(valid_start_index, valid_end_index, int(self.window_horizon * valid_fps)))

        self.num_instance_images = len(self.valid_indices)
        self._length = self.num_instance_images

        if load_co3d_files:
            with open("./co3d_train_frame_to_metadata.json", "w") as fin:
                json.dump(self.frame_to_timestamps, fin)

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

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        ref_seq = self.sequences[ref_index]
        ref_start = self.seq_to_startend[ref_seq][0]
        ref_end = self.seq_to_startend[ref_seq][1]
        if "co3d" in ref_seq:
            valid_fps = self.co3d_fps
        else:
            valid_fps = self.fps
        query_index = np.random.choice(
                np.arange(
                    max(ref_start - self.horizon * valid_fps, ref_index - self.horizon * valid_fps),
                    min(ref_end + self.horizon * valid_fps, ref_index + self.horizon * valid_fps)),
                size=(self.num_images + 1,),
                replace=False)
        """
        query_index = np.random.choice(
                np.arange(
                    max(ref_start - self.horizon * self.fps, ref_index - self.horizon * self.fps),
                    min(ref_end + self.horizon * self.fps, ref_index + self.horizon * self.fps)),
                size=(1,),
                replace=False)[0]
        context_index = sorted(np.random.choice(
                np.arange(
                    max(ref_start - self.horizon * self.fps, ref_index - self.horizon * self.fps),
                    ref_index),
                size=(self.num_images,),
                replace=False
                ))
        """
        # print("query index start and end", max(ref_start, ref_index - self.horizon * self.fps), min(ref_end, ref_index + self.horizon * self.fps))
        # print("context index start and end", max(ref_start, ref_index - self.horizon * self.fps), ref_index)
        # print("ref index", ref_index)
        # print("all index", context_index, query_index)
        # input_index = list(context_index) + [query_index]
        input_index = query_index
        depth_frames = []
        if self.load_rgb:
            rgb_frames = []
        all_filenames = []
        index_labels = []

        for i in input_index:
            query_index = int(i)

            all_filenames.append(self.filenames[query_index])

            curr_seq = self.sequences[query_index]
            assert ref_seq == curr_seq

            if "co3d" in ref_seq and self.co3d_object_crop:
                category, subseq, frame_name = str(self.filenames[query_index]).split("/")[-3:]
                frame_key = f"{category}/{subseq}/images/{frame_name}"
                mask = self.frame_to_timestamps[category][subseq][frame_key[:-4]+".jpg"]["mask"]
                mask = cv2.imread(os.path.join(self.co3d_annotations_root, mask), -1)
                bbox = utils.get_bbox_from_mask(mask, 0.4)
                bbox = utils.get_clamp_bbox(torch.Tensor(bbox), 0.3)

            depth = cv2.imread(str(self.filenames[query_index]), cv2.IMREAD_ANYDEPTH)
            if "co3d" in ref_seq and self.co3d_object_crop:
                tbox = [max(0, bbox[0]), max(0, bbox[1]), min(bbox[2], depth.shape[1]), min(bbox[3], depth.shape[0])]
                depth = depth[int(tbox[1]):int(tbox[3]), int(tbox[0]):int(tbox[2])]

            depth = depth.astype(np.float32) / self.normalization_factor
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_preprocessed = self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze()
            depth_frames.append(depth_preprocessed)

            if self.load_rgb:
                rgb = cv2.imread(str(self.rgb_filenames[query_index]), 0)
                if "co3d" in ref_seq and self.co3d_object_crop:
                    tbox = [max(0, bbox[0]), max(0, bbox[1]), min(bbox[2], rgb.shape[1]), min(bbox[3], rgb.shape[0])]
                    rgb = rgb[int(tbox[1]):int(tbox[3]), int(tbox[0]):int(tbox[2])]

                rgb_preprocessed = self.image_transforms(Image.fromarray(rgb)).squeeze()
                rgb_frames.append(rgb_preprocessed.unsqueeze(0))

            if "co3d" in ref_seq:
                category, subseq, frame_name = str(self.filenames[query_index]).split("/")[-3:]
                frame_key = f"{category}/{subseq}/images/{frame_name}"
                index_label = torch.Tensor([self.frame_to_timestamps[category][subseq][frame_key[:-4]+".jpg"]["timestamp"]])
            else:
                index_list = np.arange(self.num_images + 1)
                blah = random.choice(index_list)
                index_label = torch.Tensor([i - input_index[self.num_images-1]]) / self.fps
                # index_label = torch.Tensor([i - input_index[blah]]) / self.fps
                # index_label = torch.Tensor([i]) / self.fps

            index_labels.append(index_label)

        depth_video = torch.stack(depth_frames, axis=0)
        label_video = torch.stack(index_labels, axis=0)

        if "co3d" in ref_seq:
            label_video = label_video - label_video[self.num_images - 1]

        if self.load_rgb:
            rgb_video = torch.stack(rgb_frames, axis=0)

        if self.interpolation_baseline:
            xx, yy = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
            xx, yy = xx.reshape(-1), yy.reshape(-1)
            query = np.stack([np.full_like(yy, label_video[-1]), yy, xx], axis=1)
            interp_depth = interpn(
                    (label_video[:self.num_images], np.arange(self.resolution), np.arange(self.resolution)),
                    depth_video[:self.num_images],
                    query,
                    method='linear',
                    fill_value=None,
                    bounds_error=False)
            interp_depth = np.clip(interp_depth, 0.0, 1.0)
            interp_depth = interp_depth.reshape((self.resolution, self.resolution))

        if self.visualize:
            fig = plt.figure()
            for j in range(self.num_images+1):
                plt.subplot(1, self.num_images+1, j+1)
                plt.imshow(depth_video[j].numpy())
                plt.title(str(label_video[j]))
            plt.show()
            if self.interpolation_baseline:
                fig = plt.figure()
                for j in range(self.num_images+1):
                    plt.subplot(1, self.num_images+1, j+1)
                    if j == self.num_images:
                        plt.imshow(interp_depth)
                        plt.title("interp depth at", str(label_video[j]))
                    else:
                        plt.imshow(depth_video[j].numpy())
                        plt.title(str(label_video[j]))
                plt.show()

        example["pixel_values"] = torch.stack([depth_video] * 3, axis=1)  # video is of shape T x H x W or T x C x H x W
        # example["filenames"] = all_filenames
        # example["plucker_coords"] = label_video
        if self.interpolation_baseline:
            example["interp_depth"] = interp_depth
        if self.load_rgb:
            example["rgb_input"] = rgb_video

        return example


class TAOForecastingDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        load_rgb=False,
        rgb_data_root=None,
        co3d_annotations_root=None,
        co3d_rgb_data_root=None,
        co3d_object_crop=False,
        co3d_fps=10,
        size=256,
        num_images=3,
        split="train",
        autoregressive=False,
        num_autoregressive_frames=10,
        ext=["png"],
        offset=15,
        test_offset=None,
        fps=30,
        horizon=1,
        center_crop=True,
        interpolation_baseline=False,
        normalization_factor=20480.0,
        visualize=False,
    ):
        if load_rgb:
            assert rgb_data_root is not None

        self.load_co3d_annotations = False
        load_co3d_files = False
        if "co3d" in instance_data_root:
            self.load_co3d_annotations = True
            assert co3d_annotations_root is not None
            assert co3d_rgb_data_root is not None
            self.co3d_annotations_root = Path(co3d_annotations_root)
            self.co3d_rgb_data_root = Path(co3d_rgb_data_root)
            self.co3d_fps = co3d_fps
            self.co3d_object_crop = co3d_object_crop
            self.co3d_annotations = []
            self.co3d_annotations.extend(sorted(list(self.co3d_annotations_root.rglob(f"*.jgz"))))
            self.frame_to_timestamps = {}
            load_co3d_files = True
            if os.path.exists("./co3d_forecast_frame_to_metadata.json"):
                load_co3d_files = False
                self.frame_to_timestamps = json.load(open("./co3d_forecast_frame_to_metadata.json"))

        self.size = size
        self.center_crop = center_crop
        self.num_images = num_images
        self.split = split
        self.fps = fps
        self.load_rgb = load_rgb
        self.num_autoregressive_frames = num_autoregressive_frames
        self.autoregressive = autoregressive
        self.rgb_data_root = rgb_data_root
        self.offset = offset
        if test_offset is None:
            self.test_offset = offset
        else:
            self.test_offset = test_offset
        self.horizon = horizon
        self.visualize = visualize
        self.interpolation_baseline = interpolation_baseline
        self.normalization_factor = normalization_factor

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        all_frames = []
        seq_to_frames = {}
        self.seq_to_startend = {}
        if self.load_rgb:
            self.rgb_filenames = []

        for e in ext:
            all_frames.extend(sorted(list(self.instance_data_root.rglob(f"*.{e}"))))

        all_frames = list(all_frames)

        self.paths = all_frames

        for frame in all_frames:
            seq = str(frame)[len(str(self.instance_data_root)) + 1:str(frame).rfind("/")]

            if seq not in seq_to_frames:
                seq_to_frames[seq] = []
            seq_to_frames[seq].append(frame)

        seq_to_frames_keys = list(seq_to_frames.keys())

        # if self.autoregressive:
        #     random.shuffle(seq_to_frames_keys)
            # seq_to_frames_keys = [seq_to_frames_keys[0]]
            # seq_to_frames_keys = ["Charades/JOUM7"]
            ##### seq_to_frames_keys_ex = ['HACS/Croquet_v_vrWYdPeIUqw_scene_0_0-1779', 'HACS/Washing_dishes_v_25eIK85JWi4_scene_0_183-3069', 'HACS/Ping-pong_v_dZZqaYgPrY0_scene_0_0-800', 'HACS/Beer_pong_v_bFTTE4TV-ek_scene_0_173-3004', 'Charades/I0THD', 'ArgoVerse/4518c79d-10fb-300e-83bb-6174d5b24a45', 'AVA/YAAUPjq-L-Q_scene_1_83217-84762']
            ##### selected_vids = ['AVA/WKqbLbU68wU_scene_4_18239-19159_2_depth_nogt', 'AVA/uwW0ejeosmk_scene_3_50442-52200_2_depth_nogt', 'AVA/z-fsLpGHq6o_scene_2_40193-41361_2_depth_nogt', 'ArgoVerse/4518c79d-10fb-300e-83bb-6174d5b24a45_2_depth_nogt', 'ArgoVerse/5ab2697b-6e3e-3454-a36a-aba2c6f27818_2_depth_nogt', 'BDD/b231a630-c4522992_2_depth_nogt', 'Charades/1410C_2_depth_nogt', 'Charades/35LUV_2_depth_nogt', 'HACS/Dodgeball_v_IS3OtsJFP7Y_scene_0_2835-4590_2_depth_nogt', 'HACS/Doing_step_aerobics_v_8QyDjT0ZsHE_scene_0_0-3823_2_depth_nogt', 'HACS/Painting_furniture_v_xNxxM-OOMfw_scene_0_0-1910_2_depth_nogt', 'HACS/Washing_dishes_v_25eIK85JWi4_scene_0_183-3069_2_depth_nogt', 'LaSOT/basketball-11_2_depth_nogt', 'LaSOT/swing-12_2_depth_nogt', 'YFCC100M/v_d4fa85cf4d613518a6e9e7948102452_2_depth_nogt', 'YFCC100M/v_f729d4f362aea24236153ffc589adac_2_depth_nogt']
            ##### seq_to_frames_keys = [s[:-13] for s in selected_vids] + seq_to_frames_keys_ex

        for seq in seq_to_frames_keys:

            if self.autoregressive and "co3d" in seq:
                continue

            if "co3d" in seq and load_co3d_files:
                _, category, subseq = seq.split("/")
                annotations_root = os.path.join(str(self.co3d_annotations_root), category, "frame_annotations.jgz")
                with gzip.open(annotations_root) as ann:
                    ann = json.load(ann)
                    if category not in self.frame_to_timestamps:
                        self.frame_to_timestamps[category] = {}
                    if subseq not in self.frame_to_timestamps[category]:
                        self.frame_to_timestamps[category][subseq] = {}
                    self.frame_to_timestamps[category][subseq] = {f["image"]["path"]: {"timestamp": f["frame_timestamp"], "mask":f["mask"]["path"]} for f in ann if f["sequence_name"] == subseq}

            if "co3d" in seq:
                valid_fps = self.co3d_fps
                valid_offset = self.offset / (self.fps / self.co3d_fps)
            else:
                valid_fps = self.fps
                valid_offset = self.offset

            frames = seq_to_frames[seq]
            start_index = len(self.filenames)

            for idx in range(0, len(frames), valid_offset):
                frame_path = frames[idx]
                self.sequences.append(seq)
                self.filenames.append(frame_path)
                if self.load_rgb:
                    if "co3d" in seq:
                        rgb_path = str(self.co3d_rgb_data_root) + str(frame_path)[len(str(self.instance_data_root)):-4] + ".jpg"
                    else:
                        rgb_path = str(self.rgb_data_root) + str(frame_path)[len(str(self.instance_data_root)):-4] + ".jpg"
                    self.rgb_filenames.append(rgb_path)

            #
            end_index = len(self.filenames)

            #
            if self.split == "train":
                valid_start_index = start_index + int(self.horizon * valid_fps / valid_offset)
                valid_end_index = end_index - int(self.horizon * valid_fps / valid_offset)
            elif self.split == "val":
                choices =  np.arange(
                                start_index + int(self.horizon * valid_fps / valid_offset),
                                end_index - int(self.horizon * valid_fps / valid_offset))
                if len(choices) != 0:
                    valid_start_index = np.random.choice(
                            choices,
                            size=(1,),
                            replace=False
                        )[0]
                    valid_end_index = valid_start_index + 1
                    # elif self.autoregressive:
                    #     valid_start_index = start_index + int(self.horizon * valid_fps / valid_offset)
                    #     valid_end_index = min(valid_start_index + self.num_autoregressive_frames, end_index - int(self.horizon * valid_fps / valid_offset))
                else:
                    continue
            self.seq_to_startend[seq] = (valid_start_index, valid_end_index)
            self.valid_indices += list(range(valid_start_index, valid_end_index))

        self.num_instance_images = len(self.valid_indices)
        self._length = self.num_instance_images

        if load_co3d_files:
            with open("./co3d_forecast_frame_to_timestamps.json", "w") as fin:
                json.dump(self.frame_to_timestamps, fin)
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

    def __getitem__(self, index):
        example = {}
        ref_index = self.valid_indices[index]
        ref_seq = self.sequences[ref_index]
        depth_frames = []
        if self.load_rgb:
            rgb_frames = []
        all_filenames = []
        index_labels = []
        if "co3d" in ref_seq:
            valid_fps = self.co3d_fps
            valid_offset = self.test_offset / (self.fps / self.co3d_fps)
        else:
            valid_fps = self.fps
            valid_offset = self.test_offset

        # take 3 images from the past + a future image
        ####### change future timestep to self.num_images + 1 for evaluation
        for i in list(range(self.num_images)) + [self.num_images + k for k in range(1, 12)]:
            query_index = int(ref_index - self.num_images + i + 1)

            all_filenames.append(self.filenames[query_index])

            curr_seq = self.sequences[query_index]
            assert ref_seq == curr_seq

            if "co3d" in ref_seq and self.co3d_object_crop:
                category, subseq, frame_name = str(self.filenames[query_index]).split("/")[-3:]
                frame_key = f"{category}/{subseq}/images/{frame_name}"
                mask = self.frame_to_timestamps[category][subseq][frame_key[:-4]+".jpg"]["mask"]

                mask = cv2.imread(mask, -1)
                bbox = utils.get_bbox_from_mask(mask, 0.4)
                bbox = utils.get_clamp_bbox(bbox, 0.3)

            depth = cv2.imread(str(self.filenames[query_index]), cv2.IMREAD_ANYDEPTH)
            if "co3d" in ref_seq and self.co3d_object_crop:
                depth = depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            depth = depth.astype(np.float32) / self.normalization_factor

            # in TAO, we predicted per frame relative depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            # depth = depth * 255.0

            depth_preprocessed = self.image_transforms(Image.fromarray(depth.astype("uint8"))).squeeze()
            depth_frames.append(depth_preprocessed)

            if self.load_rgb:
                rgb = cv2.imread(str(self.rgb_filenames[query_index]), 0)
                if "co3d" in ref_seq and self.co3d_object_crop:
                    rgb = rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                rgb_preprocessed = self.image_transforms(Image.fromarray(rgb)).squeeze()
                rgb_frames.append(rgb_preprocessed)

            if "co3d" in ref_seq:
                category, subseq, frame_name = str(self.filenames[query_index]).split("/")[-3:]
                frame_key = f"{category}/{subseq}/images/{frame_name}"
                index_label = torch.Tensor([self.frame_to_timestamps[category][subseq][frame_key[:-4]+".jpg"]["timestamp"]])
            else:
                index_list = np.arange(self.num_images + 1)
                index_list[-1] += 1
                blah = random.choice(index_list)
                subtract_index = int(ref_index - self.num_images + blah + 1)
                index_label = torch.Tensor([query_index - ref_index]) / (valid_fps / valid_offset)
                # index_label = torch.Tensor([query_index - subtract_index]) / (valid_fps / valid_offset)
                # index_label = torch.Tensor([query_index]) / (valid_fps / valid_offset)

            index_labels.append(index_label)

        depth_video = torch.stack(depth_frames, axis=0)
        label_video = torch.stack(index_labels, axis=0).squeeze()
        if self.load_rgb:
            rgb_video = torch.stack(rgb_frames, axis=0)
        if "co3d" in ref_seq:
            label_video = label_video - label_video[self.num_images - 1]

        if self.interpolation_baseline:
            xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))
            xx, yy = xx.reshape(-1), yy.reshape(-1)
            query = np.stack([np.full_like(yy, label_video[-1]), xx, yy], axis=1)
            interp_depth = interpn(
                    (label_video[:self.num_images], np.arange(self.size), np.arange(self.size)),
                    depth_video[:self.num_images],
                    query,
                    method='linear',
                    fill_value=None,
                    bounds_error=False)
            interp_depth = (interp_depth - interp_depth.min()) / (interp_depth.max() - interp_depth.min())
            interp_depth = np.clip(interp_depth, 0.0, 1.0)
            interp_depth = interp_depth.reshape((self.size, self.size))
            # interp_depth = depth_video[-2:-1, :, :].numpy()

        if self.visualize:
            fig = plt.figure()
            for j in range(self.num_images+1):
                plt.subplot(1, self.num_images+1, j+1)
                plt.imshow(depth_video[j].numpy())
                plt.title(str(label_video[j]))
            plt.show()
            if self.interpolation_baseline:
                fig = plt.figure()
                for j in range(self.num_images+1):
                    plt.subplot(1, self.num_images+1, j+1)
                    if j == self.num_images:
                        plt.imshow(interp_depth)
                        plt.title("interp depth at", str(label_video[j]))
                    else:
                        plt.imshow(depth_video[j].numpy())
                        plt.title(str(label_video[j]))
                plt.show()

        example["pixel_values"] = torch.stack([depth_video] * 3, axis=1)  # video is of shape T x H x W or T x C x H x W
        # example["filenames"] = all_filenames
        # example["plucker_coords"] = label_video.unsqueeze(-1)
        if self.interpolation_baseline:
            example["interp_depth"] = torch.from_numpy(interp_depth).unsqueeze(0)
        if self.load_rgb:
            example["rgb_input"] = rgb_video

        return example


if __name__ == "__main__":
    dataset = TAOMAEDataset(
        instance_data_root="/data/tkhurana/TAO-depth/zoe/frames/val/",
        size=64,
        center_crop=False,
        num_images=3,
        load_rgb=True,
        rgb_data_root="/data3/chengyeh/TAO/frames/val/",
        split="train",
        normalization_factor=20480.0,
        visualize=False
    )
    for i in range(len(dataset)):
        print(dataset[i])

