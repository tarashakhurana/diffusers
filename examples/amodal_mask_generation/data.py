from torch.utils.data import Dataset
import cv2
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from pathlib import Path
from matplotlib import pyplot as plt
from burstapi.dataset import BURSTDataset
from burstapi.utils import rle_ann_to_mask


class TAOMasksDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        mask_annotation_root,
        size=64,
        num_images=4,
        offset=1,
        center_crop=True,
        visibility_threshold=0.8,
        visualize_batch=False
    ):
        self.size = size
        self.offset = offset
        self.center_crop = center_crop
        self.num_images = num_images
        self.visibility_threshold = visibility_threshold
        self.visualize_batch = visualize_batch

        self.mask_annotation_root = Path(mask_annotation_root)

        if not self.mask_annotation_root.exists():
            raise ValueError(f"TAO Mask annotation file at {self.mask_annotation_root} doesn't exist.")

        self.sequences = []
        self.filenames = []
        self.valid_indices = []
        self.image_size = []
        all_tracks = defaultdict(lambda: defaultdict(list))

        burstdataset = BURSTDataset(str(self.mask_annotation_root))

        category_id_to_name = burstdataset.category_names

        for video in burstdataset:
            start_index = len(self.filenames)
            segmentations = video.segmentations
            frame_indices = list(range(video.num_annotated_frames))
            video_name = video.name

            tracks = defaultdict(list)

            for t in frame_indices:

                image_size = video.image_size

                for track_id in video.track_ids:
                    if track_id in segmentations[t]:
                        tracks[track_id].append({
                            "rle": segmentations[t][track_id]["rle"],
                            "image_size": image_size,
                            "vis": segmentations[t][track_id]["visibility"],
                            "category": category_id_to_name[video._track_category_ids[track_id]]})
                    else:
                        tracks[track_id].append({
                            "rle": "",
                            "image_size": image_size,
                            "vis": 0.0,
                            "category": category_id_to_name[video._track_category_ids[track_id]]})

            all_tracks[video_name] = tracks

        self.all_subsequences = []
        for video in all_tracks:
            for track in all_tracks[video]:

                first = None
                last = None
                for frame_idx, frame in enumerate(all_tracks[video][track]):

                    if frame["vis"] >= self.visibility_threshold:
                        if first is None:
                            first = frame_idx
                        last = frame_idx
                    if frame["vis"] < self.visibility_threshold or frame_idx == len(all_tracks[video][track]) - 1:
                        if last is not None and first is not None:
                            if last - first >= self.num_images:

                                # append everything into the list of frames to select from
                                start_index = len(self.all_subsequences)
                                for idx in range(first, last+1, self.offset):
                                    frame_data = all_tracks[video][track][idx]
                                    subsequence = {
                                        "video": video,
                                        "track": track,
                                        "visibility": frame_data["vis"],
                                        "rle": frame_data["rle"],
                                        "image_size": frame_data["image_size"],
                                        "category": frame_data["category"]
                                    }
                                    self.all_subsequences.append(subsequence)

                                end_index = len(self.all_subsequences)
                                #
                                valid_start_index = start_index + self.num_images
                                valid_end_index = end_index
                                self.valid_indices += list(range(valid_start_index, valid_end_index))

                        first = None
                        last = None

        self._length = len(self.valid_indices)

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

        masks = []
        visualized = False

        for i in range(self.num_images):
            frame = self.all_subsequences[ref_index + i - self.num_images]
            full_mask = rle_ann_to_mask(frame["rle"], frame["image_size"]) * 255
            full_mask = full_mask.astype("uint8")
            per_object_mask = np.where(full_mask == 255)

            bbox = 0, 0, 0, 0
            if len(per_object_mask) != 0 and len(per_object_mask[1]) != 0 and len(per_object_mask[0]) != 0:
                x_min = int(np.min(per_object_mask[1]))
                x_max = int(np.max(per_object_mask[1]))+1
                y_min = int(np.min(per_object_mask[0]))
                y_max = int(np.max(per_object_mask[0]))+1

                bbox = x_min, x_max, y_min, y_max

            per_object_mask = full_mask[bbox[2]:bbox[3], bbox[0]:bbox[1]]

            if self.visualize_batch:
                print("bounding box", bbox)
                print("mask shape", per_object_mask.shape)
                print(np.unique(full_mask), frame["visibility"])
                plt.subplot(1, self.num_images, i + 1)
                plt.imshow(full_mask)
                plt.axis("off")
                plt.title(f"{frame['category']}_{frame['visibility']}")
                visualized = True

            masks.append(self.image_transforms(Image.fromarray(per_object_mask.astype("uint8"))).squeeze())

        if self.visualize_batch or visualized:
            plt.show()

        mask_video = torch.stack(masks, axis=0)

        example["input"] = mask_video
        return example


def collate_fn_inpainting(examples):
    inputs = torch.stack([example["input"] for example in examples])
    inputs = inputs.to(memory_format=torch.contiguous_format).float()

    return {
        "input": inputs
    }

if __name__ == "__main__":
    dataset = TAOMasksDataset(
        mask_annotation_root="/home/tkhurana/Desktop/CMU/Thesis/Everything4D/BURST-benchmark/train_visibility.json",
        abox_annotation_root="/home/tkhurana/Desktop/CMU/Thesis/TAOAmodal/Annotation_tool/validation_with_freeform.json",
        size=64,
        num_images=4,
        offset=1,
        center_crop=True,
        visibility_threshold=0.8,
        visualize_batch=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_inpainting,
        pin_memory=True,
    )

    for batch in dataloader:
        print(batch["input"].shape)

