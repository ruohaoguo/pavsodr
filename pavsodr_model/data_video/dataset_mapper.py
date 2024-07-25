import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["PAVSODRDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


class PAVSODRDatasetMapper:
    """
    A callable which takes a dataset dict in PAVSODR Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 1,
        audio_encoder_name: str = "soundnet",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.audio_encoder_name     = audio_encoder_name
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        audio_encoder_name = cfg.AUDIO_ENCODER_NAME

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "audio_encoder_name": audio_encoder_name
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in PAVSODR Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        audio_ori = dataset_dict["audio"]
        video_length = dataset_dict["length"]

        is_continuous = False
        if self.is_train:
            if is_continuous:
                selected_idx = np.arange(0, video_length)
                ref_frame = random.randint(0, video_length - self.sampling_frame_num - 1)
                selected_idx = selected_idx[ref_frame: ref_frame + self.sampling_frame_num]
            else:
                ref_frame = random.randrange(video_length)
                start_idx = max(0, ref_frame - self.sampling_frame_range)
                end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)
                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        dataset_dict["audio"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to home-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))

            # ranks
            _gt_ranks = []
            for _anno in sorted_annos:
                if _anno["id"] != -1:
                    _gt_ranks.append(_anno["rank"] - 1 if _anno["rank"] <= 5 else 5)
                else:
                    _gt_ranks.append(5)
            instances.gt_ranks = torch.tensor(_gt_ranks)

            dataset_dict["instances"].append(instances)


        # audio processing
        if self.audio_encoder_name == "soundnet" or self.audio_encoder_name == "htsat":
            audio_ori_ch1 = audio_ori[0]
            audio_ori_ch2 = audio_ori[1]
            audio_ori_ch3 = audio_ori[2]
            audio_ori_ch4 = audio_ori[3]
            audio_split_size = int(len(audio_ori[1]) / (int(video_length)))
            audio_ch1_split = torch.split(tensor=audio_ori_ch1, split_size_or_sections=audio_split_size)
            audio_ch1_split = list(audio_ch1_split)
            audio_ch1_split = audio_ch1_split[:int(video_length)]
            audio_ch2_split = torch.split(tensor=audio_ori_ch2, split_size_or_sections=audio_split_size)
            audio_ch2_split = list(audio_ch2_split)
            audio_ch2_split = audio_ch2_split[:int(video_length)]
            audio_ch3_split = torch.split(tensor=audio_ori_ch3, split_size_or_sections=audio_split_size)
            audio_ch3_split = list(audio_ch3_split)
            audio_ch3_split = audio_ch3_split[:int(video_length)]
            audio_ch4_split = torch.split(tensor=audio_ori_ch4, split_size_or_sections=audio_split_size)
            audio_ch4_split = list(audio_ch4_split)
            audio_ch4_split = audio_ch4_split[:int(video_length)]

            audios = []
            if end_idx - start_idx == 11:
                start_index = start_idx
                end_index = end_idx
                a_ch1 = audio_ch1_split[start_index: end_index]
                a_ch2 = audio_ch2_split[start_index: end_index]
                a_ch3 = audio_ch3_split[start_index: end_index]
                a_ch4 = audio_ch4_split[start_index: end_index]
            else:
                if start_idx in list(range(0, 12)) or end_idx in list(range(0, 12)):
                    start_index = 0
                    end_index = 11
                elif start_idx in list(range(video_length-11, video_length)) or end_idx in list(range(video_length-11, video_length)):
                    start_index = video_length - 11
                    end_index = video_length
                else:
                    start_index = start_idx
                    end_index = end_idx
                a_ch1 = audio_ch1_split[start_index: end_index]
                a_ch2 = audio_ch2_split[start_index: end_index]
                a_ch3 = audio_ch3_split[start_index: end_index]
                a_ch4 = audio_ch4_split[start_index: end_index]

            assert end_index - start_index == 11

            audios.append([torch.cat(a_ch1), torch.cat(a_ch2), torch.cat(a_ch3), torch.cat(a_ch4)])
            audios = torch.stack(audios[0], dim=0)
            audios = torch.mean(audios, dim=0).unsqueeze(0).unsqueeze(0)  # ambisonics to mono
            dataset_dict["audio"].append(audios)

        return dataset_dict


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to home-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict
