# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from turtle import pd

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import random
import cv2
import numpy as np
from .transforms import random_affine
from PIL import Image
from loguru import logger

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class MosaicCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(MosaicCocoDetection,self).__init__(img_folder, ann_file, 
                                                 transforms, return_masks, cache_mode=cache_mode,
                                                 local_rank=local_rank, local_size=local_size)
        normalize = T.Compose([
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        self.normalize = normalize
        self.img_size = (640,640)
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.scale = (0.1, 2)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0
    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        input_h, input_w = self.img_size

        # yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        # xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        yc = input_h
        xc = input_w
        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        mosaic_target = {'boxes': torch.zeros([0,4]), 'labels':torch.zeros([0])}
        for i_mosaic, index in enumerate(indices):
            img, target = super(MosaicCocoDetection,self).__getitem__(index)
            # pil rgb to cv2 bgr
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            if scale != 1:
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )
            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            # self.vis_boxes(img, target["boxes"], "mosaic_{}.png".format(i_mosaic))
            target = target.copy()
            if "boxes" in target:
                boxes = target["boxes"]
                if boxes.shape[0] > 0:
                    boxes[:, 0] = scale * boxes[:, 0] + padw
                    boxes[:, 1] = scale * boxes[:, 1] + padh
                    boxes[:, 2] = scale * boxes[:, 2] + padw
                    boxes[:, 3] = scale * boxes[:, 3] + padh
                    mosaic_target["boxes"] = torch.cat((mosaic_target["boxes"], boxes),dim=0)
                    mosaic_target["labels"] = torch.cat((mosaic_target["labels"], target["labels"]),dim=0)

        if len(mosaic_target["boxes"]):
            mosaic_target["boxes"][:,0] = torch.clamp(mosaic_target["boxes"][:,0],min=0,max=2 * input_w)
            mosaic_target["boxes"][:,1] = torch.clamp(mosaic_target["boxes"][:,1],min=0,max=2 * input_h)
            mosaic_target["boxes"][:,2] = torch.clamp(mosaic_target["boxes"][:,2],min=0,max=2 * input_w)
            mosaic_target["boxes"][:,3] = torch.clamp(mosaic_target["boxes"][:,3],min=0,max=2 * input_h)
            mosaic_target["labels"] = mosaic_target["labels"].long()
        # self.vis_boxes(mosaic_img, mosaic_target["boxes"], "mosaic.png")

        mosaic_img,mosaic_target = random_affine(mosaic_img,
                                                 mosaic_target,
                                                 target_size=(input_w,input_h),
                                                 degrees=self.degrees,
                                                 translate=self.translate,
                                                 scales=self.scale,
                                                 shear=self.shear,
                                                 is_cv2=True)
        mosaic_target['size'] = torch.as_tensor([input_h,input_w],dtype=target['size'].dtype)
        
        # # pil rgb to cv2 bgr
        # mosaic_img = cv2.cvtColor(np.asarray(mosaic_img), cv2.COLOR_RGB2BGR)
        # self.vis_boxes(mosaic_img, mosaic_target["boxes"], "mosaic_affined.png")
        # # cv2 bgr to pil rgb
        # mosaic_img = Image.fromarray(cv2.cvtColor(mosaic_img,cv2.COLOR_BGR2RGB))

        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            mosaic_img, mosaic_target = self.normalize(mosaic_img, mosaic_target)
        return mosaic_img, mosaic_target

    def vis_boxes(self,img,boxes,name="mosaic.png"):
        # pil rgb to cv2 bgr
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            box = boxes[i]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (0,255,0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.imwrite(name, img)

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

 
def make_rt_coco_transforms(image_set):
    # realtime data aug setting
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # scales = [256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    scales = [640]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=900),
            #     T.Compose([
            #         # T.RandomResize([400, 500, 600]),
            #         T.RandomResize([200, 300, 400]),
            #         # T.RandomSizeCrop(384, 600),
            #         T.RandomSizeCrop(180, 400),
            #         # T.RandomResize(scales, max_size=1333),
            #         T.RandomResize(scales, max_size=900),
            #     ])
            # ),
            T.RandomResize([640], max_size=640),
            # normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([640], max_size=640),
            normalize,
        ])
    logger.error(f'unknown {image_set}')
    raise ValueError(f'unknown {image_set}')


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([512], max_size=736),
            normalize,
        ])
    logger.error(f'unknown {image_set}')
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    if args.real_time:
        transforms = make_rt_coco_transforms(image_set)

    else:
        logger.error("--mosaic only support with --real_time")
        raise ValueError("--mosaic only support with --real_time")
        transforms = make_coco_transforms(image_set)
    if image_set == 'train':
        dataset = MosaicCocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    elif image_set == 'val':
        dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
