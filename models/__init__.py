# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build as build_default_defdetr
from .deformable_detr_fpn import build as build_fpn_defdetr
from loguru import logger

def build_model(args):
    if args.neck_encoder == 'deftransformer':
        logger.info("build default resnet deformable detr")
        return build_default_defdetr(args)
    elif args.neck_encoder == 'fpn':
        logger.info("build resnet deformable detr, neck encoder is FPN")
        return build_fpn_defdetr(args)
    else:
        logger.error(f"wrong neck encoder {args.neck_encoder}")
        raise ValueError(f"wrong neck encoder {args.neck_encoder}")

