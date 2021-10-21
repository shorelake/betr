# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build as build_defdetr
from .vidt import build as build_vidt
from loguru import logger

def build_model(args):
    if args.detector == 'vidt':
        logger.info("build vit backbone vidt detector")
        return build_vidt(args)
    elif args.detector == 'deformable_detr':
        logger.info("build vit backbone deformable detr detector")
        return build_defdetr(args)
    else:
        logger.error("Wrong vit backbone detector name")
        raise ValueError("Wrong detector name")

