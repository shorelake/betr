# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from dev_models import cond_detr
from loguru import logger

def build_model(args):
    if args.detector == 'deformable_detr':
        from dev_models.def_detr import build_defdetr
        logger.info("build vit backbone deformable detr detector")
        return build_defdetr(args)
    elif args.detector == 'vidt':
        from dev_models.vidt import build_vidt
        logger.info("build vit backbone vidt detector")
        return build_vidt(args)
    elif args.detector in cond_detr.__all__:
        from dev_models.cond_detr import build_conditionaldetr
        logger.info("build vit backbone conditional detr")
        return build_conditionaldetr(args)
    else:
        logger.error(f"Wrong vit backbone detector name {args.detector}")
        raise ValueError(f"Wrong vit backbone detector name {args.detector}")