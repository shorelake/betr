# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# import models.cond_detr as cond_detr
from models import cond_detr
from loguru import logger

def build_model(args):
    if args.detector == 'deformable_detr':
        from models.def_detr import build_defdetr
        logger.info("build deformable detr")
        return build_defdetr(args)
    elif args.detector == 'anchor_detr':
        from models.anchor_detr import build_anchordetr
        logger.info("build anchor detr")
        return build_anchordetr(args)
    elif args.detector in cond_detr.__all__:
        from models.cond_detr import build_conditionaldetr
        logger.info("build conditional detr")
        return build_conditionaldetr(args)
    else:
        logger.error(f"wrong detector {args.detector}")
        raise ValueError(f"wrong detector {args.detector}")

