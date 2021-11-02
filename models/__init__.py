# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from loguru import logger

def build_model(args):
    if args.detector == 'deformable_detr':
        from models.def_detr import build_defdetr
        logger.info("build deformable detr")
        return build_defdetr(args)
    else:
        logger.error(f"wrong detector {args.detector}")
        raise ValueError(f"wrong detector {args.detector}")

