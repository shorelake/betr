from .anchor_detr import build

from loguru import logger

def build_anchordetr(args):
    logger.info("build default anchor detr")
    return build(args)