from .deformable_detr import build

from loguru import logger

def build_defdetr(args):
    logger.info("building vit backbone defdetr")
    return build(args)