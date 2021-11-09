from .conditional_detr import build

from loguru import logger

def build_conditionaldetr(args):
    logger.info("build default conditional detr")
    return build(args)