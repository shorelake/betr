from .vidt import build

from loguru import logger

def build_vidt(args):
    logger.info("building vit backbone vidt")
    return build(args)