from loguru import logger
__all__ = ['default_conditional_detr']

def build_conditionaldetr(args):
    if args.detector == 'default_conditional_detr':
        from .default_cond_detr import build
        logger.info(f'build {args.num_feature_levels} scale vit backbone conditional detr')
        return build(args)