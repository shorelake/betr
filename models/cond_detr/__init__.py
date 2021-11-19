from loguru import logger

def build_conditionaldetr(args):
    if args.num_feature_levels == 1:
        from .conditional_detr import build
        from .default_cond_detr import build as build_default
        logger.info("build default conditional detr")
        # return build(args)
        return build_default(args)
    elif args.num_feature_levels == 2:
        from .multiscale_conditional_detr import build
        logger.info("build multi scale conditional detr")
        return build(args)
    else:
        logger.error(f'multi {args.num_feature_levels} scales cond_detr not supported')
        raise ValueError(f'multi {args.num_feature_levels} scales cond_detr not supported')