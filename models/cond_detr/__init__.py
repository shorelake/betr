from loguru import logger
__all__ = ['conditional_detr', 'default_conditional_detr', 'default_conditional_detr_v2']

def build_conditionaldetr(args):
    if args.detector == 'conditional_detr':
        if args.num_feature_levels == 1:
            from .conditional_detr import build
            # from .default_cond_detr import build as build_default
            logger.info("build single scale conditional detr")
            return build(args)
            # return build_default(args)
        elif args.num_feature_levels == 2:
            from .multiscale_conditional_detr import build
            logger.info("build multi scale conditional detr")
            return build(args)
        else:
            logger.error(f'multi {args.num_feature_levels} scales cond_detr not supported')
            raise ValueError(f'multi {args.num_feature_levels} scales cond_detr not supported')
    elif args.detector == 'default_conditional_detr':
        from .default_cond_detr import build
        logger.info(f'build {args.num_feature_levels} scale default conditional detr')
        return build(args)
    elif args.detector == 'default_conditional_detr_v2':
        from .default_cond_detr_v2 import build
        logger.info(f'build {args.num_feature_levels} scale default conditional detr v2')
        return build(args)