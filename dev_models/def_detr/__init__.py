from .deformable_detr import build

from loguru import logger

def build_defdetr(args):
    # if args.neck_encoder == 'fpn':
    #     logger.info("building vit backbone defdetr with fpn proj")
    #     from .deformable_detr_cnn_neck import build
    #     return build(args)
    # elif args.neck_encoder == 'panet':
    #     logger.info("building vit backbone defdetr with panet proj")
    #     from .deformable_detr_cnn_neck import build
    #     return build(args)
    # else:
    logger.info("building vit backbone defdetr")
    from .deformable_detr import build
    return build(args)