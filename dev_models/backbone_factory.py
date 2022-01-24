# import backbone
# import backbone_ram
from dev_models.backbones import  swin_transformer, swin_transformer_w_ram, swin_transformer_w_fuse, swin_transformer_w_yolos, \
                                    swin_transformer_w_yolos_v2, yolox_cspdarknet
from loguru import logger
def build_backbone(args):
    if hasattr(swin_transformer, args.vit_backbone):
        logger.info(f'build swin backbone {args.vit_backbone}')
        from .backbone import build_backbone
        return build_backbone(args)
    elif hasattr(swin_transformer_w_ram, args.vit_backbone):
        logger.info(f'build swin backbone with ram {args.vit_backbone}')
        from dev_models.backbones.swin_transformer_w_ram import build_backbone
        return build_backbone(args)
    elif hasattr(swin_transformer_w_fuse, args.vit_backbone):
        logger.info(f'build swin backbone with fuse {args.vit_backbone}')
        from dev_models.backbones.swin_transformer_w_fuse import build_backbone
        return build_backbone(args)
    elif hasattr(swin_transformer_w_yolos, args.vit_backbone):
        logger.info(f'build swin backbone with yolos {args.vit_backbone}')
        from dev_models.backbones.swin_transformer_w_yolos import build_backbone
        return build_backbone(args)
    elif hasattr(swin_transformer_w_yolos_v2, args.vit_backbone):
        logger.info(f'build swin backbone with yolosv2 {args.vit_backbone}')
        from dev_models.backbones.swin_transformer_w_yolos_v2 import build_backbone
        return build_backbone(args)   
    elif hasattr(yolox_cspdarknet, args.vit_backbone):
        logger.info(f'build swin backbone with yolosv2 {args.vit_backbone}')
        from dev_models.backbones.yolox_cspdarknet import build_backbone
        return build_backbone(args)
    else:
        logger.error(f'{args.vit_backbone} not supported!')