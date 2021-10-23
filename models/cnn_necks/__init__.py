from math import log
from .fpn import FeaturePyramidNetwork as fpn
from .PANet import fpn as panet
from loguru import logger


def build_cnn_encoder(cnn_encoder, num_backbone_outs, backbone_num_channels,hidden_dim):
    if cnn_encoder == 'fpn':
        logger.info("build cnn encoder fpn")
        return fpn(num_backbone_outs, backbone_num_channels,hidden_dim)
    elif cnn_encoder == 'panet':
        logger.info("build cnn encoder panet")
        return panet(num_backbone_outs, backbone_num_channels,hidden_dim)
    else:
        logger.error(f"cnn encoder {cnn_encoder} not support")
        raise ValueError(f"cnn encoder {cnn_encoder} not support")