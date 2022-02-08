CUDA_VISIBLE_DEVICES=1 python main.py --output_dir exps/test/ --vit_backbone swin_nano --enc_layers 0 \
--pretrained_path ./pretrained_model/swin_nano_patch4_window7_224.pth --two_stage --eff_query_init \
--eff_specific_head --proposal_net rpn_default --neck_decoder def_decoder --with_box_refine \
--dense_aux_loss dam --dense_aux_loss_coef 2 --eval --dec_layers 2 \
--resume pretrained_model/swin_nano_0enc_effrpn_om_2dec_encmatchv0_spatialpriorv0_denseauxv1.pth