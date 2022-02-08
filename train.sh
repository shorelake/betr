# best 800 133
# sn_baseline_enc0_cos_twostage_effrpn_om_defdec2_bf_encmatchv0_sppriorv0_denseauxv1test train
# encmatch cls 2 l1 5 iou 2
# loss cls 2 l1 5 iou 2
pai -name pytorch151
 -project algo_public_dev
 -Dpython=3.6
 -Dscript="file://D:/other_code/for_lbc/Deformable-DETR/deformdetr.zip"
 -DentryFile="main.py"
 -DuserDefinedParameters="--output_dir jiyang/for_lbc/workdir/sn_baseline_enc0_cos_twostage_effrpn_om_defdec2_bf_encmatchv0_sppriorv0_denseauxv1test --vit_backbone swin_nano --pretrained_path oss://jiyang/for_lbc/pretrained_model/swin_nano_patch4_window7_224.pth --batch_size 2 --enc_layers 0  --dec_layers 2 --lr_backbone 1e-4 --lr 1e-4 --lr_linear_proj_mult 1 --lr_scheduler cosinelr --two_stage --eff_query_init --eff_specific_head --proposal_net rpn_default --neck_decoder def_decoder --with_box_refine --dense_aux_loss dam --dense_aux_loss_coef 2"
 -Darn="acs:ram::1548748125132679:role/xili"
 -Dhost="cn-hangzhou.oss-internal.aliyun-inc.com"
 -Dcluster="{\"worker\":{\"cpu\":800, \"memory\":10000, \"gpu\":100}}"
 -Doversubscription=false
 -DworkerCount=8;