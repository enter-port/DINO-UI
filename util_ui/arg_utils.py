import argparse

def create_dino_args():
    args = argparse.Namespace()
    
    # 基础配置
    args.num_classes = 8
    args.param_dict_type = 'default'
    args.lr_backbone = 1e-05
    args.lr_backbone_names = ['backbone.0']
    args.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    args.lr_linear_proj_mult = 0.1
    args.ddetr_lr_param = False
    
    # 训练相关参数
    args.batch_size = 1
    args.weight_decay = 0.0001
    args.lr_drop = 11
    args.save_checkpoint_interval = 1
    args.clip_max_norm = 0.1
    args.onecyclelr = False
    args.multi_step_lr = False
    args.lr_drop_list = [33, 45]
    
    # 模型基础参数
    args.modelname = 'dino'
    # args.frozen_weights = ['backcone', 'transformer.encoder', 'transformer.decoder']
    args.frozen_weights = None
    args.backbone = 'resnet50'
    args.use_checkpoint = False
    args.dilation = False
    
    # 位置编码相关
    args.position_embedding = 'sine'
    args.pe_temperatureH = 20
    args.pe_temperatureW = 20
    args.return_interm_indices = [1, 2, 3]
    args.backbone_freeze_keywords = None
    
    # Transformer架构参数
    args.enc_layers = 6
    args.dec_layers = 6
    args.unic_layers = 0
    args.pre_norm = False
    args.dim_feedforward = 2048
    args.hidden_dim = 256
    args.dropout = 0.0
    args.nheads = 8
    args.num_queries = 900
    args.query_dim = 4
    args.num_patterns = 0
    
    # DETR特定参数
    args.pdetr3_bbox_embed_diff_each_layer = False
    args.pdetr3_refHW = -1
    args.random_refpoints_xy = False
    args.fix_refpoints_hw = -1
    args.dabdetr_yolo_like_anchor_update = False
    args.dabdetr_deformable_encoder = False
    args.dabdetr_deformable_decoder = False
    
    # 注意力机制相关
    args.use_deformable_box_attn = False
    args.box_attn_type = 'roi_align'
    args.dec_layer_number = None
    args.num_feature_levels = 4
    args.enc_n_points = 4
    args.dec_n_points = 4
    
    # 解码器噪声参数
    args.decoder_layer_noise = False
    args.dln_xy_noise = 0.2
    args.dln_hw_noise = 0.2
    args.add_channel_attention = False
    args.add_pos_value = False
    
    # 两阶段检测相关
    args.two_stage_type = 'standard'
    args.two_stage_pat_embed = 0
    args.two_stage_add_query_num = 0
    args.two_stage_bbox_embed_share = False
    args.two_stage_class_embed_share = False
    args.two_stage_learn_wh = False
    args.two_stage_default_hw = 0.05
    args.two_stage_keep_all_tokens = False
    args.num_select = 50
    
    # 其他模型参数
    args.transformer_activation = 'relu'
    args.batch_norm_type = 'FrozenBatchNorm2d'
    args.masks = False
    args.aux_loss = True
    
    # Loss相关参数
    args.set_cost_class = 2.0
    args.set_cost_bbox = 5.0
    args.set_cost_giou = 2.0
    args.cls_loss_coef = 1.0
    args.mask_loss_coef = 1.0
    args.dice_loss_coef = 1.0
    args.bbox_loss_coef = 5.0
    args.giou_loss_coef = 2.0
    args.enc_loss_coef = 1.0
    args.interm_loss_coef = 1.0
    args.no_interm_box_loss = False
    args.focal_alpha = 0.25
    
    # 解码器和匹配器参数
    args.decoder_sa_type = 'sa'
    args.matcher_type = 'HungarianMatcher'
    args.decoder_module_seq = ['sa', 'ca', 'ffn']
    args.nms_iou_threshold = 0.7
    args.dec_pred_bbox_embed_share = True
    args.dec_pred_class_embed_share = True
    
    # DN相关参数
    args.use_dn = True
    args.dn_number = 100
    args.dn_box_noise_scale = 0.4
    args.dn_label_noise_ratio = 0.5
    args.embed_init_tgt = True
    args.dn_labelbook_size = 9
    args.match_unstable_error = True
    
    # EMA相关参数
    args.use_ema = False
    args.ema_decay = 0.9997
    args.ema_epoch = 0
    
    # 其他
    args.use_detached_boxes_dec_out = False
    
    # main.py 中的参数
    args.config_file = 'path/to/config_file'
    args.options = None
    args.dataset_file = 'coco'
    args.coco_path = '/comp_robot/cv_public_dataset/COCO2017/'
    args.coco_panoptic_path = None
    args.remove_difficult = False
    args.fix_size = False
    args.output_dir = ''
    args.note = ''
    args.device = 'cuda'
    args.seed = 42
    args.resume = ''
    # args.pretrain_model_path = 'pretrained/checkpoint0011_4scale.pth'
    # args.pretrain_model_path = 'logs/2024-12-29-19-40-37/weights/checkpoint_epoch_95.pth'
    args.pretrain_model_path = None
    # args.finetune_ignore = ["label_enc.weight", "class_embed"]
    args.finetune_ignore = None
    args.start_epoch = 0
    args.eval = False
    args.num_workers = 10
    args.test = False
    args.debug = False
    args.find_unused_params = False
    args.save_results = False
    args.save_log = False
    args.world_size = 1
    args.dist_url = 'env://'
    args.rank = 0
    args.local_rank = None
    args.amp = False
    args.clip_max_norm = 0.1
    args.weight_decay = 0.0001
    args.epochs = 200
    args.lr_drop = 11
    args.save_checkpoint_interval = 1
    args.clip_max_norm = 0.1
    args.onecyclelr = False
    args.multi_step_lr = False
    args.lr_drop_list = [33, 45]
    args.distributed = False
    args.input_shape = (1280, 1920)
    args.wo_class_error = False
    args.pre_train = False 
    args.lr_warmup = True
    args.lr_warmup_target = 0.01
    args.lr = 0.0002
    args.vis = False     
    args.finetune = False
    return args