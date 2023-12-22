embed_dim=128
depths=[2, 2, 6, 2]

batch_size=128
num_device=16
n_epoch=100


head_cfg = {
    "type": "multi_output", 
    "in_features": int(embed_dim * 2 ** (len(depths) - 1)), 
    "out_features_list": [6, 3, 3],
    "task_names": ["expansion", "ICM", "TE"], 
    "dropout": 0.0
}

backbone_cfg = {
    "type": "fd_clip_B16",
    "pretrained": "./weights/esvit_300ep.pth"
}


model_cfg = {
    "backbone": backbone_cfg, 
    "head": head_cfg,
    "checkpoint_path": None 
}

iter_cfg={
    "type": "normal", 
    "accumulate_steps": 1,
    "clip_grad": 5.
}


loss_cfg = {
    "type": "multi_task", 
    "task_to_loss_cfg": {
        "expansion": dict(type="cross_entropy"),
        "ICM": dict(type="cross_entropy"),
        "TE": dict(type="cross_entropy"),
    }, 
    "task_to_weight": [1., 1., 1.]
}

lr_assigner_cfg={
    "base_lr": 1.25e-3,  
    "type": "fd_vit_ld",
    "weight_decay": 0.05,
    "layer_decay": 0.8,
    "skip_list": {'absolute_pos_embed'},
    "skip_keywords": {"rpe_mlp", "logit_scale", 'relative_position_bias_table'} # {"logit_scale"}
}


optimizer_cfg = {
    "type": "AdamW", "weight_decay": 0.05
}

scheduler_cfg={
    "type": "cosine_lr", "min_lr": 2.5e-7, 
    "warmup_lr_init": 2.5e-7, "warmup_epochs": 20,
} 

metric_cfg_list=[
    dict(type="Accuracy", tag="expansion_acc", gt_col="label_expansion", pred_col="pred_expansion"),
    # dict(type="AUC", tag="expansion_auc", multi_class="ovo", labels=[0, 1, 2, 3, 4, 5], gt_col="label_expansion", prob_col="prob_expansion"),
    dict(type="Accuracy", tag="icm_acc", gt_col="label_ICM", pred_col="pred_ICM"),
    # dict(type="AUC", tag="icm_auc", multi_class="ovo", labels=[0, 1, 2], gt_col="label_icm", prob_col="prob_icm"),
    dict(type="Accuracy", tag="te_acc", gt_col="label_TE", pred_col="pred_TE"),
    # dict(type="AUC", tag="te_auc", multi_class="ovo", labels=[0, 1, 2], gt_col="label_te", prob_col="prob_te"),
]


evaluator_cfg={
    "type": "multi_task", "metric_cfg_list": metric_cfg_list
}