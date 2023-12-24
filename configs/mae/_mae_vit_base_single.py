batch_size=1024


head_cfg = {
    "type": "single_output", 
    "in_features": 768, 
    "out_features": 555,
    "dropout": 0.0
}

backbone_cfg = {
    "type": "mae_B16",
    "pretrained": "./weights/mae_pretrain_vit_base.pth"
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
    "base_lr": 5e-4,  
    "type": "fd_vit_ld",
    "weight_decay": 0.05 ,
    "layer_decay": 0.65,
    "skip_list": {"pos_embed", "cls_token", "dist_token"}, 
    # "skip_keywords": 
}


optimizer_cfg = {
    "type": "AdamW", "weight_decay": 0.05
}

scheduler_cfg={
    "type": "cosine_lr", "min_lr": 2.5e-7, 
    "warmup_lr_init": 2.5e-7, "warmup_epochs": 100,
} 

metric_cfg_list=[
    dict(type="Accuracy", tag="acc", gt_col="label", pred_col="pred"),
    # dict(type="Recall", tag="recall", gt_col="label", preb_col="pred"),
    # dict(type="Precision", tag="precision", gt_col="label", preb_col="pred"),
]


evaluator_cfg={
    "type": "single_task", "metric_cfg_list": metric_cfg_list
}