head_cfg = {
    "type": "multi_output", 
    "out_features_list": [6, 3, 3],
    "task_names": ["expansion", "ICM", "TE"], 
    "dropout": 0.0
}

backbone_cfg = {
    "type": "fd",
    "name": "vit_base_clip", 
    "pretrain_weight": "./weights/fd/clip_300ep.pth"
}


model_cfg = {
    "backbone": backbone_cfg, 
    "head": head_cfg,
    "checkpoint_path": None 
}

iter_cfg={
    "type": "notmal", 
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
    "weight_decay": 0.05,
    "layer_decay": 0.6,
    "skip_list": {"pos_embed", "cls_token"}
}


optimizer_cfg = {
    "type": "AdamW", "lr": 1e-4, "weight_decay": 0.05
}

scheduler_cfg={
    "type": "cosine_lr", "min_lr": 2.5e-7, 
    "warmup_lr_init": 2.5e-7, "warmup_epochs": 10,
} 

metric_cfg_list=[
    dict(type="Accuracy", tag="expansion_acc", gt_col="gt_expansion", pred_col="pred_expansion"),
    dict(type="AUC", tag="expansion_auc", multi_class="ovo", labels=[0, 1, 2, 3, 4, 5], gt_col="gt_expansion", prob_col="prob_vector_expansion"),
    dict(type="Accuracy", tag="icm_acc", gt_col="gt_icm", pred_col="pred_icm"),
    dict(type="AUC", tag="icm_auc", multi_class="ovo", labels=[0, 1, 2], gt_col="gt_icm", prob_col="prob_vector_icm"),
    dict(type="Accuracy", tag="te_acc", gt_col="gt_te", pred_col="pred_te"),
    dict(type="AUC", tag="te_auc", multi_class="ovo", labels=[0, 1, 2], gt_col="gt_te", prob_col="prob_vector_te"),
]