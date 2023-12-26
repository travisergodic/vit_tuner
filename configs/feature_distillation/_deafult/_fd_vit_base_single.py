batch_size=128
num_device=16
n_epoch=100


head_cfg = {
    "type": "single_output", 
    "in_features": 768, 
    "out_features": 555,
    "dropout": 0.0
}

backbone_cfg = {
    "type": "fd_clip_B16",
    "pretrained": "./weights/clip_300ep.pth"
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


loss_cfg ={"type": "cross_entropy"}

lr_assigner_cfg={
    "base_lr": 1.25e-3, 
    "type": "fd_vit_ld",
    "weight_decay": 0.05,
    "layer_decay": 0.65,
    "skip_list": {"pos_embed", "cls_token"}
}


optimizer_cfg = {
    "type": "AdamW", "weight_decay": 0.05
}

scheduler_cfg={
    "type": "cosine_lr", "min_lr": 2.5e-7, 
    "warmup_lr_init": 2.5e-7, "warmup_epochs": 20,
} 

metric_cfg_list=[
    dict(type="Accuracy", tag="acc", gt_col="label", pred_col="pred"),
    # dict(type="Recall", tag="recall", gt_col="label", preb_col="pred"),
    # dict(type="Precision", tag="precision", gt_col="label", preb_col="pred"),
]


train_evaluator_cfg={
    "type": "single_task", "metric_cfg_list": metric_cfg_list
}

test_evaluator_cfg={
    "type": "single_task", "metric_cfg_list": metric_cfg_list
}