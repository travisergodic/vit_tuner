head_cfg = {
    "type": "single_output", 
    "in_features": 2048, 
    "out_features_list": 555,
    "dropout": 0.0
}

backbone_cfg = {
    "type": "resnet50",
    "pretrained": True
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
    "type": "default",
    "base_lr": 1e-4,
    "weight_decay": 0.05
}


optimizer_cfg = {
    "type": "AdamW"
}

scheduler_cfg={"type": None} 

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