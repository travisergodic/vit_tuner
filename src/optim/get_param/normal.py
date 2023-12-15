from src.registry import GET_PARAMS


@GET_PARAMS.register("vanilla")
def param_groups_normal(model):
    return model.parameters()


@GET_PARAMS.register("normal")
def vanilla_param_groups(model, backbone_lr, head_lr):
    return [
        {"params": getattr(model, "backbone").parameters(), "lr": backbone_lr}, 
        {"params": getattr(model, "head").parameters(), "lr": head_lr}
    ] 