from src.registry import LR_ASSIGNER


@LR_ASSIGNER.register("default")
class DefaultAssigner:
    def __init__(self, base_lr, weight_decay):
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        
    def get_params(self, model):
        return [{"params": model.parameters(), "lr": self.base_lr, "weight_decay": self.weight_decay}] 


@LR_ASSIGNER.register("custom")
class CustomAssigner:
    def __init__(self, base_lr, lr_scale_dict):
        self.base_lr = base_lr
        self.lr_scale_dict=lr_scale_dict
        
    def get_params(self, model):
        res=[]
        for name, param in model.named_parameters():
            if not (param.requires_grad) or (name not in self.lr_scale_dict):
                continue
            
            res.append(
                {"params": param, "lr_scale": self.lr_scale_dict[name], "lr": self.lr_scale_dict[name] * self.base_lr}
            )
        return res