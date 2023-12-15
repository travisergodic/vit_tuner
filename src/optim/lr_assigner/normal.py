from src.registry import LR_ASSIGNER


@LR_ASSIGNER.register("default")
class DefaultAssigner:
    def get_params(self, model):
        return model.parameters()


@LR_ASSIGNER.register("custom")
class CustomAssigner:
    def __init__(self, lr_scale_dict):
        self.lr_scale_dict=lr_scale_dict
        
    def get_params(self, model):
        res=[]
        for name, param in model.named_parameters():
            if not (param.requires_grad) or (name not in self.lr_scale_dict):
                continue
            
            res.append(
                {
                    "params": param, 
                    "lr_scale": self.lr_scale_dict[name]
                }
            )
        return res