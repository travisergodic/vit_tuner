import json
import logging

from src.registry import LR_ASSIGNER


logger=logging.getLogger(__name__)


def get_fd_vit_layer_id(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    else:
        return num_layers - 1
    

def get_mae_vit_layer_id(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers - 1
    

def get_ft_vit_layer_id(name, num_layers):
    if name in ("class_embedding", "cls_token", "mask_token", "pos_embed", "positional_embedding"):
        return 0
    elif name.startswith("patch_embed") or name.startswith("conv1"):
        return 0
    elif name.startswith("ln_pre"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split('.')[1])
        return layer_id + 1
    elif name.startswith("transformer.resblocks"):
        layer_id = int(name.split('.')[2])
        return layer_id + 1

    else:
        return num_layers - 1
    

def get_fd_swin_layer_id(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class LayerwiseDecayAssigner:
    get_layer_func=None
    def __init__(self, base_lr, weight_decay, layer_decay, skip_list=(), skip_keywords=()):
        """
        skip_list (_type_): 
            mae --> model.no_weight_decay()
            fd --> model.no_weight_decay(), model.no_weight_decay_keywords()
            ft-clip --> model.no_weight_decay(), disable weight decay on rel_pos_bias  
        """
        self.base_lr=base_lr
        self.weight_decay=weight_decay
        self.layer_decay=layer_decay
        self.skip_list=skip_list
        self.skip_keywords=skip_keywords

    def get_params(self, model):
        backbone, head=model.backbone, model.head
        parameter_group_names = {}
        parameter_group_vars = {}
        
        if isinstance(self.layer_decay, list):
            scales=self.layer_decay
        else: 
            depth = backbone.depth if hasattr(backbone, "depth") else backbone.layers
            scales=[self.layer_decay ** i for i in reversed(range(depth+2))]

        # backbone
        for name, param in backbone.named_parameters():
            if not param.requires_grad:
                continue

            # weight decay
            if param.ndim==1 or name.endswith(".bias") or (name in self.skip_list) or check_keywords_in_name(name, self.skip_keywords):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = self.weight_decay

            if self.__class__.get_layer_func is not None:
                layer_id = self.__class__.get_layer_func(name, num_layers=depth)
                group_name = f"layer_{layer_id}_{group_name}"
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if scales is not None:
                    scale = scales[layer_id]
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                    "lr": scale * self.base_lr
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                    "lr": scale * self.base_lr
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)


        # head
        parameter_group_names["head"] = {
            "weight_decay": this_weight_decay, "params": [],
            "lr_scale": 1,
            "lr": self.base_lr
        }

        parameter_group_vars["head"] = {
            "weight_decay": this_weight_decay,
            "params": [],
            "lr_scale": scale,
            "lr": self.base_lr
        }

        for name, param in head.named_parameters():
            parameter_group_vars["head"]["params"].append(param)
            parameter_group_names["head"]["params"].append(name)

        logger.info(f"Param groups = {json.dumps(parameter_group_names, indent=4)}")
        return list(parameter_group_vars.values())


@LR_ASSIGNER.register("fd_vit_ld")
class FDVITLayerwiseDecayAssigner(LayerwiseDecayAssigner):
    get_layer_func=get_fd_vit_layer_id


@LR_ASSIGNER.register("fd_swin_ld")
class FDSwinV2LayerwiseDecayAssigner(LayerwiseDecayAssigner):
    get_layer_func=get_fd_swin_layer_id


@LR_ASSIGNER.register("mae_vit_ld")
class MAELayerwiseDecayAssigner(LayerwiseDecayAssigner):
    get_layer_func=get_mae_vit_layer_id


@LR_ASSIGNER.register("ft_vit_ld")
class FTCLIPLayerwiseDecayAssigner(LayerwiseDecayAssigner):
    get_layer_func=get_ft_vit_layer_id