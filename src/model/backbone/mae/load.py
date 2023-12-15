import logging

import torch

from .pos_embed import interpolate_pos_embed


logger = logging.info(__name__)


def load_pretrained(model, pretrained): 
    checkpoint = torch.load(pretrained, map_location='cpu')
    logger.info(f"Load pre-trained checkpoint from: {pretrained}")
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(f"Load state_dict msg: {msg}")
    return model