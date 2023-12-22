import logging

from timm.utils import ModelEma

from src.registry import HOOKS


logger = logging.getLogger(__name__)


@HOOKS.register("model_ema")
class ModelEmaHook:
    def __init__(self, model_ema_decay, device):
        self.model_ema_decay = model_ema_decay
        self.device = device

    def before_run(self, trainer):
        model_ema = ModelEma(
            trainer.model, decay=self.model_ema_decay, device=self.device, resume=""
        )
        logger.info(f"Using EMA with decay = {self.model_ema_decay}")

    
            