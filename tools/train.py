import os
import logging
import argparse
import sys
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd

from src.dataset import CustomDataset 
from src.logger_helper import setup_logger
from src.trainer import Trainer
from src.model import build_model
from src.iter import ITERATION
from src.loss import LOSS
from src.optim import OPTIMIZER, SCHEDULER, LR_ASSIGNER
from src.eval import EVALUATOR
from src.transform import train_transform, test_transform
from src.utils import (get_cfg_by_file, save_performance_history_dataframe_from_trainer, save_performance_dataframe_from_trainer)


logger = setup_logger(level=logging.INFO)

def main():
    # read csv 
    df = pd.read_csv(args.csv_path)

    if args.debug: 
        df = df.loc[:500, :].copy()

    # dataset
    train_dataset = CustomDataset(
        df.loc[:, df[args.split_col]], "filename", args.y_cols, args.image_dir, 
        image_transform=train_transform
    )

    test_dataset = CustomDataset(
        df.loc[:, df[args.split_col]], "filename", args.y_cols, args.image_dir,
        image_transform=test_transform
    )

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs * 2, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    # model
    model = build_model(config.model_cfg)

    # loss
    loss_fn = LOSS.build(**config.loss_cfg)

    # evaluator
    evaluator = EVALUATOR.build(model=model, device=args.device, loss_fn=loss_fn, **config.evaluator_cfg)
    optimizer = OPTIMIZER.build(model=model, optimizer_cfg=config.optimizer_cfg)

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    # scheduler
    scheduler = SCHEDULER.build(optimizer=optimizer, **config.epoch_scheduler_cfg)

    # iter hook
    iter_hook = ITERATION.build(**config.iter_hook_cfg)
    logger.info(f"Use {type(iter_hook).__name__} object for each training iteration.")

    # build trainer
    trainer = Trainer(
        model=model,
        evaluator=evaluator, 
        optimizer=optimizer,
        iter_hook=iter_hook,
        loss_fn=loss_fn,
        device=args.device,
        n_epochs=args.n_epochs,
        iter_scheduler=iter_scheduler,
        epoch_scheduler=epoch_scheduler, 
        save_freq=args.save_freq,
        checkpoint_dir=f"./checkpoints/{args.exp_name}",
        monitor=args.monitor
    )

    # train model
    trainer.fit(train_loader, test_loader)

    # create artifacts & save to "checkpoints/{args.exp_name}"
    eval_df = evaluator.get_eval_dataframe(test_loader)
    eval_df.to_csv(f"./checkpoints/{args.exp_name}/pred_result.csv")
    logger.info(f"Save prediction result at ./checkpoints/{args.exp_name}/pred_result.csv")

    save_performance_history_dataframe_from_trainer(trainer, f"./checkpoints/{args.exp_name}/performance_history.csv")
    logger.info(f"Save performance history at ./checkpoints/{args.exp_name}/performance_history.csv")
    
    save_performance_dataframe_from_trainer(trainer, f"./checkpoints/{args.exp_name}/performance_report.csv")
    logger.info(f"Save performance report at ./checkpoints/{args.exp_name}/performance_report.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--monitor", type=str, default="loss")
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--y_cols", type=str, nargs='+', default=["expansion", "ICM", "TE"])
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optim", type=str, choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = get_cfg_by_file(args.config_file)
    main() 