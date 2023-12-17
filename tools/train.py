import os
import logging
import argparse
import sys
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd

from src.logger_helper import setup_logger
from src.trainer import Trainer
from src.model import build_model
from src.dataset import DATASET
from src.iter import ITERATION
from src.loss import LOSS
from src.optim import OPTIMIZER, SCHEDULER, LR_ASSIGNER
from src.eval import EVALUATOR
from src.transform import train_transform, test_transform
from src.utils import get_cfg_by_file, plot_loss_curve, save_train_records


logger = setup_logger(level=logging.INFO)

def main():
    df = pd.read_csv(args.csv_path)

    if args.debug: 
        df = df.loc[:500, :].copy()

    # dataset
    dataset_type="single_task" if isinstance(args.y_col, str) else "multi_task"

    train_dataset=DATASET.build(
        type=dataset_type, df=df.loc[df[args.split_col], :], filename_col="filename", 
        y_col=args.y_col, image_dir=args.image_dir, image_transform=train_transform
    )

    test_dataset=DATASET.build(
        type=dataset_type, df=df.loc[~df[args.split_col], :], filename_col="filename", 
        y_col=args.y_col, image_dir=args.image_dir, image_transform=test_transform
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
    evaluator = EVALUATOR.build(**config.evaluator_cfg)

    # optimizer
    if args.optim:
        config.optimizer_cfg["type"] = args.optim

    if args.lr:
        config.optimizer_cfg["lr"] = args.lr

    if args.weight_decay:
        config.optimizer_cfg["weight_decay"] = args.weight_decay

    # lr assigner
    lr_assigner=LR_ASSIGNER.build(**config.lr_assigner_cfg)

    optimizer = OPTIMIZER.build(
        params=lr_assigner.get_params(model), **config.optimizer_cfg
    )

    # load checkpoint
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight))
        logger.info(f"Load model weights from {args.weight} successfully.")

    # scheduler
    scheduler = SCHEDULER.build(
        optimizer=optimizer, n_iter_per_epoch=len(train_loader),  
        epochs=args.epochs, **config.epoch_scheduler_cfg
    )

    # iter hook
    iteration = ITERATION.build(**config.iter_hook_cfg)

    # build trainer
    trainer = Trainer(
        model=model, iteration=iteration, optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, evaluator=evaluator, device=args.device, n_epochs=args.n_epochs,
        checkpoint_dir=f"./checkpoints/{args.exp_name}"
    )
    # train model
    trainer.train(train_loader, test_loader)

    # create artifacts & save to "checkpoints/{args.exp_name}"
    pd.DataFrame.from_records(trainer.epoch_test_records).to_csv(f"./checkpoints/{args.exp_name}/history.csv")
    logger.info(f"save performance history at ./checkpoints/{args.exp_name}/history.csv")

    plot_loss_curve(trainer, f"./checkpoints/{args.exp_name}/loss.png")
    logger.info(f"save loss curve at ./checkpoints/{args.exp_name}/history.csv")

    save_train_records(trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--y_col", type=str, nargs='+', default=["expansion", "ICM", "TE"])
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