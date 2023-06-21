from typing import Tuple, Dict

import lightning as L
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.trainer import Trainer
import os
from pathlib import Path

from lightningtrain import utils

log = utils.get_pylogger(__name__)

def get_last_training_checkpoint_dir(path: str):
    # get all the paths wrt to date and choose latest date path
    path = os.path.join(path, max(os.listdir(path)))

    # get all child paths of the latest date path
    paths = os.listdir(path)

    # pop out the current runtime path
    paths.pop(paths.index(max(paths)))
    
    latest_checkpoint_path = None
    
    while len(paths)>0:
        p = os.path.join(path, max(paths))
        checkpoint_path = os.path.join(p, "lightning_logs", "version_0", "checkpoints")
        if os.path.exists(checkpoint_path):
            latest_checkpoint_path = checkpoint_path
            break
        else:
            paths.pop(paths.index(max(paths)))
    
    return latest_checkpoint_path


@utils.task_wrapper
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    path = Path(cfg.paths.output_dir)

    latest_path = get_last_training_checkpoint_dir(path.parent.parent.absolute()) if len(os.listdir(path.parent.parent.absolute())) > 0 else None

    checkpoint_file_path = None
    if latest_path is not None:
        checkpoint_file_path = os.path.join(latest_path, os.listdir(latest_path)[0])

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }
    
    if cfg.get("test"):
        log.info("Starting testing!")
        if checkpoint_file_path is None:
            log.warning(
                "Best ckpt not found! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_file_path)
        log.info(f"Best ckpt path: {checkpoint_file_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = eval(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
