from typing import Tuple, Dict, List

import torch
import lightning as L
from lightning.pytorch.loggers import Logger
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.trainer import Trainer
import os
from pathlib import Path
import mlflow

from lightningtrain import utils

log = utils.get_pylogger(__name__)

# def get_last_training_checkpoint_dir(path: str) -> str:
#     '''Returns the path of the last training checkpoint directory. 
#     If no checkpoint is found checks for the last training directory.'''

#     # get all the paths wrt to date and choose latest date path
#     path = os.path.join(path, max(os.listdir(path)))

#     # get all child paths of the latest date path
#     paths = os.listdir(path)

#     # pop out the current runtime path
#     paths.pop(paths.index(max(paths)))
    
#     latest_checkpoint_path = None
    
#     while len(paths)>0:
#         p = os.path.join(path, max(paths))
#         checkpoint_path = os.path.join(p, "lightning_logs", "version_0", "checkpoints")

#         # checks if path exists
#         if os.path.exists(checkpoint_path):
#             latest_checkpoint_path = checkpoint_path
#             break
#         else:
#             paths.pop(paths.index(max(paths)))
    
#     return latest_checkpoint_path


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    # path = Path(cfg.paths.output_dir)

    # # get the latest checkpoint path
    # latest_path = get_last_training_checkpoint_dir(path.parent.parent.absolute()) if len(os.listdir(path.parent.parent.absolute())) > 0 else None

    # checkpoint_file_path = None
    # if latest_path is not None:
    #     #  get the latest checkpoint file path
    #     checkpoint_file_path = os.path.join(latest_path, os.listdir(latest_path)[0])

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # if cfg.get("compile"):
    #     model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path # UPDATE ! now we can get the best model from the callbacks
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        for logger_ in logger:
            if isinstance(logger_, L.pytorch.loggers.mlflow.MLFlowLogger):
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt["state_dict"])
                os.environ['MLFLOW_RUN_ID'] = logger_.run_id
                os.environ['MLFLOW_EXPERIMENT_ID'] = logger_.experiment_id
                os.environ['MLFLOW_EXPERIMENT_NAME'] = logger_._experiment_name
                os.environ['MLFLOW_TRACKING_URI'] = logger_._tracking_uri
                mlflow.pytorch.log_model(model, "model")
                break

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = train(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
