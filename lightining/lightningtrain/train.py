from typing import Tuple, Dict, List
import optuna
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
from lightningtrain.models.gpt_module import GPTLitModule
from lightningtrain.models.components.gpt import GPT
from lightningtrain.hparam.hparam_search import PyTorchLightningPruningCallback, set_best_trial
from lightningtrain.data.hp_datamodule import HarryPotterDataModule

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

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
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, limit_train_batches=cfg.get("limit_train_batches"), limit_val_batches=cfg.get("limit_val_batches"), limit_test_batches=cfg.get("limit_test_batches"))

    log.info(f"Batch size: {datamodule.hparams.batch_size}")
    log.info(f"Block size: {datamodule.hparams.block_size}")
    log.info(f"Learning rate: {model.hparams.learning_rate}")
    log.info(f"Embed Size: {model.model.n_embed}")
    log.info(f"Block size: {model.hparams.block_size}")
    log.info(f"Block size: {model.model.block_size}")
    log.info(f"Dropout: {model.model.drop_p}")
    log.info(f"Number of Heads: {model.model.n_heads}")
    log.info(f"Number of decoder blocks: {model.model.n_decoder_blocks}")

    if cfg.get("tune_hparam"):
        def objective(trial: optuna.trial.Trial) -> float:
            n_embed = trial.suggest_int("n_embed", 32, 128, 32)
            block_power = trial.suggest_int("block_power", 2, 6, 1)
            block_size = 2 ** block_power
            n_heads_power = trial.suggest_int("n_heads_power", 2, 5, 1)
            n_heads = 2 ** n_heads_power
            drop_p = trial.suggest_float("drop_p", 0.0, 0.5)
            n_decoder_blocks = trial.suggest_int("n_decoder_blocks", 1, 5, 1)

            model_optuna = GPTLitModule(
                block_size=block_size,
                learning_rate = 0.001,
                model = GPT(
                    n_embed=n_embed,
                    block_size=block_size,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    n_decoder_blocks=n_decoder_blocks,
                )
            )
            datamodule_optuna = HarryPotterDataModule(
                data_dir = cfg.data.data_dir,
                block_size = block_size,
                batch_size=128,
                num_workers=datamodule.hparams.num_workers,
                pin_memory=datamodule.hparams.pin_memory
            )
            callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/loss"))
            trainer_optuna = L.Trainer(
                logger=logger,
                limit_train_batches=cfg.get("limit_train_batches"),
                limit_val_batches=cfg.get("limit_val_batches"),
                limit_test_batches=cfg.get("limit_test_batches"),
                max_epochs=1,
                accelerator="gpu",
                devices=1,
                callbacks=callbacks,

            )

            hyperparameters = dict(n_layers=n_embed, block_size=block_size, n_heads=n_heads, drop_p=drop_p, n_decoder_blocks=n_decoder_blocks)
            trainer.logger.log_hyperparams(hyperparameters)
            trainer_optuna.fit(model_optuna, datamodule=datamodule_optuna)

            return trainer_optuna.callback_metrics["val/loss"].item()

        pruner = optuna.pruners.MedianPruner()

        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=10, timeout=600)

        log.info(f"Best Trial: {study.best_trial.params}")

        cfg = set_best_trial(cfg, study.best_trial.params)

        log.info(f"Config Data: {cfg.data}")

        log.info(f"Config Model: {cfg.model}")

        log.info(f"Instantiating datamodule after optimization <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        log.info(f"Instantiating model after optimization <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating hyperparameter optimizer <{cfg.hparam.lr_batch_finder._target_}>")
        tuner = hydra.utils.instantiate(cfg.hparam.lr_batch_finder, trainer)

        log.info("Starting hyperparameter optimization for learning rate...")
        tuner.lr_find(model, datamodule)

        log.info("Starting hyperparameter optimization for batch size...")
        tuner.scale_batch_size(model, datamodule, mode="power")

        log.info(f"Batch size after optimization: {datamodule.hparams.batch_size}")
        log.info(f"Block size after optimization: {datamodule.hparams.block_size}")
        log.info(f"Learning rate after optimization: {model.hparams.learning_rate}")
        log.info(f"Embed Size after optimization: {model.model.n_embed}")
        log.info(f"Block size after optimization: {model.hparams.block_size}")
        log.info(f"Block size after optimization: {model.model.block_size}")
        log.info(f"Dropout after optimization: {model.model.drop_p}")
        log.info(f"Number of Heads after optimization: {model.model.n_heads}")
        log.info(f"Number of decoder blocks after optimization: {model.model.n_decoder_blocks}")

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

    if cfg.get("compile"):
        model = torch.compile(model)

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
        # for logger_ in logger:
        #     if isinstance(logger_, L.pytorch.loggers.mlflow.MLFlowLogger):
        #         ckpt = torch.load(ckpt_path)
        #         model.load_state_dict(ckpt["state_dict"])
        #         os.environ['MLFLOW_RUN_ID'] = logger_.run_id
        #         os.environ['MLFLOW_EXPERIMENT_ID'] = logger_.experiment_id
        #         os.environ['MLFLOW_EXPERIMENT_NAME'] = logger_._experiment_name
        #         os.environ['MLFLOW_TRACKING_URI'] = logger_._tracking_uri
        #         mlflow.pyfunc.log_model(model, "model")
        #         break

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict
    # return None, None


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
