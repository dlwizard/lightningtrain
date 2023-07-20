from typing import Tuple, Dict

import lightning as L
import hydra
from omegaconf import DictConfig
from lightning import LightningModule
from torchvision import transforms as T
from torch.nn.functional import softmax
import torch
import os
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path

from lightningtrain import utils

from rich.console import Console
from rich.table import Table

log = utils.get_pylogger(__name__)

def get_last_training_checkpoint_dir(path: str) -> str:
    '''Returns the path of the last training checkpoint directory. 
    If no checkpoint is found checks for the last training directory.'''

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

        # checks if path exists
        if os.path.exists(checkpoint_path):
            latest_checkpoint_path = checkpoint_path
            break
        else:
            paths.pop(paths.index(max(paths)))
    
    return latest_checkpoint_path

def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    path = Path(cfg.paths.output_dir)

    latest_path = get_last_training_checkpoint_dir(path.parent.parent.absolute()) if len(os.listdir(path.parent.parent.absolute())) > 0 else None

    checkpoint_file_path = None
    if latest_path is not None:
        checkpoint_file_path = os.path.join(latest_path, os.listdir(latest_path)[0])

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_file_path)

    image_path = cfg.get("image_path")

    if "http" in image_path:
        response = requests.get(image_path).content
        img = Image.open(BytesIO(response)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
                T.Resize((cfg.model.get("img_size"), cfg.model.get("img_size"))),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = transform(img).unsqueeze(0)
    out = model.forward(img)
    out = softmax(out, dim=1)

    table = Table(title=cfg.get("task_name") + " image: " + image_path.split("/")[-1])

    table.add_column("Class", justify="left", style="cyan", no_wrap=True)
    table.add_column("Prob", justify="left", style="magenta")

    for i in range(len(cfg.get("classes"))):
        table.add_row(cfg.get("classes")[i], str(out[0][i].item()))
    
    print("📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌\n")
    console = Console()
    console.print(table)
    print("📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌\n")

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):

    infer(cfg)



if __name__ == "__main__":
    main()
