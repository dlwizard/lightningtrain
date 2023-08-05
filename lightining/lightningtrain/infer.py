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

from lightningtrain import utils

from rich.console import Console
from rich.table import Table

log = utils.get_pylogger(__name__)

def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    assert cfg.ckpt_path

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    image_path = cfg.get("image_path")
    if "http" in image_path:
        response = requests.get(image_path).content
        img = Image.open(BytesIO(response)).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    
    transform = T.ToTensor()
    img = transform(img)
    
    # img = transform(img).unsqueeze(0)
    preds = model.forward_jit(img)
    # preds = preds[0].tolist()
    # preds =  {str(i): preds[i] for i in range(10)}

    print(preds)

    table = Table(title=cfg.get("task_name") + " image: " + image_path.split("/")[-1])

    table.add_column("Class", justify="left", style="cyan", no_wrap=True)
    table.add_column("Prob", justify="left", style="magenta")

    for i in range(len(cfg.get("classes"))):
        table.add_row(cfg.get("classes")[i], str(preds[0][i].item()))
    
    print("📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌\n")
    console = Console()
    console.print(table)
    print("📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌📌\n")

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):

    infer(cfg)



if __name__ == "__main__":
    main()
