from typing import List, Tuple

import torch
from torchvision import transforms as T
import hydra
import gradio as gr
from omegaconf import DictConfig

from lightningtrain import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info("Loading Model...")
    model = torch.jit.load(cfg.ckpt_path)
    model.eval()
    
    log.info(f"Model Loaded...")
    
    transform = T.ToTensor()

    def recognize_digit(image):
        if image is None:
            return None
        image = transform(image)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        return {cfg.get("classes")[i]: preds[i] for i in range(10)}

    im = gr.Image(source="upload", type="pil", label="Input Image")

    demo = gr.Interface(
        fn=recognize_digit,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        title="CIFAR10 Classifier",
        description="Upload a CIFAR10 image to classify it. This model was trained with VIT architecture.",
        live=True,
    )

    demo.launch(server_port=cfg.get("port", 8080))

@hydra.main(
    version_base="1.2", config_path="../configs", config_name="demo_vit.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()