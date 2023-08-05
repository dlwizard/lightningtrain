from typing import List, Tuple

import torch
from torchvision import transforms as T
import hydra
import gradio as gr
from omegaconf import DictConfig
import tiktoken

import utils

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
    
    log.info("Loading tokenizer...")

    cl100k_base = tiktoken.get_encoding("cl100k_base")

    tokenizer = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
    })

    log.info("Tokenizer Loaded...")

    def sentense_completion(text: str) -> str:
        encoded_text = tokenizer.encode(text)
        out = model.model.generate(torch.tensor(encoded_text).unsqueeze(0), max_new_tokens=256)
        log.info(f"Successfully Predicted...")
        return tokenizer.decode(out[0].cpu().numpy().tolist())

    demo = gr.Interface(
        fn=sentense_completion,
        inputs=gr.Textbox(lines=10, placeholder="Enter your text here..."),
        outputs="text"
    )

    demo.launch(server_port=cfg.get("port", 8080))

@hydra.main(
    version_base="1.2", config_path="configs", config_name="demo_gpt.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()