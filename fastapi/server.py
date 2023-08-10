import requests
import torch
import io

from pydantic import BaseModel
import numpy as np
from torch.nn import functional as F
from typing import Annotated
from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as T
import tiktoken

import utils

log = utils.Logger.create_sess("prod")

log.debug("Initializing FastAPI server...")
app = FastAPI()

log.debug("Enabling CORS for all origins...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log.debug("Loading VIT model...")
model_vit = torch.jit.load("outputs/traced_models/cifar10-vit-tr.pt")
model_vit = model_vit.eval()
log.debug("VIT model loaded.")

log.debug("Loading GPT model...")
model_gpt = torch.jit.load("outputs/traced_models/hp-gpt-tr.pt")
model_gpt = model_gpt.eval()
log.debug("GPT model loaded.")

log.debug("Loading GPT tokenizer...")
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
log.debug("GPT tokenizer loaded.")

log.debug("Initializing transforms...")
transforms = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), T.Resize((32, 32))])

@app.get("/infer_vit")
async def infer_vit(image: Annotated[bytes, File()]):
    log.debug("Received image.")
    img: Image.Image = Image.open(io.BytesIO(image))

    img_t = transforms(img).unsqueeze(0)
    log.debug("Image transformed.")

    with torch.no_grad():
        logits = model_vit(img_t)
        log.info("Inference done for VIT.")
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()

    return {str(i): preds[i] for i in range(10)}

class Text(BaseModel):
    sentense: str

@app.get("/infer_gpt")
async def infer_gpt(text: Text):

    log.debug("Received text.")

    encoded_text = tokenizer.encode(text.sentense)

    with torch.no_grad():
        out = model_gpt.model.generate(torch.tensor(encoded_text).unsqueeze(0), max_new_tokens=256)
        log.info("Inference done for GPT.")
    return {
        "sentense": text.sentense,
        "completion": tokenizer.decode(out[0].cpu().numpy().tolist())
        }

@app.get("/health")
async def health():
    log.debug("Health check passed.")
    return {"message": "ok"}