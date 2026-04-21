"""
inference.py
-------------
Run move-object inference using fine-tuned LoRA + DragonDiffusion pipeline.

Usage:
  python inference.py \
    --image ./data/images/00000.png \
    --src_box 100 120 200 220 \
    --tgt_coord 350 300 \
    --lora_ckpt ./checkpoints/lora_final.pt \
    --output ./result.png

How it works:
  1. Load SD 1.5 + inject LoRA + load MCM
  2. Encode source image
  3. Build move conditioning (text + MCM spatial token)
  4. Run DDIM denoising (50 steps)
  5. Decode latents → output image
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from lora_model import (
    inject_lora_into_unet,
    MoveConditioningModule,
    load_lora_weights,
)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE  = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      type=str, required=True,  help="Source image path")
    p.add_argument("--src_box",    type=int, nargs=4,        help="x1 y1 x2 y2 of object to move")
    p.add_argument("--tgt_coord",  type=int, nargs=2,        help="cx cy target center")
    p.add_argument("--lora_ckpt",  type=str, required=True,  help="LoRA checkpoint .pt file")
    p.add_argument("--output",     type=str, default="result.png")
    p.add_argument("--shape",      type=str, default="object", help="Shape name for caption")
    p.add_argument("--color",      type=str, default="red",    help="Color name for caption")
    p.add_argument("--num_steps",  type=int, default=50)
    p.add_argument("--guidance",   type=float, default=7.5)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print("[Load] SD 1.5 components...")
    tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device, dtype)
    vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device, dtype)
    unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device, dtype)
    scheduler    = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.eval(); text_encoder.eval(); unet.eval()

    # ── Inject and load LoRA ──────────────────────────────────────────────
    print("[Load] Injecting + loading LoRA weights...")
    inject_lora_into_unet(unet, rank=8, alpha=16.0)
    cross_attn_dim = unet.config.cross_attention_dim
    mcm = MoveConditioningModule(cross_attention_dim=cross_attn_dim).to(device, dtype)
    load_lora_weights(unet, mcm, args.lora_ckpt)
    unet.eval(); mcm.eval()

    # ── Prepare source image ──────────────────────────────────────────────
    src_pil = Image.open(args.image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    src_np  = np.array(src_pil).astype(np.float32) / 127.5 - 1.0
    src_t   = torch.from_numpy(src_np).permute(2,0,1).unsqueeze(0).to(device, dtype)

    src_latents = vae.encode(src_t).latent_dist.sample() * vae.config.scaling_factor

    # ── Build move conditioning ───────────────────────────────────────────
    x1, y1, x2, y2 = args.src_box
    tcx, tcy       = args.tgt_coord
    scx = (x1 + x2) // 2
    scy = (y1 + y2) // 2

    caption = (
        f"move the {args.color} {args.shape} "
        f"from position ({scx},{scy}) "
        f"to position ({tcx},{tcy})"
    )
    print(f"[Inference] Caption: {caption}")

    # Tokenize
    tokens = tokenizer(
        caption, padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    input_ids = tokens.input_ids.to(device)
    text_emb  = text_encoder(input_ids)[0]  # (1, seq, 768)

    # MCM spatial token
    src_box_t = torch.tensor([[x1/IMG_SIZE, y1/IMG_SIZE, x2/IMG_SIZE, y2/IMG_SIZE]], device=device, dtype=dtype)
    tgt_coord_t = torch.tensor([[tcx/IMG_SIZE, tcy/IMG_SIZE]], device=device, dtype=dtype)
    move_vec_t  = torch.tensor([[(tcx-scx)/IMG_SIZE, (tcy-scy)/IMG_SIZE]], device=device, dtype=dtype)

    move_token     = mcm(src_box_t, tgt_coord_t, move_vec_t)  # (1,1,768)
    encoder_hidden = torch.cat([text_emb, move_token], dim=1)  # (1, seq+1, 768)

    # Unconditional embedding for CFG
    uncond_tokens = tokenizer(
        "", padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_emb  = text_encoder(uncond_tokens.input_ids.to(device))[0]
    zero_token  = torch.zeros_like(move_token)
    uncond_hidden = torch.cat([uncond_emb, zero_token], dim=1)

    # ── DDIM Denoising ────────────────────────────────────────────────────
    generator = torch.Generator(device=device).manual_seed(args.seed)
    scheduler.set_timesteps(args.num_steps)

    latents = torch.randn(
        (1, unet.config.in_channels, IMG_SIZE // 8, IMG_SIZE // 8),
        generator=generator, device=device, dtype=dtype
    )
    latents = latents * scheduler.init_noise_sigma

    print(f"[Inference] Denoising {args.num_steps} steps...")
    for i, t in enumerate(scheduler.timesteps):
        # CFG: run both conditional and unconditional
        latent_input = torch.cat([latents, latents])
        hidden_input = torch.cat([uncond_hidden, encoder_hidden])

        noise_pred = unet(latent_input, t, encoder_hidden_states=hidden_input).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)

        # Classifier-free guidance
        noise_guided = noise_uncond + args.guidance * (noise_cond - noise_uncond)
        latents = scheduler.step(noise_guided, t, latents).prev_sample

        if (i + 1) % 10 == 0:
            print(f"  step {i+1}/{args.num_steps}")

    # ── Decode latents ────────────────────────────────────────────────────
    latents = latents / vae.config.scaling_factor
    image   = vae.decode(latents).sample
    image   = (image / 2 + 0.5).clamp(0, 1)
    image   = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image   = (image[0] * 255).round().astype(np.uint8)

    out_img = Image.fromarray(image)
    out_img.save(args.output)
    print(f"[Done] Saved → {args.output}")

    return out_img


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
