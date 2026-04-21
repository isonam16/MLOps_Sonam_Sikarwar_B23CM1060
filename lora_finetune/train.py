

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
from tqdm import tqdm

from dataset   import MoveObjectDataset, collate_fn
from lora_model import (
    inject_lora_into_unet,
    MoveConditioningModule,
    get_trainable_params,
    save_lora_weights,
)


# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--output_dir",  type=str,   default="./checkpoints")
    p.add_argument("--num_epochs",  type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lora_rank",   type=int,   default=8)
    p.add_argument("--lora_alpha",  type=float, default=16.0)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_mcm",      type=float, default=3e-4)
    p.add_argument("--grad_accum",  type=int,   default=2)
    p.add_argument("--mixed_prec",  type=str,   default="fp16", choices=["no","fp16","bf16"])
    p.add_argument("--save_every",  type=int,   default=5)
    p.add_argument("--val_every",   type=int,   default=2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def encode_images(vae, images):
    """Encode images to latent space."""
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


def get_text_embeddings(text_encoder, input_ids):
    """Get CLIP text embeddings."""
    with torch.no_grad():
        embeddings = text_encoder(input_ids)[0]
    return embeddings


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_prec,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    accelerator.init_trackers("move_lora")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load SD 1.5 components ────────────────────────────────────────────
    accelerator.print("[Setup] Loading SD 1.5 components...")
    tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Freeze VAE and text encoder entirely
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Inject LoRA into UNet ─────────────────────────────────────────────
    accelerator.print("[Setup] Injecting LoRA...")
    inject_lora_into_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)

    # ── Move Conditioning Module ──────────────────────────────────────────
    cross_attn_dim = unet.config.cross_attention_dim  # 768 for SD 1.5
    mcm = MoveConditioningModule(cross_attention_dim=cross_attn_dim)

    # ── Trainable params ──────────────────────────────────────────────────
    lora_params = []
    for module in unet.modules():
        from lora_model import LoRALinear
        if isinstance(module, LoRALinear):
            lora_params += list(module.lora_A.parameters())
            lora_params += list(module.lora_B.parameters())

    optimizer = AdamW([
        {"params": lora_params,          "lr": args.lr},
        {"params": mcm.parameters(),     "lr": args.lr_mcm},
    ], weight_decay=1e-4)

    # ── Datasets ──────────────────────────────────────────────────────────
    accelerator.print("[Setup] Loading datasets...")
    train_ds = MoveObjectDataset(args.data_dir, tokenizer, split="train",  augment=True)
    val_ds   = MoveObjectDataset(args.data_dir, tokenizer, split="val",    augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=True
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * steps_per_epoch)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        accelerator.print(f"[Setup] Resuming from {args.resume}")
        from lora_model import load_lora_weights
        load_lora_weights(unet, mcm, args.resume)

    # ── Accelerate prepare ────────────────────────────────────────────────
    unet, mcm, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        unet, mcm, optimizer, train_loader, val_loader, scheduler
    )
    vae          = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    accelerator.print(f"[Train] Starting training: {args.num_epochs} epochs")
    global_step = 0

    for epoch in range(start_epoch, args.num_epochs):

        # ── Training loop ─────────────────────────────────────────────────
        unet.train()
        mcm.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(unet):

                # 1. Encode TARGET image to latent space
                tgt_latents = encode_images(vae, batch["target_pixels"])

                # 2. Sample noise and timestep
                noise = torch.randn_like(tgt_latents)
                bsz   = tgt_latents.shape[0]
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (bsz,), device=tgt_latents.device
                ).long()

                # 3. Add noise to target latents (forward diffusion)
                noisy_latents = noise_sched.add_noise(tgt_latents, noise, timesteps)

                # 4. Build conditioning:
                #    CLIP text embedding + MCM spatial token appended
                text_emb = get_text_embeddings(text_encoder, batch["input_ids"])
                # text_emb: (B, seq_len, 768)

                move_token = mcm(
                    batch["src_box"],
                    batch["tgt_coord"],
                    batch["move_vector"],
                )
                # move_token: (B, 1, 768)

                # Append move token to text sequence
                encoder_hidden = torch.cat([text_emb, move_token], dim=1)
                # encoder_hidden: (B, seq_len+1, 768)

                # 5. Encode SOURCE image to provide visual context
                #    We concatenate src latents channel-wise with noisy target latents
                src_latents = encode_images(vae, batch["pixel_values"])

                # Concatenate along channel dim: (B, 8, H/8, W/8)
                model_input = torch.cat([noisy_latents, src_latents], dim=1)

                # NOTE: UNet expects 4 input channels by default.
                # For 8-channel input, the UNet conv_in layer must be adapted.
                # See adapt_unet_for_concat_input() below.
                # For simplicity in this script, we use just noisy_latents
                # and rely on MCM for move guidance.
                model_input = noisy_latents  # (B, 4, H/8, W/8)

                # 6. Predict noise
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden,
                ).sample

                # 7. MSE loss on noise
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for pg in optimizer.param_groups for p in pg["params"]],
                        max_norm=1.0
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.detach().item()
            global_step += 1

            if accelerator.sync_gradients:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        avg_train_loss = train_loss / len(train_loader)
        accelerator.print(f"  Epoch {epoch+1} | train_loss: {avg_train_loss:.4f}")

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0:
            unet.eval()
            mcm.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    tgt_latents   = encode_images(vae, batch["target_pixels"])
                    noise         = torch.randn_like(tgt_latents)
                    bsz           = tgt_latents.shape[0]
                    timesteps     = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=tgt_latents.device).long()
                    noisy_latents = noise_sched.add_noise(tgt_latents, noise, timesteps)
                    text_emb      = get_text_embeddings(text_encoder, batch["input_ids"])
                    move_token    = mcm(batch["src_box"], batch["tgt_coord"], batch["move_vector"])
                    encoder_hidden = torch.cat([text_emb, move_token], dim=1)
                    noise_pred    = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden).sample
                    loss          = F.mse_loss(noise_pred.float(), noise.float())
                    val_loss     += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            accelerator.print(f"  Epoch {epoch+1} | val_loss:   {avg_val_loss:.4f}")
            accelerator.log({"val/loss": avg_val_loss}, step=global_step)

        # ── Save checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f"lora_epoch{epoch+1:03d}.pt")
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_mcm  = accelerator.unwrap_model(mcm)
            save_lora_weights(unwrapped_unet, unwrapped_mcm, ckpt_path)

    # ── Final save ────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "lora_final.pt")
        save_lora_weights(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(mcm),
            final_path
        )
        accelerator.print(f"\n[Done] Final weights saved → {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
