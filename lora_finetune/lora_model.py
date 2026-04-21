"""
lora_model.py
--------------
LoRA injection into Stable Diffusion 1.5 UNet for move-object fine-tuning.

Architecture:
  - Injects LoRA into UNet cross-attention (Q, K, V, Out) layers
  - Optionally injects a lightweight Move Conditioning Module (MCM)
    that encodes [src_box, tgt_coord, move_vector] and adds it
    to the cross-attention context, guiding where to move the object.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


# ── LoRA Layer ────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen Linear layer with a trainable low-rank adapter."""

    def __init__(self, orig_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.orig   = orig_linear         # frozen
        self.rank   = rank
        self.scale  = alpha / rank

        in_f  = orig_linear.in_features
        out_f = orig_linear.out_features

        self.lora_A = nn.Linear(in_f,  rank,  bias=False)
        self.lora_B = nn.Linear(rank,  out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for p in self.orig.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.orig(x) + self.scale * self.lora_B(self.lora_A(x))


# ── Move Conditioning Module ───────────────────────────────────────────────────

class MoveConditioningModule(nn.Module):
    """
    Encodes spatial move instructions into a conditioning token
    that is appended to the CLIP text embeddings.

    Input:  [src_box (4), tgt_coord (2), move_vector (2)] → 8-dim vector
    Output: (B, 1, cross_attention_dim) conditioning token
    """

    def __init__(self, cross_attention_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cross_attention_dim),
        )

    def forward(self, src_box, tgt_coord, move_vector):
        """
        Args:
            src_box     : (B, 4) normalized [x1,y1,x2,y2]
            tgt_coord   : (B, 2) normalized [cx,cy]
            move_vector : (B, 2) normalized [dx,dy]
        Returns:
            token : (B, 1, cross_attention_dim)
        """
        spatial_feat = torch.cat([src_box, tgt_coord, move_vector], dim=-1)  # (B,8)
        token = self.net(spatial_feat).unsqueeze(1)  # (B,1,dim)
        return token


# ── LoRA Injection ────────────────────────────────────────────────────────────

def inject_lora_into_unet(
    unet: UNet2DConditionModel,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: list = None,
) -> dict:
    """
    Replaces Q/K/V/Out projection linears in UNet cross-attention with LoRALinear.

    Returns a dict of all injected LoRA modules for easy parameter access.
    """
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    lora_layers = {}

    for name, module in unet.named_modules():
        # Only inject into cross-attention (has encoder_hidden_states input)
        if not isinstance(module, Attention):
            continue

        for attr_name in target_modules:
            # Navigate nested attribute (e.g., "to_out.0")
            parts = attr_name.split(".")
            obj = module
            for p in parts[:-1]:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is None:
                continue

            leaf_attr = parts[-1]
            orig_layer = getattr(obj, leaf_attr, None)
            if not isinstance(orig_layer, nn.Linear):
                continue

            lora_layer = LoRALinear(orig_layer, rank=rank, alpha=alpha)
            setattr(obj, leaf_attr, lora_layer)

            key = f"{name}.{attr_name}"
            lora_layers[key] = lora_layer

    print(f"[LoRA] Injected {len(lora_layers)} LoRA layers (rank={rank}, alpha={alpha})")
    return lora_layers


def get_trainable_params(unet, mcm):
    """Returns only LoRA + MCM parameters (UNet base frozen)."""
    params = []

    # LoRA A/B matrices
    for module in unet.modules():
        if isinstance(module, LoRALinear):
            params += list(module.lora_A.parameters())
            params += list(module.lora_B.parameters())

    # Move Conditioning Module
    if mcm is not None:
        params += list(mcm.parameters())

    total = sum(p.numel() for p in params)
    print(f"[LoRA] Trainable params: {total:,}  ({total/1e6:.2f}M)")
    return params


def save_lora_weights(unet, mcm, path: str):
    """Save only LoRA + MCM weights (not full UNet)."""
    state = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            state[f"lora.{name}.lora_A"] = module.lora_A.weight.data
            state[f"lora.{name}.lora_B"] = module.lora_B.weight.data
    if mcm is not None:
        for k, v in mcm.state_dict().items():
            state[f"mcm.{k}"] = v
    torch.save(state, path)
    print(f"[LoRA] Saved weights → {path}")


def load_lora_weights(unet, mcm, path: str):
    """Load LoRA + MCM weights back into model."""
    state = torch.load(path, map_location="cpu")
    lora_modules = {name: m for name, m in unet.named_modules() if isinstance(m, LoRALinear)}

    for key, val in state.items():
        if key.startswith("lora."):
            parts = key[len("lora."):].rsplit(".", 1)
            mod_name, weight_name = parts
            if mod_name in lora_modules:
                if weight_name == "lora_A":
                    lora_modules[mod_name].lora_A.weight.data = val
                elif weight_name == "lora_B":
                    lora_modules[mod_name].lora_B.weight.data = val
        elif key.startswith("mcm.") and mcm is not None:
            mcm_key = key[len("mcm."):]
            mcm.state_dict()[mcm_key].copy_(val)

    print(f"[LoRA] Loaded weights ← {path}")
