
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer


IMG_SIZE = 512


def get_transforms(augment=True):
    """Image transforms. Both src and tgt use same spatial ops."""
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.0),  # off — flipping breaks move direction
        ]
    ops += [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ]
    return transforms.Compose(ops)


class MoveObjectDataset(Dataset):
    """
    Dataset for move-object LoRA fine-tuning on DragonDiffusion.

    Args:
        data_dir    : root directory containing images/, targets/, annotations.json
        tokenizer   : CLIPTokenizer instance
        split       : 'train' or 'val'
        val_ratio   : fraction of data to use for validation
        augment     : whether to apply data augmentation
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        split: str = "train",
        val_ratio: float = 0.1,
        augment: bool = True,
    ):
        self.data_dir  = Path(data_dir)
        self.tokenizer = tokenizer
        self.transform = get_transforms(augment and split == "train")

        with open(self.data_dir / "annotations.json") as f:
            all_ann = json.load(f)

        # Split
        n_val = int(len(all_ann) * val_ratio)
        if split == "train":
            self.annotations = all_ann[n_val:]
        else:
            self.annotations = all_ann[:n_val]

        print(f"[MoveObjectDataset] {split}: {len(self.annotations)} samples")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # ── Load images ───────────────────────────────────────────────────
        src_img = Image.open(self.data_dir / ann["image"]).convert("RGB")
        tgt_img = Image.open(self.data_dir / ann["target"]).convert("RGB")

        pixel_values  = self.transform(src_img)   # (3, H, W), [-1,1]
        target_pixels = self.transform(tgt_img)   # (3, H, W), [-1,1]

        # ── Tokenize caption ──────────────────────────────────────────────
        tokens = self.tokenizer(
            ann["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)  # (seq_len,)

        # ── Spatial annotations (normalized to [0,1]) ─────────────────────
        x1, y1, x2, y2 = ann["src_box"]
        src_box = torch.tensor([
            x1 / IMG_SIZE, y1 / IMG_SIZE,
            x2 / IMG_SIZE, y2 / IMG_SIZE,
        ], dtype=torch.float32)

        tcx, tcy = ann["tgt_coord"]
        tgt_coord = torch.tensor([tcx / IMG_SIZE, tcy / IMG_SIZE], dtype=torch.float32)

        scx, scy = ann["src_center"]
        move_vector = torch.tensor([
            (tcx - scx) / IMG_SIZE,
            (tcy - scy) / IMG_SIZE,
        ], dtype=torch.float32)

        # ── Object mask (soft box mask on source image) ───────────────────
        mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(IMG_SIZE, int(x2)), min(IMG_SIZE, int(y2))
        mask[:, iy1:iy2, ix1:ix2] = 1.0

        return {
            "pixel_values":  pixel_values,   # source image
            "target_pixels": target_pixels,  # ground truth moved image
            "input_ids":     input_ids,      # tokenized caption
            "src_box":       src_box,        # [x1,y1,x2,y2] normalized
            "tgt_coord":     tgt_coord,      # [cx,cy] normalized
            "move_vector":   move_vector,    # [dx,dy] normalized
            "object_mask":   mask,           # (1,H,W) soft mask
        }


def collate_fn(batch):
    return {
        "pixel_values":  torch.stack([b["pixel_values"]  for b in batch]),
        "target_pixels": torch.stack([b["target_pixels"] for b in batch]),
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "src_box":       torch.stack([b["src_box"]       for b in batch]),
        "tgt_coord":     torch.stack([b["tgt_coord"]     for b in batch]),
        "move_vector":   torch.stack([b["move_vector"]   for b in batch]),
        "object_mask":   torch.stack([b["object_mask"]   for b in batch]),
    }
