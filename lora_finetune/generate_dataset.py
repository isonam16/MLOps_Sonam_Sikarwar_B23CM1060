
import cv2
import numpy as np
import json
import os
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 512
NUM_SAMPLES = 2000
SEED        = 42
OUT_DIR     = Path("data")
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

SHAPES = ["circle", "rectangle", "triangle", "ellipse", "pentagon"]
COLORS = {
    "red":    (0,   0,   255),
    "green":  (0,   200, 0  ),
    "blue":   (255, 0,   0  ),
    "yellow": (0,   220, 220),
    "cyan":   (220, 220, 0  ),
    "purple": (180, 0,   180),
    "orange": (0,   140, 255),
    "white":  (240, 240, 240),
}
BG_COLORS = [
    (30,  30,  30 ),
    (200, 200, 200),
    (20,  20,  80 ),
    (80,  20,  20 ),
    (20,  80,  20 ),
    (240, 230, 200),
]


def draw_shape(img, shape, color, cx, cy, size):
    """Draw a shape centered at (cx, cy) with given size. Returns bounding box."""
    c = color
    if shape == "circle":
        r = size // 2
        cv2.circle(img, (cx, cy), r, c, -1)
        return (cx - r, cy - r, cx + r, cy + r)

    elif shape == "rectangle":
        hw, hh = size // 2, size // 3
        x1, y1 = cx - hw, cy - hh
        x2, y2 = cx + hw, cy + hh
        cv2.rectangle(img, (x1, y1), (x2, y2), c, -1)
        return (x1, y1, x2, y2)

    elif shape == "triangle":
        h = int(size * 0.866)
        pts = np.array([
            [cx,          cy - h // 2],
            [cx - size//2, cy + h // 2],
            [cx + size//2, cy + h // 2],
        ], np.int32)
        cv2.fillPoly(img, [pts], c)
        return (cx - size//2, cy - h//2, cx + size//2, cy + h//2)

    elif shape == "ellipse":
        a, b = size // 2, size // 3
        cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, c, -1)
        return (cx - a, cy - b, cx + a, cy + b)

    elif shape == "pentagon":
        pts = []
        for i in range(5):
            angle = np.radians(i * 72 - 90)
            px = int(cx + size // 2 * np.cos(angle))
            py = int(cy + size // 2 * np.sin(angle))
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], c)
        xs, ys = pts[:, 0], pts[:, 1]
        return (xs.min(), ys.min(), xs.max(), ys.max())

    return (cx - size//2, cy - size//2, cx + size//2, cy + size//2)


def make_background(bg_color):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[:] = bg_color
    # Add subtle noise for realism
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_sample():
    shape_name = random.choice(SHAPES)
    color_name, color_val = random.choice(list(COLORS.items()))
    bg_color = random.choice(BG_COLORS)
    size = random.randint(40, 100)
    margin = size + 10

    # Source position (keep object fully inside)
    src_cx = random.randint(margin, IMG_SIZE - margin)
    src_cy = random.randint(margin, IMG_SIZE - margin)

    # Target position — must be different from source
    while True:
        tgt_cx = random.randint(margin, IMG_SIZE - margin)
        tgt_cy = random.randint(margin, IMG_SIZE - margin)
        dist = np.sqrt((tgt_cx - src_cx)**2 + (tgt_cy - src_cy)**2)
        if dist > size:  # ensure meaningful move
            break

    # ── Source image ──────────────────────────────────────────────────────
    src_img = make_background(bg_color)

    # Optional: add 1-2 distractor shapes
    num_distractors = random.randint(0, 2)
    for _ in range(num_distractors):
        d_shape = random.choice(SHAPES)
        d_color = random.choice(list(COLORS.values()))
        d_size  = random.randint(20, 50)
        d_cx    = random.randint(d_size, IMG_SIZE - d_size)
        d_cy    = random.randint(d_size, IMG_SIZE - d_size)
        draw_shape(src_img, d_shape, d_color, d_cx, d_cy, d_size)

    bbox = draw_shape(src_img, shape_name, color_val, src_cx, src_cy, size)
    x1, y1, x2, y2 = [max(0, min(IMG_SIZE, v)) for v in bbox]

    # ── Target image ──────────────────────────────────────────────────────
    tgt_img = make_background(bg_color)

    # Redraw same distractors (same seed per sample — use same rng state approach)
    # For simplicity, re-draw same distractor pattern
    np.random.seed(int(src_cx * 1000 + src_cy))
    for _ in range(num_distractors):
        d_shape = random.choice(SHAPES)
        d_color = random.choice(list(COLORS.values()))
        d_size  = random.randint(20, 50)
        d_cx    = random.randint(d_size, IMG_SIZE - d_size)
        d_cy    = random.randint(d_size, IMG_SIZE - d_size)
        draw_shape(tgt_img, d_shape, d_color, d_cx, d_cy, d_size)

    draw_shape(tgt_img, shape_name, color_val, tgt_cx, tgt_cy, size)

    # ── Caption ───────────────────────────────────────────────────────────
    caption = (
        f"move the {color_name} {shape_name} "
        f"from position ({src_cx},{src_cy}) "
        f"to position ({tgt_cx},{tgt_cy})"
    )

    return {
        "src_img":   src_img,
        "tgt_img":   tgt_img,
        "src_box":   [x1, y1, x2, y2],
        "src_center":[src_cx, src_cy],
        "tgt_coord": [tgt_cx, tgt_cy],
        "shape":     shape_name,
        "color":     color_name,
        "size":      size,
        "caption":   caption,
    }


def main():
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "targets").mkdir(parents=True, exist_ok=True)

    annotations = []
    print(f"Generating {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        np.random.seed(SEED + i)
        random.seed(SEED + i)

        sample = generate_sample()

        img_name = f"{i:05d}.png"
        src_path = str(OUT_DIR / "images"  / img_name)
        tgt_path = str(OUT_DIR / "targets" / img_name)

        cv2.imwrite(src_path, sample["src_img"])
        cv2.imwrite(tgt_path, sample["tgt_img"])

        annotations.append({
            "id":         i,
            "image":      f"images/{img_name}",
            "target":     f"targets/{img_name}",
            "src_box":    sample["src_box"],    # [x1, y1, x2, y2]
            "src_center": sample["src_center"], # [cx, cy]
            "tgt_coord":  sample["tgt_coord"],  # [cx, cy]
            "shape":      sample["shape"],
            "color":      sample["color"],
            "size":       sample["size"],
            "caption":    sample["caption"],
        })

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{NUM_SAMPLES} done")

    with open(OUT_DIR / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDataset saved to ./{OUT_DIR}/")
    print(f"  images/   : {NUM_SAMPLES} source images")
    print(f"  targets/  : {NUM_SAMPLES} ground truth images")
    print(f"  annotations.json : full annotations")

    # Quick sanity check preview
    sample = annotations[0]
    print(f"\nSample[0]:")
    print(f"  caption  : {sample['caption']}")
    print(f"  src_box  : {sample['src_box']}")
    print(f"  tgt_coord: {sample['tgt_coord']}")


if __name__ == "__main__":
    main()
