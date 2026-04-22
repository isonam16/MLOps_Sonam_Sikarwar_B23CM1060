import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

NUM_CLASSES = 23
IMG_SIZE = (128, 96) 
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
SEED = 42
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=23):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.pool = nn.MaxPool2d(2)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if d4.shape != e4.shape:
            d4 = nn.functional.interpolate(d4, size=e4.shape[2:])
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


def compute_iou_dice(pred, target, num_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    dices = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            continue

        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (pred_c.sum() + target_c.sum() + 1e-8)
        ious.append(iou)
        dices.append(dice)

    mean_iou = np.mean(ious) if ious else 0.0
    mean_dice = np.mean(dices) if dices else 0.0
    return mean_iou, mean_dice

rgb_dir = os.path.join(DATA_DIR, "CameraRGB")
mask_dir = os.path.join(DATA_DIR, "CameraMask")

image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")

train_img, test_img, train_mask, test_mask = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=SEED
)

print(f"Train: {len(train_img)}, Test: {len(test_img)}")

train_dataset = CityscapesDataset(train_img, train_mask)
test_dataset = CityscapesDataset(test_img, test_mask)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
train_mious = []
train_mdices = []

print("=" * 60)
print("Starting Training...")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    num_batches = 0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clamp mask values to valid range
        masks = masks.clamp(0, NUM_CLASSES - 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_iou, batch_dice = compute_iou_dice(preds, masks, NUM_CLASSES)

        epoch_loss += loss.item()
        epoch_iou += batch_iou
        epoch_dice += batch_dice
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    avg_iou = epoch_iou / num_batches
    avg_dice = epoch_dice / num_batches

    train_losses.append(avg_loss)
    train_mious.append(avg_iou)
    train_mdices.append(avg_dice)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} | mIOU: {avg_iou:.4f} | mDice: {avg_dice:.4f}")

torch.save(model.state_dict(), "unet_cityscapes.pth")
print("Model saved to unet_cityscapes.pth")

print("\n" + "=" * 60)
print("Evaluating on Test Set...")
print("=" * 60)

model.eval()
test_iou_total = 0.0
test_dice_total = 0.0
num_test_batches = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        masks = masks.clamp(0, NUM_CLASSES - 1)

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        batch_iou, batch_dice = compute_iou_dice(preds, masks, NUM_CLASSES)
        test_iou_total += batch_iou
        test_dice_total += batch_dice
        num_test_batches += 1

test_miou = test_iou_total / num_test_batches
test_mdice = test_dice_total / num_test_batches

print(f"Test mIOU : {test_miou:.4f}")
print(f"Test mDice: {test_mdice:.4f}")

history = {
    "train_losses": train_losses,
    "train_mious": train_mious,
    "train_mdices": train_mdices,
    "test_miou": test_miou,
    "test_mdice": test_mdice,
}
with open("training_history.json", "w") as f:
    json.dump(history, f)

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, 'b-o', linewidth=2, markersize=6)
plt.title("Training Loss Curve", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_mious, 'g-o', linewidth=2, markersize=6)
plt.title("Training mIOU", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("mIOU")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_miou.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_mdices, 'r-o', linewidth=2, markersize=6)
plt.title("Training mDice", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("mDice")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_mdice.png", dpi=150)
plt.close()

print("\nPlots saved: training_loss.png, training_miou.png, training_mdice.png")
print(f"\nFinal Test Results => mIOU: {test_miou:.4f}, mDICE: {test_mdice:.4f}")

with open("test_image_paths.json", "w") as f:
    json.dump({"test_images": test_img, "test_masks": test_mask}, f)

print("Done!")
