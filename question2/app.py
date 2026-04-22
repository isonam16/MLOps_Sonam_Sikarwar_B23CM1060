import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

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

def get_color_palette(num_classes=23):
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return palette

def colorize_mask(mask, palette):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        color_mask[mask == c] = palette[c]
    return color_mask

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=23).to(device)
    model.load_state_dict(torch.load("unet_cityscapes.pth", map_location=device))
    model.eval()
    return model, device

st.set_page_config(page_title="CityScape Segmentation", page_icon="🏙️", layout="wide")

page = st.sidebar.radio("📄 Navigation", ["📊 Training Metrics", "🖼️ Segmentation Predictions"])

if page == "📊 Training Metrics":
    st.title("🏙️ CityScape Image Segmentation - Training Metrics")
    st.markdown("---")

    if os.path.exists("training_history.json"):
        with open("training_history.json", "r") as f:
            history = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Test mIOU", f"{history['test_miou']:.4f}")
        with col2:
            st.metric("🎯 Test mDice", f"{history['test_mdice']:.4f}")

        st.markdown("---")

        st.subheader("Training Loss Curve")
        if os.path.exists("training_loss.png"):
            st.image("training_loss.png", use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(history['train_losses'])+1), history['train_losses'], 'b-o')
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
            ax.grid(True)
            st.pyplot(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Training mIOU")
            if os.path.exists("training_miou.png"):
                st.image("training_miou.png", use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(history['train_mious'])+1), history['train_mious'], 'g-o')
                ax.set_xlabel("Epoch"); ax.set_ylabel("mIOU"); ax.set_title("Training mIOU")
                ax.grid(True)
                st.pyplot(fig)

        with col_b:
            st.subheader("Training mDice")
            if os.path.exists("training_mdice.png"):
                st.image("training_mdice.png", use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(history['train_mdices'])+1), history['train_mdices'], 'r-o')
                ax.set_xlabel("Epoch"); ax.set_ylabel("mDice"); ax.set_title("Training mDice")
                ax.grid(True)
                st.pyplot(fig)
    else:
        st.warning("Training history not found. Please run train.py first.")


elif page == "🖼️ Segmentation Predictions":
    st.title("🖼️ CityScape Segmentation - Predictions")
    st.markdown("---")

    model, device = load_model()
    palette = get_color_palette(23)

    if os.path.exists("test_image_paths.json"):
        with open("test_image_paths.json", "r") as f:
            test_data = json.load(f)
        test_images = test_data["test_images"]
        test_masks = test_data["test_masks"]

        st.info("Upload 4 images from the test set to see predictions. Test images available in `data/CameraRGB/`.")

        uploaded_files = st.file_uploader(
            "Upload 4 test images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files and len(uploaded_files) >= 1:
            for i, uploaded_file in enumerate(uploaded_files[:4]):
                st.markdown(f"### Image {i+1}: {uploaded_file.name}")

                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                mask_path = None
                for tp, mp in zip(test_images, test_masks):
                    if os.path.basename(tp) == uploaded_file.name:
                        mask_path = mp
                        break

                img_resized = cv2.resize(img_rgb, (128, 96), interpolation=cv2.INTER_NEAREST)
                img_input = img_resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                pred_colored = colorize_mask(pred, palette)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption("Input Image")
                    st.image(img_rgb, use_container_width=True)

                with col2:
                    st.caption("Ground Truth Mask")
                    if mask_path and os.path.exists(mask_path):
                        gt_mask = cv2.imread(mask_path)
                        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
                        gt_mask_resized = cv2.resize(gt_mask, (128, 96), interpolation=cv2.INTER_NEAREST)
                        gt_label = np.max(gt_mask_resized, axis=-1)
                        gt_colored = colorize_mask(gt_label, palette)
                        st.image(gt_colored, use_container_width=True)
                    else:
                        st.warning("Matching ground truth mask not found.")

                with col3:
                    st.caption("Predicted Mask")
                    st.image(pred_colored, use_container_width=True)

                st.markdown("---")
        else:
            st.subheader("Preview: Random test set samples")
            import random
            random.seed(42)
            sample_indices = random.sample(range(len(test_images)), min(4, len(test_images)))

            for i, idx in enumerate(sample_indices):
                st.markdown(f"### Sample {i+1}")
                img_path = test_images[idx]
                msk_path = test_masks[idx]

                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (128, 96), interpolation=cv2.INTER_NEAREST)
                img_input = img_resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

                pred_colored = colorize_mask(pred, palette)

                gt_mask = cv2.cvtColor(cv2.imread(msk_path), cv2.COLOR_BGR2RGB)
                gt_mask_resized = cv2.resize(gt_mask, (128, 96), interpolation=cv2.INTER_NEAREST)
                gt_label = np.max(gt_mask_resized, axis=-1)
                gt_colored = colorize_mask(gt_label, palette)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption("Input Image")
                    st.image(img_rgb, use_container_width=True)
                with col2:
                    st.caption("Ground Truth Mask")
                    st.image(gt_colored, use_container_width=True)
                with col3:
                    st.caption("Predicted Mask")
                    st.image(pred_colored, use_container_width=True)
                st.markdown("---")
    else:
        st.warning("Test image paths not found. Please run train.py first.")
