from PIL import Image
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import logging
from logs.logger import get_logger
logger = get_logger()


def run_segmentation(img_path, model, input_label, input_points, output_mask_dir="/app/data/masks"):

    os.makedirs(output_mask_dir, exist_ok=True)

    # Get device from model
    device = next(model.parameters()).device
    
    transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
    raw_image = Image.open(img_path).convert("RGB")
    
    # Move tensors to the same device as model
    image_tensor = transform(raw_image).unsqueeze(0).to(device) 
    points = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0).to(device)
    labels = torch.tensor(input_label, dtype=torch.int64).unsqueeze(0).to(device)

    logger.debug("[DEBUG] Preparing input to pass in efficient-sam")

    with torch.no_grad():
        pred_masks, pred_scores = model(
            image_tensor,
            points,
            labels
        )

    logger.debug("[DEBUG] Model inference completed")

    # pred_masks shape: (1, N, 256, 256)
    num_masks = pred_masks.shape[1]

    saved_paths = []

    for i in range(num_masks):
    
        mask_256 = pred_masks[0, i, 0]  

        logger.debug("[DEBUG] mask_256 shape BEFORE unsqueeze: %s", mask_256.shape)

        # now convert (256,256) → (1,1,256,256)
        mask_256 = mask_256.unsqueeze(0).unsqueeze(0)

        logger.debug("[DEBUG] mask_256 shape AFTER unsqueeze: %s", mask_256.shape)

        # upscale to original image size
        mask_resized = F.interpolate(
            mask_256,
            size=raw_image.size[::-1],   # (H, W)
            mode="bilinear",
            align_corners=False
        ).squeeze()

        # Convert to uint8
        mask_np = (mask_resized > 0.5).cpu().numpy().astype("uint8") * 255

        # Construct filename
        orig_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = f"{orig_name}_mask_{i}.png"
        output_path = os.path.join(output_mask_dir, mask_name)

        # Save mask
        try:
            success = cv2.imwrite(output_path, mask_np)
        except:
            Image.fromarray(mask_np).save(output_path)
            success = True
            
        logger.info(f"[INFO] Saved mask at {output_path}")
        print("Saved mask at ", output_path)

        saved_paths.append(output_path)

    return output_path  # return the last saved mask path

# if __name__ == "__main__":
#     img_path = "/home/b24cm1016/DATA1/FedAvg/adobe/MLpipeline/data/images/apple.jpeg"
#     run_segmentation(img_path)
