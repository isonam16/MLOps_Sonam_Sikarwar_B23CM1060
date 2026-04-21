import cv2
import numpy as np
import torch
from DragonDiffusion.src.demo.download import download_all
from DragonDiffusion.src.demo.model import DragonModels
from DragonDiffusion.src.utils.utils import resize_numpy_image
from torchvision.transforms import PILToTensor
from PIL import Image
from DragonDiffusion.src.demo.utils import get_point_move
from segcopy import run_segmentation
from DragonDiffusion.sam.efficient_sam.build_efficient_sam import build_efficient_sam_vits


from DragonDiffusion.src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset

def run_move_try(img, img_arrow, img_draw_box, label, img_path):

    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    model = DragonModels(pretrained_model_path=pretrained_model_path)

    if model is not None:
        print("[INFO] DragonDiffusion models (to move) loaded successfully.")
    
    seg_model=build_efficient_sam_vits(savedCheckpoint="DragonDiffusion/models/efficient_sam_vits.pt")
    seg_model.eval()

    prompt=f"move the following {label}"
    selected_points=[(300,300),(400,400)] 

    mask_paths = run_segmentation(img_path, seg_model, input_label=[1], input_points=[[[400, 400]]], output_mask_dir="data/masks/")
    
    resize_scale = 1.0
    w_edit = 4.0
    w_content = 6.0
    w_contrast = 0.2
    w_inpaint = 0.8
    seed = 42
    guidance_scale = 4.0
    energy_scale = 0.5
    max_resolution = 768
    SDE_strength = 0.4
    ip_scale = 0.1

    img, img_orginal, selected_points = get_point_move(img, img_arrow, selected_points)
    print(selected_points)
    global_points = []
    global_point_label = []
    
    # segment_with_points returns 7 values, the last one (img_ref) should be a dict with 'image' and 'mask'
    img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref = segment_with_points(
        img_draw_box, 
        img_orginal, 
        global_points, 
        global_point_label, 
        selected_points, 
        img
    )
    
    # img_ref should be a dict like {'image': ..., 'mask': ...}
    # If it's not, we need to construct it properly
    if not isinstance(img_ref, dict):
        print(f"[WARNING] img_ref is not a dict, it's a {type(img_ref)}")
        # Try to use the mask from segment_with_points
        img_ref = {
            'image': original_image,
            'mask': mask
        }
    
    original_image, im_w_mask_ref, mask_ref = store_img_move(img_ref)
    
    res = model.run_move(
        original_image, 
        mask, 
        mask_ref, 
        prompt, 
        resize_scale, 
        w_edit, 
        w_content, 
        w_contrast, 
        w_inpaint, 
        seed, 
        selected_points, 
        guidance_scale, 
        energy_scale, 
        max_resolution, 
        SDE_strength, 
        ip_scale
    )
    
    return res[0]


def run_move_with_loaded_models(model, seg_model, img, img_arrow, img_draw_box, label, img_path):
    """
    Run the move operation using pre-loaded models.
    
    Args:
        model: Pre-loaded DragonDiffusion model
        seg_model: Pre-loaded segmentation model
        img: Input image (numpy array)
        img_arrow: Image with arrow overlay (numpy array)
        img_draw_box: Image with drawn box (numpy array)
        label: Label for the object to move
        img_path: Path to the input image
    
    Returns:
        The processed image result
    """
    prompt = f"move the following {label}"
    selected_points = [(300, 300), (400, 400)] 

    mask_paths = run_segmentation(
        img_path, 
        seg_model, 
        input_label=[1], 
        input_points=[[[400, 400]]], 
        output_mask_dir="data/masks/"
    )
    
    # Model parameters
    resize_scale = 1.0
    w_edit = 4.0
    w_content = 6.0
    w_contrast = 0.2
    w_inpaint = 0.8
    seed = 42
    guidance_scale = 4.0
    energy_scale = 0.5
    max_resolution = 768
    SDE_strength = 0.4
    ip_scale = 0.1

    img, img_orginal, selected_points = get_point_move(img, img_arrow, selected_points)
    print(f"[INFO] Selected points: {selected_points}")
    
    global_points = []
    global_point_label = []
    
    # Segment with points
    img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref = segment_with_points(
        img_draw_box, 
        img_orginal, 
        global_points, 
        global_point_label, 
        selected_points, 
        img
    )
    
    # Ensure img_ref is properly formatted
    if not isinstance(img_ref, dict):
        print(f"[WARNING] img_ref is not a dict, it's a {type(img_ref)}")
        img_ref = {
            'image': original_image,
            'mask': mask
        }
    
    original_image, im_w_mask_ref, mask_ref = store_img_move(img_ref)
    
    # Run the move operation
    res = model.run_move(
        original_image, 
        mask, 
        mask_ref, 
        prompt, 
        resize_scale, 
        w_edit, 
        w_content, 
        w_contrast, 
        w_inpaint, 
        seed, 
        selected_points, 
        guidance_scale, 
        energy_scale, 
        max_resolution, 
        SDE_strength, 
        ip_scale
    )
    
    result = res[0]
    print(f"[DEBUG] Type of result: {type(result)}")
    
    return result