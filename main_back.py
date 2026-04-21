import os
import uuid
import base64
import json
import io
import requests
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from ultralytics import YOLO

app = FastAPI(title="JBJM Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHARED_DIR = "./shared_data"
os.makedirs(SHARED_DIR, exist_ok=True)

MOVE_SEG_URL = "http://localhost:8001/move3"
ERASE_URL = "http://localhost:8005/erase"
INPAINT_URL = "http://localhost:8002/flux-inpaint"


YOLO_MODEL = YOLO("yolov8n-seg.pt")
YOLO_MODEL.to("cpu")


STATE = {}


class ImageRequest(BaseModel):
    image: str

class MoveRequest(BaseModel):
    image_name: str
    masks_ids: List[int]
    startx: int
    starty: int
    endx: int
    endy: int
    prompt: Optional[str] = None

class EraseRequest(BaseModel):
    image_name: str
    masks_ids: List[int]

class InpaintRequest(BaseModel):
    image_name: str
    masks_ids: List[int]
    prompt: Optional[str] = None
    model_type: Optional[str] = "FREE"

class SegmentRequest(BaseModel):
    image_name: str

def b64_to_img(b64_str: str) -> Image.Image:
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_path_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_combined_mask(image_name: str, mask_ids: List[int]) -> Image.Image:
    if image_name not in STATE:
        raise ValueError("Image not found in state.")
    
    mask_paths = STATE[image_name]["masks"]
    if not mask_paths:
        raise ValueError("No masks available.")

    base_img = Image.open(STATE[image_name]["image_path"])
    width, height = base_img.size

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for i in mask_ids:
        if i < len(mask_paths):
            m = np.array(Image.open(mask_paths[i]).convert("L"))
            combined_mask = np.maximum(combined_mask, m)
            
    return Image.fromarray(combined_mask, mode="L")

@app.post("/segment")
def segment_image(req: SegmentRequest):
    """
    Run YOLOv8 instance segmentation on the stored image.
    Returns per-instance binary masks, class names, and confidence scores.
    """
    if req.image_name not in STATE:
        raise HTTPException(status_code=404, detail="Image not initialized.")
        
    img_path = STATE[req.image_name]["image_path"]
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image format: {e}")

    width, height = img.size

    results = YOLO_MODEL(img_path, task="segment")  
    result = results[0]  

    mask_paths = []
    masks_response = []

    if result.masks is None or len(result.masks) == 0:
        STATE[req.image_name]["masks"] = []
        return {"count": 0, "masks": []}

 

    seg_masks = result.masks.data.cpu().numpy()  
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names 

    for idx in range(len(seg_masks)):
      
        mask_arr = (seg_masks[idx] * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_arr, mode="L").resize(
            (width, height), resample=Image.NEAREST
        )

        mask_path = os.path.join(SHARED_DIR, f"mask_{req.image_name}_{idx}.png")
        mask_img.save(mask_path)
        mask_paths.append(mask_path)

        masks_response.append({
            "index": idx,
            "class_name": names.get(class_ids[idx], str(class_ids[idx])),
            "confidence": float(round(confs[idx], 4)),
            "png_base64": img_to_b64(mask_img)
        })

    STATE[req.image_name]["masks"] = mask_paths

    return {"count": len(masks_response), "masks": masks_response}


@app.post("/set-image")
def set_image(req: ImageRequest):
    img_id = str(uuid.uuid4())
    img_name = f"image_{img_id}"
    filename = f"{img_name}.png"
    filepath = os.path.join(SHARED_DIR, filename)
    
    try:
        img = b64_to_img(req.image)
        img.save(filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    STATE[img_name] = {"image_path": filepath, "masks": []}
    return {
        "message": "Image set successfully",
        "image_name": img_name,
        "image_path": filepath,
        "width": img.width,
        "height": img.height
    }


@app.post("/move")
def move_object(req: MoveRequest):
    img_path = STATE.get(req.image_name, {}).get("image_path")
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found.")

    try:
        base_b64 = img_path_to_b64(img_path)
        combined_mask_img = get_combined_mask(req.image_name, req.masks_ids)
        mask_b64 = img_to_b64(combined_mask_img)

        payload = {
            "image": base_b64,
            "mask": mask_b64,
            "x_in": req.startx,
            "y_in": req.starty,
            "x_f": req.endx,
            "y_f": req.endy
        }

        resp = requests.post(MOVE_SEG_URL, json=payload, timeout=300)
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
            
        data = resp.json()
        return {
            "image": data.get("result_image"),
            "message": "Move complete",
            "metrics": data.get("metrics", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/erase")
def erase_object(req: EraseRequest):
    img_path = STATE.get(req.image_name, {}).get("image_path")
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found.")
        
    try:
        combined_mask_img = get_combined_mask(req.image_name, req.masks_ids)
        
        img_buf = io.BytesIO()
        Image.open(img_path).save(img_buf, format="PNG")
        img_buf.seek(0)
        
        mask_buf = io.BytesIO()
        combined_mask_img.save(mask_buf, format="PNG")
        mask_buf.seek(0)
        
        files = {
            "image": ("image.png", img_buf, "image/png"),
            "mask": ("mask.png", mask_buf, "image/png"),
        }
        
        resp = requests.post(ERASE_URL, files=files, timeout=300)
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        result_b64 = base64.b64encode(resp.content).decode()
        return {"image": result_b64, "message": "Erase complete."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint")
def inpaint_object(req: InpaintRequest):
    img_path = STATE.get(req.image_name, {}).get("image_path")
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found.")
        
    try:
        base_b64 = img_path_to_b64(img_path)
        combined_mask_img = get_combined_mask(req.image_name, req.masks_ids)
        mask_b64 = img_to_b64(combined_mask_img)

        payload = {
            "img_base64": (None, base_b64),
            "mask_base64": (None, mask_b64),
            "prompt": (None, req.prompt or "Fill naturally")
        }
        
        resp = requests.post(INPAINT_URL, files=payload, timeout=600)
        
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
            
        data = resp.json()
        return {
            "image": data.get("image"),
            "message": "Inpaint complete.",
            "metrics": data.get("metrics", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)