#flux_service.py
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
import base64
import io
from io import BytesIO
import logging
# import gc
logger = logging.getLogger("uvicorn.error")
from monitoring import MetricsRecorder
import torch
import PIL.Image as Image
from diffusers import FluxKontextInpaintPipeline, GGUFQuantizationConfig
from diffusers.hooks import apply_group_offloading
import uvicorn
import logging
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

logging.basicConfig(level=logging.INFO, filename='/app/logs/flux_inpaint.log')
logger = logging.getLogger(__name__)

app = FastAPI(title="Flux with mask Service")


def save_base64_image(b64_str: str, filename: str = "imgb.png"):
    """Decode a base64 image string and save it as a PNG."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(filename, format="PNG")
    print(f"Saved image → {filename}")


class FluxRequest(BaseModel):
    img_base64: Optional[str] = None
    mask_base64: Optional[str] = None
    prompt: Optional[str] = None


# global model variable
pipe = None
processor = None
model = None


@app.on_event("startup")
async def load_models():
    global pipe, model, processor
    logger.info("Loading Flux inpainting model...")
    try:
        ckpt_path = ("flux1-kontext-dev-Q8_0.gguf")
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config="black-forest-labs/FLUX.1-Kontext-dev",
            subfolder="transformer",
        )
        pipe = FluxKontextInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16, transformer=transformer
        )
        # pipe.text_encoder.to("cuda")
        # pipe.text_encoder_2.to("cuda")
        # pipe.vae.to("cuda")
        apply_group_offloading(pipe.transformer,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
        apply_group_offloading(pipe.text_encoder,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
        apply_group_offloading(pipe.text_encoder_2,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
        apply_group_offloading(pipe.vae,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
        logger.info("vae offloaded")
        logger.info("Flux Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load inpainting model: {e}")
        raise RuntimeError(f"Failed to load inpainting model: {e}") 


@app.post("/flux-inpaint")
async def fluxgeneration(img_base64: str = Form(...), mask_base64: str = Form(...), prompt: str = Form(None)):
    """Accepts base64 encoded image and mask. Returns path to inpainted image."""
    recorder = MetricsRecorder(interval=1.0)
    recorder.start()
    try:
        logger.info("Received inpainting request.")
        img_data = base64.b64decode(img_base64)
        mask_data = base64.b64decode(mask_base64)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_data)).convert("RGB")
        mask = pipe.mask_processor.blur(mask, blur_factor=12)
        logger.info("Passing to model for inpainting.")
        result_image = pipe(prompt=prompt, image=img, mask_image=mask, strength=1.0, num_inference_steps=10).images[0]
        logger.info("Inpainting completed successfully.")
        buffer = BytesIO()
        result_image.save(buffer, format="PNG")  # or "JPEG" if you prefer
        buffer.seek(0)
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info("Saving inpainted image.")
        recorder.stop()
        metrics = recorder.get_metrics()
        return {
            "image": img_b64,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Invalid base64 image/mask: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image/mask: {e}")

# Convert to numpy if your inpainting function expects numpy arrays
uvicorn.run(app, host="0.0.0.0", port=8007)




