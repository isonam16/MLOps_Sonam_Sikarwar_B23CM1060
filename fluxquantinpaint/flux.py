import torch
from diffusers import FluxKontextInpaintPipeline
from diffusers.utils import load_image
from diffusers.hooks import apply_group_offloading
import time

s = time.time()

pipe = FluxKontextInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)

apply_group_offloading(pipe.transformer,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
apply_group_offloading(pipe.text_encoder,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
apply_group_offloading(pipe.text_encoder_2,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)
apply_group_offloading(pipe.vae,offload_type="leaf_level",offload_device=torch.device("cpu"),onload_device=torch.device("cuda"),use_stream=True,)

prompt = "change to joker"
img_url = "h.png"
mask_url = "insaankamask.png"
image_reference_url = "circusclown.jpeg"

source = load_image(img_url)
mask = load_image(mask_url)
# image_reference = load_image(image_reference_url)

mask = pipe.mask_processor.blur(mask, blur_factor=12)
image = pipe(
    prompt=prompt, image=source, mask_image=mask, strength=1.0,num_inference_steps=10
).images[0]
image.save("kontext_inpainting_ref.png")

print("Time taken:", time.time() - s)