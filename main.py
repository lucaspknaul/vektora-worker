import io
import base64
import torch
from diffusers import StableDiffusionPipeline
import os

pipe = None

def get_pipeline():
    global pipe
    if pipe is None:
        model_path = "/workspace/models/checkpoints/RealVisXL_V5.0_fp16.safetensors"
        # ou substitua "runwayml/stable-diffusion-v1-5" pelo nome da pasta do modelo que você subiu
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    return pipe

def handler(event):
    prompt = event.get("input", {}).get("prompt", "A cyberpunk cat sitting on a neon sign")
    steps = int(event.get("input", {}).get("steps", 11))
    guidance = float(event.get("input", {}).get("guidance", 5))

    pipeline = get_pipeline()
    result = pipeline(prompt, guidance_scale=guidance, num_inference_steps=steps)
    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"output": {"image_base64": image_b64}}
