
import os
import random
import runpod
import time
from datetime import datetime

# mock functions for development
# def create_seeds(initial_seed, quantity):
#     if initial_seed is None:
#         initial_seed = random.randint(0, 2**32 - 1)
#     return [initial_seed + i for i in range(quantity)]

# def load_model(model):
#     return model
    
# def generate_images(
#         pipeline, seeds,
#         positive_prompt, negative_prompt,
#         width, height, quantity,
#         steps, guidance):
#     result = type('obj', (object,), {'propertyName' : 'images'})
#     result.images = [
#         f"Generated Image with seed {seed} from '{positive_prompt}'"
#         for seed in seeds
#     ]
#     print(result)
#     return result
    
# real functions for serverless run

import torch
from diffusers import StableDiffusionPipeline

def create_seeds(initial_seed, quantity):
    if initial_seed is None:
        initial_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    return [initial_seed + i for i in range(quantity)]

def load_model(model):
    global pipe
    if pipe is None:
        model_path = "/workspace/models/checkpoints/" + model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_images(
        pipeline, seeds,
        positive_prompt, negative_prompt,
        width, height, quantity,
        steps, guidance):
    generators = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    result = pipeline(
        positive_prompt, 
        negative_prompt=negative_prompt,
        width=width,
        height=height,    
        num_images_per_prompt=quantity,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generators,
    )

def handler(job):
    # register initial executions time
    job_start_time_dt = datetime.utcnow()
    job_start_time_ns = time.time_ns()

    # extract inputs
    job_input = job["input"]
    model = job_input.get("model", "RealVisXL_V5.0_fp16.safetensors")
    positivePrompt = job_input.get("positivePrompt", "A cyberpunk cat sitting on a neon sign")
    negativePrompt = job_input.get("negativePrompt", "")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    quantity = job_input.get("quantity", 1)
    steps = job_input.get("steps", 11)
    guidance = job_input.get("guidance", 5)
    seed = job_input.get("seed")

    # generate images
    seeds = create_seeds(seed, quantity)
    pipeline = load_model(model)

    result = generate_images(
        pipeline, seeds,
        positivePrompt, negativePrompt,
        width, height, quantity,
        steps, guidance)

    # extract images
    images = result.images

    # register initial executions time
    job_end_time_ns = time.time_ns()

    # format time to return
    start_time_dt = job_start_time_dt.strftime("%Y-%m-%d %H:%M:%S")
    execution_time_ns = (job_start_time_ns - job_start_time_ns) / 1_000_000

    return {
        "parameters": {
            "model": model,
            "positivePrompt": positivePrompt,
            "negativePrompt": negativePrompt,
            "width": width,
            "height": height,
            "quantity": quantity,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
        },
        "execution": {
            "start_moment": start_time_dt,
            "execution_time": execution_time_ns,
        },
        "output": [
            {
                "seed": seed,
                "image": image,
            }
            for seed, image in zip(seeds, images)
        ]
    }

pipe = None

runpod.serverless.start({
    "handler": handler
})