# image-edit.py
import os
import io
import base64
import random
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# --- 环境优化 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 检查 CUDA 可用性 ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print("Loading QwenImageEditPipeline with balanced multi-GPU...")

# ✅ 使用 device_map="balanced"（diffusers 支持）
pipe = QwenImageEditPipeline.from_pretrained(
    "./Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16,
    device_map="balanced",  # 👈 关键：在多卡间平衡分配
    # low_cpu_mem_usage=True,  # 可选：减少 CPU 内存
)

print(f"✅ Pipeline loaded. Devices: {pipe.device}")

def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

app = FastAPI(title="Qwen Image Edit API (Multi-GPU Balanced)", version="1.0")

@app.post("/edit")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    seed: Optional[int] = Form(42),
    randomize_seed: bool = Form(True),
    true_guidance_scale: float = Form(4.0),
    num_inference_steps: int = Form(50),
    num_images_per_prompt: int = Form(1),
):
    try:
        # 读图
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 随机 seed
        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)

        # ⚠️ generator 必须在 CPU（因为 pipeline 已分布到多卡，无法指定单个 GPU）
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # 推理
        with torch.inference_mode():
            output = pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 返回 Base64 图像
        result_images = [pil_to_base64(img) for img in output.images]

        return JSONResponse({
            "images": result_images,
            "seed": seed,
            "prompt_used": prompt
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")
