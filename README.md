---

# 🚀 FastAPI Multi-GPU Deployment for Qwen-Image-Edit

Deploy **Qwen-Image-Edit** (the official image editing model from Alibaba's Qwen team) via **FastAPI** on **multi-GPU servers** (e.g., 8 × RTX 4090 with 48GB VRAM each).  
Supports **balanced device loading** to avoid CUDA Out-of-Memory (OOM) errors.

> ✅ No prompt rewriting  
> ✅ Multi-GPU inference (via `device_map="balanced"`)  
> ✅ Production-ready API

---

## 📦 Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3
- CUDA ≥ 12.1
- NVIDIA Driver ≥ 550
- 8+ NVIDIA GPUs with ≥ 40GB VRAM recommended

Install dependencies:

```bash
pip install fastapi uvicorn pillow torch diffusers accelerate transformers
```

> 💡 **Important**: Use the latest `diffusers` from source for Qwen-Image-Edit support:
>
> ```bash
> pip install git+https://github.com/huggingface/diffusers
> ```

---

## ⚙️ Environment Setup

Set the following environment variable to reduce CUDA memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> ⚠️ **Do NOT use `--reload`** with Uvicorn in production — it causes duplicate model loads and OOM.

---

## 🧠 Model Details

- **Model**: [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit)
- **Task**: Image editing (add/remove/replace objects, text, style transfer, etc.)
- **Input**: One image + text instruction
- **Output**: Edited image(s)

> 📌 **Note**: For better consistency, consider using `Qwen-Image-Edit-2509`, but this demo uses the base version.

---

## ▶️ Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 22001 --workers 1
```

- `--workers 1`: Required to avoid multiple model loads
- `--host 0.0.0.0`: Allows external access
- Port `22001`: Can be changed as needed

On startup, the model will automatically distribute its layers across all available GPUs using **balanced device mapping**.

You’ll see logs like:
```
Loading QwenImageEditPipeline with balanced multi-GPU...
✅ Pipeline loaded. Devices: cuda:0, cuda:1, ..., cuda:7
INFO:     Uvicorn running on http://0.0.0.0:22001
```

Use `nvidia-smi` to verify multi-GPU memory usage.

---

## 📡 API Usage

### POST `/edit`

Edit an image using a text instruction.

#### Parameters

| Field | Type | Required | Description |
|------|------|--------|-------------|
| `image` | file | ✅ | Input image (PNG/JPG) |
| `prompt` | str | ✅ | Editing instruction (raw text, no rewriting) |
| `num_inference_steps` | int | ❌ | Default: `50` |
| `true_guidance_scale` | float | ❌ | Default: `4.0` |
| `randomize_seed` | bool | ❌ | Default: `true` |
| `num_images_per_prompt` | int | ❌ | Default: `1` (max 4) |

#### Example (Python)

```python
import requests
import base64

url = "http://YOUR_SERVER_IP:22001/edit"
prompt = "Change the dog to a purple cat, keep background unchanged."

with open("input.png", "rb") as f:
    files = {"image": f}
    data = {
        "prompt": prompt,
        "num_inference_steps": "50",
        "true_guidance_scale": "4.0",
        "randomize_seed": "true"
    }
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    img_b64 = response.json()["images"][0]
    with open("output.png", "wb") as out:
        out.write(base64.b64decode(img_b64))
    print("✅ Image saved as output.png")
else:
    print("❌ Error:", response.text)
```

#### Response

```json
{
  "images": ["BASE64_ENCODED_IMAGE_STRING", ...],
  "seed": 123456789,
  "prompt_used": "Change the dog to a purple cat..."
}
```

---

## 🧪 Tips for Stability

- Keep `num_images_per_prompt=1` to reduce memory pressure.
- Reduce `num_inference_steps` to `30–40` if OOM persists.
- Ensure your input image is **not too large** (recommend ≤ 1024×1024).
- Monitor GPU memory with `nvidia-smi`.

---

## 📄 License

- Qwen-Image-Edit: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- This code: MIT License

---

## 🙏 Acknowledgements

- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Alibaba Tongyi Lab

---

> 🔗 **Online Demo**: Try it at [Qwen Chat](https://chat.qwen.ai/) → “Image Editing”  
> 🐙 **GitHub**: [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image)

---

如有问题，请提交 Issue 或联系维护者。  
Happy editing! 🎨✨
