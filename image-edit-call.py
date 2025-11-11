import base64

import requests

url = "http://localhost:22001/edit"
prompt = ("把图片里的猫换成狗")

with open("/home/dieu/桌面/03_155836_413.png", "rb") as f:
    files = {"image": f}
    data = {
        "prompt": prompt,
        "num_inference_steps": "50",
        "true_guidance_scale": "4.0",
        "randomize_seed": "true"
    }
    response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    # 保存第一张结果
    img_data = base64.b64decode(result["images"][0])
    with open("output.png", "wb") as f:
        f.write(img_data)
    print("Saved output.png")
else:
    print("Error:", response.text)
