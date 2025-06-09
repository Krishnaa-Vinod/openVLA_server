# test_api_client.py

import numpy as np
import requests
from PIL import Image

# 1) Load your sample image (e.g. robot_view.png):
img_path = "img1.jpg"              # <-- put your sample image here
pil = Image.open(img_path).convert("RGB")
image_array = np.array(pil, dtype=np.uint8)

# 2) Define your task prompt
instruction_text = "pick up the green block"  # <-- your sample prompt

# 3) Build the JSON payload
payload = {
    "instruction": instruction_text,
    "image": image_array.tolist()
}

# 4) Send the POST request to your SOL server
SERVER_IP = "127.0.0.1"  # replace with the actual IP of your SOL machine
url = f"http://{SERVER_IP}:8000/predict"

try:
    # Wait up to 60 seconds for a response:
    resp = requests.post(url, json=payload, timeout=(5.0, 60.0))
    resp.raise_for_status()
    result = resp.json()
    print("Predicted action vector:", result["action"])
except Exception as e:
    print("Error calling API:", e)
