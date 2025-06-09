from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Define request schema for automatic JSON parsing
class PredictRequest(BaseModel):
    image: List[List[List[int]]]    # 3D list for image (H x W x 3 pixels)
    instruction: str                # text instruction/prompt
    unnorm_key: Optional[str] = None  # optional key for action un-normalization

# Initialize FastAPI app
app = FastAPI()

# Load model and processor at startup
# Using Hugging Face hub model "openvla/openvla-7b"
# (trust_remote_code allows custom model code with predict_action)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# Select device (GPU if available, else CPU) and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
# Load the OpenVLA model
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

@app.post("/predict")
def predict_action(request: PredictRequest):
    # Convert image list to NumPy array and then to PIL Image
    image_array = np.array(request.image, dtype=np.uint8)
    pil_image = Image.fromarray(image_array).convert("RGB")
    # Format the prompt as per OpenVLA expected input format
    prompt_text = f"In: What action should the robot take to {request.instruction.lower()}?\nOut:"
    # Prepare model inputs (processor will resize/normalize image and tokenize text)
    inputs = processor(prompt_text, pil_image).to(device, dtype=dtype)
    # Run inference to get the 7-DoF action (model returns a NumPy array of shape (7,))
    key = request.unnorm_key or "nyu_franka_play_dataset_converted_externally_to_rlds"
    action_vector = model.predict_action(**inputs, unnorm_key=key, do_sample=False)

    # Convert NumPy output to list for JSON serialization
    action_list = action_vector.tolist()
    return {"action": action_list}
