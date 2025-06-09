#!/usr/bin/env python3
"""
remote_infer.py

Loads the OpenVLA-7B model in the `openvla` conda environment,
reads an image from disk, applies a text instruction prompt, and
outputs a 7-DoF action vector as JSON on stdout.

Usage:
    python remote_infer.py \
        --image /scratch/kvinod/mujoco_client/img1.jpg \
        --prompt "pick up the red block and place it on the green block" \
        [--unnorm_key YOUR_DATASET_KEY]
"""

import argparse
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    # 1) Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run OpenVLA inference on a single image + prompt"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Full path to the RGB image file on disk"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text instruction prompt for the model"
    )
    parser.add_argument(
        "--unnorm_key",
        default="nyu_franka_play_dataset_converted_externally_to_rlds",
        help=(
            "Normalization key for unnormalizing actions. "
            "Choose from the model’s available dataset keys if you trained on multiple."
        )
    )
    args = parser.parse_args()

    # 2) Load and preprocess the image
    pil_img = Image.open(args.image).convert("RGB")

    # 3) Initialize the processor and model
    #    trust_remote_code=True is required for OpenVLA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type=="cuda" else torch.float32

    proc = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    # 4) Tokenize the prompt + image
    #    Format: "In: <instruction>\nOut:"
    full_prompt = f"In: {args.prompt}\nOut:"
    inputs = proc(full_prompt, pil_img, return_tensors="pt").to(device, dtype=dtype)

    # 5) Run inference: generate action vector
    #    unnorm_key selects the stats for de-normalization
    outputs = model.predict_action(
        **inputs,
        unnorm_key=args.unnorm_key,
        do_sample=False
    )

    # 6) Convert to Python list and print as JSON
    action_vec = outputs.tolist()  # tensor → nested list
    print(json.dumps(action_vec))

if __name__ == "__main__":
    main()
