# -*- coding: utf-8 -*-
import argparse
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch_directml
import timm

dataset_size = 500
batch_size = 32

dataset = np.random.rand(dataset_size, 3, 224, 224).astype(np.float32)


@torch.no_grad()
def inference(model_name: str, model: nn.Module, device_name: str, device: torch.device, iterations: int = 5, precision: Literal["FP32", "FP16"] = "FP32"):
    model = model.eval().to(device)
    if precision == "FP16":
        model = model.half()
    print(f"Model: {model_name}, Device: {device_name}, Precision: {precision}")
    result = []
    for iter in range(iterations):
        t1 = time.time()
        for i in range(0, dataset_size, batch_size):
            batch = torch.from_numpy(dataset[i:i + batch_size]).to(device)
            if precision == "FP16":
                batch = batch.half()
            y: torch.Tensor = model(batch)
            if iter == 0:
                result.append(y.cpu().numpy())
        t2 = time.time()
        ips = dataset_size / (t2 - t1)
        print(f"Iteration {iter}, {ips:.2f} images/s in {t2 - t1:.3f}s.")
    return np.concatenate(result, axis=0)


parser = argparse.ArgumentParser(prog="DirectML Inference Test")

parser.add_argument('--fp32', action='store_true', help='Run FP32 inference')
parser.add_argument('--fp16', action='store_true', help='Run FP16 inference')

args = parser.parse_args()

if args.fp32:
    print("\n========== ResNet50 inference ==========")
    resnet50 = timm.create_model("resnet50", pretrained=True)
    cpu_result = inference("ResNet50", resnet50, "CPU", torch.device("cpu"), iterations=1)
    directml_result = inference("ResNet50", resnet50, "DirectML", torch_directml.device())
    if np.allclose(cpu_result, directml_result, atol=1e-3):
        print("========== PASSED ==========")
    else:
        print("========== FAILED ==========")

    print("\n========== ViT inference ==========")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    cpu_result = inference("ViT", vit, "CPU", torch.device("cpu"), iterations=1)
    directml_result = inference("ViT", vit, "DirectML", torch_directml.device())
    if np.allclose(cpu_result, directml_result, atol=1e-3):
        print("========== PASSED ==========")
    else:
        print("========== FAILED ==========")

if args.fp16:
    print("\n========== FP16 inference ==========")
    inference("ResNet50", resnet50, "DirectML", torch_directml.device(), precision="FP16")
    inference("ViT", vit, "DirectML", torch_directml.device(), precision="FP16")
