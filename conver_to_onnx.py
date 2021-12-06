import torch
import torch.nn
import numpy as np
from LPR import *

ckpt = './save_ckpt/LPR-Epoch-583-Loss-1.1.pth'
model_path = './zj-LPR.onnx'
net = LPR(num_chars=68)
torch.load(ckpt)
net.eval()
net.to("cuda")
input_data = torch.randn(1, 3, 24, 94).to("cuda")
torch.onnx.export(net, input_data, model_path, verbose=True,
                  input_names=['input'], output_names=['logits'],
                  keep_initializers_as_inputs=True)