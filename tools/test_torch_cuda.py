import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.init()
print("ok", torch.cuda.is_available(), torch.cuda.get_device_name())

torch.cuda.init()
print(f"CUDA available: {torch.cuda.is_available()}")