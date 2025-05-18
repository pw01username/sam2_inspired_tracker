import torch
from torch.profiler import profile, record_function, ProfilerActivity

q = torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(8, 16, 512, 64, device="cuda", dtype=torch.float16)

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("sdpa_test"):
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))