import torch

g = torch.load("/home/taekhyun/workspace-FL/simserver/ckpt/global_v0_20250825_015414.ckpt", map_location="cpu")
c = torch.load("/home/taekhyun/workspace-FL/simserver/ckpt/sat0_fromg0_round0_ep0_20250825_015504.ckpt", map_location="cpu")

def sd(x):
    return x.get("state_dict", x.get("model", x))

sg, sc = sd(g), sd(c)

diff_max = 0.0
for k in sg:
    if "running_" in k:  # BN 통계는 스킵
        continue
    if k in sc and sg[k].shape == sc[k].shape:
        diff_max = max(diff_max, (sc[k]-sg[k]).abs().max().item())
print("max|client - global| =", diff_max)
