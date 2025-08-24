# simserver/ckpt_check_global_delta.py
import glob, torch, math, os

def sd(ckpt):
    return ckpt.get("state_dict", ckpt.get("model", ckpt))

def rel_change(p_old, p_new):
    num = 0.0; den = 0.0
    for k,v0 in p_old.items():
        if "running_" in k or "num_batches_tracked" in k:  # BN 통계 제외
            continue
        if k in p_new and v0.shape == p_new[k].shape and v0.dtype.is_floating_point:
            d = (p_new[k]-v0).float().view(-1)
            num += float(d.dot(d))
            x = v0.float().view(-1)
            den += float(x.dot(x))
    return math.sqrt(num) / (math.sqrt(den)+1e-12)

ckpts = sorted(glob.glob("simserver/ckpt/global_v*_*.ckpt"))
for a,b in zip(ckpts, ckpts[1:]):
    A = sd(torch.load(a, map_location="cpu"))
    B = sd(torch.load(b, map_location="cpu"))
    print(os.path.basename(a), "->", os.path.basename(b), "relΔ =", rel_change(A,B))
