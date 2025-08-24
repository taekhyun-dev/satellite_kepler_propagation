# simserver/ckpt_expect_fedavg.py
import torch, math

def sd(ckpt): return ckpt.get('state_dict', ckpt.get('model', ckpt))

def merge(global_path, client_paths, server_lr=0.1):
    G = sd(torch.load(global_path, map_location='cpu'))
    Cs = [sd(torch.load(p, map_location='cpu')) for p in client_paths]
    w = 1.0 / len(Cs)
    new = {k: v.clone() for k,v in G.items()}
    with torch.no_grad():
        for k in new:
            if "running_" in k or "num_batches_tracked" in k:
                continue
            if all(k in C and C[k].shape == new[k].shape for C in Cs):
                avg = sum(w * C[k].float() for C in Cs)
                new[k].copy_((1-server_lr)*new[k].float() + server_lr*avg)
    return G, new

def rel(G,N):
    num=den=0.0
    for k in G:
        if "running_" in k or "num_batches_tracked" in k: continue
        if k in N:
            d=(N[k]-G[k]).float().view(-1); num+=float(d.dot(d))
            x=G[k].float().view(-1); den+=float(x.dot(x))
    return math.sqrt(num)/(math.sqrt(den)+1e-12)

G,N = merge(
    "simserver/ckpt/global_v0_20250825_015414.ckpt",
    [
      "simserver/ckpt/sat0_fromg0_round0_ep0_20250825_015504.ckpt",
      "simserver/ckpt/sat54_fromg0_round1_ep0_20250825_015709.ckpt",
      "simserver/ckpt/sat60_fromg0_round1_ep0_20250825_015723.ckpt",
    ],
    server_lr=0.08
)
print("expected relΔ ≈", rel(G,N))
