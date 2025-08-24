# simserver/fl/eval.py
import os, time
from typing import Optional

def evaluate_state_dict(state_dict: dict, dataset, batch_size: int = 512, device: str = "cpu",
                        num_classes: int = 10):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from .model import new_model_skeleton

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model = new_model_skeleton(num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev).eval()

    pin = (isinstance(dev, torch.device) and dev.type == "cuda")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss, total_correct, total = 0.0, 0, 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=pin)
            y = y.to(dev, dtype=torch.long, non_blocking=pin)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            pred = logits.argmax(1)
            total_correct += int((pred == y).sum().item())
            total += y.size(0)
            for c in range(num_classes):
                m = (y == c)
                if m.any():
                    conf[c] += torch.bincount(pred[m].detach().cpu(), minlength=num_classes).to(conf.dtype)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    tp = conf.diag().to(torch.float32)
    fp = conf.sum(0).to(torch.float32) - tp
    fn = conf.sum(1).to(torch.float32) - tp
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1_macro  = float((2 * precision * recall / (precision + recall + 1e-12)).mean().item() * 100.0)

    x1 = torch.randn(1, *dataset[0][0].shape, device=dev)
    with torch.no_grad():
        for _ in range(5): _ = model(x1)
        if isinstance(dev, torch.device) and dev.type == "cuda": torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(20): _ = model(x1)
        if isinstance(dev, torch.device) and dev.type == "cuda": torch.cuda.synchronize(dev)
        latency_ms = (time.perf_counter() - t0) / 20.0 * 1000.0

    madds_M, flops_M = None, None
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x1,), verbose=False)
        madds_M = macs / 1e6
        flops_M = (2 * macs) / 1e6
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            model_cpu, x_cpu = model.to("cpu"), x1.detach().to("cpu")
            flop_an = FlopCountAnalysis(model_cpu, x_cpu)
            flops_M = float(flop_an.total()) / 1e6
            madds_M = flops_M / 2.0
        except Exception:
            pass

    return avg_loss, acc, total, f1_macro, madds_M, flops_M, float(latency_ms)

def bn_recalibrate(state_dict: dict, dataset, batches: int = 20, bs: int = 256, device: Optional[str] = None,
                   num_classes: int = 10) -> dict:
    if dataset is None:
        return state_dict
    import torch
    from torch.utils.data import DataLoader
    from .model import new_model_skeleton

    model = new_model_skeleton(num_classes)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        fixed = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in state_dict.items()}
        model.load_state_dict(fixed, strict=False)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev); model.train()

    pin = (isinstance(dev, torch.device) and dev.type == "cuda")
    loader = DataLoader(dataset, batch_size=int(bs), shuffle=True, num_workers=0, pin_memory=pin, drop_last=False)

    seen = 0
    with torch.inference_mode():
        for x, *_ in loader:
            x = x.to(dev, non_blocking=pin)
            _ = model(x)
            seen += 1
            if seen >= int(batches):
                break
    model.eval()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
