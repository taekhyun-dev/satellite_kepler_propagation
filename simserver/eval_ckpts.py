# simserver/eval_ckpts.py
from __future__ import annotations
import argparse, os, re, time, hashlib, glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict

# -------- CLI --------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate existing global ckpts and write metrics CSV/XLSX.")
    p.add_argument("--ckpt-dir", type=str, default="ckpt", help="Directory containing global_v*.ckpt")
    p.add_argument("--pattern", type=str, default="global_v*.ckpt", help="Glob pattern for ckpt files")
    p.add_argument("--splits", nargs="+", default=["val","test"], choices=["val","test"],
                   help="Which splits to evaluate")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda[:idx]")
    p.add_argument("--batch-size", type=int, default=int(os.getenv("FL_EVAL_BS","1024")))
    p.add_argument("--num-classes", type=int, default=int(os.getenv("FL_NUM_CLASSES","10")))
    p.add_argument("--img-size", type=int, default=int(os.getenv("FL_IMG_SIZE","32")))
    p.add_argument("--bn-calib", action="store_true", help="Run BN recalibration before eval")
    p.add_argument("--bn-batches", type=int, default=int(os.getenv("FL_BN_CALIB_BATCHES","20")))
    p.add_argument("--bn-bs", type=int, default=int(os.getenv("FL_EVAL_BS","256")))
    p.add_argument("--metrics-dir", type=str, default="metrics/global", help="Output metrics directory")
    p.add_argument("--xlsx", action="store_true", help="Also write XLSX (requires pandas+openpyxl)")
    p.add_argument("--save-recal", action="store_true", help="Save recalibrated ckpt as *_bnrecal.ckpt")
    p.add_argument("--skip-dupes", action="store_true",
                   help="Skip evaluation for checkpoints with identical parameter fingerprints")
    p.add_argument("--arch", type=str, default="auto",
                   help="Model arch: auto | resnet18 | mobilenet_v3_small | mobilenet_v3_large | "
                        "shufflenet_v2_x1_0 | efficientnet_b0 | vgg11_bn | convnext_tiny | proj (uses project model if available)")
    return p.parse_args()

# -------- utils: CSV/XLSX --------
GLOBAL_METRICS_HEADER = [
    "timestamp", "version", "split", "num_samples",
    "loss", "acc", "f1_macro", "madds_M", "flops_M", "latency_ms",
    "ckpt_path",
]

def _append_csv_row(path: Path, header: List[str], row: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write(",".join(header) + "\n")
        def esc(x):
            s = "" if x is None else str(x)
            if ("," in s) or ("\"" in s):
                s = "\"" + s.replace("\"", "\"\"") + "\""
            return s
        f.write(",".join(esc(x) for x in row) + "\n")

def _append_excel_row(xlsx_path: Path, header: List[str], row: List[Any], sheet: str = "global"):
    try:
        import pandas as pd, zipfile
    except Exception:
        return
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if xlsx_path.exists() and not zipfile.is_zipfile(xlsx_path):
            try: xlsx_path.unlink()
            except Exception: pass
        if xlsx_path.exists():
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
            except Exception:
                df = pd.DataFrame(columns=header)
        else:
            df = pd.DataFrame(columns=header)
        s = pd.Series(row, index=header)
        df = pd.concat([df, s.to_frame().T], ignore_index=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as w:
            df.to_excel(w, sheet_name=sheet, index=False)
    except Exception as e:
        print(f"[WARN] XLSX write failed: {e}")

# -------- dataset (registry → fallback) --------
def get_eval_dataset(split: str, img_size: int) -> Optional[Any]:
    """
    split in {"val","test"}.
    1) try data.DATA_REGISTRY.get_validation/test_dataset()
    2) try data.get_validation_dataset / get_test_dataset
    3) fallback: torchvision CIFAR-10(test)
    """
    # 1) registry method
    try:
        import importlib
        data_mod = importlib.import_module("data")
        DATA_REGISTRY = getattr(data_mod, "DATA_REGISTRY", None)
        if DATA_REGISTRY is not None:
            meth = getattr(DATA_REGISTRY, f"get_{'validation' if split=='val' else 'test'}_dataset", None)
            if callable(meth):
                ds = meth()
                if ds is not None:
                    print(f"[EVAL] using DATA_REGISTRY.{meth.__name__}() for split={split}")
                    return ds
    except Exception as e:
        print(f"[INFO] registry dataset failed: {e}")

    # 2) module-level helper
    try:
        import importlib
        data_mod = importlib.import_module("data")
        fn_name = "get_validation_dataset" if split == "val" else "get_test_dataset"
        fn = getattr(data_mod, fn_name, None)
        if callable(fn):
            ds = fn()
            if ds is not None:
                print(f"[EVAL] using data.{fn_name}() for split={split}")
                return ds
    except Exception as e:
        print(f"[INFO] data.{split} fallback failed: {e}")

    # 3) fallback CIFAR-10(test)
    try:
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        mean = tuple(map(float, os.getenv("FL_NORM_MEAN", "0.4914,0.4822,0.4465").split(",")))
        std  = tuple(map(float, os.getenv("FL_NORM_STD",  "0.2470,0.2435,0.2616").split(",")))
        tfm = transforms.compose.Compose if False else transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds = CIFAR10(root=str(Path.cwd()/ "data"), train=False, download=True, transform=tfm)
        print(f"[EVAL] fallback CIFAR10(test) | size={img_size}, norm=mean{mean},std{std}")
        return ds
    except Exception as e:
        print(f"[WARN] torchvision CIFAR10 fallback failed: {e}")
        return None

# -------- model utils --------
def _set_num_classes(model, arch: str, num_classes: int):
    import torch.nn as nn
    a = arch.lower()
    if a.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif a.startswith("mobilenet_v3"):
        # classifier is Sequential, last layer is Linear
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif a.startswith("shufflenet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif a.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif a.startswith("vgg"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif a.startswith("convnext"):
        # convnext_tiny.classifier = [LayerNorm, Flatten, Linear]
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        # best effort: try to find a terminal Linear named 'head' or 'classifier'
        for name in ["head", "classifier", "fc"]:
            if hasattr(model, name):
                layer = getattr(model, name)
                if hasattr(layer, "in_features"):
                    setattr(model, name, type(layer)(layer.in_features, num_classes))  # type: ignore
                    break

def _build_torchvision(arch: str, num_classes: int):
    import torchvision.models as tvm
    arch = arch.lower()
    # instantiate with random weights (no pretrained)
    ctor = {
        "resnet18": tvm.resnet18,
        "mobilenet_v3_small": tvm.mobilenet_v3_small,
        "mobilenet_v3_large": tvm.mobilenet_v3_large,
        "shufflenet_v2_x1_0": tvm.shufflenet_v2_x1_0,
        "efficientnet_b0": tvm.efficientnet_b0,
        "vgg11_bn": tvm.vgg11_bn,
        "convnext_tiny": tvm.convnext_tiny,
    }.get(arch, None)
    if ctor is None:
        raise ValueError(f"Unknown torchvision arch: {arch}")
    m = ctor(weights=None)
    _set_num_classes(m, arch, num_classes)
    return m

def _build_project_model(num_classes: int):
    """
    If your repo exposes a project model builder (e.g., models.build_model / get_model),
    try those first so ckpts from training match exactly.
    """
    import importlib
    for modname in ["models", "model", "net", "arch"]:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for fn in ["build_model", "get_model", "create_model"]:
            try:
                f = getattr(mod, fn, None)
                if callable(f):
                    m = f(num_classes=num_classes)
                    print(f"[EVAL] using project model: {modname}.{fn}(num_classes={num_classes})")
                    return m
            except Exception:
                pass
    return None

def _unwrap_state_dict(sd_like: Any) -> Dict[str, Any]:
    if isinstance(sd_like, dict):
        for k in ("state_dict", "model", "net", "weights"):
            v = sd_like.get(k)
            if isinstance(v, dict):
                return v
    return sd_like

def _coverage_score(sd_src: Dict[str, Any], sd_dst: Dict[str, Any]) -> Tuple[float,int,int,List[str]]:
    """
    Returns: (match_ratio_by_numel, matched_keys, total_keys, top_missing_keys)
    Ignore obvious heads when scoring to be robust to different num_classes.
    """
    import torch
    ignore_tokens = (".fc.", ".classifier", ".head.")
    matched_numel = 0
    total_numel = 0
    for k, v in sd_src.items():
        if not hasattr(v, "shape"):  # skip non-tensors
            continue
        if any(t in k for t in ignore_tokens):
            continue
        total_numel += int(v.numel())
        if k in sd_dst and hasattr(sd_dst[k], "shape") and tuple(sd_dst[k].shape) == tuple(v.shape):
            matched_numel += int(v.numel())
    # collect some missing examples (for debug print)
    missing = []
    for k, v in sd_src.items():
        if not hasattr(v, "shape"):
            continue
        if k not in sd_dst:
            missing.append(k)
    return (matched_numel / max(1, total_numel), matched_numel, total_numel, missing[:8])

def _auto_pick_arch(state_dict_like: dict, num_classes: int, device: str):
    """
    Try project model first, else score several torchvision arches and pick the best.
    """
    import torch
    sd_src = _unwrap_state_dict(state_dict_like)

    # 0) project-defined model
    proj = _build_project_model(num_classes)
    if proj is not None:
        cov, m, t, miss = _coverage_score(sd_src, proj.state_dict())
        print(f"[EVAL] project model coverage: {cov*100:.1f}% (matched {m}/{t} numel)")
        return proj, "proj", cov

    # 1) torchvision candidates
    candidates = [
        "resnet18",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "shufflenet_v2_x1_0",
        "efficientnet_b0",
        "vgg11_bn",
        "convnext_tiny",
    ]
    best = None
    best_cov = -1.0
    best_name = None
    for name in candidates:
        try:
            m = _build_torchvision(name, num_classes)
        except Exception:
            continue
        cov, mnum, tnum, miss = _coverage_score(sd_src, m.state_dict())
        print(f"[EVAL] try {name:<20} coverage={cov*100:5.1f}% (matched {mnum}/{tnum} numel)")
        if cov > best_cov:
            best = m
            best_cov = cov
            best_name = name
    if best is None:
        raise RuntimeError("Failed to instantiate any candidate model.")
    return best, best_name, best_cov

def build_model(arch: str, num_classes: int, state_dict_like: Optional[dict] = None, device: str = "cpu"):
    if arch == "proj":
        m = _build_project_model(num_classes)
        if m is None:
            raise ValueError("proj arch requested but no project model builder found.")
        return m, "proj", 0.0
    if arch == "auto":
        if state_dict_like is None:
            raise ValueError("auto arch selection needs a state_dict to score.")
        return _auto_pick_arch(state_dict_like, num_classes, device)
    # torchvision explicit
    m = _build_torchvision(arch, num_classes)
    return m, arch, None

# -------- BN calib / evaluation --------
def bn_recalibrate(state_dict_like: dict, dataset, batches: int = 20, bs: int = 256,
                   device="cpu", arch="auto", num_classes: int = 10) -> dict:
    import torch
    from torch.utils.data import DataLoader
    model, used_arch, cov = build_model(arch, num_classes, state_dict_like, device)
    sd = _unwrap_state_dict(state_dict_like)
    _ = model.load_state_dict(sd, strict=False)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(dev).train()
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=(dev.type=="cuda"))
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(dev)
            _ = model(x)
            if i+1 >= batches: break
    model.eval()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def evaluate_state_dict(state_dict_like: dict, dataset, batch_size: int = 512, device: str = "cpu",
                        num_classes: int = 10, arch: str = "auto"):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader

    model, used_arch, cov = build_model(arch, num_classes, state_dict_like, device)
    sd = _unwrap_state_dict(state_dict_like)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if cov is not None:
        print(f"[EVAL] picked arch={used_arch}, coverage={cov*100:.1f}%")
    if missing:
        print(f"[INFO] missing keys (show up to 10): {list(missing)[:10]}")
    if unexpected:
        print(f"[INFO] unexpected keys (show up to 10): {list(unexpected)[:10]}")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(dev).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(dev.type=="cuda"))
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev, dtype=torch.long)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            pred = logits.argmax(1)
            total_correct += int((pred == y).sum().item())
            total += y.size(0)
            for c in range(num_classes):
                mask = (y == c)
                if mask.any():
                    binc = torch.bincount(pred[mask].detach().cpu(), minlength=num_classes)
                    conf[c] += binc.to(conf.dtype)

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    tp = conf.diag().to(torch.float32)
    fp = conf.sum(0).to(torch.float32) - tp
    fn = conf.sum(1).to(torch.float32) - tp
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1_macro  = float((2 * precision * recall / (precision + recall + 1e-12)).mean().item() * 100.0)

    # latency (bs=1)
    latency_ms: Optional[float] = None
    try:
        try:
            sample_shape = dataset[0][0].shape
        except Exception:
            sample_shape = (3, 32, 32)
        x1 = torch.randn(1, *sample_shape, device=dev)
        with torch.no_grad():
            for _ in range(5): _ = model(x1)
            if isinstance(dev, torch.device) and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(20): _ = model(x1)
            if isinstance(dev, torch.device) and dev.type == "cuda":
                torch.cuda.synchronize(dev)
            latency_ms = (time.perf_counter() - t0) / 20.0 * 1000.0
    except Exception:
        latency_ms = None

    # FLOPs/MAdds (best-effort)
    madds_M, flops_M = None, None
    try:
        from thop import profile
        macs, _params = profile(model, inputs=(x1,), verbose=False)
        madds_M = macs / 1e6
        flops_M = (2 * macs) / 1e6
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            model_cpu = model.to("cpu")
            x_cpu = x1.detach().to("cpu")
            flop_an = FlopCountAnalysis(model_cpu, x_cpu)
            flops = float(flop_an.total())
            flops_M = flops / 1e6
            madds_M = flops_M / 2.0
            model.to(dev)
        except Exception:
            pass

    return avg_loss, acc, total, f1_macro, madds_M, flops_M, (float(latency_ms) if latency_ms is not None else None)

# -------- ckpt listing/sorting & fingerprint --------
_ckpt_re = re.compile(r"global_v(\d+)(?:_(\d{8}_\d{6}))?\.ckpt$")

def _list_with_fallbacks(ckpt_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(ckpt_dir.glob(pattern))
    if paths: return paths
    script_dir = Path(__file__).resolve().parent
    alt = sorted((script_dir / "ckpt").glob(pattern))
    if alt:
        print(f"[INFO] no files under {ckpt_dir}, using {script_dir/'ckpt'}")
        return alt
    guessed = sorted(glob.glob(str((script_dir / "ckpt" / pattern))))
    return [Path(p) for p in guessed]

def list_ckpts(ckpt_dir: Path, pattern: str) -> List[Tuple[int, str, Path]]:
    paths = _list_with_fallbacks(ckpt_dir, pattern)
    out = []
    for p in paths:
        m = _ckpt_re.match(p.name)
        if not m: continue
        ver = int(m.group(1)); ts = m.group(2) or ""
        out.append((ver, ts, p))
    out.sort(key=lambda x: (x[0], x[1]))
    return out

def _unwrap_for_fp(sd_like: Any) -> Dict[str, Any]:
    sd = _unwrap_state_dict(sd_like)
    return {k:v for k,v in sd.items() if hasattr(v, "detach")}

def _tensor_fingerprint(sd_like: Any) -> str:
    import torch
    sd = _unwrap_for_fp(sd_like)
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

# -------- main --------
def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    metrics_dir = Path(args.metrics_dir).resolve()
    csv_path = metrics_dir / "global_metrics.csv"
    xlsx_path = metrics_dir / "global_metrics.xlsx"

    # datasets
    ds_by_split = {}
    for split in args.splits:
        ds = get_eval_dataset(split, img_size=args.img_size)
        if ds is None:
            print(f"[WARN] No dataset for split={split}; skip this split.")
        ds_by_split[split] = ds

    ckpts = list_ckpts(ckpt_dir, args.pattern)
    if not ckpts:
        print(f"[INFO] No ckpts matching {ckpt_dir}/{args.pattern}")
        return

    import torch
    seen_fp: Dict[str, Path] = {}
    cached_metrics: Dict[Tuple[str, str], Tuple[float, float, int, float, Optional[float], Optional[float], Optional[float]]] = {}

    print(f"[INFO] Found {len(ckpts)} checkpoint files.")
    for ver, ts, path in ckpts:
        print(f"\n[INFO] Evaluating ckpt v{ver} -> {path.name}")
        try:
            sd_raw = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[ERROR] load failed: {e}")
            continue

        # fingerprint
        try:
            fp = _tensor_fingerprint(sd_raw)
            dup_of = seen_fp.get(fp)
            if dup_of is not None:
                print(f"[INFO] fingerprint match (sha256[:16]={fp[:16]}) with {dup_of.name}")
                if args.skip_dupes:
                    print("[INFO] --skip-dupes set: will reuse metrics if cached; else compute once.")
                else:
                    print("[INFO] duplicate weights but will recompute metrics (skip-dupes disabled).")
            else:
                seen_fp[fp] = path
                print(f"[INFO] fingerprint sha256[:16]={fp[:16]}")
        except Exception as e:
            fp = None
            print(f"[WARN] fingerprint failed: {e}")

        # optional BN recalib (uses first available split dataset)
        if args.bn_calib:
            calib_ds = ds_by_split.get("val") or ds_by_split.get("test")
            if calib_ds is None:
                print("[WARN] BN recalib requested but no dataset available; skip recalib.")
            else:
                try:
                    sd_raw = bn_recalibrate(sd_raw, calib_ds, batches=args.bn_batches, bs=args.bn_bs,
                                            device=args.device, arch=args.arch, num_classes=args.num_classes)
                    if args.save_recal:
                        outp = path.with_name(path.stem + "_bnrecal.ckpt")
                        try:
                            torch.save(sd_raw, outp)
                            print(f"[INFO] saved recalibrated ckpt -> {outp}")
                        except Exception as e:
                            print(f"[WARN] failed to save recalibrated ckpt: {e}")
                    # new fp
                    try:
                        fp = _tensor_fingerprint(sd_raw)
                        seen_fp[fp] = path
                        print(f"[INFO] new fingerprint after BN recalib sha256[:16]={fp[:16]}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[WARN] BN recalibration failed: {e}")

        # evaluate per split
        for split in args.splits:
            ds = ds_by_split.get(split)
            if ds is None: continue

            if fp is not None and fp in seen_fp and seen_fp[fp] is not None and seen_fp[fp] != path and args.skip_dupes:
                cached = cached_metrics.get((fp, split))
                if cached is not None:
                    g_loss, g_acc, n, g_f1, g_madds, g_flops, g_lat = cached
                    print(f"[METRICS] v{ver} {split}: (reused) acc={g_acc:.2f}% loss={g_loss:.4f} (n={n})")
                    row = [
                        datetime.now().isoformat(), ver, split, n,
                        f"{g_loss:.6f}", f"{g_acc:.2f}",
                        (f"{g_f1:.2f}" if g_f1 is not None else ""),
                        (f"{g_madds:.2f}" if g_madds is not None else ""),
                        (f"{g_flops:.2f}" if g_flops is not None else ""),
                        (f"{g_lat:.3f}" if g_lat is not None else ""),
                        str(path),
                    ]
                    _append_csv_row(csv_path, GLOBAL_METRICS_HEADER, row)
                    if args.xlsx: _append_excel_row(xlsx_path, GLOBAL_METRICS_HEADER, row, sheet="global")
                    continue
                else:
                    print(f"[INFO] duplicate weights for split={split}, but no cached metrics yet → compute once.")

            try:
                g_loss, g_acc, n, g_f1, g_madds, g_flops, g_lat = evaluate_state_dict(
                    sd_raw, ds, batch_size=args.batch_size, device=args.device,
                    num_classes=args.num_classes, arch=args.arch
                )
                print(f"[METRICS] v{ver} {split}: acc={g_acc:.2f}% loss={g_loss:.4f} (n={n})")
                row = [
                    datetime.now().isoformat(),
                    ver,
                    split,
                    n,
                    f"{g_loss:.6f}",
                    f"{g_acc:.2f}",
                    (f"{g_f1:.2f}" if g_f1 is not None else ""),
                    (f"{g_madds:.2f}" if g_madds is not None else ""),
                    (f"{g_flops:.2f}" if g_flops is not None else ""),
                    (f"{g_lat:.3f}" if g_lat is not None else ""),
                    str(path),
                ]
                _append_csv_row(csv_path, GLOBAL_METRICS_HEADER, row)
                if args.xlsx: _append_excel_row(xlsx_path, GLOBAL_METRICS_HEADER, row, sheet="global")
                if fp is not None:
                    cached_metrics[(fp, split)] = (g_loss, g_acc, n, g_f1, g_madds, g_flops, g_lat)
            except Exception as e:
                print(f"[ERROR] eval failed for v{ver} {split}: {e}")

if __name__ == "__main__":
    main()
