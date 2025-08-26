# simserver/fl/debug_norm.py
import os
import torch
from torch.utils.data import DataLoader

# 내부 레지스트리/훅들
from ..dataio.registry import get_training_dataset, get_validation_dataset, get_test_dataset, CIFAR_ROOT
from .training import get_eval_dataset  # 이미 있는 함수 재사용

def _find_normalize_in_transform(tfm):
    """torchvision transform(또는 Compose) 안에서 Normalize(mean,std) 추출."""
    try:
        from torchvision import transforms as T
    except Exception:
        return None

    def _extract(obj):
        if obj is None:
            return None
        # Normalize 단일 객체
        if isinstance(obj, T.Normalize):
            return (tuple(float(x) for x in obj.mean), tuple(float(x) for x in obj.std))
        # Compose / Sequential 등 반복 가능한 컨테이너
        if hasattr(obj, "__iter__"):
            for sub in obj:
                found = _extract(sub)
                if found is not None:
                    return found
        # 객체가 transform 속성을 또 가지는 래퍼(Subset, 커스텀 Dataset)일 수도 있음
        if hasattr(obj, "transform"):
            return _extract(obj.transform)
        return None

    return _extract(tfm)

def _detect_effective_norm_from_dataset(ds):
    """
    데이터셋이 사용 중인 정규화(mean,std)를 '가능한 한' 추출.
    - ds.MEAN/STD 속성 있으면 그것으로 간주(CIFARSubsetDataset 스타일)
    - ds.transform에 Normalize가 있으면 그걸 사용
    - 둘 다 못 찾으면 환경변수(있으면) 또는 None
    """
    base = getattr(ds, "dataset", None)        # ★ torch.utils.data.Subset일 때 원본 접근
    if base is not None:
        ds = base                              # ★ 원본에 붙은 transform/속성 탐색
    # 1) 커스텀 데이터셋이 MEAN/STD를 노출하는 경우
    mean = std = None
    for attr_m, attr_s in (("MEAN", "STD"), ("mean", "std")):
        if hasattr(ds, attr_m) and hasattr(ds, attr_s):
            try:
                m = getattr(ds, attr_m)
                s = getattr(ds, attr_s)
                if torch.is_tensor(m): m = m.tolist()
                if torch.is_tensor(s): s = s.tolist()
                mean = tuple(float(x) for x in m)
                std  = tuple(float(x) for x in s)
                return ("dataset_attr", mean, std)
            except Exception:
                pass

    # 2) torchvision transform에서 Normalize 찾기
    tfm = getattr(ds, "transform", None)
    found = _find_normalize_in_transform(tfm)
    if found is not None:
        return ("transform", found[0], found[1])

    # 3) 환경변수(있으면)
    mean_env = os.getenv("FL_NORM_MEAN")
    std_env  = os.getenv("FL_NORM_STD")
    if mean_env and std_env:
        try:
            mean = tuple(map(float, mean_env.split(",")))
            std  = tuple(map(float, std_env.split(",")))
            return ("env_fallback", mean, std)
        except Exception:
            pass

    return ("unknown", None, None)

@torch.no_grad()
def _channel_stats_from_loader(ds, batch_size=128, max_batches=2):
    """
    배치 몇 개 가져와서 채널별 mean/std를 실제로 계산.
    (정규화 결과가 실제로 0/1 근처인지 확인하는 용도)
    """
    # 텐서 이미지(CHW)만 가정
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    n = 0
    sum_c = None
    sumsq_c = None
    for i, (x, y) in enumerate(dl):
        if not torch.is_tensor(x):
            # PIL/ndarray라면 정규화 전 상태일 수 있음
            return None, None
        # (N,C,H,W) -> 채널 축만 남기고 나머지 축 평균/분산
        x = x.float()
        N, C = x.shape[0], x.shape[1]
        cur = x.view(N, C, -1)
        s = cur.sum(dim=(0,2))              # (C,)
        ss = (cur * cur).sum(dim=(0,2))     # (C,)
        if sum_c is None:
            sum_c = s
            sumsq_c = ss
        else:
            sum_c += s
            sumsq_c += ss
        n += cur.shape[0] * cur.shape[2]
        if i+1 >= max_batches:
            break
    if n == 0 or sum_c is None:
        return None, None
    mean_c = (sum_c / n).cpu().tolist()
    var_c  = (sumsq_c / n - (sum_c / n)**2).clamp_min(0).cpu().tolist()
    std_c  = [float(v**0.5) for v in var_c]
    return tuple(float(m) for m in mean_c), tuple(std_c)

def log_norm_consistency(ctx, sat_id: int, split: str = "val", batch_size: int = 128, atol: float = 1e-3):
    """
    - 학습 데이터셋과 평가 데이터셋이 사용하는 (mean,std)를 추출해서 비교
    - 실제 배치에서 채널별 평균/표준편차도 계산해 비교
    """
    # 1) 데이터셋 가져오기
    train_ds = get_training_dataset(sat_id)
    eval_ds  = get_eval_dataset(ctx, split) or get_test_dataset()

    # 2) '의도된' 정규화 파라미터 비교
    t_src, t_mean, t_std = _detect_effective_norm_from_dataset(train_ds)
    e_src, e_mean, e_std = _detect_effective_norm_from_dataset(eval_ds)

    print("=== Intended (configured) normalization ===")
    print(f"[TRAIN] source={t_src} mean={t_mean} std={t_std}")
    print(f"[EVAL ] source={e_src} mean={e_mean} std={e_std}")

    same_config = (t_mean is not None and e_mean is not None and
                   t_std is not None and e_std is not None and
                   len(t_mean) == len(e_mean) == 3 and len(t_std) == len(e_std) == 3 and
                   all(abs(t_mean[i] - e_mean[i]) <= atol for i in range(3)) and
                   all(abs(t_std[i]  - e_std[i])  <= atol for i in range(3)))

    # 3) 실제 데이터 흐름에서의 통계 비교(채널별)
    t_m, t_s = _channel_stats_from_loader(train_ds, batch_size=batch_size, max_batches=2)
    e_m, e_s = _channel_stats_from_loader(eval_ds , batch_size=batch_size, max_batches=2)

    print("\n=== Observed (on-sample) channel stats ===")
    print(f"[TRAIN] mean≈{t_m} std≈{t_s}")
    print(f"[EVAL ] mean≈{e_m} std≈{e_s}")

    observed_close = False
    if t_m is not None and e_m is not None and t_s is not None and e_s is not None:
        observed_close = (
            all(abs(t_m[i] - e_m[i]) <= 0.15 for i in range(3)) and  # 평균은 대략 0±0.15 정도면 충분
            all(abs(t_s[i] - e_s[i]) <= 0.25 for i in range(3))      # 표준편차는 대략 1±0.25 정도면 충분
        )

    print("\n=== Verdict ===")
    if same_config and observed_close:
        print("✔ 학습과 평가 경로가 같은 정규화를 사용하고 있으며, 실제 배치 통계도 근접합니다.")
    elif same_config and not observed_close:
        print("△ 설정(mean/std)은 동일하지만, 실제 배치 통계가 다소 차이가 있습니다. (배치 구성/리사이즈/증강 확인 필요)")
    elif not same_config and observed_close:
        print("△ 설정(mean/std)은 다르게 보이지만, 실제 배치 통계는 유사합니다. (중간에 추가 정규화/전처리 래퍼가 있을 수 있음)")
    else:
        print("✘ 학습과 평가 정규화가 불일치할 가능성이 높습니다. 환경변수 FL_NORM_MEAN/STD 또는 transform 구성을 확인하세요.")

    return dict(
        same_config=same_config,
        observed_close=observed_close,
        train_cfg=(t_src, t_mean, t_std),
        eval_cfg=(e_src, e_mean, e_std),
        train_obs=(t_m, t_s),
        eval_obs=(e_m, e_s),
    )
