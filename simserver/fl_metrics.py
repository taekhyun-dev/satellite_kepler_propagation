#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FL 로그(csv)에서 학습 지표 그래프 생성 스크립트

사용 예)
  python plot_fl_metrics.py --csv /path/to/metrics.csv --out ./plots
  python plot_fl_metrics.py --csv fl.csv --out ./plots --sat-id 0 --show
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(csv_path: str, sat_id: int | None = None) -> pd.DataFrame:
    """CSV 로드 및 전처리(정렬/지표 파생)."""
    df = pd.read_csv(csv_path)
    # 필수 컬럼 체크
    required = ["timestamp", "sat_id", "round", "epoch", "num_samples", "loss", "acc", "ckpt_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼 누락: {missing}")

    # 타입 캐스팅
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sat_id"] = pd.to_numeric(df["sat_id"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")

    # sat_id 필터
    if sat_id is not None:
        df = df[df["sat_id"] == sat_id]

    # 시간 정렬 및 전역 스텝/누적 최고 정확도 계산
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    df["step"] = range(1, len(df) + 1)
    df["best_acc_so_far"] = df["acc"].cummax()

    # 라운드 내 스텝 번호(라운드별 epoch 연속선 그릴 때 유용)
    df["round_step"] = df.groupby("round").cumcount() + 1

    return df


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss_vs_step(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.plot(df["step"], df["loss"], marker="o")
    plt.xlabel("step")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Step")
    plt.tight_layout()
    out = outdir / "loss_vs_step.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_acc_vs_step(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.plot(df["step"], df["acc"], marker="o", label="Acc (%)")
    plt.plot(df["step"], df["best_acc_so_far"], linestyle="--", label="Best so far (%)")
    plt.xlabel("step")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Step")
    plt.legend()
    plt.tight_layout()
    out = outdir / "acc_vs_step.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_acc_over_time(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.plot(df["timestamp"], df["acc"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Time")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    out = outdir / "acc_over_time.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_loss_over_time(df: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    plt.plot(df["timestamp"], df["loss"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    out = outdir / "loss_over_time.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_within_round(df: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    # Loss within each round
    plt.figure()
    for r, sub in df.groupby("round"):
        plt.plot(sub["round_step"], sub["loss"], marker="o", label=f"round {int(r)}")
    plt.xlabel("Step within round")
    plt.ylabel("Loss")
    plt.title("Loss within Each Round")
    plt.legend()
    plt.tight_layout()
    out1 = outdir / "loss_within_round.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # Acc within each round
    plt.figure()
    for r, sub in df.groupby("round"):
        plt.plot(sub["round_step"], sub["acc"], marker="o", label=f"round {int(r)}")
    plt.xlabel("Step within round")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy within Each Round")
    plt.legend()
    plt.tight_layout()
    out2 = outdir / "acc_within_round.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    return out1, out2


def main():
    parser = argparse.ArgumentParser(description="FL CSV에서 학습 지표 그래프 생성")
    parser.add_argument("--csv", required=True, help="입력 CSV 경로")
    parser.add_argument("--out", default="./plots", help="그래프 저장 폴더 (기본 ./plots)")
    parser.add_argument("--sat-id", type=int, default=None, help="특정 sat_id만 필터 (생략 시 전체)")
    parser.add_argument("--show", action="store_true", help="저장 후 그래프 창 표시")

    args = parser.parse_args()

    outdir = ensure_outdir(args.out)
    df = load_and_prepare(args.csv, sat_id=args.sat_id)

    if df.empty:
        raise SystemExit("조건에 맞는 데이터가 없습니다. (필터가 너무 좁은지 확인하세요)")

    paths = []
    paths.append(plot_loss_vs_step(df, outdir))
    paths.append(plot_acc_vs_step(df, outdir))
    paths.append(plot_acc_over_time(df, outdir))
    paths.append(plot_loss_over_time(df, outdir))
    p1, p2 = plot_within_round(df, outdir)
    paths.extend([p1, p2])

    print("\n[Saved plots]")
    for p in paths:
        print(str(p))

    if args.show:
        # show는 마지막 그림만 뜨기 쉬우니, 주요 1~2개 다시 렌더링해서 표시
        # (필요 시 커스텀 가능)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["step"], df["loss"], marker="o")
        plt.xlabel("step"); plt.ylabel("Loss"); plt.title("Training Loss vs Step")
        plt.tight_layout()

        plt.figure()
        plt.plot(df["step"], df["acc"], marker="o", label="Acc (%)")
        plt.plot(df["step"], df["best_acc_so_far"], linestyle="--", label="Best so far (%)")
        plt.xlabel("step"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs Step")
        plt.legend(); plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
