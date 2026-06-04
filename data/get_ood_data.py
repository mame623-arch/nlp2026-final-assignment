"""PAWS·MRPC OOD 평가셋을 HuggingFace에서 받아 Quora 포맷(tab-separated)으로 저장.
이 데이터는 학습에 쓰지 않고 OOD(out-of-distribution) 평가 전용이다.

  출력(data/ 직접):
    paws_dev.csv (8000) · paws_test.csv (8000)
    mrpc_traindev.csv (train+dev, 4076) · mrpc_test.csv (1725)

  실행: python data/get_ood_data.py        (필요: datasets, pandas — env.yml 포함)
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parent  # data/


def save_tsv(df, name):
  p = OUT / name
  df.to_csv(p, sep="\t", index=False, encoding="utf-8")
  print(f"  saved {name:20s} rows={len(df):6d}  dup={int(df['is_duplicate'].sum())}")


def to_quora(df, prefix, id_col):
  """HF split → Quora 포맷 (id, sentence1, sentence2, is_duplicate)."""
  df = df.rename(columns={"label": "is_duplicate"})
  df["id"] = df[id_col].apply(lambda i: f"{prefix}_{i}")
  df["is_duplicate"] = df["is_duplicate"].astype(float)
  return df[["id", "sentence1", "sentence2", "is_duplicate"]]


def get_paws():
  print("Downloading PAWS (labeled_final)...")
  try:
    ds = load_dataset("paws", "labeled_final")
  except Exception:
    ds = load_dataset("google-research-datasets/paws", "labeled_final")
  save_tsv(to_quora(ds["validation"].to_pandas(), "paws", "id"), "paws_dev.csv")
  save_tsv(to_quora(ds["test"].to_pandas(), "paws", "id"), "paws_test.csv")


def get_mrpc():
  print("Downloading MRPC (glue/mrpc)...")
  ds = load_dataset("glue", "mrpc")
  train = to_quora(ds["train"].to_pandas(), "mrpc", "idx")
  dev = to_quora(ds["validation"].to_pandas(), "mrpc", "idx")
  test = to_quora(ds["test"].to_pandas(), "mrpc", "idx")
  # MRPC dev(408)는 너무 작아 OOD dev = train+dev 합본 사용
  save_tsv(pd.concat([train, dev], ignore_index=True), "mrpc_traindev.csv")
  save_tsv(test, "mrpc_test.csv")


if __name__ == "__main__":
  get_paws()
  get_mrpc()
  print("\nOOD 평가셋 준비 완료: paws_dev / paws_test / mrpc_traindev / mrpc_test")
