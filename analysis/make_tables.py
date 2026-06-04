"""결과표 생성 — baseline / 비교1(증강) / 비교2(symmetric) / ours(스태커) 를
ID(quora) + OOD(paws, mrpc) 에서 평가. seed 평균±std. metrics.py 사용.

입력: run.sh 가 dump한 predictions/*.csv
  cloze-base-<size>-s<seed>-<dom>.csv   (yes_ab,no_ab,yes_ba,no_ba,prior_yes,prior_no)
  cloze-aug-<size>-s<seed>-<dom>.csv
  align-step3-<size>-s<seed>-<dom>.csv  (yes_logit,no_logit)  -> alignment 멤버
  align-step1-<size>-s<seed>-<dom>.csv  (yes_logit,no_logit)  -> bi-enc 멤버
  external-<dom>.csv                    (e_ab,n_ab,c_ab,e_ba,n_ba,c_ba,cos,jac)

방법 ↔ 신호:
  baseline = cloze margin (yes_ab-no_ab)
  비교1    = cloze-aug margin
  비교2    = cloze symmetric ((yes_ab+yes_ba)-(no_ab+no_ba))
  ours     = 스태커: [cloze, alignment, bi-enc] margin (+외부 5피처) 2-fold cross-fit

사용:
  python analysis/make_tables.py                       # 기본 metric acc,f1
  python analysis/make_tables.py --metrics acc f1 mcc  # metric 선택
  SIZES="gpt2" SEEDS="0" python analysis/make_tables.py
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "predictions")
sys.path.insert(0, ROOT)
import metrics as M  # noqa: E402

norm = lambda x: x.strip().lower()

# 결과표 도메인: name -> (gold tsv, dump dom 태그)
DOMAINS = {
  "quora (ID)":  ("data/quora-dev.csv",  "quora-dev"),
  "paws (OOD)":  ("data/paws_test.csv",  "paws-test"),
  "mrpc (OOD)":  ("data/mrpc_test.csv",  "mrpc-test"),
}


def read_gold(path):
  g = {}
  full = os.path.join(ROOT, path)
  if not os.path.exists(full):
    return g                       # OOD gold 미생성(get_ood_data 전) -> 빈 dict
  with open(full, encoding="utf-8-sig") as f:
    next(f)
    for line in f:
      x = line.rstrip("\n").split("\t")
      if len(x) < 4:
        continue
      g[norm(x[0])] = int(float(x[3]))
  return g


def read_cloze(path):
  m, sym = {}, {}
  for r in csv.DictReader(open(path)):
    i = norm(r["sent_id"])
    yab, nab, yba, nba = (float(r[k]) for k in ("yes_ab", "no_ab", "yes_ba", "no_ba"))
    m[i] = yab - nab
    sym[i] = (yab + yba) - (nab + nba)
  return m, sym


def read_margin(path):
  return {norm(r["sent_id"]): float(r["yes_logit"]) - float(r["no_logit"])
          for r in csv.DictReader(open(path))}


def read_ext(path):
  d = {}
  for r in csv.DictReader(open(path)):
    d[norm(r["sent_id"])] = [float(r[k]) for k in ("e_ab", "e_ba", "c_ab", "c_ba", "cos")]
  return d


def stacker_2fold(cols, y):
  """2-fold cross-fit logreg -> (pred, proba). 평가 fold는 다른 fold 라벨로만 학습."""
  X = np.column_stack(cols).astype(float)
  X = (X - X.mean(0)) / (X.std(0) + 1e-8)
  n = len(y); half = n // 2; idx = np.arange(n)
  pred = np.zeros(n, int); proba = np.zeros(n)
  for tr, te in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    pred[te] = lr.predict(X[te])
    proba[te] = lr.predict_proba(X[te])[:, 1]
  return pred, proba


def eval_one(size, seed, dom, gold):
  """한 (size, seed, dom) -> {방법: (y, pred, score)} or None(파일 누락)."""
  T = f"{size}-s{seed}"
  tag = DOMAINS[dom][1]
  try:
    base_m, base_sym = read_cloze(f"{PR}/cloze-base-{T}-{tag}.csv")
    align = read_margin(f"{PR}/align-step3-{T}-{tag}.csv")
    bienc = read_margin(f"{PR}/align-step1-{T}-{tag}.csv")
    ext = read_ext(f"{PR}/external-{tag}.csv")
  except FileNotFoundError:
    return None
  # cloze-aug(비교1)는 선택: 증강 비교를 아직 안 돌렸으면 그 칸만 비운다.
  try:
    aug_m, _ = read_cloze(f"{PR}/cloze-aug-{T}-{tag}.csv")
  except FileNotFoundError:
    aug_m = None
  ids = [i for i in gold if i in base_m and i in align and i in bienc and i in ext
         and (aug_m is None or i in aug_m)]
  if not ids:
    return None
  y = np.array([gold[i] for i in ids])
  bm = np.array([base_m[i] for i in ids])
  res = {
    "baseline":            (y, (bm > 0).astype(int), bm),
    "비교2: symmetric":     (y, (np.array([base_sym[i] for i in ids]) > 0).astype(int),
                              np.array([base_sym[i] for i in ids])),
  }
  if aug_m is not None:
    res["비교1: 증강"] = (y, (np.array([aug_m[i] for i in ids]) > 0).astype(int),
                          np.array([aug_m[i] for i in ids]))
  cols = ([[base_m[i] for i in ids], [align[i] for i in ids], [bienc[i] for i in ids]]
          + [[ext[i][k] for i in ids] for k in range(5)])
  op, opr = stacker_2fold(cols, y)
  res["ours: 스태커"] = (y, op, opr)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--metrics", nargs="+", default=["acc", "f1"], choices=M.available())
  a = ap.parse_args()
  sizes = os.environ.get("SIZES", "gpt2 gpt2-medium").split()
  seeds = os.environ.get("SEEDS", "0 1 2").split()
  methods = ["baseline", "비교1: 증강", "비교2: symmetric", "ours: 스태커"]

  for size in sizes:
    print("\n" + "=" * 78)
    print(f"  MODEL = {size}   (seed={','.join(seeds)} 평균±std)")
    print("=" * 78)
    header = f"{'method':18s}" + "".join(f"| {d:24s}" for d in DOMAINS)
    print(header)
    print(f"{'':18s}" + "".join(f"| {'  '.join(a.metrics):24s}" for _ in DOMAINS))
    print("-" * len(header))
    for meth in methods:
      cells = []
      for dom in DOMAINS:
        gold = read_gold(DOMAINS[dom][0])
        per_seed = []
        for seed in seeds:
          r = eval_one(size, seed, dom, gold)
          if r is None or meth not in r:
            continue
          y, pred, score = r[meth]
          per_seed.append(M.compute_metrics(y, pred, score, which=a.metrics))
        if not per_seed:
          cells.append(f"{'(no dump)':24s}")
          continue
        txt = "  ".join(
          f"{np.mean([d[m] for d in per_seed]):.3f}±{np.std([d[m] for d in per_seed]):.3f}"
          for m in a.metrics)
        cells.append(f"{txt:24s}")
      print(f"{meth:18s}" + "".join(f"| {c}" for c in cells))


if __name__ == "__main__":
  main()
