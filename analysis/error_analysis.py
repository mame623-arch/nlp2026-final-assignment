"""오류 분석 — judge_cache(A/B/OK) 기반. quora-dev 기준, 한 모델(--size --seed)의 dump 신호 사용.

  1) 천장 분해   : A/B/OK 비율 + cleanAcc(= A·B 제외 baseline 정확도). 줄일 여지 = 1-cleanAcc
  2) 멤버 상보성 : cloze/alignment/bi-enc/cloze+NLI 의 tag별 정확도·오류 Jaccard·oracle
  3) 표적 회수   : baseline(cloze)이 틀린 OK-tag 케이스를 ours(스태커)가 회수하는 비율 (closed-loop)
  4) κ           : human_verify_50.csv(사람) vs judge(OK↔C 매핑) Cohen's κ — judge 신뢰성 검증

입력: predictions/{cloze-base,align-step3,align-step1}-<size>-s<seed>-quora-dev.csv,
      predictions/external-quora-dev.csv, analysis/judge_cache.csv, analysis/human_verify_50.csv
사용: python analysis/error_analysis.py --size gpt2 --seed 0
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "predictions")
ANA = os.path.join(ROOT, "analysis")
norm = lambda x: x.lower().strip()
TAGS = ["scope", "lexical_gap", "entity", "negation", "quantity", "word_order", "identical", "unrelated"]


def read_gold():
  g = {}
  with open(os.path.join(ROOT, "data", "quora-dev.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter="\t"):
      g[norm(r["id"])] = int(float(r["is_duplicate"]))
  return g


def read_judge():
  verdict, tag = {}, {}
  with open(os.path.join(ANA, "judge_cache.csv")) as f:
    for r in csv.DictReader(f):
      if r["verdict"] in ("A", "B", "OK"):
        verdict[norm(r["id"])] = r["verdict"]
        tag[norm(r["id"])] = (r["tag"] or "untagged")
  return verdict, tag


def read_cloze(fn):
  d = {}
  for r in csv.DictReader(open(os.path.join(PRED, fn))):
    d[norm(r["sent_id"])] = float(r["yes_ab"]) - float(r["no_ab"])
  return d


def read_margin(fn):
  return {norm(r["sent_id"]): float(r["yes_logit"]) - float(r["no_logit"])
          for r in csv.DictReader(open(os.path.join(PRED, fn)))}


def read_ext(fn):
  d = {}
  for r in csv.DictReader(open(os.path.join(PRED, fn))):
    d[norm(r["sent_id"])] = [float(r[k]) for k in ("e_ab", "e_ba", "c_ab", "c_ba", "cos")]
  return d


def cross_fit(cols, y):
  X = np.column_stack(cols).astype(float)
  X = (X - X.mean(0)) / (X.std(0) + 1e-8)
  n = len(y); half = n // 2; idx = np.arange(n)
  pred = np.zeros(n, int)
  for tr, te in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
    pred[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict(X[te])
  return pred


def section_kappa(verdict):
  """human_verify_50 vs judge (judge OK->C)."""
  path = os.path.join(ANA, "human_verify_50.csv")
  if not os.path.exists(path):
    print("  (human_verify_50.csv 없음 — κ 생략)")
    return
  jv = {i: ("C" if v == "OK" else v) for i, v in verdict.items()}
  hum = {}
  for r in csv.DictReader(open(path)):
    v = (r.get("verdict") or "").strip().upper()[:1]
    if v in ("A", "B", "C"):
      hum[norm(r["id"])] = v
  pairs = [(hum[i], jv[i]) for i in hum if i in jv]
  n = len(pairs)
  if n == 0:
    print("  (human 라벨 비어있음 — human_verify_50.csv verdict 칸을 A/B/C로 채우세요)")
    return
  agree = sum(h == j for h, j in pairs)
  po = agree / n
  cats = ["A", "B", "C"]
  ph, pj = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
  pe = sum((ph[c] / n) * (pj[c] / n) for c in cats)
  kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
  print(f"  n={n}  관측일치={po*100:.1f}%  Cohen's κ={kappa:.3f}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--size", default="gpt2")
  ap.add_argument("--seed", default="0")
  a = ap.parse_args()
  T = f"{a.size}-s{a.seed}"
  gold = read_gold()
  verdict, tag = read_judge()
  try:
    cloze_m = read_cloze(f"cloze-base-{T}-quora-dev.csv")
    align = read_margin(f"align-step3-{T}-quora-dev.csv")
    bienc = read_margin(f"align-step1-{T}-quora-dev.csv")
    ext = read_ext("external-quora-dev.csv")
  except FileNotFoundError as e:
    print(f"dump 누락: {e}\n먼저 run.sh 로 quora-dev 신호를 dump하세요.")
    sys.exit(1)

  ids = [i for i in gold if i in cloze_m and i in align and i in bienc and i in ext]
  y = np.array([gold[i] for i in ids]); n = len(ids)
  cloze = {i: int(cloze_m[i] > 0) for i in ids}
  alignp = {i: int(align[i] > 0) for i in ids}
  biencp = {i: int(bienc[i] > 0) for i in ids}
  nli = cross_fit([[cloze_m[i] for i in ids]] + [[ext[i][k] for i in ids] for k in range(5)], y)
  clozeNLI = {ids[k]: int(nli[k]) for k in range(n)}
  op = cross_fit([[cloze_m[i] for i in ids], [align[i] for i in ids], [bienc[i] for i in ids]]
                 + [[ext[i][k] for i in ids] for k in range(5)], y)
  ours = {ids[k]: int(op[k]) for k in range(n)}
  models = {"cloze": cloze, "alignment": alignp, "bi-enc": biencp, "cloze+NLI": clozeNLI}
  print(f"공통 id = {n}  (모델 {T})\n")

  # 1) 천장 분해
  vc = Counter(verdict.get(i, "untagged") for i in ids)
  okids = [i for i in ids if verdict.get(i) == "OK"]
  clean_acc = sum(cloze[i] == gold[i] for i in okids) / len(okids) * 100 if okids else 0
  judged = sum(vc[v] for v in ("A", "B", "OK"))
  print("=== 1) 천장 분해 (judged = {} ) ===".format(judged))
  for v in ("A", "B", "OK"):
    print(f"  {v:3s} {vc[v]:6d}  ({vc[v]/judged*100:.1f}%)")
  print(f"  cleanAcc(baseline, A·B 제외) = {clean_acc:.2f}%   줄일 여지 = {100-clean_acc:.2f}%\n")

  # 2) 멤버 상보성
  print("=== 2) 멤버 상보성 — OK tag별 정확도 (굵게=최강) ===")
  print(f"{'tag':<12}" + "".join(f"{k:>11}" for k in models) + "   n   best")
  for t in TAGS:
    tid = [i for i in okids if tag[i] == t]
    if len(tid) < 15:
      continue
    accs = {k: sum(m[i] == gold[i] for i in tid) / len(tid) for k, m in models.items()}
    best = max(accs, key=accs.get)
    print(f"{t:<12}" + "".join(f"{accs[k]*100:>10.1f} " for k in models) + f"{len(tid):>4}  {best}")
  err = {k: set(i for i in ids if m[i] != gold[i]) for k, m in models.items()}
  print("\n  오류 Jaccard (낮을수록 상보):")
  ks = list(models)
  print("    " + "".join(f"{k:>11}" for k in ks))
  for x in ks:
    print(f"    {x:<8}" + "".join(
      f"{len(err[x]&err[z])/len(err[x]|err[z])*100:>10.1f} " if (err[x] | err[z]) else f"{0:>10.1f} "
      for z in ks))
  oracle = sum(any(m[i] == gold[i] for m in models.values()) for i in ids) / n * 100
  print(f"  oracle(한명이라도 맞힘) = {oracle:.2f}%\n")

  # 3) 표적 회수 (baseline OK-오류를 ours가 회수)
  print("=== 3) 표적 회수 — baseline(cloze) 틀린 OK-케이스를 ours가 회수한 비율 ===")
  print(f"{'tag':<12}{'baseline틀림':>12}{'ours회수':>10}{'회수율':>9}")
  for t in TAGS:
    bad = [i for i in okids if tag[i] == t and cloze[i] != gold[i]]
    if len(bad) < 5:
      continue
    rec = sum(ours[i] == gold[i] for i in bad)
    print(f"{t:<12}{len(bad):>12}{rec:>10}{rec/len(bad)*100:>8.1f}%")
  print("  (A·B는 비가역이라 회수 대상 아님 — 정상이면 불변)\n")

  # 4) κ
  print("=== 4) judge 신뢰성 — human vs judge Cohen's κ ===")
  section_kappa(verdict)


if __name__ == "__main__":
  main()
