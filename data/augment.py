"""Quora-train 데이터 증강 — 비교메소드(데이터증강)용. 학습=quora-only 원칙 유지.
  swap    : (s2, s1, label) 순서 뒤집기 -> 순서 대칭성
  bt      : en->de->en back-translation (label=1 페어의 sentence2) -> 패러프레이즈 다양성
  hardneg : TF-IDF로 어휘는 유사하나 의미가 다른 페어 생성 (label=0) -> 어려운 음성
  merge   : quora-train + 증강분 -> data/quora_aug_train.csv

실행:
  python data/augment.py --steps swap bt hardneg merge
  ※ bt는 GPU 권장. hardneg는 quora 283k 코퍼스라 메모리·시간 소요.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA = Path(__file__).resolve().parent          # data/
SRC = DATA / "quora-train.csv"
WS = re.compile(r"\s+")


def _norm(s):
  return WS.sub(" ", s.strip().lower()) if isinstance(s, str) else ""


def _load(path):
  return pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype={"id": str}, keep_default_na=False)


def _save(df, name):
  df.to_csv(DATA / name, sep="\t", index=False, encoding="utf-8")
  print(f"  saved {name:24s} rows={len(df)}")


def augment_swap():
  """모든 quora-train 행을 (s2, s1, label)로 뒤집어 추가 (순서 대칭성)."""
  df = _load(SRC)
  swap = df.copy()
  swap["id"] = swap["id"].apply(lambda i: f"{i}_swap")
  swap["sentence1"], swap["sentence2"] = df["sentence2"].values, df["sentence1"].values
  _save(swap, "quora_train_swap.csv")


def augment_bt(batch_size=64, max_length=128, num_beams=2, device=None):
  """label=1 페어의 sentence2 만 BT -> 새로운 (s1, s2_bt, 1) 페어."""
  import torch
  from transformers import MarianMTModel, MarianTokenizer
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  df = _load(SRC)
  pos = df[df["is_duplicate"].astype(float) == 1.0].reset_index(drop=True)
  print(f"BT 대상(label=1) = {len(pos)} pairs  (device={device})")

  def translate(texts, tok, mdl, desc):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
      chunk = list(texts[i:i + batch_size])
      inp = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
      with torch.no_grad():
        gen = mdl.generate(**inp, max_length=max_length * 2, num_beams=num_beams)
      out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out

  tok_f = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
  mdl_f = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device).eval()
  tok_b = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
  mdl_b = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device).eval()

  src = pos["sentence2"].tolist()
  bt = translate(translate(src, tok_f, mdl_f, "en->de"), tok_b, mdl_b, "de->en")
  aug = pos.copy()
  aug["sentence2"] = bt
  aug["id"] = aug["id"].apply(lambda i: f"{i}_bt")

  def keep(o, n):
    if not isinstance(n, str) or not n.strip():
      return False
    a, b = _norm(o), _norm(n)
    if a == b:
      return False
    return 0.5 <= len(b) / max(len(a), 1) <= 2.0

  mask = [keep(o, n) for o, n in zip(src, bt)]
  kept = aug[mask].reset_index(drop=True)
  print(f"BT 필터 통과 = {len(kept)}/{len(aug)}")
  _save(kept, "quora_train_bt.csv")


def augment_hardneg(topk=20, jaccard_min=0.5, per_anchor=1, batch_size=1000):
  """quora 문장을 TF-IDF 인덱싱 -> 각 s1 anchor에 어휘 유사·다른 페어를 retrieve -> (s1, retrieved, 0)."""
  from sklearn.feature_extraction.text import TfidfVectorizer
  df = _load(SRC)
  pool = pd.concat([
    df[["id", "sentence1"]].rename(columns={"sentence1": "text"}),
    df[["id", "sentence2"]].rename(columns={"sentence2": "text"}),
  ], ignore_index=True)
  pool["tn"] = pool["text"].map(_norm)
  print(f"TF-IDF 인덱싱 (코퍼스 {len(pool)} 문장) ...")
  vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
  Xp = vec.fit_transform(pool["tn"].tolist())
  anc = df["sentence1"].map(_norm).tolist()
  Xa = vec.transform(anc)
  pid, ptext, pnorm = pool["id"].tolist(), pool["text"].tolist(), pool["tn"].tolist()
  XPT = Xp.T.tocsr()
  tok = lambda s: set(s.split())

  rows = []
  for st in tqdm(range(0, Xa.shape[0], batch_size), desc="hardneg retrieval"):
    en = min(st + batch_size, Xa.shape[0])
    sims = (Xa[st:en] @ XPT).toarray()          # 청크만 dense화 (OOM 회피)
    for k, i in enumerate(range(st, en)):
      at = tok(anc[i])
      if not at:
        continue
      pi = df.iloc[i]["id"]
      found = 0
      for j in sims[k].argsort()[::-1][:topk]:
        if pid[j] == pi:
          continue
        ct = tok(pnorm[j])
        if not ct or len(at & ct) / len(at | ct) < jaccard_min:
          continue
        rows.append({"id": f"{pi}_hardneg{found}", "sentence1": df.iloc[i]["sentence1"],
                     "sentence2": ptext[j], "is_duplicate": 0.0})
        found += 1
        if found >= per_anchor:
          break
  print(f"hardneg 생성 = {len(rows)}")
  _save(pd.DataFrame(rows), "quora_train_hardneg.csv")


def build_merged(parts):
  """quora-train(raw) + 선택 증강분 -> quora_aug_train.csv."""
  dfs = [_load(SRC)]
  for p in parts:
    f = DATA / f"quora_train_{p}.csv"
    if not f.exists():
      raise FileNotFoundError(f"{f} 없음 — 먼저 `--steps {p}` 를 실행하세요.")
    dfs.append(_load(f))
  _save(pd.concat(dfs, ignore_index=True), "quora_aug_train.csv")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--steps", nargs="+", choices=["swap", "bt", "hardneg", "merge"],
                  default=["swap", "bt", "hardneg", "merge"], help="실행할 증강 단계")
  ap.add_argument("--merge_parts", nargs="+", choices=["swap", "bt", "hardneg"],
                  default=["swap", "bt", "hardneg"], help="merge 시 합칠 증강분")
  a = ap.parse_args()
  if "swap" in a.steps:
    augment_swap()
  if "bt" in a.steps:
    augment_bt()
  if "hardneg" in a.steps:
    augment_hardneg()
  if "merge" in a.steps:
    build_merged(a.merge_parts)


if __name__ == "__main__":
  main()
