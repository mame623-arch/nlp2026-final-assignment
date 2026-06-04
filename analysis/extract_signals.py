"""학습된 모델/외부 모델에서 예측 신호를 추출(dump)해 csv로 저장.
스태커·결과표(make_tables)·분석(error_analysis)의 입력이 된다. 재학습 없음, 추론만.

  --kind cloze    : ParaphraseGPT(cloze) ckpt -> yes_ab,no_ab,yes_ba,no_ba,prior_yes,prior_no
                    (원본/swap 양방향 + 빈입력 prior; symmetric·baseline 신호 모두 도출)
  --kind align    : AlignmentParaphraseGPT(alignment/bi-enc) ckpt -> yes_logit,no_logit
  --kind external : roberta-large-mnli + all-MiniLM-L6-v2 (ckpt 불필요)
                    -> e_ab,n_ab,c_ab,e_ba,n_ba,c_ba,cos,jac

사용:
  python analysis/extract_signals.py --kind cloze    --ckpt <pt> --data data/quora-dev.csv --out predictions/cloze-...-quora.csv
  python analysis/extract_signals.py --kind align    --ckpt <pt> --data <devfile>          --out <csv>
  python analysis/extract_signals.py --kind external              --data <devfile>          --out <csv>
"""
import argparse
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

YES, NO = 8505, 3919   # cloze verbalizer 토큰 id


def read_pairs(path):
  """tsv (id, sentence1, sentence2, [is_duplicate]) -> ids, s1, s2."""
  ids, s1, s2 = [], [], []
  with open(path, encoding='utf-8-sig') as f:
    next(f)
    for line in f:
      x = line.rstrip('\n').split('\t')
      if len(x) < 3:
        continue
      ids.append(x[0].strip())
      s1.append(x[1])
      s2.append(x[2])
  return ids, s1, s2


def extract_cloze(ckpt, data, out):
  from transformers import GPT2Tokenizer
  from paraphrase_detection import ParaphraseGPT
  from datasets import PROMPT_TEMPLATES
  tpl = PROMPT_TEMPLATES['default']                   # 학습·평가와 동일 prompt
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  saved = torch.load(ckpt, map_location=dev, weights_only=False)
  model = ParaphraseGPT(saved['args']).to(dev)
  model.load_state_dict(saved['model'])
  model.eval()
  tok = GPT2Tokenizer.from_pretrained('gpt2')
  tok.pad_token = tok.eos_token
  ids, s1, s2 = read_pairs(data)

  @torch.no_grad()
  def yn(prompts, B=32):
    ys, ns = [], []
    for i in range(0, len(prompts), B):
      e = tok(prompts[i:i + B], return_tensors='pt', padding=True, truncation=True, max_length=128)
      lo = model(e['input_ids'].to(dev), e['attention_mask'].to(dev))
      ys += lo[:, YES].tolist()
      ns += lo[:, NO].tolist()
    return ys, ns

  py, pn = yn([tpl.format(s1='', s2='')])              # 빈입력 prior
  prior_yes, prior_no = py[0], pn[0]
  ab_y, ab_n = yn([tpl.format(s1=a, s2=b) for a, b in zip(s1, s2)])   # 원본
  ba_y, ba_n = yn([tpl.format(s1=b, s2=a) for a, b in zip(s1, s2)])   # swap
  with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sent_id', 'yes_ab', 'no_ab', 'yes_ba', 'no_ba', 'prior_yes', 'prior_no'])
    for k in range(len(ids)):
      w.writerow([ids[k], f"{ab_y[k]:.4f}", f"{ab_n[k]:.4f}", f"{ba_y[k]:.4f}", f"{ba_n[k]:.4f}",
                  f"{prior_yes:.4f}", f"{prior_no:.4f}"])
  print(f"[cloze] {len(ids)} rows -> {out}")


def extract_align(ckpt, data, out):
  from torch.utils.data import DataLoader
  from alignment_paraphrase import AlignmentParaphraseGPT, AlignmentParaphraseDataset
  from datasets import load_paraphrase_data
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  saved = torch.load(ckpt, map_location=dev, weights_only=False)
  model = AlignmentParaphraseGPT(saved['args']).to(dev)
  model.load_state_dict(saved['model'])
  model.eval()
  ds = AlignmentParaphraseDataset(load_paraphrase_data(data))
  loader = DataLoader(ds, batch_size=128, collate_fn=ds.collate_fn, shuffle=False)
  rows = []
  with torch.no_grad():
    for b in loader:
      logits, _, _ = model(b['ids1'].to(dev), b['mask1'].to(dev),
                           b['ids2'].to(dev), b['mask2'].to(dev),
                           b['cloze_ids'].to(dev), b['cloze_mask'].to(dev))
      rows += list(zip(b['sent_ids'], logits[:, 1].cpu().tolist(), logits[:, 0].cpu().tolist()))
  with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sent_id', 'yes_logit', 'no_logit'])
    w.writerows(rows)
  print(f"[align] {len(rows)} rows -> {out}")


def extract_external(data, out):
  import torch.nn.functional as F
  from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("loading roberta-large-mnli ...")
  nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
  nli = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(dev).eval()
  id2l = {i: l.upper() for i, l in nli.config.id2label.items()}
  ENT = [i for i, l in id2l.items() if "ENTAIL" in l][0]
  NEU = [i for i, l in id2l.items() if "NEUTRAL" in l][0]
  CON = [i for i, l in id2l.items() if "CONTRA" in l][0]
  print("loading all-MiniLM-L6-v2 ...")
  emb_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
  emb = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(dev).eval()

  @torch.no_grad()
  def nli_probs(prem, hyp, bs=64):
    o = []
    for i in range(0, len(prem), bs):
      enc = nli_tok(prem[i:i + bs], hyp[i:i + bs], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
      o.append(torch.softmax(nli(**enc).logits, dim=-1).cpu())
    return torch.cat(o)

  @torch.no_grad()
  def embed(texts, bs=128):
    vs = []
    for i in range(0, len(texts), bs):
      enc = emb_tok(texts[i:i + bs], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(dev)
      h = emb(**enc).last_hidden_state
      m = enc["attention_mask"].unsqueeze(-1).float()
      v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
      vs.append(F.normalize(v, dim=-1).cpu())
    return torch.cat(vs)

  def jac(a, b):
    A = set(a.lower().split()); B = set(b.lower().split())
    return len(A & B) / len(A | B) if (A | B) else 0.0

  ids, s1, s2 = read_pairs(data)
  ab = nli_probs(s1, s2)
  ba = nli_probs(s2, s1)
  e1, e2 = embed(s1), embed(s2)
  cos = (e1 * e2).sum(-1).tolist()
  with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["sent_id", "e_ab", "n_ab", "c_ab", "e_ba", "n_ba", "c_ba", "cos", "jac"])
    for k in range(len(ids)):
      w.writerow([ids[k],
                  f"{ab[k, ENT]:.4f}", f"{ab[k, NEU]:.4f}", f"{ab[k, CON]:.4f}",
                  f"{ba[k, ENT]:.4f}", f"{ba[k, NEU]:.4f}", f"{ba[k, CON]:.4f}",
                  f"{cos[k]:.4f}", f"{jac(s1[k], s2[k]):.4f}"])
  print(f"[external] {len(ids)} rows -> {out}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--kind', required=True, choices=['cloze', 'align', 'external'])
  ap.add_argument('--ckpt', default=None, help="cloze/align 에 필요")
  ap.add_argument('--data', required=True, help="평가 tsv (id, s1, s2, [label])")
  ap.add_argument('--out', required=True)
  a = ap.parse_args()
  os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
  if a.kind == 'cloze':
    assert a.ckpt, "--ckpt 필요"
    extract_cloze(a.ckpt, a.data, a.out)
  elif a.kind == 'align':
    assert a.ckpt, "--ckpt 필요"
    extract_align(a.ckpt, a.data, a.out)
  else:
    extract_external(a.data, a.out)


if __name__ == '__main__':
  main()
