"""alignment(②) 모델의 soft-attention 행렬을 heatmap으로 시각화 — 보고서 정성 증거.
s1의 각 토큰이 s2의 어느 토큰에 정렬되는지(a1 [L1,L2])를 보여준다.
scope 케이스: s2에만 있는 절은 대응 토큰이 없어 그 열의 정렬이 흩어진다.

사용:
  python analysis/visualize_attention.py --ckpt checkpoints/align-step3-gpt2-s0.pt \
      --s1 "How to learn Python" --s2 "How to learn Python for data science" \
      --out figures/attn_scope.png
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, help="alignment(step3) 체크포인트")
  ap.add_argument("--s1", required=True)
  ap.add_argument("--s2", required=True)
  ap.add_argument("--out", default="figures/attn.png")
  a = ap.parse_args()

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from transformers import GPT2Tokenizer
  from alignment_paraphrase import AlignmentParaphraseGPT
  from models.alignment import masked_softmax

  dev = "cuda" if torch.cuda.is_available() else "cpu"
  saved = torch.load(a.ckpt, map_location=dev, weights_only=False)
  model = AlignmentParaphraseGPT(saved["args"]).to(dev)
  model.load_state_dict(saved["model"])
  model.eval()
  tok = GPT2Tokenizer.from_pretrained("gpt2")
  tok.pad_token = tok.eos_token

  def enc(s):
    e = tok(s, return_tensors="pt", truncation=True, max_length=64)
    return e["input_ids"].to(dev), e["attention_mask"].to(dev)

  ids1, m1 = enc(a.s1)
  ids2, m2 = enc(a.s2)
  with torch.no_grad():
    H1 = model._encode_tokens(ids1, m1)
    H2 = model._encode_tokens(ids2, m2)
    P1 = model.align._encode(H1)
    P2 = model.align._encode(H2)
    e = torch.bmm(P1, P2.transpose(1, 2))              # [1,L1,L2]
    a1 = masked_softmax(e, m2.unsqueeze(1), dim=2)     # s1 토큰별 s2 분포
  A = a1[0].cpu().numpy()

  def toks(ids):
    return [t.replace("Ġ", " ").strip() or t for t in tok.convert_ids_to_tokens(ids[0])]
  t1, t2 = toks(ids1), toks(ids2)

  fig, ax = plt.subplots(figsize=(max(4, len(t2) * 0.6), max(3, len(t1) * 0.5)))
  im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
  ax.set_xticks(range(len(t2))); ax.set_xticklabels(t2, rotation=45, ha="right")
  ax.set_yticks(range(len(t1))); ax.set_yticklabels(t1)
  ax.set_xlabel("s2 tokens (aligned to)")
  ax.set_ylabel("s1 tokens (anchor)")
  ax.set_title("alignment soft-attention  (s1 -> s2)")
  fig.colorbar(im, fraction=0.046, pad=0.04)
  fig.tight_layout()
  os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"saved {a.out}  (s1 {len(t1)} tok x s2 {len(t2)} tok)")


if __name__ == "__main__":
  main()
