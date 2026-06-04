"""
ESIM 계열 정렬-인지(alignment-aware) 모듈.

오류 분석(scope 46% · lexical_gap 35%)에서 연역한 구조:
  token 표현 → BiLSTM 재인코딩 → 문장 간 soft-alignment → enhance → compose → pool.

- 정렬 안 된 토큰  → 한쪽에만 있는 내용 (scope 신호)
- 정렬됐는데 표면형 다름 → 동의표현 (lexical_gap 신호)

ablation_step:
  1 = bi-encoder (정렬 없음, 대칭 |u-v|) — 대조군
  2 = + soft-alignment (비대칭 도입, enhance=[P, P~])
  3 = + full enhance [P, P~, P-P~, P*P~] + compose BiLSTM
"""

import torch
from torch import nn
import torch.nn.functional as F


def masked_softmax(scores, mask, dim):
  """scores 를 mask(0=pad)가 가리키는 위치에서 -inf 처리 후 softmax."""
  scores = scores.masked_fill(mask == 0, -1e9)
  return F.softmax(scores, dim=dim)


def pool(x, mask):
  """x [B,L,D], mask [B,L] → mean+max+last concat [B,3D] (pad 위치 제외).
  GPT-2는 causal LM이라 마지막 비-pad 토큰이 문장 전체 문맥을 봄 → last 추가가 핵심."""
  m = mask.unsqueeze(-1).to(x.dtype)
  summed = (x * m).sum(dim=1)
  cnt = m.sum(dim=1).clamp(min=1.0)
  mean = summed / cnt
  mx = x.masked_fill(m == 0, -1e9).max(dim=1).values
  last_idx = (mask.sum(dim=1).long().clamp(min=1) - 1)        # 마지막 유효 토큰 위치
  last = x[torch.arange(x.size(0), device=x.device), last_idx]
  return torch.cat([mean, mx, last], dim=-1)


class AlignmentModule(nn.Module):
  def __init__(self, d, hidden=256, ablation_step=3, dropout=0.2, gpt_skip=False):
    super().__init__()
    self.step = ablation_step
    self.gpt_skip = gpt_skip   # True 면 pool(GPT-2 출력 H)를 분류 피처에 직접 concat (병렬)
    self.d = d
    self.input_lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
    enc = 2 * hidden

    if self.step >= 3:
      self.compose_proj = nn.Linear(4 * enc, enc)
      self.compose_lstm = nn.LSTM(enc, hidden, batch_first=True, bidirectional=True)
      self.pooled = 2 * hidden
    elif self.step == 2:
      self.compose_proj = nn.Linear(2 * enc, enc)  # enhance = [P, P~]
      self.compose_lstm = nn.LSTM(enc, hidden, batch_first=True, bidirectional=True)
      self.pooled = 2 * hidden
    else:  # step 1: 정렬 없음
      self.pooled = enc

    self.dropout = nn.Dropout(dropout)
    if gpt_skip:
      self.gpt_ln = nn.LayerNorm(d * 3)   # pool(H)=mean+max+last 크기 폭발 방지
    # 문장 벡터 = pool(BiLSTM/compose 출력) [3*pooled] (+gpt_skip 시 pool(H) [3*d])
    per_sent = self.pooled * 3 + (d * 3 if gpt_skip else 0)
    self.feat_dim = per_sent * 4   # [v1, v2, |v1-v2|, v1*v2]
    self.sent_dim = per_sent        # contrastive 용 문장 임베딩 차원

  def _encode(self, H):
    out, _ = self.input_lstm(H)
    return out

  def _align(self, P1, P2, mask1, mask2):
    e = torch.bmm(P1, P2.transpose(1, 2))               # [B,L1,L2]
    a1 = masked_softmax(e, mask2.unsqueeze(1), dim=2)    # L2 분포
    P1t = torch.bmm(a1, P2)                              # [B,L1,enc]
    a2 = masked_softmax(e, mask1.unsqueeze(2), dim=1)    # L1 분포
    P2t = torch.bmm(a2.transpose(1, 2), P1)             # [B,L2,enc]
    return P1t, P2t

  def forward(self, H1, mask1, H2, mask2):
    P1 = self._encode(H1)
    P2 = self._encode(H2)

    if self.step == 1:
      v1, v2 = pool(P1, mask1), pool(P2, mask2)
    else:
      P1t, P2t = self._align(P1, P2, mask1, mask2)
      if self.step == 2:
        m1 = torch.cat([P1, P1t], dim=-1)
        m2 = torch.cat([P2, P2t], dim=-1)
      else:  # step >= 3
        m1 = torch.cat([P1, P1t, P1 - P1t, P1 * P1t], dim=-1)
        m2 = torch.cat([P2, P2t, P2 - P2t, P2 * P2t], dim=-1)
      c1, _ = self.compose_lstm(F.relu(self.compose_proj(m1)))
      c2, _ = self.compose_lstm(F.relu(self.compose_proj(m2)))
      v1, v2 = pool(c1, mask1), pool(c2, mask2)

    if self.gpt_skip:                       # GPT-2 출력(H)을 직접 pool 해 병렬 결합
      v1 = torch.cat([v1, self.gpt_ln(pool(H1, mask1))], dim=-1)
      v2 = torch.cat([v2, self.gpt_ln(pool(H2, mask2))], dim=-1)
    feats = torch.cat([v1, v2, (v1 - v2).abs(), v1 * v2], dim=-1)
    return self.dropout(feats), v1, v2
