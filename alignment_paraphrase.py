"""
정렬-인지 임베딩 모듈 학습/평가 (보고서 ② 탭 A장).

cloze baseline(paraphrase_detection.py)과 분리된 별도 경로:
  - s1, s2 를 따로 토크나이즈 (bi-encoder 입력)
  - 공유 GPT-2 로 토큰 표현 → AlignmentModule → binary [B,2] head
  - ablation_step 1~4 사다리

실행 예:
  python alignment_paraphrase.py --use_gpu --model_size gpt2-medium --ablation_step 3 \
      --para_train data/quora-train.csv \
      --para_dev data/quora-dev.csv,data/paraphrase_extra_data/paws_dev.csv,data/paraphrase_extra_data/mrpc_traindev.csv
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from datasets import load_paraphrase_data, preprocess_string  # noqa: F401
from models.gpt2 import GPT2Model

YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919
TQDM_DISABLE = bool(os.environ.get('TQDM_DISABLE'))


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Dataset — s1, s2 를 따로 토크나이즈 + (step4 용) cloze prompt 도 함께 생성
# --------------------------------------------------------------------------- #
class AlignmentParaphraseDataset(Dataset):
  def __init__(self, dataset, has_labels=True):
    self.dataset = dataset
    self.has_labels = has_labels
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def _tok(self, sents):
    enc = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=128)
    return enc['input_ids'].long(), enc['attention_mask'].long()

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    ids1, mask1 = self._tok(sent1)
    ids2, mask2 = self._tok(sent2)
    cloze = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n'
             for s1, s2 in zip(sent1, sent2)]
    cids, cmask = self._tok(cloze)
    batch = {'ids1': ids1, 'mask1': mask1, 'ids2': ids2, 'mask2': mask2,
             'cloze_ids': cids, 'cloze_mask': cmask}
    if self.has_labels:
      batch['labels'] = torch.LongTensor([int(x[2]) for x in all_data])
      batch['sent_ids'] = [x[3] for x in all_data]
    else:
      batch['sent_ids'] = [x[2] for x in all_data]
    return batch


# --------------------------------------------------------------------------- #
# Model — 공유 GPT-2 + AlignmentModule + binary head (+ optional cloze 융합)
# --------------------------------------------------------------------------- #
from models.alignment import AlignmentModule  # noqa: E402


class AlignmentParaphraseGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.freeze_backbone = args.freeze_backbone
    if self.freeze_backbone:
      for p in self.gpt.parameters():
        p.requires_grad = False

    self.align = AlignmentModule(args.d, hidden=args.lstm_hidden,
                                 ablation_step=args.ablation_step, dropout=args.dropout,
                                 gpt_skip=getattr(args, 'gpt_skip', False))
    self.fuse_cloze = args.ablation_step >= 4
    cls_in = self.align.feat_dim + (2 if self.fuse_cloze else 0)
    self.classifier = nn.Sequential(
        nn.Linear(cls_in, args.cls_hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.cls_hidden, 2))

  def _encode_tokens(self, ids, mask):
    emb = self.gpt.embed(ids)
    seq = self.gpt.encode(emb, mask)
    return self.gpt.final_layer_norm(seq)

  def forward(self, ids1, mask1, ids2, mask2, cloze_ids=None, cloze_mask=None):
    def enc():
      return self._encode_tokens(ids1, mask1), self._encode_tokens(ids2, mask2)
    if self.freeze_backbone:
      with torch.no_grad():
        H1, H2 = enc()
    else:
      H1, H2 = enc()

    feats, v1, v2 = self.align(H1, mask1, H2, mask2)

    if self.fuse_cloze and cloze_ids is not None:
      out = self.gpt(cloze_ids, cloze_mask)
      vocab = self.gpt.hidden_state_to_token(out['last_token'])
      yn = torch.stack([vocab[:, YES_TOKEN_ID], vocab[:, NO_TOKEN_ID]], dim=-1)
      feats = torch.cat([feats, yn], dim=-1)

    logits = self.classifier(feats)
    return logits, v1, v2


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(loader, model, device):
  model.eval()
  y_true, y_pred, sids = [], [], []
  for batch in tqdm(loader, desc='eval', disable=TQDM_DISABLE):
    logits, _, _ = model(batch['ids1'].to(device), batch['mask1'].to(device),
                         batch['ids2'].to(device), batch['mask2'].to(device),
                         batch['cloze_ids'].to(device), batch['cloze_mask'].to(device))
    preds = logits.argmax(dim=1).cpu().numpy()
    y_pred.extend(preds.tolist())
    y_true.extend(batch['labels'].numpy().tolist())
    sids.extend(batch['sent_ids'])
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred, average='macro')
  return acc, f1, y_pred, y_true, sids


def save_preds(path, sids, preds):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w+') as f:
    f.write("id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(sids, preds):
      f.write(f"{p}, {s} \n")


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def split_csv(s):
  return [t.strip() for t in s.split(',') if t.strip()]


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  train_raw = load_paraphrase_data(args.para_train)
  if args.max_train and args.max_train < len(train_raw):
    rng = random.Random(args.seed)
    train_raw = rng.sample(train_raw, args.max_train)
    print(f"[subset] train -> {len(train_raw)}")

  dev_files = split_csv(args.para_dev)
  dev_loaders = []
  for fp in dev_files:
    raw = load_paraphrase_data(fp)
    if args.max_dev and args.max_dev < len(raw):
      raw = raw[:args.max_dev]
    ds = AlignmentParaphraseDataset(raw)
    dev_loaders.append((fp, DataLoader(ds, shuffle=False, batch_size=args.eval_batch_size,
                                       collate_fn=ds.collate_fn, num_workers=2)))

  train_ds = AlignmentParaphraseDataset(train_raw)
  if args.balanced_sampler:
    labels = [int(x[2]) for x in train_raw]
    c1 = sum(labels); c0 = len(labels) - c1
    w = [(0.5 / c1) if l == 1 else (0.5 / c0) for l in labels]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_ds, sampler=sampler, batch_size=args.batch_size,
                              collate_fn=train_ds.collate_fn, num_workers=4)
  else:
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size,
                              collate_fn=train_ds.collate_fn, num_workers=4)

  model = AlignmentParaphraseGPT(args).to(device)

  # param group: backbone(작은 lr) vs 새 모듈(큰 lr)
  bk, hd = [], []
  for n, p in model.named_parameters():
    if not p.requires_grad:
      continue
    (bk if n.startswith('gpt.') else hd).append(p)
  groups = [{'params': hd, 'lr': args.head_lr}]
  if bk:
    groups.append({'params': bk, 'lr': args.lr})
  optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

  from transformers import get_linear_schedule_with_warmup
  total_steps = len(train_loader) * args.epochs
  scheduler = get_linear_schedule_with_warmup(
      optimizer, int(total_steps * args.warmup_ratio), total_steps)

  cos_loss = nn.CosineEmbeddingLoss(margin=0.0)
  best_acc, best_epoch, no_improve = 0.0, -1, 0

  for epoch in range(args.epochs):
    model.train()
    tot, ncorr, ntot, nb = 0.0, 0, 0, 0
    for batch in tqdm(train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      ids1, mask1 = batch['ids1'].to(device), batch['mask1'].to(device)
      ids2, mask2 = batch['ids2'].to(device), batch['mask2'].to(device)
      cids, cmask = batch['cloze_ids'].to(device), batch['cloze_mask'].to(device)
      labels = batch['labels'].to(device)

      optimizer.zero_grad()
      logits, v1, v2 = model(ids1, mask1, ids2, mask2, cids, cmask)
      loss = F.cross_entropy(logits, labels)
      if args.contrastive_lambda > 0:
        tgt = (labels * 2 - 1).float()  # 1 / -1
        loss = loss + args.contrastive_lambda * cos_loss(v1, v2, tgt)
      loss.backward()
      torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
      optimizer.step()
      scheduler.step()

      tot += loss.item(); nb += 1
      ncorr += (logits.argmax(1) == labels).sum().item(); ntot += labels.size(0)

    metrics = {}
    for fp, loader in dev_loaders:
      acc, f1, *_ = evaluate(loader, model, device)
      metrics[fp] = (acc, f1)
      tag = 'primary' if fp == dev_files[0] else 'aux'
      print(f"  [{tag}] {fp}  acc={acc:.4f} f1={f1:.4f}")
    dev_acc = metrics[dev_files[0]][0]

    if dev_acc > best_acc:
      best_acc, best_epoch, no_improve = dev_acc, epoch, 0
      torch.save({'model': model.state_dict(), 'args': args}, args.filepath)
      print(f"  >> saved (best dev acc {best_acc:.4f})")
    else:
      no_improve += 1

    print(f"Epoch {epoch}: loss={tot/max(nb,1):.4f} train_acc={ncorr/max(ntot,1):.4f} "
          f"dev_acc={dev_acc:.4f} (best {best_acc:.4f}@{best_epoch})")
    if args.patience and no_improve >= args.patience:
      print(f"Early stopping @ epoch {epoch}")
      break

  return best_acc, best_epoch


@torch.no_grad()
def test(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath, weights_only=False)
  model = AlignmentParaphraseGPT(saved['args']).to(device)
  model.load_state_dict(saved['model'])
  model.eval()
  print(f"Loaded {args.filepath}")

  dev_files = split_csv(args.para_dev)
  dev_outs = split_csv(args.para_dev_out) if args.para_dev_out else []
  for i, fp in enumerate(dev_files):
    raw = load_paraphrase_data(fp)
    ds = AlignmentParaphraseDataset(raw)
    loader = DataLoader(ds, shuffle=False, batch_size=args.eval_batch_size,
                        collate_fn=ds.collate_fn, num_workers=2)
    acc, f1, preds, _, sids = evaluate(loader, model, device)
    print(f"dev [{fp}] acc={acc:.4f} f1={f1:.4f}")
    if i < len(dev_outs):
      save_preds(dev_outs[i], sids, preds)
      print(f"  preds -> {dev_outs[i]}")


def add_model_dims(args):
  if args.model_size == 'gpt2':
    args.d, args.l, args.num_heads = 768, 12, 12
  elif args.model_size == 'gpt2-medium':
    args.d, args.l, args.num_heads = 1024, 24, 16
  elif args.model_size == 'gpt2-large':
    args.d, args.l, args.num_heads = 1280, 36, 20
  else:
    raise ValueError(args.model_size)
  return args


def get_args():
  p = argparse.ArgumentParser()
  p.add_argument('--para_train', default='data/quora-train.csv')
  p.add_argument('--para_dev', default='data/quora-dev.csv')
  p.add_argument('--para_dev_out', default='')
  p.add_argument('--model_size', choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-medium')
  p.add_argument('--ablation_step', type=int, choices=[1, 2, 3, 4], default=3)
  p.add_argument('--freeze_backbone', action='store_true')
  p.add_argument('--lstm_hidden', type=int, default=256)
  p.add_argument('--cls_hidden', type=int, default=256)
  p.add_argument('--dropout', type=float, default=0.2)
  p.add_argument('--contrastive_lambda', type=float, default=0.0)
  p.add_argument('--gpt_skip', action='store_true',
                 help='pool(GPT-2 출력)을 분류 피처에 직접 concat (병렬 결합)')

  p.add_argument('--epochs', type=int, default=4)
  p.add_argument('--patience', type=int, default=2)
  p.add_argument('--batch_size', type=int, default=32)
  p.add_argument('--eval_batch_size', type=int, default=64)
  p.add_argument('--lr', type=float, default=1e-5, help='backbone lr')
  p.add_argument('--head_lr', type=float, default=1e-3, help='alignment+head lr')
  p.add_argument('--weight_decay', type=float, default=0.01)
  p.add_argument('--warmup_ratio', type=float, default=0.1)
  p.add_argument('--balanced_sampler', action='store_true')

  p.add_argument('--max_train', type=int, default=0)
  p.add_argument('--max_dev', type=int, default=0)
  p.add_argument('--seed', type=int, default=11711)
  p.add_argument('--use_gpu', action='store_true')
  p.add_argument('--filepath', default='')
  p.add_argument('--eval_only', action='store_true')
  return p.parse_args()


if __name__ == '__main__':
  args = get_args()
  args = add_model_dims(args)
  if not args.filepath:
    args.filepath = f'align-step{args.ablation_step}-{args.model_size.replace("gpt2","s")}.pt'
  seed_everything(args.seed)
  if not args.eval_only:
    train(args)
  test(args)
