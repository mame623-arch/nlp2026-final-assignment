'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import math
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import (
  model_eval_paraphrase,
  model_test_paraphrase,
  model_eval_paraphrase_calibrated,
  model_test_paraphrase_calibrated,
  estimate_prior,
  model_eval_paraphrase_symmetric,
  model_test_paraphrase_symmetric,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

from pathlib import Path # 나중에 삭제

TQDM_DISABLE = bool(os.environ.get('TQDM_DISABLE'))

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection 의 출력은 두 가지: 1 (yes) or 0 (no).

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask, return_lm_logits=False):
    """
    TODO: paraphrase_detection_head Linear layer를 사용하여 토큰의 레이블을 예측하시오.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다.
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가,
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.

    return_lm_logits=True 일 때 (class_logits, lm_logits) 튜플 반환.
    lm_logits 는 전체 시퀀스에 대한 vocab logits [B, S, V] — 보조 LM loss 계산용.
    """
    ### 완성시켜야 할 빈 코드 블록
    outputs = self.gpt(input_ids, attention_mask)
    last_token = outputs['last_token']
    logits = self.gpt.hidden_state_to_token(last_token)

    if return_lm_logits:
      lm_logits = self.gpt.hidden_state_to_token(outputs['last_hidden_state'])
      return logits, lm_logits
    return logits
  

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def _split_csv_arg(s):
  """콤마 구분 문자열을 리스트로 분해 (공백 제거, 빈 토큰 제외)."""
  return [t.strip() for t in s.split(",") if t.strip()]


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_raw = load_paraphrase_data(args.para_train)

  # --para_dev 는 콤마 구분 다중 파일을 허용. 첫 번째 dev 가 model selection 기준.
  dev_files = _split_csv_arg(args.para_dev)
  para_dev_raw_map = {}
  para_dev_loaders = []
  for fp in dev_files:
    raw = load_paraphrase_data(fp)
    para_dev_raw_map[fp] = raw
    ds = ParaphraseDetectionDataset(raw, args)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                        collate_fn=ds.collate_fn)
    para_dev_loaders.append((fp, loader))

  # symmetric eval 용 swap-변환 dev loader (학습 epoch 평가용)
  para_dev_swap_loaders = {}
  if args.symmetric_eval:
    for fp, raw in para_dev_raw_map.items():
      ds_swap = ParaphraseDetectionDataset(raw, args, swap=True)
      para_dev_swap_loaders[fp] = DataLoader(
          ds_swap, shuffle=False, batch_size=args.batch_size,
          collate_fn=ds_swap.collate_fn)

  para_train_data = ParaphraseDetectionDataset(para_train_raw, args)
  if args.balanced_sampler:
    labels = [int(x[2]) for x in para_train_data.dataset]
    cnt0 = sum(1 for l in labels if l == 0)
    cnt1 = sum(1 for l in labels if l == 1)
    # epoch 당 pos:neg ≈ 50:50 강제. num_samples 는 원본 크기 유지.
    w0 = 0.0 if cnt0 == 0 else 0.5 / cnt0
    w1 = 0.0 if cnt1 == 0 else 0.5 / cnt1
    weights = [w1 if l == 1 else w0 for l in labels]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    para_train_dataloader = DataLoader(para_train_data, sampler=sampler, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    print(f"balanced_sampler: pos={cnt1}, neg={cnt0} → epoch 당 50:50 sampling")
  else:
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)

  args = add_arguments(args)

  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

  # LR 스케줄러 (default none → 기존 고정 lr 과 동일). warmup 후 linear/cosine decay.
  # grad_accum 적용: 스케줄러는 optimizer step(= micro-batch/grad_accum) 기준.
  total_steps = (len(para_train_dataloader) // args.grad_accum) * args.epochs
  warmup_steps = int(args.warmup_ratio * total_steps)
  def _lr_lambda(step):
    if warmup_steps > 0 and step < warmup_steps:
      return step / max(1, warmup_steps)
    if args.lr_schedule == 'none':
      return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if args.lr_schedule == 'linear':
      return 1.0 - progress
    if args.lr_schedule == 'cosine':
      return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0
  scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
               if (args.lr_schedule != 'none' or warmup_steps > 0) else None)
  if scheduler is not None:
    print(f"lr_schedule={args.lr_schedule}, warmup_ratio={args.warmup_ratio} "
          f"(total_steps={total_steps}, warmup_steps={warmup_steps})")

  use_lm_loss = args.lm_lambda > 0
  if use_lm_loss:
    print(f"보조 LM loss 사용: lm_lambda={args.lm_lambda}")

  best_dev_acc = 0
  best_epoch = -1
  no_improvement = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    train_correct = 0
    train_total = 0
    optimizer.zero_grad()
    for step_i, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 손실, 그래디언트를 계산하고 grad_accum 단위로 파라미터 업데이트.
      with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=getattr(args, 'bf16', False)):
        if use_lm_loss:
          # 분류 손실 + 보조 LM(next-token) 손실. lm_lambda 로 가중.
          logits, lm_logits = model(b_ids, b_mask, return_lm_logits=True)
          shift_logits = lm_logits[:, :-1, :].contiguous()
          shift_labels = b_ids[:, 1:].contiguous()
          shift_mask = b_mask[:, 1:].contiguous()
          shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)  # padding ignore
          lm_loss = F.cross_entropy(
              shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1), ignore_index=-100, reduction='mean')
          class_loss = F.cross_entropy(logits, labels, reduction='mean')
          loss = class_loss + args.lm_lambda * lm_loss
        else:
          logits = model(b_ids, b_mask)
          loss = F.cross_entropy(logits, labels, reduction='mean')
      preds = torch.argmax(logits, dim=1)
      (loss / args.grad_accum).backward()
      if (step_i + 1) % args.grad_accum == 0:
        optimizer.step()
        if scheduler is not None:
          scheduler.step()
        optimizer.zero_grad()

      train_loss += loss.item()
      num_batches += 1
      train_correct += (preds == labels).sum().item()
      train_total += labels.size(0)

    train_loss = train_loss / num_batches
    train_acc = train_correct / train_total

    # 다중 dev: 첫 번째 = primary (early stopping / 저장 기준), 나머지는 모니터링용
    dev_metrics = {}
    sym_dev_metrics = {}
    for fp, loader in para_dev_loaders:
      acc, f1, *_ = model_eval_paraphrase(loader, model, device)
      dev_metrics[fp] = (acc, f1)
      if fp != para_dev_loaders[0][0]:
        print(f"  aux dev [{fp}] acc :: {acc :.3f}, f1 :: {f1 :.3f}")
      if args.symmetric_eval:
        sym_acc, sym_f1, *_ = model_eval_paraphrase_symmetric(
            loader, para_dev_swap_loaders[fp], model, device)
        sym_dev_metrics[fp] = (sym_acc, sym_f1)
        print(f"  sym dev [{fp}] acc :: {sym_acc :.3f}, f1 :: {sym_f1 :.3f}")
    dev_acc, dev_f1 = dev_metrics[para_dev_loaders[0][0]]

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      best_epoch = epoch
      no_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      no_improvement += 1

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f} (best {best_dev_acc :.3f} @ epoch {best_epoch})")

    if args.patience is not None and no_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch} (best dev acc {best_dev_acc :.3f} @ epoch {best_epoch})")
      break

@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  # 다중 dev / test: 콤마 구분 리스트 허용. 입력 - 출력 파일 수는 정확히 매칭되어야 함.
  dev_files = _split_csv_arg(args.para_dev)
  dev_out_files = _split_csv_arg(args.para_dev_out)
  test_files = _split_csv_arg(args.para_test)
  test_out_files = _split_csv_arg(args.para_test_out)
  assert len(dev_files) == len(dev_out_files), \
      f"--para_dev ({len(dev_files)}) 와 --para_dev_out ({len(dev_out_files)}) 길이가 다릅니다."
  assert len(test_files) == len(test_out_files), \
      f"--para_test ({len(test_files)}) 와 --para_test_out ({len(test_out_files)}) 길이가 다릅니다."

  # --prior_calibration: 첫 dev set 의 dataset 으로 dummy pair 한 번 forward → prior 추정.
  # prior_yes / prior_no 는 모든 dev/test 셋에 동일하게 차감됨.
  prior_yes, prior_no = 0.0, 0.0
  if args.prior_calibration:
    first_ds = ParaphraseDetectionDataset(load_paraphrase_data(dev_files[0]), args)
    prior_yes, prior_no = estimate_prior(model, first_ds, device)
    print(f"prior_calibration: prior_yes={prior_yes:.4f}, prior_no={prior_no:.4f} "
          f"(차이 {prior_yes - prior_no:+.4f} — yes 쪽으로 편향이면 양수)")

  for dev_fp, dev_out_fp in zip(dev_files, dev_out_files):
    data = load_paraphrase_data(dev_fp)
    ds = ParaphraseDetectionDataset(data, args)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    if args.prior_calibration:
      dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase_calibrated(
          loader, model, device, prior_yes=prior_yes, prior_no=prior_no)
    else:
      dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(loader, model, device)
    print(f"dev paraphrase acc [{dev_fp}] :: {dev_para_acc :.3f}")
    with open(dev_out_fp, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
        f.write(f"{p}, {s} \n")

    # symmetric_eval: 같은 dev 에 대해 swap 평균 예측을 *-symmetric.csv 로 따로 저장
    # prior_calibration 과 함께 켜면 평균 logit 에서 prior 도 차감됨 (진짜 결합 효과).
    if args.symmetric_eval:
      ds_swap = ParaphraseDetectionDataset(data, args, swap=True)
      loader_swap = DataLoader(ds_swap, shuffle=False, batch_size=args.batch_size,
                               collate_fn=ds_swap.collate_fn)
      sym_acc, _, sym_y_pred, _, sym_sent_ids = model_eval_paraphrase_symmetric(
          loader, loader_swap, model, device,
          prior_yes=prior_yes, prior_no=prior_no)
      print(f"dev paraphrase acc (symmetric) [{dev_fp}] :: {sym_acc :.3f}")
      sym_out = dev_out_fp.replace('.csv', '-symmetric.csv')
      with open(sym_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(sym_sent_ids, sym_y_pred):
          f.write(f"{p}, {s} \n")

  for test_fp, test_out_fp in zip(test_files, test_out_files):
    data = load_paraphrase_data(test_fp, split='test')
    ds = ParaphraseDetectionTestDataset(data, args)
    # symmetric 이면 shuffle 끄기 (orig/swap 페어링이 같은 순서여야 함)
    test_shuffle = not args.symmetric_eval
    loader = DataLoader(ds, shuffle=test_shuffle, batch_size=args.batch_size, collate_fn=ds.collate_fn)
    if args.prior_calibration:
      test_para_y_pred, test_para_sent_ids = model_test_paraphrase_calibrated(
          loader, model, device, prior_yes=prior_yes, prior_no=prior_no)
    else:
      test_para_y_pred, test_para_sent_ids = model_test_paraphrase(loader, model, device)
    print(f"test predictions saved [{test_fp}] -> {test_out_fp}")
    with open(test_out_fp, "w+") as f:
      f.write(f"id \t Predicted_Is_Paraphrase \n")
      for p, s in zip(test_para_sent_ids, test_para_y_pred):
        f.write(f"{p}, {s} \n")

    if args.symmetric_eval:
      ds_swap = ParaphraseDetectionTestDataset(data, args, swap=True)
      loader_swap = DataLoader(ds_swap, shuffle=False, batch_size=args.batch_size,
                               collate_fn=ds_swap.collate_fn)
      sym_test_y_pred, sym_test_sent_ids = model_test_paraphrase_symmetric(
          loader, loader_swap, model, device,
          prior_yes=prior_yes, prior_no=prior_no)
      sym_out = test_out_fp.replace('.csv', '-symmetric.csv')
      print(f"test predictions saved (symmetric) [{test_fp}] -> {sym_out}")
      with open(sym_out, "w+") as f:
        f.write(f"id \t Predicted_Is_Paraphrase \n")
        for p, s in zip(sym_test_sent_ids, sym_test_y_pred):
          f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--bf16", action="store_true", help="BF16 autocast 학습 가속(H100 등). 기본 off.")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--grad_accum", type=int, default=1,
                      help='gradient accumulation steps (effective batch = batch_size * grad_accum)')
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--patience", type=int, default=None,
                      help="early stopping patience (epochs without dev improvement). 지정하지 않으면 비활성화")
  parser.add_argument("--weight_decay", type=float, default=0.01)
  parser.add_argument("--lr_schedule", type=str, choices=['none', 'linear', 'cosine'], default='none',
                      help="warmup 후 lr decay 형태. none=고정(기존 동작). linear/cosine=0까지 감소.")
  parser.add_argument("--warmup_ratio", type=float, default=0.0,
                      help="전체 step 중 선형 warmup 비율 (예: 0.06, 0.1). 0이면 warmup 없음.")
  parser.add_argument("--lm_lambda", type=float, default=0.0,
                      help="보조 LM(next-token) loss 가중치. 0이면 분류 loss만(기존 동작). >0이면 정규화로 추가.")
  parser.add_argument("--balanced_sampler", action='store_true',
                      help="WeightedRandomSampler 로 epoch 당 pos:neg 50:50 강제 (학습 데이터 mix 가 한쪽으로 쏠릴 때 FPR/FNR 균형 회복)")
  parser.add_argument("--prior_calibration", action='store_true',
                      help="추론 시 빈 문장 페어로 prior (yes/no logit) 추정 후 실제 logit 에서 차감. 사전 편향 제거 — bt 셀처럼 yes 쪽으로 쏠린 모델 보정에 사용.")
  parser.add_argument("--eval_only", action='store_true',
                      help='train() 건너뛰고 기존 체크포인트로 test() 만 실행 (inference-only).')
  parser.add_argument("--symmetric_eval", action='store_true',
                      help="dev/test 평가 시 (S1,S2)와 (S2,S1) yes/no logit 평균으로 추가 예측 → *-symmetric.csv 로 저장")
  parser.add_argument("--prompt_style", type=str, default='default',
                      choices=['default', 'original', 'same_meaning', 'semantic', 'duplicate', 'fewshot'],
                      help="cloze prompt 템플릿 ('prompt 변경' 비교메소드용). default=학습·평가 기본 prompt. datasets.PROMPT_TEMPLATES 참조")
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--filepath", type=str, default="",
                      help="체크포인트 저장 경로 override (미지정 시 자동 생성).")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  # args.filepath = f'{args.epochs}-{args.lr}-wd{args.weight_decay}-pat{args.patience}-paraphrase.pt'  # 경로명 저장.
  if not args.filepath:
    args.filepath = f'{Path(args.para_train).stem}-{args.epochs}-...-paraphrase.pt'
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  if not args.eval_only:
    train(args)
  else:
    # eval_only: model_size 등 add_arguments 로만 채워지는 값 보충
    args = add_arguments(args)
    print(f"[eval_only] train() 스킵, 기존 체크포인트 {args.filepath} 로 test() 만 실행")
  test(args)
