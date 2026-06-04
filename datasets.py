# !/usr/bin/env python3


"""
이 파일은 Quora의 Paraphrase Detection을 위한 Dataset 클래스를 포함한다. 추가 데이터 소스로 훈련시키거나
Quora 데이터셋의 처리 방식(예: 데이터 증강 등)을 변경하려는 경우 이 파일을 수정할 수 있다.
"""

import csv

import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


# cloze prompt 템플릿. 'default' = 학습·평가에 실제 사용하는 기본 prompt.
# 'prompt 변경' 비교메소드용으로 나머지 스타일 제공. (--prompt_style 로 선택)
PROMPT_TEMPLATES = {
  'default':      'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n',
  'original':     'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ',
  'same_meaning': 'Question 1: {s1}\nQuestion 2: {s2}\nDo these two questions have the same meaning? Answer: ',
  'semantic':     'Determine whether the following two questions are semantically equivalent.\nQuestion A: {s1}\nQuestion B: {s2}\nAnswer: ',
  'duplicate':    'Question 1: {s1}\nQuestion 2: {s2}\nAre these duplicate questions? Answer: ',
  'fewshot': (
    'Example 1:\nQuestion 1: How do I learn Python?\nQuestion 2: What is the best way to study Python?\nAnswer: yes\n\n'
    'Example 2:\nQuestion 1: How do I learn Python?\nQuestion 2: How do I cook pasta?\nAnswer: no\n\n'
    'Now classify:\nQuestion 1: {s1}\nQuestion 2: {s2}\nAnswer: '
  ),
}


def build_cloze_sents(sent1, sent2, prompt_style='default'):
  """prompt_style 템플릿으로 (s1,s2) 쌍을 cloze 입력 문자열 리스트로 변환."""
  template = PROMPT_TEMPLATES[prompt_style]
  return [template.format(s1=s1, s2=s2) for s1, s2 in zip(sent1, sent2)]


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args, swap=False):
    self.dataset = dataset
    self.p = args
    self.swap = swap
    self.prompt_style = getattr(args, 'prompt_style', 'default')
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    if self.swap:
      sent1, sent2 = sent2, sent1
    # labels = torch.LongTensor([x[2] for x in all_data])
    labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = build_cloze_sents(sent1, sent2, self.prompt_style)
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args, swap=False):
    self.dataset = dataset
    self.p = args
    self.swap = swap
    self.prompt_style = getattr(args, 'prompt_style', 'default')
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    if self.swap:
      sent1, sent2 = sent2, sent1
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = build_cloze_sents(sent1, sent2, self.prompt_style)

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(file=paraphrase_filename, mode='r', encoding="utf-8-sig") as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(file=paraphrase_filename, mode='r', encoding="utf-8-sig") as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file=file_path, mode='r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data
