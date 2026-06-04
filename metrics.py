"""평가 metric 후보 모음. 필요한 것만 which 로 골라 호출한다.

  compute_metrics(y_true, y_pred, y_score=None, which=['acc','f1']) -> dict

  y_true, y_pred : 0/1 정수 배열
  y_score        : positive(중복=1) 확률/점수 — AUROC/AUPRC 에만 필요(없으면 nan)
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, roc_auc_score,
                             average_precision_score, cohen_kappa_score)


def _acc(yt, yp, ys):      return accuracy_score(yt, yp)
def _f1(yt, yp, ys):       return f1_score(yt, yp, average='macro')          # macro-F1
def _mcc(yt, yp, ys):      return matthews_corrcoef(yt, yp)
def _bal_acc(yt, yp, ys):  return balanced_accuracy_score(yt, yp)
def _kappa(yt, yp, ys):    return cohen_kappa_score(yt, yp)
def _auroc(yt, yp, ys):    return float('nan') if ys is None else roc_auc_score(yt, ys)
def _auprc(yt, yp, ys):    return float('nan') if ys is None else average_precision_score(yt, ys)

# 결과표에 쓸 수 있는 metric 후보 전부. acc·f1 이 기본, 나머지는 선택.
REGISTRY = {
  'acc':     _acc,      # accuracy
  'f1':      _f1,       # macro-F1
  'mcc':     _mcc,      # Matthews 상관 (불균형 견고)
  'bal_acc': _bal_acc,  # balanced accuracy
  'auroc':   _auroc,    # AUROC (y_score 필요)
  'auprc':   _auprc,    # average precision (y_score 필요)
  'kappa':   _kappa,    # Cohen's kappa
}


def compute_metrics(y_true, y_pred, y_score=None, which=('acc', 'f1')):
  yt = np.asarray(y_true).astype(int)
  yp = np.asarray(y_pred).astype(int)
  ys = None if y_score is None else np.asarray(y_score, dtype=float)
  out = {}
  for name in which:
    if name not in REGISTRY:
      raise KeyError(f"unknown metric '{name}'. 후보: {list(REGISTRY)}")
    out[name] = float(REGISTRY[name](yt, yp, ys))
  return out


def available():
  return list(REGISTRY)
