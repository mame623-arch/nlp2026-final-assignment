#!/bin/bash
# 학습 + 예측 신호 dump 까지. 결과표·분석은 analysis/ 스크립트에서 별도로 수행.
#
#   학습 모델 (각 SIZE × SEED):
#     cloze-base   = baseline (+ 비교2 symmetric, + ours의 cloze 멤버)   ← 같은 ckpt 재사용
#     cloze-aug    = 비교1 (데이터 증강; data/quora_aug_train.csv 필요)
#     align-step3  = ours의 alignment 멤버
#     align-step1  = ours의 bi-enc 멤버
#   외부 NLI(roberta-mnli) 신호는 모델과 무관하므로 평가셋당 1회.
#
# 사용:
#   ./run.sh                                  # 전체 (small+medium, seed 0/1/2)
#   SIZES="gpt2" SEEDS="0" EPOCHS=1 ./run.sh  # 빠른 smoke test
#   RUN_AUG=1 ./run.sh                         # 비교1(증강)까지 포함 (data/quora_aug_train.csv 필요)
#
# 선행: python data/get_ood_data.py  (paws/mrpc) · [RUN_AUG=1 일 때] python data/augment.py  (증강)
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=${TQDM_DISABLE:-1}

SIZES=${SIZES:-"gpt2 gpt2-medium"}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-10}
RUN_AUG=${RUN_AUG:-0}   # 1이면 비교1(증강) cloze-aug 학습·dump 포함. 0이면 baseline+ours만.

QTRAIN=data/quora-train.csv
AUG=data/quora_aug_train.csv
# dump 대상 평가셋 (name:path). quora=dev / paws·mrpc = dev(analysis)+test(결과표)
EVALSETS="quora-dev:data/quora-dev.csv \
          paws-dev:data/paws_dev.csv paws-test:data/paws_test.csv \
          mrpc-dev:data/mrpc_traindev.csv mrpc-test:data/mrpc_test.csv"

CK=checkpoints; PR=predictions; mkdir -p $CK $PR
CLOZE="--use_gpu --epochs $EPOCHS --batch_size 16 --grad_accum 4 --lr 1e-5 --lr_schedule linear --warmup_ratio 0.1"
ALIGN="--use_gpu --epochs $EPOCHS --batch_size 64 --contrastive_lambda 0.1"

for SIZE in $SIZES; do
  for SEED in $SEEDS; do
    T="$SIZE-s$SEED"
    echo "==================== train  $T  $(date) ===================="
    $PY paraphrase_detection.py $CLOZE --model_size $SIZE --seed $SEED --para_train $QTRAIN --filepath $CK/cloze-base-$T.pt
    if [ "$RUN_AUG" = 1 ]; then
      $PY paraphrase_detection.py $CLOZE --model_size $SIZE --seed $SEED --para_train $AUG    --filepath $CK/cloze-aug-$T.pt
    fi
    $PY alignment_paraphrase.py $ALIGN --model_size $SIZE --seed $SEED --ablation_step 3    --filepath $CK/align-step3-$T.pt
    $PY alignment_paraphrase.py $ALIGN --model_size $SIZE --seed $SEED --ablation_step 1    --filepath $CK/align-step1-$T.pt

    echo "-------------------- dump  $T --------------------"
    for KV in $EVALSETS; do
      DOM=${KV%%:*}; DATA=${KV##*:}
      $PY analysis/extract_signals.py --kind cloze --ckpt $CK/cloze-base-$T.pt --data $DATA --out $PR/cloze-base-$T-$DOM.csv
      [ "$RUN_AUG" = 1 ] && $PY analysis/extract_signals.py --kind cloze --ckpt $CK/cloze-aug-$T.pt  --data $DATA --out $PR/cloze-aug-$T-$DOM.csv
      $PY analysis/extract_signals.py --kind align --ckpt $CK/align-step3-$T.pt --data $DATA --out $PR/align-step3-$T-$DOM.csv
      $PY analysis/extract_signals.py --kind align --ckpt $CK/align-step1-$T.pt --data $DATA --out $PR/align-step1-$T-$DOM.csv
    done
  done
done

echo "==================== external NLI dump (평가셋당 1회) ===================="
for KV in $EVALSETS; do
  DOM=${KV%%:*}; DATA=${KV##*:}
  $PY analysis/extract_signals.py --kind external --data $DATA --out $PR/external-$DOM.csv
done

echo "학습+dump 완료. 다음: python analysis/make_tables.py  (결과표)"
