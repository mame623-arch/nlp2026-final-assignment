#!/bin/bash
# =====================================================================
#  run_h100.sh — H100 재실험 진입점 (baseline + ours + 비교메소드, seed 0)
#
#  run.sh 와 동일 파이프라인이되 H100 가속을 위해:
#    - BF16 autocast (--bf16)
#    - cloze 를 실배치 64(=유효배치 64, grad_accum 1)로 — 누적 제거해 커널 런치↓
#    - cloze early stopping patience 3 (보고서 Appendix C 레시피와 일치)
#  기본 seed 0 1회. OOD/증강 데이터가 없으면 자동 생성.
#
#  선행 1회:  conda env create -f env.yml && conda activate nlp_final
#  실행:      bash run_h100.sh
#  결과표:    python analysis/make_tables.py --metrics acc f1 mcc
#
#  ※ 더 공격적으로 키우려면(레시피에서 벗어남, 결과 미세 변동 가능):
#     CLOZE_BS=128 ALIGN_BS=128 bash run_h100.sh
# =====================================================================
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=${TQDM_DISABLE:-1}

SIZES=${SIZES:-"gpt2 gpt2-medium"}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-10}
RUN_AUG=${RUN_AUG:-1}          # 1=비교1(데이터 증강) 포함
CLOZE_BS=${CLOZE_BS:-64}       # 유효배치 64 유지(누적 1). H100 80GB면 128도 가능(레시피 벗어남)
ALIGN_BS=${ALIGN_BS:-64}

# ---- 데이터 준비(없으면 생성) — "데이터셋 준비 미비" 실패 규정 방어 ----
if [ ! -f data/paws_test.csv ] || [ ! -f data/mrpc_test.csv ]; then
  echo "[prep] OOD(PAWS/MRPC) 다운로드 ..."; $PY data/get_ood_data.py
fi
if [ "$RUN_AUG" = 1 ] && [ ! -f data/quora_aug_train.csv ]; then
  echo "[prep] Quora 증강 생성 ..."; $PY data/augment.py --steps swap bt hardneg merge
fi

QTRAIN=data/quora-train.csv; AUG=data/quora_aug_train.csv
EVALSETS="quora-dev:data/quora-dev.csv \
          paws-dev:data/paws_dev.csv paws-test:data/paws_test.csv \
          mrpc-dev:data/mrpc_traindev.csv mrpc-test:data/mrpc_test.csv"
CK=checkpoints; PR=predictions; mkdir -p $CK $PR

CLOZE="--use_gpu --bf16 --epochs $EPOCHS --batch_size $CLOZE_BS --grad_accum 1 \
       --lr 1e-5 --lr_schedule linear --warmup_ratio 0.1 --patience 3"
ALIGN="--use_gpu --bf16 --epochs $EPOCHS --batch_size $ALIGN_BS --contrastive_lambda 0.1"

for SIZE in $SIZES; do
  for SEED in $SEEDS; do
    T="$SIZE-s$SEED"
    echo "==================== train $T $(date) ===================="
    $PY paraphrase_detection.py $CLOZE --model_size $SIZE --seed $SEED --para_train $QTRAIN --filepath $CK/cloze-base-$T.pt
    if [ "$RUN_AUG" = 1 ]; then
      $PY paraphrase_detection.py $CLOZE --model_size $SIZE --seed $SEED --para_train $AUG --filepath $CK/cloze-aug-$T.pt
    fi
    $PY alignment_paraphrase.py $ALIGN --model_size $SIZE --seed $SEED --ablation_step 3 --filepath $CK/align-step3-$T.pt
    $PY alignment_paraphrase.py $ALIGN --model_size $SIZE --seed $SEED --ablation_step 1 --filepath $CK/align-step1-$T.pt

    echo "-------------------- dump $T --------------------"
    for KV in $EVALSETS; do
      DOM=${KV%%:*}; DATA=${KV##*:}
      $PY analysis/extract_signals.py --kind cloze --ckpt $CK/cloze-base-$T.pt --data $DATA --out $PR/cloze-base-$T-$DOM.csv
      [ "$RUN_AUG" = 1 ] && $PY analysis/extract_signals.py --kind cloze --ckpt $CK/cloze-aug-$T.pt --data $DATA --out $PR/cloze-aug-$T-$DOM.csv
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

echo "완료. 결과표:  $PY analysis/make_tables.py --metrics acc f1 mcc"
echo "오류분석(Fig1/Table2/κ):  $PY analysis/error_analysis.py --size gpt2 --seed 0"
