# PART-II — GPT-2 Paraphrase Detection (제출본 · 재현 가이드)

오류 분석에서 연역한 **다중 구조 앙상블(스태커)** 로 Quora paraphrase를 탐지하고,
동일하게 Quora만으로 학습한 채 **OOD(PAWS·MRPC)** 일반화까지 평가한다.

## 데이터셋 & 실행 파이프라인 (채점 실행 안내)
- **Quora(QQP)**: 과제 제공본을 그대로 리포지토리에 포함한다 (`data/quora-{train,dev,test-student}.csv`).
- **OOD(PAWS·MRPC)**: 리포지토리에 포함하지 않고 `python data/get_ood_data.py` 로 HuggingFace에서 내려받아 `data/` 에 생성한다(학습에는 쓰지 않는 평가 전용).
- **증강(비교메소드용)**: `python data/augment.py` 로 `data/quora_aug_train.csv` 생성.
- 학습/평가 진입점(`run.sh`)은 **OOD 파일이 없으면 자동으로 `get_ood_data.py` 를 먼저 실행**하므로, 데이터 준비 미비로 인한 실행 실패가 없도록 설계되어 있다.
- 모든 데이터 처리(다운로드·전처리·분할·로드)는 `datasets.py` + `data/get_ood_data.py` + `data/augment.py` 파이프라인으로 코드화되어 있다.

## 방법 요약
- **baseline** : GPT-2 cloze (yes/no verbalizer)
- **ours** : 4멤버 스태커 — `cloze` · `alignment`(ESIM 정렬, scope) · `bi-enc`(임베딩, lexical_gap)
  · `cloze+외부NLI`(roberta-mnli) 를 **2-fold cross-fit logistic regression** 으로 결합
- 학습 = **Quora-only** / 평가 = ID(Quora) + OOD(PAWS, MRPC)

## 환경
```bash
conda env create -f env.yml
conda activate nlp_final
# (선택) LLM judge 재실행 시:  echo "sk-..." > ~/.openai_key
```

## 재현 순서
```bash
# 1) 데이터 — OOD 평가셋 + (비교메소드용) Quora 증강
python data/get_ood_data.py                              # PAWS·MRPC -> data/*.csv
python data/augment.py --steps swap bt hardneg merge     # -> data/quora_aug_train.csv

# 2) 학습 + 예측 신호 dump
#    ※ predictions/ 에 신호 dump(*-gpt2*-s0-*.csv, external-*.csv)가 이미 있으면 이 단계는 건너뛰고 바로 3)으로 — 재학습은 기존 결과를 덮어쓰며 수십 시간 소요.
./run.sh                                                 # 표준 (small+medium, seed 0/1/2)
#   빠른 점검:  SIZES="gpt2" SEEDS="0" EPOCHS=1 ./run.sh

# 3) 결과표 (ID/OOD, baseline/비교/ours)
python analysis/make_tables.py --metrics acc f1 mcc

# 4) 분석
python analysis/judge.py                                 # (선택) LLM judge. 미실행 시 analysis/judge_cache.csv 사용
python analysis/error_analysis.py --size gpt2 --seed 0   # 천장분해·상보성·표적회수·κ
python analysis/visualize_attention.py --ckpt checkpoints/align-step3-gpt2-s0.pt \
    --s1 "How to learn Python" --s2 "How to learn Python for data science" --out figures/attn.png
```

## 주요 파일
```
paraphrase_detection.py   cloze 학습/평가  (--prompt_style --symmetric_eval
                          --balanced_sampler --prior_calibration --lm_lambda)
alignment_paraphrase.py   ours: alignment(--ablation_step 3) / bi-enc(--ablation_step 1)
datasets.py               데이터 로더 + cloze PROMPT_TEMPLATES
metrics.py                metric 후보 (acc·f1·mcc·bal_acc·auroc·auprc·kappa)
run.sh                    학습 + dump 일괄
data/get_ood_data.py      PAWS·MRPC 다운로드          data/augment.py  Quora 증강
analysis/extract_signals.py  ckpt → 신호 csv          analysis/make_tables.py  결과표
analysis/judge.py            LLM judge                analysis/error_analysis.py  분석
analysis/visualize_attention.py  attention heatmap
```

## 비교 메소드 (택2)
데이터 증강 / prompt 변경(`--prompt_style`) / symmetric(`--symmetric_eval`)
/ prior calibration / 보조 LM loss(`--lm_lambda`) 중 선택 — 어느 단일 축도 천장을 못 뚫고
**아키텍처 다양성(ours)만 유효**함을 대비한다.

---

# 자연어처리 2026-1 지정주제 기말 프로젝트: GPT-2 구축

## PART-I:

#### 다음 각 모듈에서 누락된 코드 블록을 완성해야 한다.
* `modules/attention.py`
* `modules/gpt2_layer.py`
* `models/gpt2.py`
* `classifier.py`
* `optimizer.py`

#### 다음 모듈들을 실행하여 PART-I의 구현을 테스트한다.

* `optimizer_test.py`: `optimizer.py` 구현을 테스트.
* `sanity_check.py`: GPT 모델 구현을 테스트.
* `classifier.py`: 모델을 사용한 감정 분류 수행.

## PART-II

#### 다음 모듈들을 실행하여 PART-II의 구현을 테스트한다.

* `paraphrase_detection.py`: 패러프레이즈 탐지 수행.
* `sonnet_generation.py`: 소네트 생성 수행.

**주목**: 사용하는 GPU 사양에 따라 batch_size 같은 하이퍼파라미터를 조정하여 성능을 최적화하고 메모리 부족 오류를 방지해야 한다.

#### PART-II 테스트의 핵심 포인트

두 파일에 있는 누락된 코드 블록을 완성하는 것도 중요하지만, PART-II의 핵심은 기능의 확장에 있다. GPT-2 모델을 수정하여 한 문장이 다른 문장의 패러프레이즈인지 판단하는 능력과 소네트를 생성하는 능력을 개선하는 방법에 촛점을 맞추도록 하자.

## 환경 설정
**주목**: .yml 파일의 버전을 변경하지 말것.

#### GitHub에서 Source code 내려 받기:
* 단순히 압축 파일 내려받아서 풀지 말고 GitHub의 프로젝트 리포지토리를 클론할 것.
* 프로젝트 폴더를 만들 폴더로 가서 다음 명령을 터미널에서 실행한다.
```
git clone https://github.com/kikim6114/nlp2026-final.git
```
* 소스코드 변경 사항이 있을 경우 공지가 나가므로, 그 경우 `git pull` 하여 PC의 로컬 리포지토리를 업데이트할 수 있다.

#### 파이썬 설치
* anaconda3 를 설치한다.

#### 환경 및 패키지 설치

* conda env create -f env.yml
* conda activate nlp_final  

**주의**:
* 프로젝트 PART-I을 수행하면서, 위에서 설치된 패키지만을 사용해야 하며, 별도의 다른 패키지는 허용되지 않는다.
* 모든 command 옵션이나 파라미터는 변경/추가하면 안된다.


