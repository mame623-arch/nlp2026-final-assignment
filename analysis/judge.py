"""LLM judge — quora-dev 전수를 A/B/OK + tag 로 분류해 천장 분해(error_analysis)의 입력 생성.

  판정(데이터셋 라벨에 대한):
    A  = 라벨 오류(명백히 반대)  B = 모호(합의 불가)  OK = 정상(라벨 맞음, 모델이 틀린 것)
  tag = scope / lexical_gap / entity / negation / quantity / word_order / identical / unrelated / other

결과: analysis/judge_cache.csv (id, verdict, tag, conf, reason) — 캐시·재개 지원(중단 후 재실행 가능).
API 키: ~/.openai_key. 동시 호출(ThreadPool). 대량(40k)·rate-limit 우려 시 OpenAI Batch API로 옮겨도 됨.

사용: python analysis/judge.py [--limit N] [--model gpt-4o-mini] [--workers 4] [--max_per_run N]
※ judge_cache.csv 가 이미 있으면 재실행 불필요 — 그대로 error_analysis.py 가 사용한다.
"""
import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "data", "quora-dev.csv")
CACHE = os.path.join(ROOT, "analysis", "judge_cache.csv")
KEYFILE = os.path.expanduser("~/.openai_key")

RUBRIC = """You classify a Quora question pair for paraphrase-detection error analysis.
You are given two questions and the DATASET label (1 = the dataset says they are duplicates/paraphrases, 0 = not).

Decide a verdict about the DATASET LABEL (not about a model):
- "A" (label error): the dataset label is clearly WRONG — the correct answer is the opposite. Bar: you would confidently bet the label is a mistake.
- "B" (ambiguous): genuinely contested — competent readers could reasonably disagree; no single correct answer. You must be able to state two readings.
- "OK" (clean): the dataset label is correct and a competent reader would agree.
Tie-break: if you can write one sentence explaining why the label is correct, it is NOT "A".

Also give the dominant linguistic relation/difference:
tag in {scope, lexical_gap, entity, negation, quantity, word_order, identical, unrelated, other}
- scope: one question is more specific/general; differ in coverage/quantifier or an extra clause.
- lexical_gap: same meaning, different words (synonyms/paraphrase).
- entity: differ in a named entity. quantity: differ in a number/amount.
- negation: a negation/antonym flips meaning. word_order: same words, different order/structure.
- identical: essentially the same surface. unrelated: about different things. other: none fits.

Reply ONLY as compact JSON: {"verdict":"A|B|OK","tag":"...","conf":0.0-1.0,"reason":"<=12 words"}"""


def load_key():
  with open(KEYFILE) as f:
    return f.read().strip()


def load_dev():
  rows = []
  with open(DEV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter="\t"):
      rows.append((r["id"].lower().strip(), r["sentence1"], r["sentence2"], int(float(r["is_duplicate"]))))
  return rows


def load_done():
  done = set()
  if os.path.exists(CACHE):
    with open(CACHE) as f:
      for r in csv.DictReader(f):
        if r["verdict"] in ("A", "B", "OK"):     # ERR 은 재judge 대상
          done.add(r["id"])
  return done


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--model", default="gpt-4o-mini")
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--max_per_run", type=int, default=0, help="이번 실행 최대 판정 수(rate 보호, 0=무제한)")
  args = ap.parse_args()

  client = OpenAI(api_key=load_key(), max_retries=0, timeout=60)
  rows = load_dev()
  done = load_done()
  todo = [r for r in rows if r[0] not in done]
  if args.limit:
    todo = todo[:args.limit]
  if args.max_per_run:
    todo = todo[:args.max_per_run]
  print(f"dev={len(rows)}  done={len(done)}  todo={len(todo)}  model={args.model}", flush=True)

  new_file = not os.path.exists(CACHE)
  fh = open(CACHE, "a", newline="")
  w = csv.writer(fh)
  if new_file:
    w.writerow(["id", "verdict", "tag", "conf", "reason"])
  lock = threading.Lock()
  counts = {"n": 0, "err": 0}

  def judge(row):
    sid, s1, s2, gold = row
    user = (f'Question 1: "{s1}"\nQuestion 2: "{s2}"\n'
            f'Dataset label: {gold} ({"duplicates" if gold == 1 else "not duplicates"})')
    for attempt in range(4):
      try:
        resp = client.chat.completions.create(
            model=args.model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": user}],
            max_tokens=80)
        d = json.loads(resp.choices[0].message.content)
        return (sid, d.get("verdict", "?"), d.get("tag", "?"),
                d.get("conf", ""), str(d.get("reason", ""))[:80])
      except Exception as e:
        if attempt == 3:
          return (sid, "ERR", "ERR", "", str(e)[:60])
        time.sleep(2 * (attempt + 1))

  t0 = time.time()
  with ThreadPoolExecutor(max_workers=args.workers) as ex:
    for res in as_completed([ex.submit(judge, r) for r in todo]):
      sid, v, tag, conf, reason = res.result()
      with lock:
        w.writerow([sid, v, tag, conf, reason]); fh.flush()
        counts["n"] += 1
        if v == "ERR":
          counts["err"] += 1
        if counts["n"] % 500 == 0:
          el = time.time() - t0
          print(f"  {counts['n']}/{len(todo)}  err={counts['err']}  {el:.0f}s", flush=True)
  fh.close()
  print(f"[done] judged {counts['n']}  errors {counts['err']}  -> {CACHE}", flush=True)


if __name__ == "__main__":
  main()
