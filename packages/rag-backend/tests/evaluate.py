"""
evaluate.py — app.models 연동(Type-safe 파싱) + reference_documents 기반 평가
- 가능하면 from app.models import GenerateResponse, DocumentChunk를 불러와
  응답을 Pydantic 모델로 검증/파싱합니다.
- 불러올 수 없으면 동일 스키마의 로컬 모델로 fallback.

기본 전략:
  1) /api/generate(use_rag=True) 1회 호출 → answer + reference_documents 확보
  2) reference_documents의 id/url/title에서 doc_id 후보를 추출
  3) Retrieval 유사 지표(Precision@k/Recall@k/MRR), Generation 지표(Relevance/Faithfulness) 계산

옵션:
  --retrieval-source generate|retrieve  (기본: generate)

JSONL 입력:
  - RAG 레코드: {"id","query","expected_points"[], "expected_doc_ids"[], "must_cite", "negative_cases"[]}
  - MCQ 레코드: {"question","choices"[],"answer"}

OpenAPI 스펙 참고: /api/generate, /api/retrieve
"""
import argparse
import json
import statistics
import time
import re
from typing import List, Dict, Any, Optional

try:
    import requests
except Exception:
    requests = None

# ---------- app.models 사용 시도 ----------
GenerateResponseModel = None
DocumentChunkModel = None

try:
    # 앱 환경에서 실행되는 경우
    from app.models import GenerateResponse as _AppGenerateResponse, DocumentChunk as _AppDocumentChunk
    GenerateResponseModel = _AppGenerateResponse
    DocumentChunkModel = _AppDocumentChunk
except Exception:
    # 독립 실행 시 로컬 Pydantic 모델 사용
    try:
        from pydantic import BaseModel
    except Exception:
        BaseModel = object  # 안전장치 (pydantic 미설치 환경에서도 동작은 하게)

    class DocumentChunkFallback(BaseModel):
        id: Optional[str] = None
        chunk_text: str
        chunk_index: int
        title: str
        url: str
        source_type: str
        score: Optional[float] = None

    class GenerateResponseFallback(BaseModel):
        response: str
        reference_documents: Optional[List[DocumentChunkFallback]] = []
        prompt: Optional[str] = None
        question: Optional[str] = None
        elapsed_ms: int

    GenerateResponseModel = GenerateResponseFallback
    DocumentChunkModel = DocumentChunkFallback

# ---------- 공통 유틸 ----------
def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

def token_set(text: str) -> set:
    for ch in [",", ".", ":", ";", "?", "!", "(", ")", "[", "]", "{", "}", "\"", "'"]:
        text = text.replace(ch, " ")
    return set([t for t in text.lower().split() if t])

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ---------- Metrics ----------
def precision_at_k(ids: List[str], expected_ids: List[str], k: int) -> float:
    topk = ids[:k]
    if not topk:
        return 0.0
    hit = sum(1 for x in topk if x in expected_ids)
    return hit / len(topk)

def recall_at_k(ids: List[str], expected_ids: List[str], k: int) -> float:
    if not expected_ids:
        return 1.0
    topk = ids[:k]
    hit = sum(1 for x in expected_ids if x in topk)
    return hit / len(expected_ids)

def mrr(ids: List[str], expected_ids: List[str]) -> float:
    for rank, rid in enumerate(ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0

def relevance(answer: str, expected_points: List[str]) -> float:
    if not expected_points:
        return 1.0
    ans_norm = normalize(answer)
    c = 0
    for p in expected_points:
        if normalize(p) in ans_norm:
            c += 1
        else:
            ptoks = token_set(p)
            if ptoks and (ptoks & token_set(ans_norm)):
                c += 1
    return c / len(expected_points)

def faithfulness(answer: str, reference_docs: List[Dict[str, Any]], expected_points: List[str]) -> float:
    if not expected_points:
        return 1.0
    if not reference_docs:
        return 0.0
    ref_text = " ".join([d.get("chunk_text", "") for d in reference_docs]).lower()
    ans_norm = normalize(answer)
    credit = 0
    for p in expected_points:
        p_norm = normalize(p)
        if p_norm in ans_norm or (token_set(p_norm) & token_set(ans_norm)):
            if (p_norm in ref_text) or (token_set(p_norm) & token_set(ref_text)):
                credit += 1
    return credit / len(expected_points)

def contains_negative(answer: str, negatives: List[str]) -> bool:
    ans = normalize(answer)
    for n in negatives or []:
        if normalize(n) in ans:
            return True
    return False

# ---------- API ----------
def call_retrieve(base_url: str, query: str, top_k: int = 5, session=None) -> List[Dict[str, Any]]:
    if session is None:
        import requests as _r
        session = _r.Session()
    payload = {"query": query, "top_k": top_k}
    r = session.post(f"{base_url}/api/retrieve", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["chunks"]

def call_generate(base_url: str, query: str, max_tokens: int = 512, use_rag: bool = True, session=None) -> Dict[str, Any]:
    if session is None:
        import requests as _r
        session = _r.Session()
    payload = {"query": query, "use_rag": use_rag, "max_tokens": max_tokens}
    r = session.post(f"{base_url}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

# ---------- Helper ----------
def ids_from_refs(refs: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for d in refs or []:
        # id 필드가 있으면 최우선
        did = d.get("id") or ""
        if did:
            ids.append(str(did))
            continue
        # url/title fallback
        url = d.get("url") or ""
        candidate = url.split("/")[-1] if url else d.get("title") or ""
        ids.append(str(candidate))
    return ids


def _first_hit_rank(ids: List[str], expected_ids: List[str]):
    for i, rid in enumerate(ids, start=1):
        if rid in expected_ids:
            return i
    return None

def _print_rag_sample(idx, item_id, q, expected_ids, used_ids, p_at_k, r_at_k, rr, rel, faith, must_cite, neg_hit, negatives, answer, refs, used_k):
    print(f"\n--- [{idx}] {item_id} ------------------------------")
    print(f"Q: {q}")
    if answer:
        ans_short = re.sub(r"\s+", " ", str(answer))[:240]
        print(f"A: {ans_short}")
    print(f"expected_doc_ids: {expected_ids}")
    print(f"used_doc_ids    : {used_ids}")
    fh = _first_hit_rank(used_ids, expected_ids)
    print(f"first_hit_rank  : {fh if fh is not None else 'None'} (k={used_k})")
    print(f"precision@k={p_at_k:.3f} | recall@k={r_at_k:.3f} | RR={rr:.3f}")
    print(f"relevance={rel:.3f} | faithfulness={faith:.3f}")
    if must_cite and not refs:
        print("! must_cite violated: no reference_documents")
    if neg_hit:
        print(f"! negative matched: {negatives}")

def _print_mcq_sample(idx, q, gold, pred, correct, latency_ms, raw_answer):
    print(f"\n--- [{idx}] MCQ -----------------------------------")
    print(f"Q: {q}")
    print(f"Gold: {gold}")
    print(f"Pred: {pred} | correct={correct} | latency_ms={latency_ms}")
    body = re.sub(r"\s+", " ", str(raw_answer))[:240]
    print(f"Model raw: {body}")

# ---------- RAG 평가 ----------
def eval_rag_records(rows: List[Dict[str, Any]], base_url: str, k: Optional[int], dry_run: bool,
                     retrieval_source: str = "generate", verbose: bool = False) -> Dict[str, Any]:
    ret_precs, ret_recalls, ret_mrrs = [], [], []
    gens_rel, gens_faith = [], []
    neg_violations = []
    eff_ks: List[int] = []
    sess = None
    if not dry_run:
        import requests as _r
        sess = _r.Session()

    for item in rows:
        if "query" not in item:
            continue
        q = item["query"]
        expected_ids = item.get("expected_doc_ids", [])
        expected_points = item.get("expected_points", [])
        negatives = item.get("negative_cases", [])
        must_cite = item.get("must_cite", False)

        # 1) /api/generate 호출
        if dry_run:
            refs = [{"id": eid, "chunk_text": "dummy text", "chunk_index": 0, "title": f"title_{eid}", "url": f"http://example/{eid}", "source_type": "dryrun", "score": 1.0} for eid in expected_ids]
            gen_obj = GenerateResponseModel.model_validate({  # type: ignore
                "response": " ".join(expected_points) or "모의 응답",
                "reference_documents": refs,
                "prompt": f"[PROMPT]{q}",
                "question": q,
                "elapsed_ms": 0
            }) if hasattr(GenerateResponseModel, "model_validate") else GenerateResponseModel(
                response=" ".join(expected_points) or "모의 응답",
                reference_documents=refs,
                prompt=f"[PROMPT]{q}",
                question=q,
                elapsed_ms=0
            )
        else:
            raw = call_generate(base_url, q, max_tokens=512, use_rag=True, session=sess)
            # 타입 세이프 파싱
            if hasattr(GenerateResponseModel, "model_validate"):
                gen_obj = GenerateResponseModel.model_validate(raw)  # pydantic v2
            else:
                gen_obj = GenerateResponseModel(**raw)  # pydantic v1 or fallback

        answer = gen_obj.response
        # gen_obj.reference_documents는 이미 모델 객체이므로 dict 리스트로 변환
        refs_list: List[Dict[str, Any]] = []
        for d in (gen_obj.reference_documents or []):
            # pydantic model일 수도, dict일 수도 있음
            if hasattr(d, "model_dump"):
                refs_list.append(d.model_dump())
            elif hasattr(d, "dict"):
                refs_list.append(d.dict())
            elif isinstance(d, dict):
                refs_list.append(d)
            else:
                refs_list.append(json.loads(json.dumps(d, default=lambda o: getattr(o, "__dict__", {}))))

        # Auto-k 기준: generate가 준 reference_documents 개수
        auto_k = max(len(refs_list), 1)

        # 2) Retrieval 지표용 used_ids 선택
        if retrieval_source == "retrieve" and not dry_run:
            # retrieve 호출도 k(or auto_k)에 맞춰 동기화
            top_k = k or auto_k
            retrieved = call_retrieve(base_url, q, top_k=top_k, session=sess)
            retrieved_ids = ids_from_refs(retrieved)
        else:
            retrieved_ids = ids_from_refs(refs_list)

        # 최종 평가 k 결정
        effective_k = k or max(len(retrieved_ids), 1)
        eff_ks.append(effective_k)

        p_at_k = precision_at_k(retrieved_ids, expected_ids, effective_k)
        r_at_k = recall_at_k(retrieved_ids, expected_ids, effective_k)
        rr = mrr(retrieved_ids, expected_ids)
        ret_precs.append(p_at_k)
        ret_recalls.append(r_at_k)
        ret_mrrs.append(rr)

        # 3) Generation & 안전성
        neg_hit = False
        if must_cite and not refs_list:
            neg_violations.append((item.get("id", q), "must_cite_but_no_refs"))
            neg_hit = True
        if contains_negative(answer, negatives):
            neg_violations.append((item.get("id", q), "answer_contains_negative_case"))
            neg_hit = True

        rel = relevance(answer, expected_points)
        fai = faithfulness(answer, refs_list, expected_points)
        gens_rel.append(rel)
        gens_faith.append(fai)

        if verbose:
            _print_rag_sample(idx=item.get('id', '?'), item_id=item.get('id', '?'), q=q, expected_ids=expected_ids,
                              used_ids=retrieved_ids, p_at_k=p_at_k, r_at_k=r_at_k, rr=rr,
                              rel=rel, faith=fai, must_cite=must_cite, neg_hit=neg_hit, negatives=negatives,
                              answer=answer, refs=refs_list, used_k=effective_k)

    summary = {
        "retrieval_source": retrieval_source,
        "retrieval": {
            "used_k": (k if k is not None else "auto"),
            "avg_eval_k": round(statistics.mean(eff_ks), 3) if eff_ks else None,
            "precision@k_mean": round(statistics.mean(ret_precs), 3) if ret_precs else 0.0,
            "recall@k_mean": round(statistics.mean(ret_recalls), 3) if ret_recalls else 0.0,
            "mrr_mean": round(statistics.mean(ret_mrrs), 3) if ret_mrrs else 0.0,
        },
        "generation": {
            "relevance_mean": round(statistics.mean(gens_rel), 3) if gens_rel else 0.0,
            "faithfulness_mean": round(statistics.mean(gens_faith), 3) if gens_faith else 0.0,
        },
        "neg_violations": neg_violations
    }
    return summary

# ---------- MCQ 그대로 유지 ----------
ANSWER_PATTERN = re.compile(r"\[정답\]\s*(.+?)(?:\n|\r|$)", re.IGNORECASE)

def build_prompt_mcq(q: Dict[str, Any]) -> str:
    return (
        "다음 질문에 대해 보기 중 정답을 한 개 고르고, 이유를 간단히 설명하세요.\n"
        f"[질문] {q['question']}\n"
        f"[보기] {q['choices']}\n\n"
        "반드시 아래 형식을 정확히 지키세요.\n"
        "[정답] (위 보기 중 하나 그대로)\n"
        "[이유] (간단 설명)\n"
    )

def extract_answer(text: str) -> str:
    m = ANSWER_PATTERN.search(text or "")
    return m.group(1).strip() if m else ""

def normalize_strict(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def is_correct(pred: str, gold: str, choices: List[str]) -> bool:
    pred_n = normalize_strict(pred)
    gold_n = normalize_strict(gold)
    if pred_n == gold_n:
        return True
    for c in choices:
        c_n = normalize_strict(c)
        if c_n == gold_n and c_n in pred_n:
            return True
    return False

def eval_mcq_records(rows: List[Dict[str, Any]], base_url: str, k: Optional[int], verbose: bool = False) -> Dict[str, Any]:
    import requests as _r
    sess = _r.Session()
    correct = 0
    latencies_ms = []
    failures = []

    for q in rows:
        if not {"question","choices","answer"} <= q.keys():
            continue
        prompt = build_prompt_mcq(q)
        try:
            t0 = time.perf_counter()
            resp = sess.post(f"{base_url}/api/generate",
                             json={"query": prompt, "use_rag": True, "max_tokens": 1024},
                             timeout=60)
            t1 = time.perf_counter()
        except _r.RequestException as e:
            failures.append({"q": q["question"], "error": f"request_error: {e}"})
            continue

        if resp.status_code != 200:
            failures.append({"q": q["question"], "error": f"status_{resp.status_code}", "body": resp.text[:300]})
            continue

        try:
            data = resp.json()
        except Exception as e:
            failures.append({"q": q["question"], "error": f"json_error: {e}", "body": resp.text[:300]})
            continue

        # 타입 세이프 파싱(선택)
        try:
            if hasattr(GenerateResponseModel, "model_validate"):
                gen_obj = GenerateResponseModel.model_validate(data)  # v2
            else:
                gen_obj = GenerateResponseModel(**data)  # v1/fallback
            raw_answer = gen_obj.response
            elapsed_ms = gen_obj.elapsed_ms
        except Exception:
            raw_answer = data.get("response","")
            elapsed_ms = data.get("elapsed_ms")

        latencies_ms.append(int(elapsed_ms) if isinstance(elapsed_ms,int) else int((t1-t0)*1000))

        pred = extract_answer(raw_answer)
        ok = is_correct(pred, q["answer"], q["choices"])
        if ok:
            correct += 1
        if verbose:
            lat = (elapsed_ms if isinstance(elapsed_ms, int) else int((t1-t0)*1000))
            _print_mcq_sample(idx=q.get('id', '?'), q=q['question'], gold=q['answer'], pred=pred, correct=ok, latency_ms=lat, raw_answer=raw_answer)
        time.sleep(0.05)

    total = len([r for r in rows if {"question","choices","answer"} <= r.keys()])
    return {
        "total": total,
        "correct": correct,
        "accuracy": round((correct/total) if total else 0.0, 4),
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p50": round(statistics.median(latencies_ms), 2) if latencies_ms else None,
        "latency_ms_p95": round(statistics.quantiles(latencies_ms, n=20)[18], 2) if len(latencies_ms) >= 20 else None,
        "failures": failures[:10],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="JSONL 경로")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--k", type=int, default=None,
                        help="precision/recall/MRR 계산 시 사용할 k (생략 시 reference_documents 길이를 자동 사용)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", choices=["auto","rag","mcq"], default="auto")
    parser.add_argument("--retrieval-source", choices=["generate","retrieve"], default="generate")
    parser.add_argument("--verbose", action="store_true", help="샘플별 상세 비교 출력")
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    mode = args.task
    if mode == "auto":
        first = next((r for r in rows if r), {})
        mode = "mcq" if {"question","choices","answer"} <= first.keys() else "rag"

    t0 = time.time()
    if mode == "mcq":
        result = eval_mcq_records(rows, base_url=args.base_url, k=args.k, verbose=args.verbose)
    else:
        result = eval_rag_records(rows, base_url=args.base_url, k=args.k, dry_run=args.dry_run,
                                  retrieval_source=args.retrieval_source, verbose=args.verbose)
    elapsed = time.time() - t0
    print("# Evaluation Summary")
    print(json.dumps({"mode": mode, "result": result, "elapsed_s": round(elapsed,2)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
