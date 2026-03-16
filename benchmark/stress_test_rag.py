"""
RAG retrieval E2E stress test.

Hits POST /api/search with fact-type questions from golden_dataset.json
at increasing concurrency. foc_enhance=False to keep all requests short-context
(intent classification + query rewrite + Milvus hybrid search).

Measures retrieval latency, throughput, and success rate only.

Usage:
    python -m benchmark.stress_test_rag \
        --base-url http://localhost:8000 \
        --levels 1 2 4 8 16 20 \
        --rounds 3
"""

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from itertools import cycle

import httpx


GOLDEN_DATASET = Path(__file__).parent.parent / "rag" / "evaluation" / "golden_dataset.json"

RESULT_FIELDS = [
    "concurrency",
    "round",
    "request_id",
    "question",
    "question_type",
    "status_code",
    "success",
    "error",
    "latency_ms",
    "retrieval_chunks",
    "intent",
    "timestamp",
]


@dataclass
class RequestResult:
    concurrency: int = 0
    round: int = 0
    request_id: int = 0
    question: str = ""
    question_type: str = ""
    status_code: int = 0
    success: bool = False
    error: str = ""
    latency_ms: float = 0.0
    retrieval_chunks: int = 0
    intent: str = ""
    timestamp: str = ""


def load_questions(qtype: str | None = None) -> list[dict]:
    with open(GOLDEN_DATASET) as f:
        data = json.load(f)
    samples = data["samples"]
    if qtype:
        samples = [s for s in samples if s.get("type") == qtype]
    return [{"question": s["question"], "type": s.get("type", "unknown")} for s in samples]


# ---------------------------------------------------------------------------
# Single request: POST /api/search
# ---------------------------------------------------------------------------


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    kb_id: str = "default_kb",
    doc_id: str | None = None,
    timeout: float = 60.0,
) -> RequestResult:
    result = RequestResult(
        question=question[:80],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    filters = {}
    if doc_id:
        filters["doc_id"] = doc_id

    payload = {
        "query": question,
        "kb_id": kb_id,
        "top_k": 5,
        "filters": filters,
        "mode": "hybrid",
        "foc_enhance": False,
    }

    t_start = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/api/search",
            json=payload,
            timeout=timeout,
        )

        t_end = time.perf_counter()

        result.status_code = resp.status_code
        result.latency_ms = (t_end - t_start) * 1000

        if resp.status_code == 200:
            body = resp.json()
            result.success = True
            result.retrieval_chunks = len(body.get("results", []))
            result.intent = body.get("intent", "")
        else:
            result.error = resp.text[:200]

    except Exception as e:
        result.latency_ms = (time.perf_counter() - t_start) * 1000
        result.error = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def _fmt(values: list[float], pct: int) -> str:
    if not values:
        return "—"
    return f"{percentile(values, pct):.0f}ms"


# ---------------------------------------------------------------------------
# Concurrency driver
# ---------------------------------------------------------------------------


async def run_level(
    base_url: str,
    questions: list[dict],
    concurrency: int,
    round_num: int,
    kb_id: str,
    doc_id: str | None,
) -> list[RequestResult]:
    q_iter = cycle(questions)
    batch = [next(q_iter) for _ in range(concurrency)]

    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, base_url, q["question"], kb_id, doc_id) for q in batch]
        results = await asyncio.gather(*tasks)

    for i, (r, q) in enumerate(zip(results, batch)):
        r.concurrency = concurrency
        r.round = round_num
        r.request_id = i + 1
        r.question_type = q["type"]

    return list(results)


def print_summary(results: list[RequestResult], concurrency: int):
    ok = [r for r in results if r.success]
    total = len(results)
    rate = len(ok) / total if total else 0
    lats = [r.latency_ms for r in ok]

    print(f"\n  c={concurrency}  ok={len(ok)}/{total} ({rate:.0%})")
    if lats:
        qps = len(ok) / (max(lats) / 1000) if max(lats) > 0 else 0
        print(f"    latency  p50={_fmt(lats, 50)}  p95={_fmt(lats, 95)}  max={max(lats):.0f}ms")
        print(f"    QPS      {qps:.1f} req/s")
        print(f"    chunks   avg={sum(r.retrieval_chunks for r in ok)/len(ok):.1f}")


def export_csv(results: list[RequestResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            writer.writerow({k: row[k] for k in RESULT_FIELDS})
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="RAG retrieval stress test")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--kb-id", default="default_kb")
    parser.add_argument("--doc-id", default=None, help="Filter by doc_id (optional)")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 20],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--question-type",
        default="fact",
        choices=["fact", "logic", "all"],
        help="Question type filter (default: fact for short-context only)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    qtype = None if args.question_type == "all" else args.question_type
    questions = load_questions(qtype)
    print(f"Loaded {len(questions)} questions (type={args.question_type})")
    print(f"Target: {args.base_url}/api/search")
    print("Config: foc_enhance=False, mode=hybrid, top_k=5")
    print(f"Levels: {args.levels}  Rounds: {args.rounds}")

    # Warmup
    print("\n--- Warmup ---")
    async with httpx.AsyncClient() as client:
        w = await send_request(client, args.base_url, questions[0]["question"], args.kb_id, args.doc_id)
        print(f"  {w.latency_ms:.0f}ms  chunks={w.retrieval_chunks}  {'OK' if w.success else w.error}")

    if not w.success:
        print("Warmup failed.")
        return

    # Scaling
    all_results: list[RequestResult] = []

    for level in args.levels:
        print(f"\n{'='*50}")
        print(f"Concurrency = {level}  ({args.rounds} rounds)")
        print(f"{'='*50}")

        level_results: list[RequestResult] = []
        for r in range(1, args.rounds + 1):
            results = await run_level(
                args.base_url,
                questions,
                level,
                r,
                args.kb_id,
                args.doc_id,
            )
            level_results.extend(results)

            ok = [x for x in results if x.success]
            if ok:
                avg = sum(x.latency_ms for x in ok) / len(ok)
                print(f"  round {r}: avg={avg:.0f}ms  ok={len(ok)}/{len(results)}")

        all_results.extend(level_results)
        print_summary(level_results, level)

    # Summary table
    print(f"\n{'='*75}")
    print("Retrieval Concurrency Scaling (POST /api/search, foc_enhance=False)")
    print(f"{'='*75}")
    print(f"  {'Conc':>5} | {'OK':>7} | {'p50':>8} | {'p95':>8} | {'max':>8} | {'QPS':>8} | {'Chunks':>6}")
    print(f"  {'-'*62}")

    for level in args.levels:
        lr = [r for r in all_results if r.concurrency == level]
        ok = [r for r in lr if r.success]
        lats = [r.latency_ms for r in ok]
        rate = f"{len(ok)}/{len(lr)}"

        if not lats:
            print(f"  {level:>5} | {rate:>7} | {'FAIL':>8} |")
            continue

        qps = len(ok) / (max(lats) / 1000) if max(lats) > 0 else 0
        avg_chunks = sum(r.retrieval_chunks for r in ok) / len(ok)
        print(
            f"  {level:>5} | {rate:>7} | {_fmt(lats, 50):>8} | {_fmt(lats, 95):>8} | "
            f"{max(lats):>6.0f}ms | {qps:>6.1f}/s | {avg_chunks:>5.1f}"
        )

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else (Path(__file__).parent / "results" / f"stress_retrieval_{ts}.csv")
    export_csv(all_results, out_path)


if __name__ == "__main__":
    asyncio.run(main())
