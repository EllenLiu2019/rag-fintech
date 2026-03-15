"""
vLLM Minimal Benchmark Runner
Three-phase performance test: Cold Start / Warm Start / Long Context Pressure
Usage: python run_benchmark.py [--config config/vllm_benchmark_config_minimal.yaml]
"""

import argparse
import asyncio
import csv
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from benchmark.prompt_builder import Qwen3PromptBuilder, PromptBuilderConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    phase: str
    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    decode_tps: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    success: bool = True
    error: str = ""
    timestamp: str = ""


@dataclass
class PhaseReport:
    phase: str
    records: list[RequestMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU memory polling (nvidia-smi)
# ---------------------------------------------------------------------------


class GpuMemoryMonitor:
    """Poll GPU peak memory via nvidia-smi in background."""

    def __init__(self, interval: float = 0.3):
        self.interval = interval
        self.peak_mb: float = 0.0
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _poll(self):
        while not self._stop.is_set():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                mem_mb = float(stdout.decode().strip().split("\n")[0])
                self.peak_mb = max(self.peak_mb, mem_mb)
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def start(self):
        self._stop.clear()
        self.peak_mb = 0.0
        self._task = asyncio.create_task(self._poll())

    async def stop(self) -> float:
        self._stop.set()
        if self._task:
            await self._task
        return self.peak_mb


# ---------------------------------------------------------------------------
# Streaming request with TTFT measurement
# ---------------------------------------------------------------------------


async def streaming_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0,
) -> RequestMetrics:
    """Send one streaming request; measure TTFT and decode throughput."""
    metrics = RequestMetrics(
        phase="",
        label="",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    t_first_token = None
    output_text_parts: list[str] = []
    t_start = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                output_text_parts.append(chunk.choices[0].delta.content)

            if hasattr(chunk, "usage") and chunk.usage:
                metrics.input_tokens = chunk.usage.prompt_tokens
                metrics.output_tokens = chunk.usage.completion_tokens

        t_end = time.perf_counter()

        metrics.latency_ms = (t_end - t_start) * 1000
        if t_first_token is not None:
            metrics.ttft_ms = (t_first_token - t_start) * 1000

        decode_duration_s = t_end - (t_first_token or t_start)
        if decode_duration_s > 0 and metrics.output_tokens > 1:
            metrics.decode_tps = (metrics.output_tokens - 1) / decode_duration_s

    except Exception as e:
        metrics.success = False
        metrics.error = f"{type(e).__name__}: {e}"
        metrics.latency_ms = (time.perf_counter() - t_start) * 1000

    return metrics


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def summarize(metrics_list: list[RequestMetrics]) -> dict:
    ok = [m for m in metrics_list if m.success]
    if not ok:
        return {"count": 0, "success_rate": 0}
    ttfts = [m.ttft_ms for m in ok]
    tps_list = [m.decode_tps for m in ok]
    return {
        "count": len(ok),
        "success_rate": len(ok) / len(metrics_list),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "decode_tps_p50": percentile(tps_list, 50),
        "decode_tps_p95": percentile(tps_list, 95),
        "latency_p50": percentile([m.latency_ms for m in ok], 50),
        "gpu_mem_peak_mb": max((m.gpu_mem_peak_mb for m in ok), default=0),
    }


# ---------------------------------------------------------------------------
# Phase A: Cold Start
# ---------------------------------------------------------------------------


async def phase_cold(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    gpu_mon: GpuMemoryMonitor,
    max_tokens: int = 256,
) -> PhaseReport:
    print("\n" + "=" * 60)
    print("Phase A: Cold Start (first request after server ready)")
    print("=" * 60)

    gpu_mon.start()
    m = await streaming_request(client, model, prompt, max_tokens=max_tokens)
    peak = await gpu_mon.stop()

    m.phase = "cold"
    m.label = "first_request"
    m.gpu_mem_peak_mb = peak

    print(f"  TTFT           : {m.ttft_ms:.1f} ms")
    print(f"  Total latency  : {m.latency_ms:.1f} ms")
    print(f"  Decode TPS     : {m.decode_tps:.1f} tokens/s")
    print(f"  Output tokens  : {m.output_tokens}")
    print(f"  GPU mem peak   : {peak:.0f} MB")
    print(f"  Success        : {m.success}")

    return PhaseReport(phase="cold", records=[m])


# ---------------------------------------------------------------------------
# Phase B: Warm Start
# ---------------------------------------------------------------------------


async def phase_warm(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    gpu_mon: GpuMemoryMonitor,
    total_runs: int = 5,
    eval_from: int = 3,
    max_tokens: int = 256,
) -> PhaseReport:
    print("\n" + "=" * 60)
    print(f"Phase B: Warm Start ({total_runs} runs, eval runs {eval_from}-{total_runs})")
    print("=" * 60)

    all_metrics: list[RequestMetrics] = []

    for i in range(1, total_runs + 1):
        gpu_mon.start()
        m = await streaming_request(client, model, prompt, max_tokens=max_tokens)
        peak = await gpu_mon.stop()

        m.phase = "warm"
        m.label = f"run_{i}"
        m.gpu_mem_peak_mb = peak
        all_metrics.append(m)
        print(f"  Run {i}: TTFT={m.ttft_ms:.1f}ms  TPS={m.decode_tps:.1f}  latency={m.latency_ms:.1f}ms")

    eval_metrics = all_metrics[eval_from - 1 :]
    stats = summarize(eval_metrics)
    print(f"\n  --- Eval runs {eval_from}-{total_runs} summary ---")
    print(f"  TTFT  p50={stats['ttft_p50']:.1f}ms  p95={stats['ttft_p95']:.1f}ms")
    print(f"  TPS   p50={stats['decode_tps_p50']:.1f} tokens/s  p95={stats['decode_tps_p95']:.1f} tokens/s")
    print(f"  GPU mem peak: {stats['gpu_mem_peak_mb']:.0f} MB")

    return PhaseReport(phase="warm", records=all_metrics)


# ---------------------------------------------------------------------------
# Phase C: Long Context Pressure
# ---------------------------------------------------------------------------


async def phase_long_context(
    client: AsyncOpenAI,
    model: str,
    gpu_mon: GpuMemoryMonitor,
    prompt_builder: Qwen3PromptBuilder,
    input_token_targets: list[int],
    output_max_tokens: int = 128,
) -> PhaseReport:
    print("\n" + "=" * 60)
    print(f"Phase C: Long Context Pressure (output fixed {output_max_tokens} tokens)")
    print("=" * 60)

    all_metrics: list[RequestMetrics] = []

    for i, target in enumerate(input_token_targets):
        prompt = prompt_builder.build(target_tokens=target, seed=i)
        actual_tokens = prompt_builder.count_tokens(prompt)
        label = f"input_{target}"
        print(f"\n  Target {target} tokens (actual={actual_tokens}, chars={len(prompt)})")

        gpu_mon.start()
        m = await streaming_request(client, model, prompt, max_tokens=output_max_tokens)
        peak = await gpu_mon.stop()

        m.phase = "long_context"
        m.label = label
        m.gpu_mem_peak_mb = peak
        all_metrics.append(m)

        print(f"    Input tokens : {m.input_tokens}")
        print(f"    Output tokens: {m.output_tokens}")
        print(f"    TTFT         : {m.ttft_ms:.1f} ms")
        print(f"    Decode TPS   : {m.decode_tps:.1f} tokens/s")
        print(f"    GPU mem peak : {peak:.0f} MB")
        print(f"    Success      : {m.success}")

    return PhaseReport(phase="long_context", records=all_metrics)


# ---------------------------------------------------------------------------
# Phase D: prefix cache
# ---------------------------------------------------------------------------


async def phase_prefix_cache(
    client: AsyncOpenAI,
    model: str,
    gpu_mon: GpuMemoryMonitor,
    prompt_builder: Qwen3PromptBuilder,
    output_max_tokens: int = 128,
) -> PhaseReport:
    print("\n" + "=" * 60)
    print("Phase D: Prefix Cache)")
    print("=" * 60)

    all_metrics: list[RequestMetrics] = []

    prompt = prompt_builder.build(target_tokens=8192, seed=0)
    _prompt = prompt_builder.build(target_tokens=8192, seed=1)
    actual_tokens = prompt_builder.count_tokens(prompt)
    _actual_tokens = prompt_builder.count_tokens(_prompt)
    label = "prefix_cache_8192"
    print(f"\n  Target 8192 tokens (actual={actual_tokens}, {_actual_tokens}, chars={len(prompt)}, {len(_prompt)})")

    for i in range(5):
        gpu_mon.start()
        if i == 4:
            m = await streaming_request(client, model, _prompt, max_tokens=output_max_tokens)
        else:
            m = await streaming_request(client, model, prompt, max_tokens=output_max_tokens)
        peak = await gpu_mon.stop()

        m.phase = "prefix_cache"
        m.label = f"{label}_run_{i + 1}"
        m.gpu_mem_peak_mb = peak
        all_metrics.append(m)

        print(
            f"    Run {i + 1}: Input tokens={m.input_tokens}  TTFT={m.ttft_ms:.1f}ms  Output tokens={m.output_tokens}  TPS={m.decode_tps:.1f}  Success={m.success}"
        )

    # Summary: prefix cache effect = TTFT drop from run1 to run2-5
    ok = [m for m in all_metrics if m.success]
    if len(ok) >= 2:
        ttft_run1 = ok[0].ttft_ms
        ttft_cached = [m.ttft_ms for m in ok[1:]]
        avg_cached = sum(ttft_cached) / len(ttft_cached)
        drop_pct = (1 - avg_cached / ttft_run1) * 100 if ttft_run1 > 0 else 0
        print("\n  --- Prefix cache effect ---")
        print(f"  TTFT run1 (cold) : {ttft_run1:.1f} ms")
        print(f"  TTFT run2-5 avg  : {avg_cached:.1f} ms")
        print(f"  TTFT reduction   : {drop_pct:.1f}% (expect >0 if prefix cache hits)")

    return PhaseReport(phase="prefix_cache", records=all_metrics)


# ---------------------------------------------------------------------------
# Phase E: Concurrency Scaling
# ---------------------------------------------------------------------------


async def phase_concurrent_scaling(
    client: AsyncOpenAI,
    model: str,
    gpu_mon: GpuMemoryMonitor,
    prompt_builder: Qwen3PromptBuilder,
    concurrent_levels: list[int] | None = None,
    input_tokens: int = 4096,
    output_max_tokens: int = 256,
    rounds: int = 3,
) -> PhaseReport:
    if concurrent_levels is None:
        concurrent_levels = [1, 2, 4, 8]

    print("\n" + "=" * 60)
    print(f"Phase E: Concurrency Scaling  (input={input_tokens} tok, output≤{output_max_tokens} tok, {rounds} rounds)")
    print("=" * 60)

    all_metrics: list[RequestMetrics] = []
    level_summaries: list[dict] = []

    for level in concurrent_levels:
        print(f"\n  --- concurrency = {level} ---")
        level_metrics: list[RequestMetrics] = []

        for r in range(rounds):
            prompts = [
                prompt_builder.build(target_tokens=input_tokens, seed=level * 100 + r * 10 + j) for j in range(level)
            ]

            gpu_mon.start()
            results = await asyncio.gather(
                *[streaming_request(client, model, p, max_tokens=output_max_tokens) for p in prompts]
            )
            peak = await gpu_mon.stop()

            for j, m in enumerate(results):
                m.phase = "concurrent_scaling"
                m.label = f"c{level}_r{r + 1}_req{j + 1}"
                m.gpu_mem_peak_mb = peak
                level_metrics.append(m)

            ok = [m for m in results if m.success]
            if ok:
                avg_ttft = sum(m.ttft_ms for m in ok) / len(ok)
                avg_tps = sum(m.decode_tps for m in ok) / len(ok)
                print(
                    f"    round {r + 1}: avg_TTFT={avg_ttft:.1f}ms  avg_TPS={avg_tps:.1f}  "
                    f"peak_GPU={peak:.0f}MB  ok={len(ok)}/{len(results)}"
                )

        all_metrics.extend(level_metrics)

        ok = [m for m in level_metrics if m.success]
        stats = summarize(ok) if ok else {}
        stats["concurrency"] = level
        level_summaries.append(stats)

        if ok:
            print(
                f"    level summary: TTFT p50={stats['ttft_p50']:.1f}ms p95={stats['ttft_p95']:.1f}ms  "
                f"TPS p50={stats['decode_tps_p50']:.1f} p95={stats['decode_tps_p95']:.1f}  "
            )

    # ---- trend table ----
    _print_scaling_trend(level_summaries)

    return PhaseReport(phase="concurrent_scaling", records=all_metrics)


def _print_scaling_trend(level_summaries: list[dict]) -> None:
    hdr = f"  {'Conc':>5} | {'TTFT_p50':>10} | {'TTFT_p95':>10} | {'TPS_p50':>10} | {'TPS_p95':>10} | {'GPU_MB':>8}"
    sep = "  " + "-" * len(hdr)

    print("\n  " + "=" * len(hdr))
    print("  Phase E  —  Concurrency Scaling Trend")
    print("  " + "=" * len(hdr))
    print(hdr)
    print(sep)

    prev_ttft: float | None = None
    prev_tps: float | None = None

    for s in level_summaries:
        c = s.get("concurrency", 0)
        ttft_p50 = s.get("ttft_p50", 0)
        ttft_p95 = s.get("ttft_p95", 0)
        tps_p50 = s.get("decode_tps_p50", 0)
        tps_p95 = s.get("decode_tps_p95", 0)
        gpu = s.get("gpu_mem_peak_mb", 0)

        ttft_tag = ""
        tps_tag = ""
        if prev_ttft and prev_ttft > 0:
            ttft_tag = f"  ({(ttft_p50 - prev_ttft) / prev_ttft * 100:+.0f}%)"
        if prev_tps and prev_tps > 0:
            tps_tag = f"  ({(tps_p50 - prev_tps) / prev_tps * 100:+.0f}%)"

        print(
            f"  {c:>5} | {ttft_p50:>8.1f}ms | {ttft_p95:>8.1f}ms | "
            f"{tps_p50:>8.1f}/s | {tps_p95:>8.1f}/s | {gpu:>7.0f}"
        )
        if ttft_tag or tps_tag:
            print(f"  {'':>5} | {ttft_tag:>10} | {'':>10} | {tps_tag:>10} |")

        prev_ttft = ttft_p50
        prev_tps = tps_p50

    # ---- inflection detection ----
    if len(level_summaries) < 3:
        return

    ttft_vals = [s.get("ttft_p50", 0) for s in level_summaries]
    tps_vals = [s.get("decode_tps_p50", 0) for s in level_summaries]
    conc_vals = [s.get("concurrency", 0) for s in level_summaries]

    print("\n  Inflection detection (TTFT jump / TPS plateau):")
    found = False

    for i in range(1, len(ttft_vals) - 1):
        prev_r = (ttft_vals[i] - ttft_vals[i - 1]) / max(ttft_vals[i - 1], 0.01)
        next_r = (ttft_vals[i + 1] - ttft_vals[i]) / max(ttft_vals[i], 0.01)
        if next_r > prev_r * 2 and next_r > 0.3:
            print(
                f"    ⚠ TTFT inflection at concurrency={conc_vals[i + 1]}: "
                f"jump {next_r * 100:.0f}% (prior step {prev_r * 100:.0f}%) — prefill contention likely"
            )
            found = True

    for i in range(1, len(tps_vals) - 1):
        prev_g = (tps_vals[i] - tps_vals[i - 1]) / max(tps_vals[i - 1], 0.01)
        next_g = (tps_vals[i + 1] - tps_vals[i]) / max(tps_vals[i], 0.01)
        if prev_g > 0.1 and next_g < prev_g * 0.3:
            print(
                f"    ⚠ TPS plateau at concurrency={conc_vals[i + 1]}: "
                f"gain dropped to {next_g * 100:.0f}% (prior step {prev_g * 100:.0f}%) — scheduler/KV budget saturated"
            )
            found = True

    if not found:
        print("    No clear inflection detected in this range.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "phase",
    "label",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "latency_ms",
    "decode_tps",
    "gpu_mem_peak_mb",
    "success",
    "error",
    "timestamp",
]


def export_csv(reports: list[PhaseReport], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for report in reports:
            for rec in report.records:
                row = asdict(rec)
                writer.writerow({k: row[k] for k in RESULT_FIELDS})
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(config_path: str):
    config_file = Path(__file__).parent / config_path
    with open(config_file) as f:
        config = yaml.safe_load(f)

    model = config["experiment"]["model"]
    base_url = config["experiment"]["base_url"]
    max_tokens = config["run"].get("max_new_tokens", 256)
    sample_prompt = config["run"]["sample_prompt"]
    long_ctx_targets = config["run"].get("long_context_tokens", [1024, 2048, 4096, 8192])

    print(f"Model      : {model}")
    print(f"Base URL   : {base_url}")
    print(f"Max tokens : {max_tokens}")

    # Initialize prompt builder with exact tokenizer for Phase C
    print(f"Loading Qwen3PromptBuilder for {model} ...")
    prompt_builder = Qwen3PromptBuilder(PromptBuilderConfig(model_id=model))
    print(f"Tokenizer loaded: vocab_size={prompt_builder.tok.vocab_size}")

    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
    gpu_mon = GpuMemoryMonitor(interval=0.1)
    reports: list[PhaseReport] = []

    # Phase A: Cold Start
    r_cold = await phase_cold(client, model, sample_prompt, gpu_mon, max_tokens=max_tokens)
    reports.append(r_cold)

    # Phase B: Warm Start
    warm_cfg = config["run"].get("warm", {})
    r_warm = await phase_warm(
        client,
        model,
        sample_prompt,
        gpu_mon,
        total_runs=warm_cfg.get("total_runs", 5),
        eval_from=warm_cfg.get("eval_from", 3),
        max_tokens=max_tokens,
    )
    reports.append(r_warm)

    # Phase C: Long Context Pressure
    r_long = await phase_long_context(
        client,
        model,
        gpu_mon,
        prompt_builder=prompt_builder,
        input_token_targets=long_ctx_targets,
        output_max_tokens=config["run"].get("long_context_output_tokens", 128),
    )
    reports.append(r_long)

    # # Phase D: Prefix Cache (same 8192-token prompt x5 → expect TTFT drop after run 1)
    # r_prefix = await phase_prefix_cache(
    #     client,
    #     model,
    #     gpu_mon,
    #     prompt_builder=prompt_builder,
    #     output_max_tokens=config["run"].get("long_context_output_tokens", 128),
    # )
    # reports.append(r_prefix)

    # Phase E: Concurrency Scaling (fixed input, increasing concurrency)
    r_concurrent = await phase_concurrent_scaling(
        client,
        model,
        gpu_mon,
        prompt_builder=prompt_builder,
        concurrent_levels=config["run"].get("concurrent_levels", [1, 2, 4, 8]),
        input_tokens=config["run"].get("concurrent_input_tokens", 4096),
        output_max_tokens=config["run"].get("concurrent_output_tokens", 256),
        rounds=config["run"].get("concurrent_rounds", 3),
    )
    reports.append(r_concurrent)

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / "results" / f"benchmark_{ts}.csv"
    export_csv(reports, out_path)

    await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Minimal Benchmark")
    parser.add_argument(
        "--config",
        default="config/vllm_benchmark_config_minimal.yaml",
        help="Path to benchmark config YAML (relative to benchmark/)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
