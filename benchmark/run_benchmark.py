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

    t_start = time.perf_counter()
    t_first_token = None
    output_text_parts: list[str] = []

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
    print(f"  TPS   p50={stats['decode_tps_p50']:.1f}  p95={stats['decode_tps_p95']:.1f}")
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
        print(f"    TTFT         : {m.ttft_ms:.1f} ms")
        print(f"    Decode TPS   : {m.decode_tps:.1f} tokens/s")
        print(f"    GPU mem peak : {peak:.0f} MB")
        print(f"    Success      : {m.success}")

    return PhaseReport(phase="long_context", records=all_metrics)


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
    gpu_mon = GpuMemoryMonitor(interval=0.2)
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
