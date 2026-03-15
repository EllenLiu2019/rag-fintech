#!/usr/bin/env bash
# ==============================================================================
# vLLM inference optimization experiments on AutoDL (NVIDIA H20 96GB × 2-4)
#
# Experiments conducted while studying vLLM source code to understand:
#   - Tensor parallelism behavior at different shard counts
#   - AWQ quantization memory savings vs quality trade-offs
#   - Speculative decoding acceptance rates (ngram vs draft model)
#   - Serving throughput under RAG-like workloads
#
# Hardware: AutoDL H20 96GB instances
# ==============================================================================

# ---------- vLLM bench sweep (0.5B sanity check, validating sweep tooling) ----

vllm bench sweep serve \
    --serve-cmd 'vllm serve Qwen/Qwen2.5-0.5B-Instruct' \
    --bench-cmd 'vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/tests/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/tests/serve_hparams.json \
    --bench-params benchmarks/tests/bench_hparams.json \
    -o benchmarks/tests/results/basic_test

vllm bench sweep serve \
    --serve-cmd 'vllm serve Qwen/Qwen2.5-0.5B-Instruct' \
    --bench-cmd 'vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/tests/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/tests/serve_hparams_medium.json \
    --bench-params benchmarks/tests/bench_hparams_medium.json \
    -o benchmarks/tests/results/basic_test_medium

vllm bench sweep serve \
    --serve-cmd 'vllm serve Qwen/Qwen2.5-0.5B-Instruct' \
    --bench-cmd 'vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/tests/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/tests/serve_hparams_large.json \
    --bench-params benchmarks/tests/bench_hparams_large.json \
    -o benchmarks/tests/results/basic_test_large

# ---------- 72B dense: tensor-parallel-size 4 (dummy weights, memory layout test) ----

vllm serve "Qwen/Qwen2.5-72B-Instruct" \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --max-model-len 4096 \
  --enforce-eager \
  --load-format dummy

# ---------- 72B AWQ: baseline serving (TP=2, no speculation) ----------

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve "Qwen/Qwen2.5-72B-Instruct-AWQ" \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --quantization awq \
  2>&1 | tee /root/projects/vllm/vllm_serve_basic.log

# ---------- 72B AWQ: speculative decoding — ngram (no draft model) ----------

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# vllm serve "Qwen/Qwen2.5-3B-Instruct" \
#   --gpu-memory-utilization 0.75 \
#   --max-num-seqs 5 \
#   --generation-config vllm \
#   --speculative_config '{"method": "ngram", "num_speculative_tokens": 5}' \
#   2>&1 | tee /root/projects/vllm/vllm_serve_speculative_ngram.log

# ---------- 72B AWQ: speculative decoding — 7B draft model ----------

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve "Qwen/Qwen2.5-72B-Instruct-AWQ" \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --quantization awq \
  --speculative_config '{ "method": "draft_model", "model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "num_speculative_tokens": 5}' \
  2>&1 | tee /root/projects/vllm/vllm_serve_speculative_draft_model.log

# ---------- GuideLLM: RAG scenario benchmark (against 72B AWQ baseline) ----------
# Report output: benchmarks.html (interactive GuideLLM report)

guidellm benchmark \
  --target "http://localhost:8000" \
  --scenario rag \
  --rate 16 \
  --max-seconds 120 \
  --warmup 0.1 \
  --cooldown 0.1 \
  --max-errors 10 \
  --request-formatter-kwargs '{"extras": {"body": {"temperature": 0}}}'