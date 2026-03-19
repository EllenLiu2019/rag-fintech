# RAG 检索压测报告

## 测试环境

| 组件 | 规格 | 备注 |
|------|------|------|
| **Kubernetes** | 阿里云 ACK (K8s 1.35) | cn-hangzhou |
| **vLLM** | Qwen3.5-9B / Qwen3.5-35B-A3B-FP8 | 详见各轮次 |
| **GPU** | NVIDIA A10 24GB / L20 48GB | 详见各轮次 |
| **Milvus** | 2.6.11 standalone | HNSW + BM25 hybrid search (RRF) |
| **Dense Embedding** | Voyage AI (voyage-3.5-lite) | remote API (cross-border) |
| **Sparse Embedding** | BGE-M3 (local) | FP16, MilvusBgeM3Embed |
| **API Pod** | 2 vCPU / 4Gi | FastAPI, single replica |

## 测试方案

- **目标 API**: `POST /api/search` (retrieval-only, 不含 LLM 生成)
- **参数**: `foc_enhance=False`, `mode=hybrid`, `top_k=5`
- **问题来源**: `golden_dataset.json` 中 33 个 fact 类问题
- **缓存状态**: retriever 层缓存已关闭，embedding 层 Redis 缓存已关闭
- **执行方式**: K8s Job (集群内网调用，无外网延迟)
- **Query Rewrite**: ~840 input tokens + ~7 output tokens per request

---

## Round 1: A10 24GB · FP8 KV Cache

**配置**: `--kv-cache-dtype fp8`, `--max-model-len 4096`, `--max-num-seqs 16`
**KV Cache**: ~2.6 GiB, 最大并发 ~16

### E2E 压测结果 (c=1/2/4, 3 轮)

| 并发 | 成功率 | p50 | p95 | max | QPS | chunks |
|------|--------|-----|-----|-----|-----|--------|
| 1 | 3/3 (100%) | 759ms | 781ms | 781ms | 3.8/s | 5.0 |
| 2 | 6/6 (100%) | 1,219ms | 2,202ms | 2,202ms | 2.7/s | 5.0 |
| 4 | 12/12 (100%) | 1,413ms | 3,945ms | 3,945ms | 3.0/s | 5.0 |

### Grafana vLLM 指标

| 指标 | 值 | 含义 |
|------|-----|------|
| TTFT P99 | 995ms | 首 token 延迟接近 1s |
| TPOT P99 | 95ms | FP8 反量化开销明显 |
| GPU SM Active | **100%** | 算力已打满 |
| DRAM Active | 60% | 显存带宽有余量 |

**判定: Compute-Bound** — A10 的 FP8 Tensor Core 性能不足，且量化/反量化运算把 GPU 算力打满。

### A10 FP8 KV 量化 Trade-off

| 配置 | TTFT | 并发上限 | GPU Util | 结论 |
|------|------|---------|----------|------|
| 不开 FP8 KV | ~500ms | 11 (KV cache 仅 1.3GB) | ~70-80% | 延迟好，并发低 |
| 开 FP8 KV | ~995ms | 16+ (KV cache 翻倍) | 100% | 延迟差，compute 打满 |

**结论: A10 不适合该场景。** 无论是否开启 FP8 KV 量化都存在明显 trade-off，query rewrite 短输出场景下延迟比并发更重要。

---

## Round 2: L20 48GB · 独占 · 无 KV 量化

**配置**: `--max-model-len 1536`, `--max-num-seqs 64`, `--max-num-batched-tokens 8192`, `--enable-chunked-prefill`
**KV Cache**: 17.55 GiB / 143,616 tokens, 理论最大并发 98.9x

### Grafana vLLM 指标对比

| 指标 | L20 (c=4) | L20 (c=32) | A10 (c=4) |
|------|-----------|------------|-----------|
| TTFT P99 | **495ms** | 4.76s | 995ms |
| TPOT P99 | **35.5ms** | 175ms | 95ms |
| Prefill Mean | 211ms | 348ms | — |
| Decode Mean | 337ms | 351ms | — |
| GPU SM Active | **33%** | **75%** | 100% |
| DRAM Active | 45% | 60% | 60% |

**判定: 非 Compute-Bound** — L20 在 c=4 时 GPU 仅用 33%，有大量扩展空间。

### 延迟分解 (API 日志)

```
POST /api/search  E2E 请求链路:

  ┌──────────────────────────┐
  │ Query Rewrite            │  0.45s (c=1) → 0.97s (c=4) → 排队增长 (c=32)
  │ vLLM 9B · L20 · BF16    │  ← 可控瓶颈, 受 prefill 串行调度影响
  └──────────────────────────┘
  ┌──────────────────────────┐
  │ Voyage Dense Embedding   │  0.35s ~ 0.99s 
  │ Remote API (cross-border)│  ← 不可控瓶颈, 跨境网络
  └──────────────────────────┘
  ┌──────────────────────────┐
  │ BGE-M3 Sparse Embedding  │  0.08 ~ 0.32s (local, 并发时排队)
  └──────────────────────────┘
  ┌──────────────────────────┐
  │ Milvus Hybrid Search     │  0.002 ~ 0.020s  ✅ 无瓶颈
  └──────────────────────────┘
```

### Query Rewrite 延迟: L20 vs A10

| 并发 | L20 | A10 (FP8 KV) | 改善 |
|------|-----|-------------|------|
| c=1 | **0.452s** | 0.710s | **36%** |
| c=2 | 0.586-0.615s | 1.14s | **46%** |
| c=4 | 0.830-0.974s | 1.37-1.47s | **34%** |

延迟随并发线性增长， **单卡 GPU 算力固定**：并发请求在 vLLM scheduler 中排队等待 prefill，chunked prefill 允许每 step batch 多个 prefill（~8192/840 ≈ 9 个），但总计算量不变。

### 高并发吞吐测试 (c=1/4/8/16/32)

**Scheduler State (c=32)**:
- Running 峰值: 16 (`max-num-batched-tokens=8192` 限制)
- Waiting 峰值: 6

**`max-num-batched-tokens` 调优对比**:

| 参数值 | SM Active | DRAM Active | Running | Waiting | 吞吐 | 延迟特征 |
|--------|-----------|-------------|---------|---------|------|---------|
| 8192 | 75% | 50% | 16 | 有排队 | ~1.3/s | 平滑 |
| 32768 | **90%** | **50%** | **32** | **0** | ~1.3/s | 锯齿状 |

**关键发现**: 两种配置的吞吐天花板相同（~1.3 req/s），因为 GPU 总算力不变。`32768` 只是把更多请求塞进单个 step，每 step 耗时更长，总产出一致。`8192` 的延迟更可控（小 batch、快 step），是更优配置。

### L20 吞吐天花板

**vLLM Query Rewrite 稳态吞吐: ~1.3 req/s (~1,100 tokens/s)**

| 工作区间 | 并发 | 延迟 | 适用场景 |
|---------|------|------|---------|
| **最佳** | c ≤ 4 | < 1s | 在线交互 |
| **可接受** | c ≤ 8 | < 2s | 后台批量 |
| **吞吐极限** | c=32 | ~4-5s | 压力测试 |

---

## Round 3: L20 48GB · 独占 · 35B MoE · FoC 条款检索

**模型**: Qwen3.5-35B-A3B-FP8 (MoE, 每 token 仅激活 3B 参数)
**配置**: `--max-model-len 15000`, `--max-num-batched-tokens 8192`, `--max-num-seqs 20`, `--gpu-memory-utilization 0.95`, `--dtype auto` (BF16 KV cache)
**场景**: FoC 条款森林分析 — ~7,000+ input tokens + ~200+ output tokens/request
**问题类型**: logic 类 (需条款推理)

### KV Cache 量化对比 (FP8 vs BF16)

| 维度 | FP8 KV Cache | BF16 KV Cache (无量化) |
|------|-------------|----------------------|
| **性能指标** | 无显著差异 | 无显著差异 |
| **条款选择质量** | 高并发下偶发输出漂移 (tokens 膨胀到 11k-13k, 耗时 50-63s) | 更稳定 |
| **JSON 解析失败** | 略多 | 略少 |

MoE 模型每 token 仅激活 3B 参数，KV cache 规模远小于 dense 模型，FP8 量化的显存节省效果有限，性能收益不显著。**推荐使用 BF16 KV cache (不量化)**，以获得更稳定的输出质量。

### `max-num-batched-tokens` 对比 (8192 vs 2048)

| 参数值 | E2E 延迟特征 | 适用场景 |
|--------|------------|---------|
| **8192** | **更平滑稳定** | ✅ 推荐 — 长 context FoC 场景 |
| 2048 | 波动较大 | 短 context 场景 |

8192 允许每个 scheduler step 处理更多 prefill tokens，减少 FoC 长输入 (~7000+ tokens) 的分片次数，E2E 延迟更可预测。

### vLLM 性能指标 (BF16 KV, max-num-batched-tokens=8192)

| 指标 | c=2 | c=3 | c=4 |
|------|-----|-----|-----|
| **Prefill Mean** | 335 ms | 417 ms | 471 ms |
| **Decode Mean** | 1.61 s | 2.71 s | 3.04 s |
| **TTFT P99** | 995ms | 2.22 s | 2.28 s |
| **TPOT P99** | 24.9 ms | 24.9 ms | 24.9 ms |
| **E2E P50** | 3.75 s | 4.03 s | 6.48 s |
| **E2E P95** | 4.88 s | 10.2 s | 12.2 s |
| **E2E P99** | 4.97 s | 11.7 s | 14.0 s |
| **Input tps (max)** | 3,768 | 4,522 | 5,612 |
| **Output tps (max)** | 174 | 192 | 225 |
| **Throughput** | — | — | 0.224 req/s (peak) |
| **GPU SM Active** | — | — | 60% |

### 关键发现

**1. TPOT 恒定 24.9ms — MoE 架构 decode 极其轻量**

35B-A3B 每 token 仅激活 3B 参数，decode 算力需求极低。无论并发多少，单 token 生成速度恒定 ~40 tokens/s，这是 MoE 模型的核心优势。

**2. E2E 尾延迟随并发急剧恶化**

| 并发 | P50 → P99 放大倍数 | 原因 |
|------|-------------------|------|
| c=2 | 3.75s → 4.97s (1.3x) | 稳定，2 请求交替 decode |
| c=3 | 4.03s → 11.7s (**2.9x**) | 部分请求排队等 prefill |
| c=4 | 6.48s → 14.0s (**2.2x**) | 全面拥塞，prefill + decode 竞争 |

P50 → P99 的差距在 c=3 时急剧放大，说明第 3 个请求触发了 prefill 排队。

**3. Decode Mean 是延迟增长的主因**

Prefill 增长幅度温和 (335ms → 471ms, +40%)，但 Decode Mean 从 1.61s 飙升到 3.04s (+89%)。这是因为 continuous batching 下更多并发请求共享 decode 时间片，每个请求完成全部 ~200 tokens 的总耗时被拉长。

**4. 吞吐线性增长但有上限**

Input tps 从 3,768 增长到 5,612 (+49%)，GPU SM Active 70% 说明理论还有余量，但 E2E 延迟已不可接受。这是 **延迟-吞吐 trade-off** 的典型表现。

### FoC 并发建议

| 场景 | 推荐并发 | E2E P99 | 理由 |
|------|---------|---------|------|
| **用户交互** (低延迟优先) | **c ≤ 2** | < 5s | P50/P99 差距小，体验稳定 |
| **批量分析** (吞吐优先) | c=4 | ~14s | 0.224 req/s 峰值吞吐 |
| **折中** | c=3 | ~12s | P50 可控，但尾延迟波动大 |

---

## 瓶颈总结

### 1. Milvus: 无瓶颈 ✅ (当前规模)

Hybrid search (HNSW dense + BM25 sparse, RRF fusion) 仅 2-20ms，c=32 时仍稳定在 20ms 以内。

> **注意**: 当前测试集合数据量较小，HNSW index 和 inverted index 完全常驻内存，OS page cache 命中率接近 100%。生产环境中若数据量增长到百万级向量、多 collection 并发访问、或 pod 冷启动后首次 load，延迟可能显著高于本测试值。需在数据规模增长后补充测试。

### 2. Query Rewrite (vLLM): 可控瓶颈

- L20 单卡吞吐天花板 ~1.3 req/s，受 GPU 算力限制
- Chunked prefill 已默认开启，`max-num-batched-tokens=8192` 是最优配置
- 扩容方案: 多副本 vLLM（水平扩展）或更快 GPU

### 3. Voyage Dense Embedding: 不可控瓶颈

- 中国大陆 → Voyage AI (美西) 跨境网络，延迟 0.35-2.55s，方差大
- 白天 (0.35-0.99s) 显著优于夜间 (0.76-2.55s)
- 优化方向: 替换为国内 embedding API 或本地部署

### 4. A10 vs L20 硬件选型结论

| 维度 | A10 24GB | L20 48GB |
|------|----------|----------|
| TTFT (c=1) | 500-995ms | **495ms** |
| TPOT P99 | 95ms | **35.5ms** |
| KV Cache | 1.3-2.6 GiB | **17.55 GiB** |
| 理论并发 | 11-16 | **98** |
| GPU 饱和点 | c=4 (100% SM) | c=32 (90% SM) |
| 吞吐天花板 | ~1.3/s (估) | **~1.3/s (实测)** |
| FP8 KV 量化 | 反效果 (compute-bound) | **不需要** |

两者吞吐天花板接近（9B 模型，短 output ，瓶颈在 prefill 计算而非 KV cache），但 **L20 在延迟指标表现较优**：单请求 TTFT 快 2x，TPOT 快 2.7x， GPU 利用率未打满。

### 5. FoC 条款分析 (35B MoE): 长上下文瓶颈

- 35B-A3B MoE 每 token 仅激活 3B，decode 极快 (TPOT 24.9ms)
- Prefill ~8,500 tokens 耗时 335-471ms，是 TTFT 的主要来源
- **FoC 并发上限 c=2** (用户交互场景)，E2E P99 < 5s
- c=3 时尾延迟骤增到 11.7s，不适合实时交互

### 硬件选型结论

| GPU | 模型 | 用途 | 推荐并发 |
|-----|------|------|---------|
| L20 #1 (共享) | 9B query rewrite | 查询改写 + 意图识别 + Sparse Embedding (800 context) | c ≤ 4 |
| L20 #2 (独占) | 35B-A3B-FP8 | 条款森林分析(8k+ context) | c ≤ 2 |

## 附：vLLM 推理优化与压测

### GPU 算力基准

部署前先对目标 GPU 做 FP16 GEMM 基准测试，确认硬件实际算力与理论峰值的偏差：

| GPU             | 矩阵规模 | 精度 | 实测 TFLOPS | 理论峰值 | 利用率 | 脚本                            |
| --------------- | -------- | ---- | ----------- | -------- | ------ | ------------------------------- |
| NVIDIA H20 96GB | 16384³   | FP16 | 139.35      | ~148     | 94%    | `benchmark/benchmark_tflops.py` |

### 推理优化策略探索 (H20 × 2)

基于 Qwen2.5-72B-Instruct-AWQ 在 H20 双卡上系统验证了多种 vLLM 推理优化方案：

| 策略                               | 配置                                          | 目标                        |
| ---------------------------------- | --------------------------------------------- | --------------------------- |
| AWQ 4-bit 量化                     | `--quantization awq`                          | 72B 模型压缩至双卡可加载    |
| Tensor Parallelism                 | `--tensor-parallel-size 2` / `4`              | 多卡并行拆分模型权重        |
| Speculative Decoding (Draft Model) | 7B-AWQ 作为 draft，`num_speculative_tokens=5` | 加速自回归解码              |
| Speculative Decoding (N-gram)      | `method: ngram`                               | 无 draft 模型的轻量投机方案 |

详见 `benchmark/basic_test.sh`。