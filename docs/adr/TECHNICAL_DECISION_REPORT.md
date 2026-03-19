# 技术选型与性能优化报告

## 摘要

本报告系统阐述保险智能问答 RAG 系统在**检索架构设计、推理引擎选型、GPU 硬件适配、性能调优**方面的技术决策与实验依据。系统面向保险条款问答与理赔辅助场景，需要在**结构化条款推理准确性**与**低延迟在线服务**之间取得平衡。

核心结论：
1. **传统向量检索不足以覆盖保险条款的层级推理需求**，FoC (Forest of Clauses) 条款森林检索将条款选择准确率提升至可用水平
2. **自部署 vLLM 在延迟可控性和数据隐私上优于 API 调用**，配合 MoE 模型可在单卡上实现合理的质量-成本平衡
3. **GPU 选型需匹配任务特征**：A10 适合极低并发短文本，L20 是中等负载的最优性价比选择
4. **Guided Decoding 从根本上解决高并发下的 JSON 输出合规问题**，消除了 regex fallback 的不确定性

---

## 1. 问题定义：为什么保险场景需要特殊的 RAG 架构

### 1.1 保险条款的结构化特征

保险条款文档具有严格的层级结构（部分 → 条 → 款 → 项），条款之间存在**引用、排除、附加**等逻辑关系。一个用户问题（如"主险和附加险的保障范围有什么区别？"）可能需要同时检索：

- 主险保险责任条款（第六条）
- 主险责任免除条款（第十条）
- 附加险保险责任条款（附加条款第二条）
- 附加险责任免除条款（附加条款第五条）

这些条款**分布在文档的不同位置**，语义相似度不一定高，但在逻辑上构成完整答案。

### 1.2 通用方案的局限性

| 方案 | 问题 |
|------|------|
| 直接调用通用 LLM API（ChatGPT / 文心一言） | 无法获取私有保险条款知识；幻觉风险高；数据合规问题 |
| 标准向量检索 RAG | 依赖语义相似度，无法捕捉条款间的层级与逻辑关系；跨章节检索命中率低 |
| 全文检索 + 关键词匹配 | 无法理解用户意图的语义层面；"保障范围区别"无法匹配具体条款标题 |

### 1.3 系统目标

| 维度 | 指标 | 目标值 |
|------|------|--------|
| **准确性** | FoC 条款命中率 | ≥ 90%（相关条款全部命中） |
| **延迟** | 在线问答 E2E P95 | < 5s（含检索 + 生成） |
| **并发** | 在线交互并发 | ≥ 2 用户同时查询 |
| **成本** | GPU 资源 | 单卡 L20 可承载核心负载 |
| **合规** | 数据驻留 | 模型自部署，数据不出境 |

---

## 2. 架构决策：三路并发检索 + 模型路由

### 2.1 检索架构：为什么需要三条路径

系统采用 `asyncio.gather()` 并发执行三条独立检索路径，每条路径解决一类特定的检索需求：

```
用户问题
    │
    ├─→ FoC Retriever     条款结构推理，精准定位条款 ID
    ├─→ Graph Retriever    条款实体关系遍历，保障/除外责任路径
    └─→ Vector Retriever   语义相似度兜底，覆盖长尾问题
    │
    └─→ 结果合并 → LLM 推理生成最终答案
```

**决策依据：**

| 路径 | 解决的问题 | 不可替代性 |
|------|-----------|-----------|
| **FoC** | 条款层级推理（"主险 vs 附加险"） | 向量检索无法感知文档结构 |
| **Graph** | 条款实体关系链（沿 INCLUDE/NOT_INCLUDE 边遍历保障→除外→豁免路径） | 双重否定结构无法用相似度表征 |
| **Vector** | 语义模糊查询（"这个产品怎么样"） | FoC/Graph 需要明确意图，无法处理开放式问题 |

### 2.2 FoC 条款森林：核心创新点

**Forest of Clauses (FoC)** 是本系统的核心检索策略。它将保险条款文档的层级目录结构（Markdown heading 层级）构建为一棵树，每个节点记录条款 ID、标题、层级和关联的文本块数量。

**工作流程：**

```
1. 文档 Ingestion 阶段
   PDF → LlamaParse → Markdown
       → ClauseForestBuilder 解析层级标题（第X部分 → 第X条 → (X)）
       → 构建 ClauseForest 树结构，持久化到 PostgreSQL (JSONB)
       → 每个叶子节点的文本块 → Milvus 向量化存储

2. 查询阶段
   用户问题 + ClauseForest Markdown 树 → LLM（35B MoE）
       → 输出: {"relevant_clause_ids": [11, 12, 13], "reasoning": "..."}
       → 按 clause_id 批量从 Milvus 拉取原文 chunk
```

**为什么让 LLM 做条款路由而非规则匹配：**

- 用户问题是自然语言，与条款标题之间存在语义 gap
- 条款选择需要理解层级关系（父条款 = 统领性定义，子条款 = 具体细则）
- `<no chunk>` 节点的处理需要推理能力（选其父节点）

### 2.3 模型路由：按任务复杂度分级

```
                        ┌─────────────────────┐
                        │   Intent Router     │
                        │   (9B: fact/logic)  │
                        └──────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ Tier 1: 9B  │  │ Tier 2: 35B  │  │ Tier 3: DeepSeek │
    │ Query Rewrite│  │ FoC 条款推理  │  │ 理赔资格推理      │
    │ 意图识别     │  │ 智能问答      │  │ 医学编码 · TNM    │
    │ HyDE        │  │              │  │                  │
    └─────────────┘  └──────────────┘  └──────────────────┘
      高频 · 低延迟     中频 · 中等推理     低频 · 高精度 · 0 容错
      ~840 tokens in    ~8000 tokens in     多轮工具调用
      ~7 tokens out     ~200 tokens out     CoT 推理链
```

**决策依据：**

- 9B Dense 模型处理查询改写等结构化短任务，延迟 < 500ms，无需大参数量
- 35B MoE（每 token 激活 3B）处理 FoC 条款推理，兼顾推理能力与推理速度
- DeepSeek-Reasoner 处理理赔最终决策，0 容错场景直接使用最强模型 API

---

## 3. 推理引擎决策：自部署 vLLM vs API 调用

### 3.1 为什么选择自部署

| 维度 | 自部署 vLLM | API 调用 (DashScope等) |
|------|-----------|---------------------|
| **数据隐私** | 数据不出本地集群 ✅ | 数据经过第三方 |
| **延迟可控** | P99 可预测，无网络抖动 ✅ | 受 API 排队和网络影响 |
| **成本模型** | 固定 GPU 租赁费 | 按 token 计费，高频场景成本不可控 |
| **调优自由度** | 可调 KV cache、batched tokens 等 ✅ | 黑箱 |
| **可观测性** | Prometheus 指标全链路可见 ✅ | 有限 |
| **运维复杂度** | 需自建 K8s + GPU 调度 ❌ | 零运维 |

**结论：** 对于保险等金融场景，数据隐私和延迟可控性是硬性要求，自部署是必要选择。运维复杂度通过 K8s + HAMi + HPA 实现了自动化。

### 3.2 vLLM 关键配置与调优

vLLM 的核心优势是 **PagedAttention** + **Continuous Batching**，但具体参数需根据任务特征调整：

| 参数 | 9B (查询改写) | 35B (FoC 条款分析) | 调优依据 |
|------|-------------|-------------------|---------|
| `--max-model-len` | 1536 | 15000 | 按任务最大 context 长度设置，减少 KV cache 浪费 |
| `--max-num-batched-tokens` | 8192 | 8192 | 8192 是长 context 场景的最优值（详见 §4.3） |
| `--max-num-seqs` | 64 | 20 | 9B 短任务可高并发；35B 长 context 需限制 |
| `--gpu-memory-utilization` | 0.90 | 0.95 | 35B 模型权重更大，需更高显存利用率 |
| `--kv-cache-dtype` | auto (BF16) | auto (BF16) | FP8 量化对 MoE 模型无显著收益（详见 §4.4） |
| `--enable-chunked-prefill` | ✅ | ✅ | 允许多请求 prefill 交叉，减少排队 |

### 3.3 Guided Decoding：结构化输出的工程保障

高并发下 LLM 输出 JSON 的解析失败率显著上升（`reasoning` 字段截断、markdown 包装、引号不配对）。

**Guided Decoding** 在 token 采样阶段用有限状态机 (FSM) 约束输出，保证 100% 合法 JSON：

```python
CLAUSE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_clause_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "reasoning": {"type": "string"},
    },
    "required": ["relevant_clause_ids", "reasoning"],
}

# VLLm.generate() 通过 extra_body 传递
extra_body["guided_json"] = CLAUSE_SELECTION_SCHEMA
```

| 方案 | JSON 合规率 | 性能开销 | 局限 |
|------|-----------|---------|------|
| 无约束 + regex fallback | ~85-90% | 0 | 仍有 10-15% 解析失败 |
| Guided Decoding (outlines/xgrammar) | **100%** | 首次编译 ~1-3s，后续缓存 | 与 thinking mode 不兼容 |

**冷启动风险与缓解：** Guided Decoding 的 FSM 编译发生在首次遇到某个 JSON Schema 时。在 HPA 扩容场景中，新 Pod 启动后首个请求会同时遭遇**模型权重加载 + CUDA Graph 预热 + FSM 编译**的三重冷启动惩罚。缓解方案：在 K8s ReadinessProbe 通过后、切入生产流量前，由 `postStart` 生命周期钩子向新 Pod 发送一次携带完整 `CLAUSE_SELECTION_SCHEMA` 的 Dummy 推理请求，确保 FSM 缓存预构建完毕。

---

## 4. GPU 硬件选型：数据驱动的决策

### 4.1 实验设计

在三种硬件配置上进行了系统压测，覆盖两种核心任务场景：

| 轮次 | GPU | 模型 | 任务 | 输入/输出规模 |
|------|-----|------|------|-------------|
| Round 1 | A10 24GB | Qwen3.5-9B | Query Rewrite | ~840 / ~7 tokens |
| Round 2 | L20 48GB | Qwen3.5-9B | Query Rewrite | ~840 / ~7 tokens |
| Round 3 | L20 48GB | Qwen3.5-35B-A3B-FP8 | FoC 条款分析 | ~7,000+ / ~200 tokens |

### 4.2 核心发现一：A10 在 FP8 量化下 Compute-Bound

A10 (24GB) 的 Tensor Core FP8 性能不足以支撑即使是 9B 模型的量化推理：

| 指标 | A10 (FP8 KV) | A10 (无量化) |
|------|-------------|-------------|
| TTFT P99 | 995ms | ~500ms |
| TPOT P99 | 95ms | — |
| GPU SM Active | **100%** ⚠️ | ~70-80% |
| KV Cache | 2.6 GiB / 并发16 | 1.3 GiB / 并发11 |

**Trade-off 困境：** 不开量化延迟好但并发极低（11）；开量化并发提升但 GPU 算力打满，延迟恶化 2x。无论哪种配置都不理想。

**结论：A10 不适合生产环境下的 LLM 推理服务。**

### 4.3 核心发现二：L20 是 9B 模型的最优性价比

L20 (48GB) 在相同 9B 模型上：

| 指标 | L20 (c=4) | A10 (c=4) | 改善 |
|------|-----------|-----------|------|
| TTFT P99 | **495ms** | 995ms | **2x** |
| TPOT P99 | **35.5ms** | 95ms | **2.7x** |
| GPU SM Active | **33%** | 100% | 大量余量 |
| KV Cache | 17.55 GiB / 并发98 | 2.6 GiB / 并发16 | **6x** |

L20 在 c=4 时 GPU 仅用 33%，说明算力远未饱和。吞吐天花板 ~1.3 req/s 是因为 Query Rewrite 的 prefill 计算量固定，与 GPU 型号无关。L20 的优势在于**延迟指标全面碾压 A10**，且有充足的并发扩展空间。

**`max-num-batched-tokens` 调优实验：**

| 参数值 | SM Active | Running | Waiting | 吞吐 | 延迟特征 |
|--------|-----------|---------|---------|------|---------|
| 8192 | 75% | 16 | 有排队 | ~1.3/s | **平滑** ✅ |
| 32768 | 90% | 32 | 0 | ~1.3/s | 锯齿状 |

两者吞吐天花板相同（GPU 总算力不变），但 `8192` 的延迟更可控——小 batch 快 step 优于大 batch 慢 step。

### 4.4 核心发现三：MoE 模型的量化策略

35B-A3B MoE 每 token 仅激活 3B 参数，KV cache 规模远小于 dense 模型。FP8 KV 量化的测试结果：

| 维度 | FP8 KV Cache | BF16 KV Cache (无量化) |
|------|-------------|----------------------|
| 性能指标 | 无显著差异 | 无显著差异 |
| 输出质量 | 高并发下偶发输出漂移 (token 膨胀至 11k-13k) | **更稳定** ✅ |
| JSON 解析 | 偶发解析失败 | 更可靠 |

**结论：MoE 模型不建议 KV cache 量化。** 显存节省效果有限（激活参数少），反而引入输出不稳定风险。

### 4.5 核心发现四：FoC 长上下文的并发瓶颈

35B MoE 在 FoC 场景（~8,500 input tokens）下：

| 并发 | E2E P50 | E2E P99 | P50→P99 放大 |
|------|---------|---------|-------------|
| c=2 | 3.75s | 4.97s | 1.3x ✅ |
| c=3 | 4.03s | **11.7s** | **2.9x** ⚠️ |
| c=4 | 6.48s | **14.0s** | 2.2x |

c=3 时尾延迟骤增到 11.7s，是因为第 3 个请求触发了 prefill 排队。TPOT 恒定 24.9ms（MoE decode 极轻），延迟增长的主因是 Decode Mean 从 1.61s 飙升到 3.04s（continuous batching 下多请求共享 decode 时间片）。

**FoC 并发上限：在线交互场景 c ≤ 2（E2E P99 < 5s）。**

### 4.6 硬件选型总结

| GPU | 适用场景 | 不适用场景 | 推荐配置 |
|-----|---------|-----------|---------|
| **A10 24GB** | ❌ 不推荐生产使用 | LLM 推理（compute-bound） | — |
| **L20 48GB** | 9B 查询改写 (c ≤ 4)、35B FoC (c ≤ 2) | 122B 模型 | `max-num-batched-tokens=8192` |
| **H20 96GB × 2** | 122B MoE (TP=2) | — | 待验证 |

---

## 5. 可观测性与弹性伸缩

### 5.1 全链路可观测性

```
GPU 硬件层     DCGM Exporter → SM Active / DRAM Active / 显存利用率
                    ↓
vLLM 引擎层    /metrics → TTFT / TPOT / num_requests_waiting / KV Cache 使用率
                    ↓
应用层         LangSmith → LLM 调用链 / Token 消耗 / 延迟分布
                    ↓
大盘           Grafana → vLLM 性能大盘 / GPU 资源大盘 / Milvus 大盘
```

自定义的 Grafana vLLM 大盘支持**级联变量筛选**（Namespace → Job → Model → Instance），可精确定位到单个 vLLM Pod 的指标，避免多实例数据混淆。

### 5.2 基于 vLLM 指标的 HPA

```yaml
# 扩容信号
vllm:num_requests_waiting > 5    # 队列积压
vllm:num_requests_running > 10   # 并发饱和

# 策略
scaleUp:   稳定窗口 30s,  每 60s  最多 +1 Pod
scaleDown: 稳定窗口 300s, 每 120s 最多 -1 Pod  # 模型加载慢，避免抖动
```

缩容窗口 300s 是关键设计：vLLM 冷启动（模型下载 + CUDA Graph 编译）耗时数分钟，过早缩容会导致反复冷启动。

---

## 6. 工程实践与问题解决

### 6.1 关键工程问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| JSON 解析失败率高 (~10-15%) | LLM 自由输出格式不稳定 | Guided Decoding (FSM 约束) → 100% 合规 |
| A10 量化后延迟恶化 2x | FP8 Tensor Core 性能不足 | 迁移至 L20，不使用量化 |
| MoE 模型 KV FP8 输出漂移 | 激活参数少，量化收益低但精度损失仍在 | 使用 BF16 KV cache |
| Voyage Embedding 跨境延迟 0.35-2.55s | 中国大陆 → 美西网络不稳定 | 规划替换为国内 Embedding 或本地部署 |
| vLLM Pod 冷启动慢 | 模型下载 + CUDA Graph 编译 | NAS PVC 缓存模型 + HPA 保守缩容策略 |
| 多 vLLM 实例 Grafana 指标混淆 | Prometheus 多 Job 重复采集 | 级联变量 + Job 过滤 + Instance 重命名 |
| K8s Service 路由错误 (9B/35B 混淆) | 共用 label selector | 独立 component label |

### 6.2 多平台 GPU 部署架构

实际部署中面临阿里云 GPU 资源抢占问题，采用了混合部署策略：

```
阿里云 ACK (K8s)                     AutoDL (按需)
┌────────────────────────┐          ┌─────────────────────┐
│ L20 裸金属 (8卡)       │          │ H20 96GB × 2        │
│ ├─ vLLM 9B (32GB vGPU) │          │ ├─ vLLM 35B          │
│ ├─ rag-api (3GB vGPU)  │          │ └─ vLLM 122B (TP=2)  │
│ ├─ rag-worker (3GB)    │          │   (按需开机)          │
│ └─ HAMi 显存隔离       │          │                     │
│                        │          │ Grafana Alloy        │
│ Prometheus ← metrics ──┼──────────┤ → Remote Write       │
│ Grafana                │          │   → ARMS Prometheus   │
└────────────────────────┘          └─────────────────────┘
```

跨平台监控通过 **Grafana Alloy** 实现：AutoDL 上的 vLLM `/metrics` 通过 Remote Write 推送到阿里云 ARMS Prometheus，统一在 Grafana 大盘展示。

---

## 7. 性能基准数据汇总

### 7.1 检索链路延迟分解

```
POST /api/search  E2E 请求链路 (c=1):

  Query Rewrite (vLLM 9B · L20)      0.45s    ← 可控，GPU 算力瓶颈
  Voyage Dense Embedding (跨境 API)   0.35s    ← ⚠️ 木桶短板，不可控网络瓶颈
  BGE-M3 Sparse Embedding (本地)      0.08s    ✅
  Milvus Hybrid Search (HNSW + BM25)  0.002s   ✅
  ─────────────────────────────────────────
  总计                                ~0.88s
```

> **木桶效应警告：** Voyage Dense Embedding 作为检索的前置强依赖，其跨境延迟在夜间可飙升至 2.55s，直接击穿 `E2E P95 < 5s` 的 SLA 目标。本地化部署 BGE-Large-zh-v1.5 或 GTE-Qwen2 可将该环节延迟稳定压缩至 < 50ms（利用 CPU AVX-512 向量化指令或极少量 GPU 资源），是当前链路中 **ROI 最高的优化点**。

### 7.2 模型推理性能矩阵

| 模型 | GPU | 任务 | TTFT P99 | TPOT P99 | E2E P99 (最优并发) | 吞吐天花板 |
|------|-----|------|----------|----------|-------------------|-----------|
| Qwen3.5-9B | L20 48GB | Query Rewrite (840→7 tokens) | 495ms | 35.5ms | < 1s (c ≤ 4) | ~1.3 req/s |
| Qwen3.5-35B-A3B | L20 48GB | FoC 条款分析 (7k→200 tokens) | 995ms | 24.9ms | < 5s (c ≤ 2) | ~0.22 req/s |

### 7.3 GPU 利用率对比

| GPU | 模型 | c=4 SM Active | c=32 SM Active | 是否饱和 |
|-----|------|--------------|----------------|---------|
| A10 24GB | 9B (FP8) | 100% | — | ⚠️ 已饱和 |
| L20 48GB | 9B (BF16) | 33% | 75% | ✅ 大量余量 |
| L20 48GB | 35B MoE (BF16) | — | 60% (c=4) | ✅ 有余量 |

---

## 8. 成本分析

### 8.1 GPU 租赁成本（阿里云按量付费参考）

| GPU | 单价 (约) | 可承载任务 |
|-----|----------|---------------|
| A10 24GB | ¥8/h | ❌ 不推荐 |
| L20 48GB | ¥15/h | 9B + 35B 共存 |
| H20 96GB × 2 | - | 122B MoE |

### 8.3 成本优化方向：Serverless GPU 与 Scale-to-Zero

在 NAS PVC 模型预缓存的基础上，可进一步探索基于 `Knative` 的 Serverless GPU 架构：

| 节点 | 策略 | 启动时间 | 预期收益 |
|------|------|---------|---------|
| 9B 查询改写 | Scale-to-Zero + mmap 模型加载 | ~10-20s | 闲时 0 成本 |
| 35B FoC 分析 | Min Replicas = 1 (保活) | 0 (已就绪) | 兼顾延迟与成本 |
| 122B 理赔推理 | 按需启停 (AutoDL) | ~3-5min | 低频场景可接受 |

9B 规模模型通过内存映射（mmap）或 CPU 卸载（offload）可实现秒级冷启动；35B 核心节点保持最小副本数常驻以保障 SLA；122B 按实际理赔需求在 AutoDL 上按需开机。

---

## 9. 待完成与下一步

### 9.1 已验证 ✅

- [x] FoC 条款森林检索流水线（LLM 条款路由 + chunk 批量拉取）
- [x] 三路并发检索架构（FoC ∥ Graph ∥ Vector）
- [x] vLLM 9B / 35B 性能基准（A10 / L20 对比）
- [x] GPU 选型依据（数据驱动，非经验猜测）
- [x] vLLM 参数调优（KV cache 量化、batched tokens）
- [x] Guided Decoding 结构化输出
- [x] K8s 全链路部署（HAMi 显存隔离 + HPA 弹性伸缩）
- [x] 跨云可观测性（Grafana Alloy → ARMS Remote Write）

### 9.2 待完成

| 项目 | 优先级 | 预期收益 |
|------|--------|---------|
| 三路检索消融实验（MRR / NDCG@K + 工程指标） | **P0** | 用数据证明 FoC + Graph 的增量价值 |
| Dense Embedding 本地化 (BGE-Large-zh-v1.5 / GTE-Qwen2) | **P0** | 消除跨境延迟瓶颈 (0.35-2.55s → < 50ms)，彻底移除外部 API 依赖 |
| 替换 Voyage Embedding 为国内方案 | P1 | 作为本地化前的过渡方案 |
| 122B 模型验证（H20 × 2 或 API） | P1 | 验证复杂理赔推理的质量提升 |
| 端到端问答质量评估（RAGAS 指标） | P1 | 量化生成质量（Faithfulness / Context Recall） |
| NAS PVC 模型预缓存 + FSM 预热 ReadinessProbe | **P1** | 冷启动时间从分钟级降至秒级，消除 Guided Decoding 首次编译毛刺 |

### 9.3 消融实验设计（P0）

消融实验是量化三路检索增量价值的关键。基于已有的 148 条标注样本 (`golden_dataset.json`)。

**观测矩阵**需同时覆盖**检索质量**与**工程代价**两个维度——组合检索在提升命中率的同时，会导致 Context Payload 急剧膨胀，可能突破 L20 在 c=2 时的 KV Cache 承载力：

| 检索策略 | 检索质量指标 | 工程代价指标 |
|---------|-----------|-----------|
| Vector only | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |
| FoC only | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |
| Graph only | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |
| Vector + FoC | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |
| Vector + Graph | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |
| FoC + Graph + Vector | MRR · NDCG@5 · Hit Rate@5 | Avg Input Tokens · TPOT · E2E P95 |

**关键验证点：**

1. **检索质量**：FoC 在 logic 类问题上预期显著优于 Vector only；Graph 在涉及除外责任的问题上提供不可替代的增量
2. **Token 膨胀风险**：FoC + Graph 叠加后的平均输入长度，是否超过 `--max-model-len 15000` 限制
3. **KV Cache 溢出**：组合检索的 Prompt 在 c=2 并发下是否触发 KV Cache Swapping，导致 TPOT 退化
4. **质量-代价 Pareto 最优**：如果 Vector + FoC 已达到 95% 的质量水平但 Token 消耗仅为三路组合的 60%，则可在生产中省略 Graph 路径以降低延迟

---

## 附录

### A. 测试环境

| 组件 | 规格 |
|------|------|
| Kubernetes | 阿里云 ACK (K8s 1.35), cn-hangzhou |
| GPU | NVIDIA A10 24GB / L20 48GB |
| vLLM | 最新稳定版 |
| Milvus | 2.6.11 standalone, HNSW + BM25 |
| Dense Embedding | Voyage AI (voyage-3.5-lite) |
| Sparse Embedding | BGE-M3 (FP16, local) |

### B. 压测工具

- `benchmark/stress_test_rag.py`：自定义并发压测框架，支持多并发级别、cooldown、时间戳
- `benchmark/benchmark_tflops.py`：GPU GEMM TFLOPS 基准测试
- Grafana + Prometheus：vLLM 引擎级指标实时监控

### C. 关键代码引用

| 组件 | 文件 |
|------|------|
| FoC 检索器 | `rag/retrieval/foc_retriever.py` |
| ClauseForest 数据结构 | `rag/entity/clause_tree.py` |
| VLLm 客户端 (含 Guided Decoding) | `rag/llm/chat_model.py` |
| 条款选择 Prompt | `conf/prompts/clause_selection.yaml` |
| 三路并发检索 | `agent/tools/clause_matcher.py` |
| HPA 配置 | `ci/k8s/vllm-hpa.yml` |
| vLLM 部署 (9B) | `ci/k8s/vllm-9B-deployment.yml` |
| vLLM 部署 (35B) | `ci/k8s/vllm-35B-deployment.yml` |
