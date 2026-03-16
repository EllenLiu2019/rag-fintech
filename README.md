# RAG-Fintech：保险领域全链路 RAG 系统

基于 RAG + GraphRAG + Multi-Agent 的保险智能问答与理赔辅助系统。覆盖**文档解析 → 信息抽取 → 向量/图谱检索 → 大模型推理 → 理赔决策**全链路，并在 Kubernetes 上实现异构 GPU 算力调度与弹性伸缩。

---

## 1. 系统架构

```mermaid
flowchart TB
    subgraph Client["客户端"]
        UI["React SPA"]
    end

    subgraph K8s["Kubernetes 集群 · ACK (Alibaba Cloud)"]
        direction TB
        Ingress["Ingress Controller (nginx)\n/api/ → rag-api · / → rag-ui"]

        subgraph App["应用层"]
            direction LR
            API["rag-api (FastAPI)\nSSE Streaming · LangSmith Tracing"]
            Worker["rag-worker (RQ)\n异步文档 Ingestion"]
        end

        subgraph LLM_Tier["多级模型路由层"]
            direction TB

            subgraph Tier1["Tier 1 · 轻量推理 (A10 24GB)"]
                vLLM_9B["vLLM · Qwen3.5-9B\nCompute-Bound · CUDA Graphs"]
            end

            subgraph Tier2["Tier 2 · 中等推理 (L20 48GB / H20 96GB / A100 80GB)"]
                vLLM_35B["vLLM · Qwen3.5-35B-A3B-FP8\nMoE · 激活 3B"]
                vLLM_122B["vLLM · Qwen3.5-122B-A10B-FP8\nMoE · 激活 10B"]
            end

            subgraph Tier3["Tier 3 · 深度推理 (API)"]
                DS["DeepSeek-Reasoner\nCoT 推理 · 理赔决策"]
            end
        end

        subgraph HPA["弹性伸缩"]
            direction LR
            Prom["Prometheus\n+ DCGM Exporter"]
            Adapter["Prometheus Adapter\n自定义指标 → K8s metrics API"]
            HPA_ctrl["HPA\nvllm:num_requests_waiting\nvllm:num_requests_running"]
            Grafana["Grafana\nvLLM 观测大盘"]
            Prom --> Adapter --> HPA_ctrl
            Prom --> Grafana
        end

        subgraph GPU["GPU 资源管理"]
            HAMi["HAMi 调度器\n显存隔离 · 算力配额"]
            NAS["NAS PVC 30Gi\nHuggingFace 模型缓存 (共享)"]
        end
    end

    subgraph Data["数据层"]
        direction LR
        Milvus["Milvus\n向量检索"]
        PG["PostgreSQL\n业务数据"]
        Neo4j["Neo4j\n知识图谱"]
        Redis["Redis\n任务队列"]
        OSS["阿里云 OSS\n文档存储"]
    end

    UI --> Ingress --> App
    API --> LLM_Tier
    Worker --> LLM_Tier
    HPA_ctrl -.->|扩缩| Tier1
    vLLM_9B & vLLM_35B & vLLM_122B -- "/metrics" --> Prom
    App --> Data
```

### 多平台 GPU 部署

模型按参数量分布在不同算力平台，按需启停控制成本：

| 模型 | GPU | 平台 | 部署方式 | 配置文件 |
|------|-----|------|----------|----------|
| Qwen3.5-9B | A10 24GB × 1 | 阿里云 ACK | K8s Deployment + HPA | `ci/k8s/vllm-deployment.yml` |
| Qwen3.5-35B-A3B-FP8 | H20 96GB × 1 | AutoDL | docker compose (profile: 35b) | `vllm/docker-compose.autodl.yml` |
| Qwen3.5-122B-A10B-FP8 | H20 96GB × 2 | AutoDL | docker compose + tensor-parallel=2 | `vllm/docker-compose.autodl.yml` |

9B 常驻 K8s 集群处理高频请求；35B/122B 在 AutoDL 按需开机，通过 `base_url` 切换即可接入系统；DeepSeek-Reasoner 直接走 API，无需自部署。

---

## 2. 存储层与检索架构

```mermaid
flowchart TB
    subgraph Ingestion["文档 Ingestion 流水线"]
        direction LR
        RawDoc["原始文档\n(PDF / Text)"]
        Parser["LlamaParse\n(Claude Sonnet 4.5\nAgent 模式解析)"]
        Extract["信息抽取\n(规则优先 · LLM 补漏)"]
        Forest["ClauseForest 构建\n(Markdown 层级解析)"]
        Chunk["MarkdownNodeParser\n(分块 · 条款关联)"]
        Embed["双路 Embedding\n(Voyage Dense ∥ BGE-M3 Sparse)"]
        GraphBuild["GraphRAG 索引\n(异步 RQ 任务)"]

        RawDoc --> Parser --> Extract --> Forest --> Chunk --> Embed
        Forest --> GraphBuild
    end

    subgraph Storage["持久化存储层"]
        direction LR
        OSS[("阿里云 OSS\n原始文档 · 解析结果")]
        PG[("PostgreSQL\n结构化数据 · ClauseForest\n文档元数据 · 理赔评估")]
        Milvus[("Milvus\nDense HNSW + BM25 Sparse\nRRF 融合检索")]
        Neo4j[("Neo4j\n条款实体图谱\nINCLUDE / NOT_INCLUDE 边")]
        Redis[("Redis\nRQ 任务队列 · Embedding 缓存\n搜索结果缓存 · 进度追踪")]
    end

    Embed -->|"向量 + 元数据"| Milvus
    Extract -->|"保单/被保人/险种"| PG
    Forest -->|"clause_forest JSONB"| PG
    GraphBuild -->|"实体对齐 · PageRank"| Neo4j
    GraphBuild -->|"实体向量 (graph KB)"| Milvus
    RawDoc -->|"原始上传"| OSS

    subgraph Retrieval["三路并发检索 (ClauseMatcher.match)"]
        direction TB
        Query["理赔请求\n(MedicalEntity · HumanDecision)"]

        subgraph Path1["路径 1 · FoC Retriever"]
            FoC_Load["加载 ClauseForest\n(PostgreSQL)"]
            FoC_LLM["LLM 条款 ID 路由\n(Markdown 树 → 相关条款)"]
            FoC_Fetch["按 chunk_id 批量拉取\n(Milvus)"]
            FoC_Load --> FoC_LLM --> FoC_Fetch
        end

        subgraph Path2["路径 2 · Graph Retriever"]
            G_Search["诊断实体混合检索\n(Milvus graph KB)"]
            G_Cypher["Neo4j 多跳遍历\n(INCLUDE / NOT_INCLUDE\n最大深度 5)"]
            G_Evidence["路径提取\n(保障 vs 除外)"]
            G_Search --> G_Cypher --> G_Evidence
        end

        subgraph Path3["路径 3 · Vector Retriever"]
            V_Embed["查询向量化\n(Dense ∥ Sparse)"]
            V_Search["Milvus Hybrid Search\n(RRF k=60)"]
            V_Embed --> V_Search
        end

        Query --> Path1 & Path2 & Path3

        FoC_Fetch --> Merge["结果合并\n(coverage · exclusion · clause_ids)"]
        G_Evidence --> Merge
        V_Search --> Merge
        Merge --> Reasoner["EligibilityReasoner\n(LLM 最终决策)"]
    end

    Storage --> Retrieval

    classDef db fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#1a237e;
    classDef engine fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;
    classDef input fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;

    class OSS,PG,Milvus,Neo4j,Redis db;
    class Parser,Extract,Forest,Chunk,Embed,GraphBuild engine;
    class FoC_Load,FoC_LLM,FoC_Fetch,G_Search,G_Cypher,G_Evidence,V_Embed,V_Search engine;
    class RawDoc,Query input;
    class Merge,Reasoner output;
```

### 向量检索的局限与 GraphRAG 补位

保险条款中存在大量"双重否定"与"条件豁免"结构（如"除外责任中不包含以下情形"），高维空间相似度无法表征因果逻辑关系。GraphRAG 在此场景的作用：Milvus Hybrid Search 负责在海量文本中快速锚定目标实体（如"甲状腺恶性肿瘤"），返回的 entity_name 作为 Neo4j 的查询入口；Neo4j 沿 `INCLUDE` / `NOT_INCLUDE` 边做多跳遍历（最大深度 5），提取完整的保障-除外责任路径链，最终生成结构化的 coverage / exclusion 证据。

### 三路并发检索

`ClauseMatcher.match()` 通过 `asyncio.gather()` 并发执行三条检索路径：

| 路径 | 数据源 | 检索方式 | 产出 |
|------|--------|----------|------|
| FoC Retriever | PostgreSQL → Milvus | LLM 在 ClauseForest 的 Markdown 树上做条款 ID 路由，再按 chunk_id 批量拉取原文 | 精确条款原文 |
| Graph Retriever | Milvus (graph KB) → Neo4j | 诊断实体混合检索定位图谱锚点，Cypher 多跳遍历提取路径 | 保障/除外责任路径 |
| Vector Retriever | Milvus | Dense + BM25 Sparse 混合检索，RRF (k=60) 融合排序 | 语义相关片段 |

三路结果合并后，由 `EligibilityReasoner` 基于完整证据生成最终理赔决策。

### Ingestion 数据流

```
原始 PDF → OSS (持久存储)
         → LlamaParse (解析为 Markdown + HTML 表格)
         → RuleExtractor (正则 + 表头匹配 + HTML Grid 解析)
         → LLMExtractor (仅补缺失字段，Qwen3.5-9B)
         → ClauseForestBuilder (中文层级标题识别：第X部分 → 第X条 → （X）)
         → MarkdownNodeParser (分块，每个 chunk 关联 clause_id + clause_path)
         → Voyage Dense ∥ BGE-M3 Sparse (并行向量化)
         → Milvus (HNSW + BM25 Sparse Inverted Index)
         → PostgreSQL (结构化实体 + ClauseForest JSONB + 置信度评分)
         → Neo4j (GraphRAG 异步构建，实体对齐 + PageRank)
```

---

## 3. 理赔 Multi-Agent 流程

```mermaid
flowchart TB
    subgraph API["API 端点 (/api/claim)"]
        direction LR
        Submit["POST /submit\n发起初审"]
        Approve["POST /approve\n人工确认实体"]
        Replay["POST /subgraph-replay/{thread_id}\n时间旅行 · 状态覆写"]
        Checkpoints["GET /checkpoints/{thread_id}\n检查点时间线"]
    end

    subgraph Phase1["Phase 1 · ClaimsOrchestrator.start_evaluation()"]
        direction TB
        Start((ClaimRequest))

        subgraph Parallel["两个子图并行执行 (asyncio.gather)"]
            direction LR

            subgraph Encode["encode_graph (ICD-10 编码)"]
                direction TB
                E_Agent["encode_agent\n(deepseek-chat)\n从 KB 候选中选择 ICD/SNOMED"]
                A_Agent["align_agent\n(deepseek-chat)\n+ tool: align_medical_concepts"]
                A_Tool["ToolNode\n查询 Neo4j MAPS_TO/ISA 关系"]
                E_Approve{"interrupt()\n等待人工确认 ICD 编码"}
                E_Agent --> A_Agent --> A_Tool --> A_Agent
                A_Agent -->|agent_output| E_Approve
            end

            subgraph Stage["stage_graph (TNM 分期)"]
                direction TB
                S_Agent["stage_agent\n(deepseek-chat)\n+ tool: calculate_thyroid_tnm_stage"]
                S_Tool["ToolNode\nAJCC 8th 确定性分期算法"]
                S_Approve{"interrupt()\n等待人工确认 TNM 分期"}
                S_Agent --> S_Tool --> S_Agent
                S_Agent -->|agent_output| S_Approve
            end
        end

        Start --> Parallel
    end

    subgraph Phase2["Phase 2 · ClaimsOrchestrator.complete_evaluation()"]
        direction TB
        Resume["graph.resume()\nCommand(resume=HumanDecision)"]
        Match["ClauseMatcher.match()\n三路并发检索\n(FoC ∥ Graph ∥ Vector)"]
        Reason["EligibilityReasoner.reason()\n(deepseek-reasoner)\n生成 ClaimDecision"]
        Resume --> Match --> Reason
    end

    subgraph Persist["持久化层"]
        direction LR
        PG_Check[("PostgresSaver\nLangGraph Checkpoint\n每个节点自动快照")]
        PG_Eval[("claim_evaluations\n评估记录 · 人工决策\nsubgraph_configs JSONB")]
    end

    Submit --> Start
    E_Approve & S_Approve ==>|挂起 · 返回 pending_reviews| PG_Check
    Approve -.->|注入 HumanDecision| Resume
    Replay -.->|aupdate_state → ainvoke| Phase1

    Parallel ==>|自动 checkpoint| PG_Check
    Phase2 ==>|状态更新| PG_Eval

    classDef api fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef node fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef interrupt fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100;
    classDef db fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c;

    class Submit,Approve,Replay,Checkpoints api;
    class E_Agent,A_Agent,A_Tool,S_Agent,S_Tool,Resume,Match,Reason node;
    class E_Approve,S_Approve interrupt;
    class PG_Check,PG_Eval db;
```

### Checkpoint 与容错

LangGraph 的 `PostgresSaver` 在每个节点转换时自动将图状态序列化并持久化到 PostgreSQL。当网络中断或模型调用失败时，可从最近的 checkpoint 恢复执行，无需从头开始。

### Human-in-the-Loop 中断

两个并行子图各自在 `approval` 节点调用 `interrupt()`，向外抛出 AI 生成的候选结果（ICD-10 编码 + TNM 分期）。此时 LangGraph 状态被挂起并 checkpoint 到 PostgreSQL。前端展示候选结果供理赔员审核确认后，通过 `POST /api/claim/approve` 注入 `HumanDecision`，状态机恢复执行并进入 Phase 2 的条款检索与推理。

### 时间旅行 (Time Travel)

`POST /api/claim/subgraph-replay/{thread_id}` 可将子图状态回退到任意 checkpoint，注入修正后的变量，通过 `aupdate_state` + `ainvoke(None, forked_config)` 从该节点 fork 出新的执行分支。适用于事后审计中发现实体判断有误的场景——无需整体重跑，只需回溯到出错节点并修正。

### 理赔 API 端点

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/claim/upload` | POST | 上传理赔材料 PDF |
| `/api/claim/submit` | POST | 发起初审 (Phase 1) |
| `/api/claim/approve` | POST | 人工确认实体，触发 Phase 2 |
| `/api/claim/{doc_id}/evaluations` | GET | 查询评估历史 |
| `/api/claim/checkpoints/{thread_id}` | GET | 检查点时间线 |
| `/api/claim/state/{thread_id}` | GET | 指定检查点的完整图状态 |
| `/api/claim/subgraph-checkpoints/{thread_id}` | GET | 子图检查点时间线 |
| `/api/claim/subgraph-state/{thread_id}` | GET | 子图状态 |
| `/api/claim/subgraph-replay/{thread_id}` | POST | Fork + Replay 子图 |
| `/api/claim/subgraph-resume/{thread_id}` | POST | 恢复 Replay 后的中断 |

---

## 4. 模型路由矩阵

系统按任务复杂度和风险等级将请求路由到不同参数量的模型：

| 任务 | 模型 | 参数量 | 部署方式 | 选型依据 |
|------|------|--------|----------|----------|
| 查询改写 · HyDE · 意图识别 | Qwen3.5-9B | 9B Dense | vLLM on A10/L20 | 低延迟高频任务，结构化输出有 few-shot 约束 |
| 保单信息抽取 (rule + LLM fallback) | Qwen3.5-9B | 9B Dense | vLLM on A10/L20 | 规则优先，LLM 仅补缺失字段 |
| 条款检索 · 智能问答 | Qwen3.5-35B-A3B | 35B MoE (3B active) | vLLM on L20/H20 | 中等推理复杂度，MoE 兼顾速度和质量 |
| 理赔 Agent (医学编码 · TNM 分期) | Qwen3.5-122B-A10B | 122B MoE (10B active) | vLLM on H20 * 2 | 需要医学领域知识，工具调用能力 |
| GraphRAG 实体关系抽取 | DeepSeek-Reasoner | 闭源 | API | 最复杂的 prompt，续提取循环 |
| 理赔疾病抽取 | DeepSeek-Reasoner | 闭源 | API | 高风险环节，0 容错 |
| 理赔资格推理 (最终决策) | DeepSeek-Reasoner | 闭源 | API | 多证据交叉推理，CoT |

---

## 5. GPU 算力调度与可观测性

### 异构资源隔离

通过 HAMi 在 L20 上实现单卡多 Pod 共享，按需分配显存和算力：

| Pod | 显存 | 算力 | 用途 |
|-----|------|------|------|
| vLLM (Qwen3.5-9B) | 32 GB / 70% cores | Compute-Bound | 推理服务 |
| rag-api | 3 GB / 15% cores | — | Sparse Embedding (BGE-M3) |
| rag-worker | 3 GB / 15% cores | — | Sparse Embedding (BGE-M3) |

### 弹性伸缩

基于 vLLM 原生指标的 AI-native HPA：

```
vLLM /metrics → Prometheus → Prometheus Adapter → custom.metrics.k8s.io → HPA
```

- **扩容信号**：`vllm:num_requests_waiting > 5`（队列积压）或 `vllm:num_requests_running > 10`（并发饱和）
- **扩容策略**：稳定窗口 30s，每 60s 最多 +1 Pod
- **缩容策略**：稳定窗口 300s，每 120s 最多 -1 Pod（模型加载慢，避免抖动）
- **节点扩容**：HPA 触发新 Pod → Pending → Cluster Autoscaler 自动扩 GPU 节点池

### Compute-Bound vs Memory-Bound

| 场景 | 瓶颈类型 | 特征 | 优化手段 |
|------|----------|------|----------|
| 高并发短文本 (查询改写) | Compute-Bound | TPOT 稳定，TTFT 随队列增长 | CUDA Graphs · enforce-eager |
| 低并发长上下文 (理赔分析) | Memory-Bound | KV Cache 占满显存，TPOT 上升 | PagedAttention · max-model-len 调控 |

### 可观测性

| 层级 | 工具 | 采集指标 |
|------|------|----------|
| GPU 硬件 | DCGM Exporter | 显存利用率 · SM 利用率 · PCIe 带宽 (DCGM_FI_DEV_GPU_UTIL · DCGM_FI_PROOF_DRAAM_ACTIVE) |
| vLLM 引擎 | Prometheus `/metrics` | num_requests_waiting · TTFT · TPOT · KV Cache 使用率 |
| 应用链路 | LangSmith | LLM 调用链 · Token 消耗 · 延迟分布 |
| 大盘 | Grafana | vLLM 性能大盘 · GPU 资源大盘 |

压测期间的 Grafana 面板截图见 [`benchmark/results/grafana/`](benchmark/results/grafana/)。


---

## 6. 质量评估

### RAG 评估框架

基于 golden dataset（148 条标注样本）的端到端评估，覆盖检索质量和生成质量两个维度：

| 维度 | 指标 | 工具 |
|------|------|------|
| 检索质量 | MRR · NDCG@K · Hit Rate@K · Precision@K · Recall@K | 自定义 `RetrievalMetrics` |
| 生成质量 | Context Recall · Answer Correctness · Faithfulness | RAGAS 框架 |
| 意图识别 | Intent Accuracy（fact vs logic 分类准确率） | 评估器内置 |

评估流程：`RAGEvaluator.generate_answer()` 并发调用检索 + 生成管线 → `RAGEvaluator()` 调用 RAGAS 计算指标 → 输出 `eval_result.csv`。评估模型使用 DeepSeek-Chat，embedding 使用 BGE-M3。



## 7. CI/CD

```
git push master
  → GitHub Actions (deploy.yml)
    → Build backend/frontend images
    → Push to Alibaba Cloud ACR
    → envsubst 替换镜像 tag
    → kubectl apply (api · worker · ui · vllm · ingress)
    → kubectl rollout status (等待就绪)
```

vLLM 基础镜像通过 `mirror-vllm.yml` 从 Docker Hub 同步到 ACR（解决国内拉取问题）。

---

## 8. 技术栈

| 类别 | 技术 |
|------|------|
| 推理引擎 | vLLM · PagedAttention · CUDA Graphs · AWQ 量化 · Speculative Decoding |
| 模型 | Qwen3.5 系列 (9B / 35B-A3B / 122B-A10B) · DeepSeek-Reasoner |
| 压测 | GuideLLM (vLLM 官方) · 自定义并发 Scaling 框架 · GEMM TFLOPS 基准 |
| RAG | LlamaParse · Milvus (Dense + Sparse) · Voyage Embedding · BGE-M3 · Jina Reranker |
| GraphRAG | Neo4j · 实体对齐 · SNOMED-ICD10 医学本体映射 |
| Agent | LangGraph · 多 Agent 协作 (医学编码 · TNM 分期 · 条款匹配 · 资格推理) |
| 评估 | RAGAS (Context Recall · Faithfulness) · 自定义检索指标 (MRR · NDCG@K) |
| 微调 | HuggingFace Trainer · BERT Claim Detection (F1 0.91) |
| 后端 | FastAPI · Redis (RQ) · PostgreSQL · SQLAlchemy |
| 前端 | React · Vite · SSE Streaming |
| 基础设施 | Kubernetes (ACK) · HAMi GPU 共享 · HPA · Prometheus · Grafana |
| CI/CD | GitHub Actions · Alibaba Cloud ACR · Alibaba Cloud ACK |

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

### 端到端压测
#### RAG 问答并发
用 locust （Python 生态，能处理 SSE streaming）：
测量端到端延迟（问题发出 → 最后一个 token 返回）、TTFB（首字节）、吞吐量
并发从 1 → 5 → 10 → 20 → 50 爬坡


### 三路检索消融实验

| 检索策略             | MRR  | NDCG@5 | Hit Rate@5 |
| :------------------- | :--- | :----- | :--------- |
| Vector only          | 0.xx | 0.xx   | 0.xx       |
| FoC only             | 0.xx | 0.xx   | 0.xx       |
| Graph only           | 0.xx | 0.xx   | 0.xx       |
| Vector + FoC         | 0.xx | 0.xx   | 0.xx       |
| Vector + Graph       | 0.xx | 0.xx   | 0.xx       |
| FoC + Graph + Vector | 0.xx | 0.xx   | 0.xx       |

### Embedding 微调/Onnx