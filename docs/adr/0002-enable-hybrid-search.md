# 0002-enable-hybrid-search

Date: 2025-12-04

## Status

Accepted

## Context

### 业务背景

在金融保单问答场景中，用户查询通常包含两类信息：
- **专业术语**：保单号（如 `POL-2024-001`）、条款名称、被保险人姓名
- **自然语言描述**：关于保障范围、理赔条件的问题

### 当前问题

纯语义检索（Dense Vector Search）存在以下局限：
- 对专业术语、编号的精确匹配能力弱
- 无法有效捕捉领域特定词汇的重要性

### 解决方案

**混合检索（Hybrid Search）** 结合了：
- Dense Vector Search（语义搜索）
- Sparse Vector Search（BGE-M3）
- RRF (Reciprocal Rank Fusion) 重排序

实现语义理解与精确匹配的最佳平衡：
- **语义理解**：捕捉用户查询的深层意图（Dense）
- **精确匹配**：匹配专业术语（Sparse/BM25）

## Technical Choices

### 稀疏检索技术对比

| 特性维度 | 全文检索 (ElasticSearch) | 统计型稀疏 (Milvus Native BM25) | 学习型稀疏 (SPLADE/BGE-M3) |
|---------|-------------------------|-------------------------------|---------------------------|
| **匹配逻辑** | 精确关键词 (Exact Keyword) | 精确关键词 (Exact Keyword) | 关键词 + 语义扩展 (Keyword + Semantic) |
| **索引结构** | Lucene 倒排索引 | 稀疏倒排索引 / WAND | 稀疏倒排索引 / WAND |
| **语义理解** | 弱 (仅靠同义词库) | 弱 | 强 (自动扩展) |
| **索引大小** | 中等 | 小 (仅非零元素) | 中等 (扩展导致非零元素增加) |
| **查询灵活性** | 极高 (Wildcard, Phrase, Fuzzy) | 中等 (主要靠关键词匹配) | 中等 |
| **系统架构** | 独立组件 (需同步) | 融合组件 (数据库内置) | 融合组件 (数据库内置) |
| **生成方式** | 统计规则 (TF-IDF) | 统计规则 (内部函数) | 深度学习模型推理 |

### SPLADE vs BGE-M3

| 特性 | SPLADE | BGE-M3 |
|------|--------|--------|
| **全称** | Sparse Lexical and Expansion Model | Multi-Linguality, Multi-Functionality, Multi-Granularity |
| **生成原理** [1]| 基于 BERT，利用 MLM 策略，预测每个 token 的激活值 | 基于 BERT，评估每个 token 上下文的重要性权重 |
| **词汇扩展能力** | 向文档/查询中"引入"相关词，缓解词汇错配问题 | 适合词汇错配严重、语义匹配要求更高的场景 |
| **多语言支持** | 主要针对英文 | 支持长文本（8192 tokens），多语言支持（100+） |
| **开发者** | Naver | BAAI (Beijing Academy of Artificial Intelligence) |

### 数据库层融合：RRF

在 Milvus 中实现混合检索的最佳实践是利用其内置的 RRF Ranker。

对于一个 `hybrid_search` 请求，Milvus 的查询节点（Query Node）会：
1. 执行多路检索（Dense + Sparse）
2. 合并结果
3. RRF 重新排序
4. 返回最终的 Top-K

## Decision

| 组件 | 推荐方案 | 理由 |
|------|---------|------|
| **嵌入模型** | BAAI/bge-m3 | 同时生成高质量的稠密和稀疏向量，支持多语言 |
| **向量数据库** | Milvus (v2.5+) | 支持 Dense/Sparse 双向量存储，支持服务器端 RRF 融合 |
| **稀疏索引类型** | SPARSE_INVERTED_INDEX | 针对稀疏数据优化的倒排索引，平衡存储与查询速度 |
| **稀疏检索性能权衡** | drop_ratio_search | 剪枝算法加速查询 |
| **融合策略** | RRF (k=60) | 鲁棒性最强的无监督融合算法，无需训练权重 |
| **重排序 (Rerank)** | Cross-Encoder (如 bge-reranker-v2) | 在 RRF 召回 Top-50 后，使用交叉编码器进行精排 |

> **Key Insight** [2]: RRF is more robust but weighted allows fine-tuning.
> RRF = robust default. No calibration needed. This is a great method when score scales are incomparable or volatile across queries, or when you're fusing many signals.

### 替代方案考虑

| 方案 | 描述 | 未选择原因 |
|------|------|-----------|
| ElasticSearch + Faiss | 独立全文检索 + 向量检索 | 架构复杂，需要数据同步 |
| SPLADE | 学习型稀疏模型 | 不支持中文 |
| Weighted Fusion | 加权融合 | 需要调参，不如 RRF 鲁棒 |

## Implementation Reference

### Milvus 索引类型对比

| 索引类型 | 适用场景 | 召回率 (Recall) | 查询速度 (Speed) | 内存消耗 |
|---------|---------|----------------|-----------------|---------|
| **SPARSE_INVERTED_INDEX** | 数据集较小，要求 100% 召回 | 极高 | 中等 | 低 |
| **SPARSE_WAND** | 大规模数据集，追求高吞吐 | 高 (有微小损失) | 极高 (Block-Max 剪枝) | 低 |

## Consequences

### Positive

- 混合检索在处理专业术语密集的领域文档时表现更优
- 语义搜索与关键词匹配互补，提高召回率和准确率
- 为后续添加更多高级检索选项（如重排序策略等）打下基础

### Negative

- 混合检索计算成本更高（需要同时执行多路检索、向量计算，以及后续重排、融合步骤）
- 需要维护两套向量索引（Dense + Sparse）
- BGE-M3 模型推理增加额外延迟

## References

1. [Exploring BGE-M3 and Splade: Two Machine Learning Models for Generating Sparse Embeddings](https://medium.com/@zilliz_learn/exploring-bge-m3-and-splade-two-machine-learning-models-for-generating-sparse-embeddings-0772de2c52a7)
2. [RAG Series – Hybrid Search with Re-ranking](https://www.dbi-services.com/blog/rag-series-hybrid-search-with-re-ranking/)
