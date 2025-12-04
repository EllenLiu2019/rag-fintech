# 0001-introduce-observation-module.md

Date: 2025-12-03

## Status

Accepted

## Context

为了构建企业级金融 RAG 系统，需要对系统的底层推理性能（Inference Performance）有清晰的量化指标。目前缺乏统一的基准测试（Benchmark）环境来评估不同硬件（CPU vs GPU）和不同优化技术（如 ONNX Runtime）对模型延迟和吞吐量的影响。

此外，作为架构演进的一部分，需要从单纯的功能实现转向对系统性能和可观测性的全面掌控。

## Decision

我们决定在项目中引入 `observation` 模块（目录），用于存放：

1.  **性能基准测试 (Benchmarking)**：评估不同模型在不同硬件上的表现。
2.  **优化实验 (Optimization Experiments)**：记录如量化 (Quantization)、算子融合等优化手段的效果。
3.  **可观测性原型 (Observability Prototyping)**：早期探索代码。

目前已初始包含 CPU 和 GPU 的优化实验 Notebooks。

## Consequences

### Positive
- 基于数据（Data-Driven）而非直觉来选择模型和硬件配置。
- 为后续引入更复杂的生产级可观测性系统（Tracing/Metrics）打下基础。
- 方便复现和对比不同优化策略的效果。

### Negative
- 需要维护额外的实验代码和环境依赖（如 `onnxruntime`, `optimum` 等）。
- 实验代码可能与生产代码产生割裂，需要定期清理或迁移有效结论到生产逻辑中。

