# vLLM 最小化性能实验设计（Qwen 7B）

## 1. 实验目标

在 `4090 24G` 单卡环境下，对比以下三组配置在短文本结构化抽取任务上的推理性能：

- Baseline: Hugging Face text generation inference
- 对比组 A: vLLM（参数调优）
- 对比组 B: vLLM（量化）

核心指标：

- `TTFT`（time to first token）
- `TPS`（tokens per second）
- `GPU 显存占用`（峰值/稳态）

质量护栏（最小必需）：

- 字段级准确率（`field_acc`）

## 2. 文件结构（最小可执行）

- `docs/vllm_dataset_minimal.jsonl`：输入样本与期望输出（逐行 JSON）
- `docs/vllm_benchmark_config_minimal.yaml`：实验配置
- `docs/vllm_metrics_template.csv`：结果记录模板

## 3. 数据格式说明（对齐 `_fallback_to_llm_extraction`）

每条样本建议包含以下字段：

- `id`: 样本唯一标识
- `documents`: 文本片段列表（与 `Extractor` 入参对齐）
- `hints`: 已抽取字段（可为空对象）
- `missing_fields`: 需要 LLM 补全的字段
- `expected`: 期望字段（用于自动评估）

评估规则（最小版本）：

- 对 `missing_fields` 中每个字段做 exact match
- `field_acc = 正确字段数 / missing_fields 字段总数`

## 4. 执行建议（最小版）

固定参数后分别跑三组：

- 并发：`1 / 4 / 8`
- 每组预热：`20` 请求
- 每组正式：`100` 请求
- 采样参数：`temperature=0`

输出 `p50/p95`：

- `ttft_ms_p50 / ttft_ms_p95`
- `decode_tps_p50 / decode_tps_p95`
- `gpu_mem_peak_mb`
- `field_acc_avg`

## 5. 结论口径（建议）

报告时用同一口径输出：

- 在并发 `<=4` 时，哪组 `TTFT` 最优
- 在并发 `8` 时，哪组 `TPS` 最优
- 在满足 `field_acc` 不显著下降前提下，推荐配置
