# model-explorer 对接契约

本项目向 `model-explorer` 暴露的是解释性 JSON 摘要，不暴露在线重规划、完整自主探索状态机或 Hybrid A* 接口。

## 稳定接口

- `schema_version`：当前固定为 `model-explorer-contract/v1`。
- `grid`：地图摘要，包含 `width`、`height`、`resolution`、`frame_id`、`origin` 和已输出图层名。
- `constraints`：硬约束摘要，包含违规模格数、可通行比例和约束原因计数。
- `top_goals`：Top-K 离散探索目标，稳定字段为 `cell`、`utility` 和 `reachable`。
- `top_sequences`：Top 多步目标序列，稳定字段为 `cells`、`utility` 和 `coverage_area`。
- `observation_update`：观测更新摘要，沿用当前报告中的可信度均值、`delta_c`、可见格和更新格计数字段。

## 实验字段

以下字段用于解释消融实验和研究原型，字段名可能随下一轮指标定义调整：

- `top_goals.information_gain`
- `top_goals.value`
- `top_goals.confidence_gain`
- `top_sequences.delta_c`
- `top_sequences.value_coverage`
- `top_sequences.risk`
- `top_sequences.segment_path_costs`
- `top_sequences.unreachable_reasons`

完整 JSON 示例见 [model-explorer-contract-example.json](model-explorer-contract-example.json)。
