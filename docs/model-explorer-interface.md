# model-explorer 对接契约

本项目向 `model-explorer` 暴露的是解释性 JSON 摘要，不暴露在线重规划、完整自主探索状态机或 Hybrid A* 接口。

## 稳定接口

- `schema_version`：当前固定为 `model-explorer-contract/v1`。
- `grid`：地图摘要，包含 `width`、`height`、`resolution`、`frame_id`、`origin` 和已输出图层名。
- `constraints`：硬约束摘要，包含违规模格数、可通行比例和约束原因计数。
- `top_goals`：Top-K 离散探索目标，稳定字段为 `cell`、`utility` 和 `reachable`。
- `top_sequences`：Top 多步目标序列，稳定字段为 `cells`、`utility` 和 `coverage_area`。
- `observation_update`：观测更新摘要，沿用当前报告中的可信度均值、`delta_c`、可见格和更新格计数字段；覆盖率统计作为实验字段追加。

这些字段由 `MODEL_EXPLORER_STABLE_FIELDS` 常量和契约测试保护。`model-explorer-contract/v1` 版本内不得删除或改名这些顶层字段；如需要新增解释性指标，默认进入 `experimental_fields`。

## 实验字段

以下字段用于解释消融实验和研究原型，字段名可能随下一轮指标定义调整：

- `top_goals.information_gain`
- `top_goals.value`
- `top_goals.confidence_gain`
- `top_goals.risk`
- `top_goals.path_cost`
- `top_goals.energy_cost`
- `top_goals.coverage_area`
- `top_goals.expected_new_coverage_area`
- `top_goals.expected_coverage_rate_delta`
- `observation_update.total_valid_area`
- `observation_update.covered_valid_area`
- `observation_update.coverage_rate`
- `observation_update.coverage_rate_delta`
- `observation_update.total_valid_cell_count`
- `observation_update.covered_valid_cell_count`
- `top_sequences.delta_c`
- `top_sequences.value_coverage`
- `top_sequences.risk`
- `top_sequences.segment_path_costs`
- `top_sequences.unreachable_reasons`
- `top_sequences.risk_reasons`

## 示例

- 完整 JSON 示例见 [model-explorer-contract-example.json](model-explorer-contract-example.json)，用于展示当前报告可提供的全部解释字段。
- 最小可消费示例见 [model-explorer-minimal-example.json](model-explorer-minimal-example.json)，用于上层消费方做轻量解析和字段兼容性检查。

## 版本规则

- `model-explorer-contract/v1` 只保证稳定接口字段不破坏；实验字段可新增或调整，但必须列入 `experimental_fields`。
- 若稳定接口字段需要改名、删除或改变语义，应新增 `model-explorer-contract/v2`，并保留 v1 示例与测试，直到上层完成迁移。
- 本契约只描述报告输出，不承诺在线重规划、任务状态机或路径执行接口。
