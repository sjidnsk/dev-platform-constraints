# P1 可信度最小闭环验证报告

## 基本信息

- 验证日期：2026-05-27
- 验证范围：P0 最小闭环 + P1 可信度分量、配置化融合、一次局部观测更新、显式观测模型、数据契约报告、障碍/通行概率贝叶斯边界、离散地形类别后验、可配置多场景可信度权重消融和 P2 footprint 候选目标生成/多步序列评估起点
- 输入平台配置：`configs/platforms/yutu2.json`
- 输入地图：`scripts/run_minimal_closure.py` 内的确定性示例地图

## 验证命令

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python scripts\run_minimal_closure.py
python scripts\run_confidence_ablation.py --output-dir outputs\ablation
python scripts\visualize_minimal_closure.py --output-dir outputs\visualization --confidence-config configs\confidence\default.json
```

## 当前结果快照

| 指标 | 结果 |
| --- | --- |
| 数据契约是否有效 | `true` |
| 路径是否可达 | `true` |
| 硬约束违规数 | `13` |
| 路径节点数 | `32` |
| 路径总代价 | `49.20605767223693` |
| 更新前平均可信度 | `0.73125` |
| 更新后平均可信度 | `0.6063961833368423` |
| 平均可信度变化 | `-0.12485381666315765` |
| 低可信区域面积 | `10.0 -> 9.75` |
| `ΔC` | `0.3617140436661379` |
| 可见栅格数 | `53` |
| 已更新栅格数 | `53` |
| 数据契约报告 | `data_contract.issue_summary.errors = 0` |
| 探索目标摘要 | `top_exploration_goals` 由可视化脚本输出 |
| 消融报告 | `outputs/ablation/confidence_ablation.json`、`.csv`、`.png`、`.html`，包含 `scenarios`、`runs`、`aggregate`、`recommendation` |

平均可信度下降是预期结果：当前脚本同时模拟了可见区观测提升和未观测区域时间衰减。`ΔC` 和低可信区域面积变化用于确认局部观测确实带来了正向可信度提升。

消融实验默认从 `configs/ablation/scenarios.json` 读取九个场景，其中六个为确定性样例场景，三个为固定随机种子的 `seeded_synthetic` 半合成场景。实验比较三组可信度权重：默认配置、偏观测配置、偏一致性/时间衰减配置。每次运行记录路径代价、`ΔC`、低可信高风险路径比例、Top-K 探索目标排序和离线多步目标序列，并额外输出配置级聚合指标、推荐配置和 HTML/PNG 可视化，用于后续参数选择和实验复现。

## 多场景可信度权重消融实验结果与结论

数据来源：`outputs/ablation/confidence_ablation.json`，记录日期：2026-05-27。

默认场景矩阵：

- `baseline_gap`：32 x 20 栅格，观测位姿 `(0, 10)`，目标 `(31, 19)`。
- `upper_observation`：32 x 20 栅格，观测位姿 `(0, 6)`，目标 `(31, 15)`。
- `compact_value`：24 x 16 栅格，观测位姿 `(0, 8)`，目标 `(23, 15)`。
- `risk_conflict_gap`：在路径必经横向带设置低可信高风险区域，验证低可信高风险路径比例。
- `simple_occlusion`：在高价值区域前放置简单遮挡障碍，候选收益使用 `use_simple_occlusion=true`。
- `multi_observation`：使用两个观测位姿，并将候选目标收益设置为两步离散 lookahead。
- `synthetic_crater_shadow`：`seeded_synthetic` 地图，固定种子 `101`，覆盖随机坡面、坑洼和阴影风险区域。
- `synthetic_multi_pose`：`seeded_synthetic` 地图，固定种子 `202`，覆盖多观测位姿和两步 lookahead。
- `synthetic_risk_band`：`seeded_synthetic` 地图，固定种子 `303`，覆盖低可信高风险横向带。

配置级聚合结果：

| 配置 | 路径总代价均值 | 路径总代价标准差 | `ΔC` 均值 | `ΔC` 标准差 | 更新后低可信区域面积均值 | 配置胜率 | 风险冲突命中率 | Top-K 稳定性 | 序列目标稳定性 | 失败场景 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `default.json` | `45.130596108035895` | `7.995616720709458` | `1.1804746191998106` | `0.7514575381066648` | `14.555555555555555` | `1/9` | `0.2222222222222222` | `0.1111111111111111` | `0.1111111111111111` | 无 |
| `observation_focused.json` | `45.230605328303014` | `7.944625126419484` | `0.4200272742285056` | `0.3719208531389355` | `17.583333333333332` | `1/9` | `0.2222222222222222` | `0.1111111111111111` | `0.1111111111111111` | 无 |
| `consistency_recency_focused.json` | `44.99428769584678` | `8.057454237143219` | `2.4969659078022812` | `1.0555111192978415` | `14.38888888888889` | `9/9` | `0.2222222222222222` | `0.1111111111111111` | `0.1111111111111111` | 无 |

推荐配置：`recommendation.confidence_config = consistency_recency_focused.json`。推荐理由为该配置在九个场景中取得最低或并列最低路径总代价，`ΔC` 均值最高，且无失败场景。按地图族统计，该配置在 `sample` 场景中胜出 `6` 次，在 `seeded_synthetic` 场景中胜出 `3` 次。

实验结论：

- 三组配置均通过数据契约与路径验证，九个场景均可达且无失败场景，说明扩展到半合成地图族后仍未破坏 P0/P1 最小闭环。
- `consistency_recency_focused.json` 在 `9/9` 个场景中取得最低或并列最低路径总代价，路径总代价均值比默认配置低约 `0.30%`，优势仍然较小，但在 sample 与 seeded_synthetic 两类地图中方向一致。
- `consistency_recency_focused.json` 的 `ΔC` 均值最高，约为默认配置的 `2.12` 倍、偏观测配置的 `5.94` 倍，说明当前简化观测模型下，时间衰减和一致性权重更能放大可解释的正向可信度提升。
- `observation_focused.json` 仍不适合作为默认策略：它的路径总代价均值最高，`ΔC` 均值最低，更新后低可信区域面积均值最高。
- `risk_conflict_gap` 和 `synthetic_risk_band` 均覆盖低可信高风险路径，配置级风险冲突命中率为 `0.2222222222222222`。
- Top-K 稳定性和序列目标稳定性均为 `0.1111111111111111`，主要因为九个场景的价值区域、遮挡、地图族和观测位姿不同，首选目标随场景变化；在同一场景内，三组权重配置的 Top-3 目标和多步序列仍基本一致。

推荐结论：下一轮实验继续优先使用 `consistency_recency_focused.json` 作为 P1/P2 探索基线，保留 `default.json` 作为对照；`observation_focused.json` 仅用于观测模型增强后的再评估。暂不把推荐基线自动改成默认配置，原因是当前虽然加入了固定种子的半合成地图族，但仍不是真实遥感数据或长序列在线观测验证。

局限性：当前结果来自六个确定性样例场景和三个固定种子半合成场景，覆盖了不同观测位姿、目标点、地图尺寸、低可信区域分布、价值图分布、简单遮挡、两步 lookahead 和离线多步目标序列，但还没有覆盖真实遥感数据、长序列观测或复杂三维遮挡。

## 本轮增量验证

本轮已完成外部 `.npz` 地图输入、离散地形类别观测似然、P2 多步目标序列解释和 `model-explorer` JSON 契约。消融脚本的逐次 `runs` 现在包含：

- `map_source_kind`：区分 `sample`、`seeded_synthetic` 和 `npz_grid`。
- `terrain_model_confidence_mean`：由 `safe_regolith`、`rough`、`obstacle`、`shadow_risk` 类别后验熵派生的 `model` 可信度均值。
- `top_goal_sequence_details`：Top 序列的坐标、`utility`、`delta_c`、价值覆盖、风险、路径代价、覆盖面积、每段路径代价、累计风险和不可达原因。

默认 9 场景矩阵重新运行后仍保持上一轮结论：`consistency_recency_focused.json` 在 `9/9` 个场景中取得最低或并列最低路径总代价，`confidence_delta_c_mean = 2.4969659078022812`，无失败场景；`default.json` 的 `confidence_delta_c_mean = 1.1804746191998106`，`observation_focused.json` 的 `confidence_delta_c_mean = 0.4200272742285056`。

额外使用一个临时 `npz_grid` 外部地图场景对比 `default.json` 与 `consistency_recency_focused.json`，结果如下：

| 配置 | 地图来源 | 路径可达 | 路径总代价 | `ΔC` | 类别后验 `model` 可信度均值 | Top 序列数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `default.json` | `npz_grid` | `true` | `17.86029704719798` | `1.5965623630228316` | `0.37667757942364055` | `2` |
| `consistency_recency_focused.json` | `npz_grid` | `true` | `17.728274604176686` | `2.6826057450069047` | `0.37667757942364055` | `2` |

外部地图单场景中，`consistency_recency_focused.json` 仍取得较低路径总代价和更高 `ΔC`，推荐结论没有被打破。但该验证仍只是首版 `.npz` 接入烟测，不足以把 `consistency_recency_focused.json` 自动切换为默认配置。

`model-explorer` 对接契约已固定 `schema_version = model-explorer-contract/v1`，稳定字段见 `docs/model-explorer-interface.md`，示例见 `docs/model-explorer-contract-example.json`。该接口只输出地图摘要、约束摘要、Top-K 目标、Top 序列和观测更新报告，不引入在线重规划或任务状态机。

## 下一步

- 扩展 `.npz` 外部地图验证集，覆盖多尺寸、多光照带、多障碍分布和不同观测位姿。
- 用真实/半真实 DEM 或遥感派生栅格继续校验 `consistency_recency_focused.json` 的稳定性。
- 在不改变项目边界的前提下，为 `model-explorer` 增加更多报告消费侧样例和字段兼容性检查。
