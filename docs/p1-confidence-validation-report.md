# P1 可信度最小闭环验证报告

## 基本信息

- 验证日期：2026-05-26
- 验证范围：P0 最小闭环 + P1 可信度分量、配置化融合、一次局部观测更新、数据契约报告、障碍/通行概率贝叶斯边界、可信度权重消融和 P2 候选目标生成起点
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
| 消融报告 | `outputs/ablation/confidence_ablation.json`、`.csv`、`.png`、`.html` |

平均可信度下降是预期结果：当前脚本同时模拟了可见区观测提升和未观测区域时间衰减。`ΔC` 和低可信区域面积变化用于确认局部观测确实带来了正向可信度提升。

消融实验默认比较三组可信度权重：默认配置、偏观测配置、偏一致性/时间衰减配置。每组结果记录路径代价、`ΔC`、低可信高风险路径比例和 Top-K 探索目标排序，并额外输出 HTML/PNG 可视化，用于后续参数选择和实验复现。

## 下一步

- 为贝叶斯更新增加离散地形类别或更细的观测似然模型。
- 将候选目标生成从当前启发式扩展到多步观测收益评估，但仍暂不引入在线重规划。
- 增加简单遮挡或观测几何质量接口，保持与当前 `sensor_range`、`sensor_fov` 假设参数兼容。
