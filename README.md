# dev-platform-constraints

本项目实现 `docs/confidence-metrics-design.md` 中定义的 P0
“环境-平台-约束模型”最小闭环，并提供 P1 可信度更新能力和 P2 探索目标排序起点。
正式项目边界见根目录 `PROJECT_BOUNDARY.md`；本目录只增强建模底座、可解释指标和可复现实验，不承载完整自主探索主循环。

## 范围

已完成的 P0 内容：

- `GridMap` 与每层元数据：`source_id`、`timestamp`、`resolution`、`frame_id`、`unit`。
- 核心地图层命名：`elevation`、`slope`、`roughness`、`obstacle`、`obstacle_height`、`illumination`、`confidence`、`value`、`traversability`、`cost`、`valid_mask`。
- 数据契约校验：尺寸、元数据、单位、`NaN` 处理、归一化范围和非负代价。
- 地形特征派生：以度为单位的坡度和归一化局部崎岖度。
- 平台参数 JSON 加载，并区分 `public`、`estimated`、`assumed` 来源标记。
- `configs/platforms/` 下的玉兔与玉兔二号近似平台配置。
- 硬约束掩膜和原因位图层，覆盖无效栅格、坡度、障碍、障碍高度和可选禁行区。
- 基础非负 `cost` 输出，以及与 `confidence` 分离的 `traversability` 输出。
- 用 A* 验证最小闭环路径是否可达。

当前 P1 实现可解释的可信度分量、配置化融合、一次局部观测更新、显式观测模型、数据契约报告、障碍概率与通行概率贝叶斯后验边界；
暂不实现复杂三维遮挡、高级观测调度或风险传播。`confidence` 表示信息可靠程度，不等同于
`traversability`。`value` 默认视为科学或任务效用，不进入通行代价，除非后续实验显式启用价值奖励权重。

## 开发状态

- P0：已完成，包含栅格数据契约、地形特征、平台配置、硬约束、非负代价图和 A* 最小闭环。
- P1：已达到可运行研究原型阶段，包含 `c_resolution`、`c_observation`、`c_recency`、`c_consistency`、缺失分量重归一化、配置化权重、局部观测更新、数据契约报告、可配置多场景可信度权重消融脚本、最小贝叶斯状态后验、离散地形类别后验和 JSON 可配置类别观测似然。
- P2：已达到可运行研究原型阶段，提供离散候选观测目标生成与排序起点，候选收益按传感器 footprint 和 `coverage_mask` 新增覆盖估计，并支持两步离散 lookahead、离线多步目标序列评估、序列去重覆盖和风险/不可达解释；暂不实现 Hybrid A*、动力学约束、在线重规划和完整工程部署能力。
- 生成数据：scripts/generate_example_data.py 是生成型脚本，用于刷新开发示例数据，不是核心运行依赖；生成的 `data/sample_grid.npz` 不提交版本库。
- 当前边界内完成度：按 `PROJECT_BOUNDARY.md` 衡量约 `95%`；剩余约 `5%` 主要留给真实/半真实外部数据接入和上层 `model-explorer` 联调反馈。

`src/dev_platform_constraints/` 按职责分为：

- `core/`：栅格地图数据结构、图层元数据和数据契约校验。
- `platforms/`：平台参数模型、配置加载和默认配置路径。
- `terrain/`：坡度、崎岖度等地形特征派生。
- `confidence/`：可信度分量计算、加权融合、局部观测更新和最小贝叶斯状态后验。
- `exploration/`：候选观测目标生成、效用评分和排序。
- `experiments/`：消融实验场景配置加载和默认场景配置路径。
- `mapping/`：硬约束掩膜、原因位图和通行代价图生成。
- `path_planning/`：A* 等路径规划算法。
- `sample_data/`：开发示例地图生成。
- `reporting/`：静态可视化和报告输出。

旧的平铺模块导入路径已移除，代码统一使用上述分组包导入。

验证命令：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python scripts\run_minimal_closure.py
python scripts\run_confidence_ablation.py --output-dir outputs\ablation
python scripts\generate_npz_validation_maps.py --output-dir data\validation_maps
python scripts\run_confidence_ablation.py --scenario-config configs\ablation\npz_validation_scenarios.json --output-dir outputs\ablation_npz
python scripts\generate_example_data.py
```

预期结果：

- 单元测试输出 `OK`。
- `scripts\run_minimal_closure.py` 输出 JSON 摘要，其中 `validation_valid` 和 `path_reachable` 均为 `true`，并包含 `data_contract`、`confidence_delta_c`、低可信区域面积、局部观测更新指标、`coverage_rate` 和 `coverage_rate_delta`。
- `scripts\run_confidence_ablation.py` 输出 JSON/CSV 数据报告和 HTML/PNG 可视化报告，比较不同可信度权重在可配置多场景下对路径代价、`ΔC`、覆盖率、低可信高风险路径比例、配置胜率、风险冲突命中率和 Top-K 探索目标稳定性的影响。
- `scripts\generate_npz_validation_maps.py` 生成固定种子的外部 `.npz` 验证地图；生成物位于 `data/validation_maps/`，不提交版本库。
- `configs\ablation\npz_validation_scenarios.json` 可驱动 `npz_grid` 外部地图矩阵，验证 `.npz` 输入、类别似然配置和 P2 序列解释在外部地图上的稳定性。
- `scripts\generate_example_data.py` 写入 `data/sample_grid.npz`，输出写入路径和图层列表。

## 运行

PowerShell：

```powershell
.\scripts\setup_env.ps1 -RunValidation
conda activate lunar-explorer
```

Ubuntu Bash：

```bash
bash scripts/setup_env.sh --run-validation
conda activate lunar-explorer
```

部署脚本会使用 `environment.yml` 创建或更新 Python 3.12 Conda 环境，
默认环境名为 `lunar-explorer`。脚本会对已有环境显式执行
`python=3.12` 约束校验，但不会以 editable 模式安装本仓库；验证命令通过
`PYTHONPATH=src` 读取当前源码。
可以使用 dry-run 模式只查看将执行的命令：

```powershell
.\scripts\setup_env.ps1 -DryRun -RunValidation
```

```bash
bash scripts/setup_env.sh --dry-run --run-validation
```

手动运行命令：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python scripts\run_minimal_closure.py
python scripts\run_confidence_ablation.py --output-dir outputs\ablation
python scripts\generate_npz_validation_maps.py --output-dir data\validation_maps
python scripts\run_confidence_ablation.py --scenario-config configs\ablation\npz_validation_scenarios.json --output-dir outputs\ablation_npz
python scripts\generate_example_data.py
```

生成最小闭环静态可视化报告：

```powershell
$env:PYTHONPATH='src'
python scripts\visualize_minimal_closure.py --output-dir outputs\visualization --confidence-config configs\confidence\default.json
```

可通过 `--confidence-config configs\confidence\default.json` 指定可信度融合权重配置。

该命令会生成 `outputs\visualization\minimal_closure.png` 和
`outputs\visualization\minimal_closure.html`，展示 `elevation`、`slope`、`roughness`、
`obstacle`、`confidence`、`traversability`、`cost + path` 和硬约束结果。HTML 报告会按中文分组展示：

- 路径与约束摘要：可达性、路径节点数、路径总代价、硬约束违规数和低可信高风险路径比例。
- 地图与代价摘要：栅格尺寸、分辨率和可通行代价范围。
- 可信度更新摘要：更新前后平均可信度、低可信区域面积、`ΔC`、可见栅格数和已更新栅格数。
- 覆盖率摘要：有效栅格数、累计已覆盖栅格数、新增覆盖栅格数、累计覆盖面积、覆盖率和覆盖率增量。
- 数据契约摘要：图层数量、缺失核心图层数、契约错误数和契约警告数。
- 探索目标摘要：Top-K 候选目标的栅格、效用、可达性、信息增益、价值、可信度提升、风险、路径代价、总覆盖面积、预计新增覆盖面积和预计覆盖率增量。
- 约束原因计数：无效、坡度、障碍、障碍高度和禁行区触发次数。

最小闭环脚本会执行：

1. 生成 2.5D 月面示例栅格。
2. 派生 `slope` 和 `roughness`。
3. 加载玉兔二号近似平台配置。
4. 加载可信度融合权重配置。
5. 模拟一次局部观测并更新 `confidence`。
6. 更新运行时 `coverage_mask` 并输出覆盖率指标。
7. 生成硬约束掩膜。
8. 生成非负 `cost` 和独立的 `traversability`。
9. 运行 A* 验证起终点可达性。
10. 生成候选探索目标并按效用排序。
11. 输出 JSON 摘要、数据契约报告和 Top-K 探索目标。

运行可信度权重消融实验：

```powershell
$env:PYTHONPATH='src'
python scripts\run_confidence_ablation.py --output-dir outputs\ablation
```

可通过 `--scenario-config configs\ablation\scenarios.json` 指定消融场景配置。默认会在
`baseline_gap`、`upper_observation`、`compact_value`、`risk_conflict_gap`、
`simple_occlusion`、`multi_observation` 和三个 `seeded_synthetic` 半合成场景中比较
`configs\confidence\default.json`、`configs\confidence\observation_focused.json`
和 `configs\confidence\consistency_recency_focused.json`，并写入
`outputs\ablation\confidence_ablation.json`、`outputs\ablation\confidence_ablation.csv`、
`outputs\ablation\confidence_ablation.png` 和 `outputs\ablation\confidence_ablation.html`。
JSON 报告包含 `scenarios`、逐次 `runs`、配置级 `aggregate` 和 `recommendation`；CSV 保留逐次实验记录。
HTML/PNG 报告展示路径总代价分布、可信度正向提升总量、覆盖率、配置胜率、风险冲突命中率、Top-K 稳定性、序列目标稳定性和逐场景 Top-K 探索目标。
可通过 `--terrain-likelihood-config configs\confidence\terrain_likelihood_default.json` 显式指定离散地形类别观测似然配置；也可以在场景配置顶层设置
`terrain_likelihood_config`，命令行参数优先级更高。

当前多场景结果中，`consistency_recency_focused.json` 在 `9/9` 个场景取得最低或并列最低路径总代价，
且 `confidence_delta_c_mean = 2.4969659078022812`，高于 `default.json` 的
`1.1804746191998106` 和 `observation_focused.json` 的 `0.4200272742285056`。
报告中的 `recommendation.confidence_config` 因此推荐以 `consistency_recency_focused.json` 作为下一轮 P1/P2 实验基线，
但默认配置暂不自动切换，直到更多地图和观测位姿验证完成。

`scripts/generate_example_data.py` 会写入 `data/sample_grid.npz`，其中包含开发用示例图层。
该文件由脚本生成，不作为核心运行依赖或版本库输入。

生成外部 `.npz` 验证地图：

```powershell
$env:PYTHONPATH='src'
python scripts\generate_npz_validation_maps.py --output-dir data\validation_maps
python scripts\run_confidence_ablation.py --scenario-config configs\ablation\npz_validation_scenarios.json --output-dir outputs\ablation_npz
```

该验证集当前包含 `npz_shadow_corridor`、`npz_rock_field_multi_pose` 和
`npz_low_confidence_risk_band` 三个固定种子地图，覆盖阴影走廊、岩块/多观测位姿和低可信风险带。
`.npz` 生成物由脚本刷新，不纳入版本库；入库的是场景配置和生成脚本。

## 本轮新增能力

- 外部地图输入与验证集：消融场景支持 `map_source.kind = "npz_grid"`，从 `.npz` 地图包读取 `elevation`、`obstacle`、`obstacle_height`、`illumination`、`confidence`、`value` 和 `valid_mask`；新增 `scripts/generate_npz_validation_maps.py` 和 `configs/ablation/npz_validation_scenarios.json`，用于固定复现三类外部地图验证输入。
- P1 类别后验和似然标定：新增 `TerrainLikelihoodRules`、`compute_terrain_category_likelihood(...)`、`load_terrain_likelihood_rules(...)` 和 `configs/confidence/terrain_likelihood_default.json`，从坡度、崎岖度、障碍概率和光照层生成 `safe_regolith`、`rough`、`obstacle`、`shadow_risk` 四类观测似然，再由类别后验熵派生 `model` 可信度分量。
- 覆盖率状态维护：新增运行时 `coverage_mask`、覆盖率更新和观测覆盖收益估计能力，只统计 `valid_mask` 内累计唯一可见栅格；候选目标同步输出 `expected_new_coverage_area` 和 `expected_coverage_rate_delta`，供上层 `model-explorer` 做覆盖率优先排序。
- P2 序列解释：`evaluate_goal_sequences(...)` 输出去重后的序列覆盖面积、每段路径代价、累计风险、不可达原因和风险原因；消融 JSON/CSV/HTML 同步保留 Top 序列解释字段。
- `model-explorer` 对接：稳定 JSON 契约由 `build_model_explorer_contract(...)` 生成，字段说明见 `docs/model-explorer-interface.md`，完整示例见 `docs/model-explorer-contract-example.json`，最小可消费示例见 `docs/model-explorer-minimal-example.json`。
- `path-planner` sidecar 对接：`build_path_planner_sidecar(...)` 和 `scripts/export_path_planner_sidecars.py` 可为半真实场景导出 `path-planner-sidecar/v1`，补充完整 `cost`、`passable_mask` 和可选 terrain layers，避免 `model-explorer` 在可信实验中使用 open-grid fallback。

外部 `.npz` 输入首版只依赖 `numpy`，不引入 GIS 重型依赖；默认可信度配置仍不自动切换。

导出半真实联调输入：

```bash
python scripts/generate_npz_validation_maps.py --scenario-set smoke --output-dir data/validation_maps --scenario-config outputs/npz_validation_scenarios.generated.json
PYTHONPATH=src python scripts/export_path_planner_sidecars.py \
  --scenario-config outputs/npz_validation_scenarios.generated.json \
  --output-dir outputs/path_planner_sidecars
```

输出目录包含每个场景的 `*.contract.json` 和 `*.path-planner-sidecar.json`，以及 `manifest.json`。当前固定覆盖：

- `npz_shadow_corridor`
- `npz_rock_field_multi_pose`
- `npz_low_confidence_risk_band`

`scripts/generate_npz_validation_maps.py` 还支持 `--scenario-set stress` 和
`--scenario-set all`。stress 集包含近阻断走廊、高风险价值陷阱、密集岩石收缩通道和
`mixed_stress` 绕行场景。`mixed_stress` 会同时保留 path-planner 可达候选、会触发
failure/replan 的候选和契约层 blocked 候选，用于确认上层反馈既能解释失败，又能比较可达替代目标，
而不是只重复 easy smoke 验证。

## 假设

- 单位遵循设计文档：长度使用米，角度使用度。
- `NaN` 表示未知或不可用数据，只允许出现在 `valid_mask == false` 的位置。
- 低 `confidence` 不会自动把栅格变成不可通行，只会按地形基础风险增加代价风险。
- P0 使用普通栅格 A*，因此 `min_turning_radius` 只加载和校验为假设占位值，不实际约束路径。
- 规划代价会被限制为非负值。被硬约束排除的栅格使用大的有限代价表示，同时由硬约束掩膜从 A* 扩展中排除。
