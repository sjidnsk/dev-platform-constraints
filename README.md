# dev-platform-constraints

本项目实现 `docs/confidence-metrics-design.md` 中定义的 P0
“环境-平台-约束模型”最小闭环。

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

P1/P2 暂不实现；当前只保留最小闭环所需的 costmap 和 A*。`confidence`
表示信息可靠程度，不等同于 `traversability`。`value` 默认视为科学或任务效用，
不进入通行代价，除非后续实验显式启用价值奖励权重。

## 开发状态

- P0：已完成，包含栅格数据契约、地形特征、平台配置、硬约束、非负代价图和 A* 最小闭环。
- P1：未完成，暂不实现高级观测、价值奖励、风险传播和面向实验的扩展代价模型。
- P2：未完成，暂不实现 Hybrid A*、动力学约束、在线重规划和完整工程部署能力。
- 生成数据：scripts/generate_example_data.py 是生成型脚本，用于刷新开发示例数据，不是核心运行依赖；生成的 `data/sample_grid.npz` 不提交版本库。

`src/dev_platform_constraints/` 按职责分为：

- `core/`：栅格地图数据结构、图层元数据和数据契约校验。
- `platforms/`：平台参数模型、配置加载和默认配置路径。
- `terrain/`：坡度、崎岖度等地形特征派生。
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
python scripts\generate_example_data.py
```

预期结果：

- 单元测试输出 `OK`。
- `scripts\run_minimal_closure.py` 输出 JSON 摘要，其中 `validation_valid` 和 `path_reachable` 均为 `true`。
- `scripts\generate_example_data.py` 写入 `data/sample_grid.npz`，输出写入路径和图层列表。

## 运行

PowerShell：

```powershell
.\scripts\setup_env.ps1 -RunValidation
conda activate dev-platform-constraints
```

Ubuntu Bash：

```bash
bash scripts/setup_env.sh --run-validation
conda activate dev-platform-constraints
```

部署脚本会使用 `environment.yml` 创建或更新 Python 3.12 Conda 环境，
并对已有环境显式执行 `python=3.12` 约束校验，然后以 editable 模式安装本仓库。
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
python scripts\generate_example_data.py
```

生成最小闭环静态可视化报告：

```powershell
$env:PYTHONPATH='src'
python scripts\visualize_minimal_closure.py --output-dir outputs\visualization
```

该命令会生成 `outputs\visualization\minimal_closure.png` 和
`outputs\visualization\minimal_closure.html`，展示 `elevation`、`slope`、`roughness`、
`obstacle`、`confidence`、`traversability`、`cost + path` 和硬约束结果。

最小闭环脚本会执行：

1. 生成 2.5D 月面示例栅格。
2. 派生 `slope` 和 `roughness`。
3. 加载玉兔二号近似平台配置。
4. 生成硬约束掩膜。
5. 生成非负 `cost` 和独立的 `traversability`。
6. 运行 A* 验证起终点可达性。
7. 输出 JSON 摘要。

`scripts/generate_example_data.py` 会写入 `data/sample_grid.npz`，其中包含开发用示例图层。
该文件由脚本生成，不作为核心运行依赖或版本库输入。

## 假设

- 单位遵循设计文档：长度使用米，角度使用度。
- `NaN` 表示未知或不可用数据，只允许出现在 `valid_mask == false` 的位置。
- 低 `confidence` 不会自动把栅格变成不可通行，只会按地形基础风险增加代价风险。
- P0 使用普通栅格 A*，因此 `min_turning_radius` 只加载和校验为假设占位值，不实际约束路径。
- 规划代价会被限制为非负值。被硬约束排除的栅格使用大的有限代价表示，同时由硬约束掩膜从 A* 扩展中排除。
