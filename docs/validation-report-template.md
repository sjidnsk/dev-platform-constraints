# 验证报告模板

## 基本信息

- 项目版本或提交：`<commit>`
- 验证日期：`<YYYY-MM-DD>`
- 验证人：`<name>`
- Python 环境：`<python --version>`
- 输入平台配置：`configs/platforms/<name>.json`
- 输入地图或生成脚本：`<path or command>`

## 验证命令

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
python scripts\run_minimal_closure.py
python scripts\visualize_minimal_closure.py --output-dir outputs\visualization
```

## 指标结果

| 指标 | 结果 | 备注 |
| --- | --- | --- |
| 数据契约是否有效 | `<true/false>` | 来自 `ValidationReport` |
| 路径是否可达 | `<true/false>` | 来自 A* |
| 路径节点数 | `<int>` | 规划输出 |
| 路径总代价 | `<float>` | 规划输出 |
| 硬约束违规数 | `<int>` | 原因位图非零数量 |
| 更新前平均可信度 | `<float>` | 局部观测更新前 |
| 更新后平均可信度 | `<float>` | 局部观测更新后 |
| 低可信区域面积变化 | `<before -> after>` | 单位平方米 |
| `ΔC` | `<float>` | 可信度正向提升总量 |
| 低可信高风险路径比例 | `<float>` | 路径节点比例 |

## 失败案例

- 失败命令：`<command>`
- 失败现象：`<summary>`
- 可能原因：`<hypothesis>`
- 下一步处理：`<action>`

## 参数调整记录

| 参数 | 原值 | 新值 | 调整原因 |
| --- | --- | --- | --- |
| `<parameter>` | `<old>` | `<new>` | `<reason>` |
