# AGENTS.md

## 语言约定

- README 使用中文编写。
- 代码注释和 docstring 使用中文编写。
- 新增面向开发者的说明性文档时，默认使用中文；保留必要的代码标识符、命令、配置键和第三方工具名原文。

## 文件删除约束

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

正确示例：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除，应停止操作，并请求用户手动删除。
