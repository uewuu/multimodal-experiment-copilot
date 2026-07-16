# Day 01：读取实验配置

## 今天完成的功能

使用 Python 和 PyYAML 读取：

`examples/demo_experiment/hparams.yaml`

程序能够进入 YAML 最外层的 `config` 节点，并输出核心实验配置。

## 程序执行流程

1. 使用 Path 表示 YAML 文件路径。
2. 使用 yaml.safe_load() 读取 YAML。
3. 检查最外层数据是不是字典。
4. 从最外层字典中取得 config 节点。
5. 遍历 fields 列表。
6. 使用 config.get() 读取字段值。
7. 在终端显示字段名称和字段值。

## 今天理解的 Python 内容

- Python 列表
- for 循环
- Python 字典
- dict.get()
- f-string
- try/except
- 函数参数和返回值

## 我亲自完成的修改

在 fields 列表中增加：

`attention_type`

程序随后输出：

`attention_type: linear`

这个修改只影响程序显示的内容，不会修改 YAML 或改变模型训练配置。

## 下一步

读取 history.json，并寻找最佳验证 R² 对应的 Epoch。