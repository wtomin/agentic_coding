# Agentic Coding：PyTorch 到 MindSpore 转换

[English](README.md) | [中文](README_CN.md)

# 项目简介

本工具可自动将 PyTorch 的模型与配置代码转换为 MindSpore，特别适用于 `transformers` 相关模型。

## 安装说明

### 先决条件
- Python 3.7 及以上
- PyTorch（用于测试和验证）
- MindSpore（目标框架）
- 智能体（如 Claude 或 Cursor）

### 依赖安装
安装所需依赖：
```bash
pip install libcst # 规则自动转换
pip install torch  # 验证用
pip install mindspore  # 目标框架
```

# 与 mindone 快速集成

如需在 [mindone](https://github.com/mindspore-lab/mindone.git) 仓库中进行模型转换和测试，请按如下步骤操作：

## 克隆 mindone
```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/
```

## 拷贝转换相关文件
将本仓库中的以下文件和文件夹拷贝到 `mindone/` 目录下：
- `plans/`（整个文件夹）
- `CLAUDE.md`
- `auto_convert.py`

拷贝后目录结构如下：
```bash
mindone/
├── examples/     # mindone 原有
├── mindone/      # mindone 原有
├── scripts/      # mindone 原有
├── ...
├── plans/
│   ├── phase1_modeling_convert.md
│   └── phase2_test_script.md
├── CLAUDE.md
└── auto_convert.py
```

## 准备模型源码

确定目标模型名称（如 `bert`）。
将 PyTorch 模型代码和配置文件放置于：
```
mindone/mindone/transformers/models/bert/
```
至少应包含：
- `configuration_bert.py`
- `modeling_bert.py`
- `__init__.py`

## 运行规则转换

执行如下命令进行规则转换并更新模型文件夹：
```bash
python auto_convert.py --src_root mindone/mindone/transformers/models/bert/ --dst_root mindone/mindone/transformers/models/bert_ms/
mv mindone/mindone/transformers/models/bert_ms/ mindone/mindone/transformers/models/bert/
```

## 编辑模型计划
- 打开 `plans/phase1_modeling_convert.md`，将所有 `{model-name}` 替换为你的目标模型名（如 `bert`）。
- 打开 `plans/phase2_test_script.md`，同样替换。

## 启动智能体
启动你的智能体（如 Claude Code）。Claude Code 会自动加载 `CLAUDE.md` 作为系统提示。

如使用其他智能体，请相应地设置系统提示。

## 执行转换与测试步骤
可按如下方式指示智能体：

### a. 转换模型
```bash
╭────────────────────────────────────────────────────────────────────────────╮
│ >   Convert the modeling script following @plans/phase1_modeling_convert.md. │
╰────────────────────────────────────────────────────────────────────────────╯
```

### b. 编写并运行测试脚本
```bash
╭────────────────────────────────────────────────────────────────────────────╮
│ >   Write the test script following @plans/phase2_test_script.md.           │
╰────────────────────────────────────────────────────────────────────────────╯
```

---


## 文档

- **`CLAUDE.md`**：详细转换规则与技术指南
- **`plans/`**：模型转换与测试分步计划

## 常见问题

1. **导入错误**：请确保已正确安装 MindSpore
2. **设备错误**：检查设备相关代码是否已移除
3. **API 不匹配**：请查阅 `CLAUDE.md` 的 API 映射

## 支持

如有问题：
1. 先查阅常见问题部分
2. 阅读 `CLAUDE.md` 转换规则
3. 在 GitHub 提 issue

---

**注意**：本工具可自动化大部分转换流程，但复杂模型仍需人工复查和调整。请务必在生产环境前充分验证转换结果。

