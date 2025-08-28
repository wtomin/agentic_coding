# Agentic 代码转换工具

[English](README.md) | [中文](readme_cn.md)

## 项目简介

本项目是一个自动化的 PyTorch 到 MindSpore 代码转换工具，支持多种输入方式，旨在最大程度保持代码功能和结构的同时，适配 MindSpore 实现。工具基于 AST（抽象语法树）转换和模式匹配，确保高质量转换。并且利用AI coding agent的代码能力，对复杂转换的情况进行处理。

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

## 快速开始

1. 准备待转换的 PyTorch 仓库：
```bash
# 克隆本项目
 git clone https://github.com/wtomin/agentic_coding
 git checkout flex-inputs
 cd agentic_coding
 cd example_inputs
 bash download.sh  # 下载示例 PyTorch 仓库 https://github.com/ivanwhaf/yolov1-pytorch/tree/master
 cd ..
 mv example_inputs/ inputs/
```
2. 启动自动转换：
```bash
mv example_inputs/ inputs/
python auto_convert.py --src_root ./inputs --dst_root ./outputs  # 部分转换 torch 脚本为 mindspore
```
3. 生成任务文件：
```bash
python task_generator.py
```
任务文件会保存在 tasks/ 目录下，每个 .json 文件对应一个待转换的 python 脚本。

4. 启动智能体进行转换：
```text
当前任务文件为 @tasks/task_001.json。请阅读 @CONVERT.md 并开始转换任务。
```
建议任务较多时并行独立转换，任务较少时可合并处理。

## 项目结构

```
agentic_coding/
├── task_generator.py        # 任务生成主控
├── convert_folder.py        # 批量转换实现
├── convert_single_file.py   # 单文件转换工具
├── examples/                # 示例代码与转换参考
│   ├── dataset/             # 数据集相关示例
│   ├── inference/           # 推理代码示例
│   ├── modeling/            # 模型结构示例
│   └── training/            # 训练脚本示例
├── example_inputs/          # PyTorch 示例文件
├── inputs/                  # 输入 PyTorch 文件
└── outputs/                 # MindSpore 转换输出
```

## 主要功能
- 自动识别 PyTorch 代码并生成转换任务
- 基于规则的 API 映射与语法转换
- 支持多类别（建模、数据集、训练、推理等）
- 转换日志与详细变更记录
- 生成测试脚本，支持数值一致性验证

## 转换规则（详见 CLAUDE.md）
- **最小化修改**：保留变量名和结构，仅改动必要部分
- **设备相关代码移除**：去除 .to(device)、.device 等 CUDA 相关逻辑
- **框架命名替换**：torch → mindspore
- **API 映射**：如 torch.nn.Module.forward → mindspore.nn.Cell.construct
- **参数初始化**：nn.init.constant_ → tensor.set_data(initializer(Constant(val), ...))
- **梯度检查点**：移除相关逻辑，MindSpore 不支持时抛出 NotImplementedError
- **Tokenizer 输出**：需 return_tensors="np" 并转为 Tensor
- **常见 API 替换**：如 .expand → .broadcast_to，.detach → .clone

## 常见问题与支持
- MindSpore 未安装或 API 不兼容：请先检查环境
- 设备/导入错误：确认设备相关代码已移除
- API 映射不全：参考 CLAUDE.md
- 复杂自定义操作、未支持特性需人工复查

如有疑问：
1. 先查阅本说明和 CLAUDE.md
2. 查看 examples/ 目录
3. 在 GitHub 提 issue

---

**注意**：本工具可大幅自动化转换流程，但复杂模型仍需人工复查和调优。请务必在生产环境前充分验证转换结果。
