# GELU Baseline

<div align="right">
  <a href="README.md">English</a>
</div>

GELU Baseline 是一个基于 PyTorch 和 Transformers 的高级库，专门用于在 GLUE 数据集上进行多任务评估。该项目旨在提供一个清晰、模块化的框架，方便研究人员和开发者快速测试和验证不同模型在 GLUE 任务上的表现。

## 功能特性

- **多任务支持**：支持 GLUE 数据集中的多个任务，包括 CoLA、SST-2、MRPC、QQP、QNLI、RTE、STS-B 和 WNLI。
- **模块化设计**：通过配置文件管理不同任务的测试，保持代码的清晰和易用性。
- **灵活配置**：通过 YAML 配置文件轻松切换任务和模型配置。
- **评估指标**：自动计算任务相关指标，如准确率、F1 值、Matthews 相关系数等。
- **日志记录**：详细的日志记录，方便跟踪和调试。

## 快速开始

### 版本依赖

```bash
pip install -r requirements.txt
```

### 运行测试

1. 修改 `configs/base.yaml` 中的任务名称，例如将 `task.name` 改为 `"rte"`。
2. 运行主程序：

```bash
python src/main.py
```

### 配置文件示例

```yaml:configs/base.yaml
model:
  name: "roberta-large"  # 可选 roberta-base / roberta-large
  pretrained_path: "roberta-base"  # HuggingFace模型名称或本地路径
  save_dir: "./saved_models"

training:
  batch_size: 32
  learning_rate: 2e-5
  max_epochs: 10
  early_stop_patience: 3

task:
  name: "rte"  # 修改此处切换任务
```

## 目前支持的模型

- RoBERTa-Base
- RoBERTa-Large

## 目前支持的任务

- CoLA
- SST-2
- MRPC
- QQP
- QNLI
- RTE
- STS-B
- WNLI

## 评估指标

根据任务类型，自动计算以下指标：

- **分类任务**：准确率、F1 值
- **回归任务**：Pearson 相关系数
- **CoLA 任务**：Matthews 相关系数

## 日志记录

日志文件保存在 `logs/` 目录下，格式为 `YYYYMMDD_HHMM.log`。日志内容包括模型名称、任务名称和评估结果。

## 贡献

欢迎提交 Issue 和 Pull Request，共同改进 GELU Baseline。

## 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。 