# GELU Baseline

<div align="right">
  <a href="README_zh.md">中文</a>
</div>

GELU Baseline is a high-level library based on PyTorch and Transformers, specifically designed for multi-task evaluation on the GLUE dataset. This project aims to provide a clear and modular framework, enabling researchers and developers to quickly test and validate the performance of different models on GLUE tasks.

## Features

- **Multi-task Support**: Supports multiple tasks in the GLUE dataset, including CoLA, SST-2, MRPC, QQP, QNLI, RTE, STS-B, and WNLI.
- **Modular Design**: Manages different task tests through configuration files, ensuring code clarity and usability.
- **Flexible Configuration**: Easily switch tasks and model configurations via YAML configuration files.
- **Evaluation Metrics**: Automatically calculates task-related metrics such as accuracy, F1 score, Matthews correlation coefficient, etc.
- **Logging**: Detailed logging for easy tracking and debugging.

## Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

### Running Tests

1. Modify the task name in `configs/base.yaml`, for example, change `task.name` to `"rte"`.
2. Run the main program:

```bash
python src/main.py
```

### Example Configuration

```yaml:configs/base.yaml
model:
  name: "roberta-large"  # Options: roberta-base / roberta-large
  pretrained_path: "roberta-base"  # HuggingFace model name or local path
  save_dir: "./saved_models"

training:
  batch_size: 32
  learning_rate: 2e-5
  max_epochs: 10
  early_stop_patience: 3

task:
  name: "rte"  # Modify this to switch tasks
```

## Currently Supported Models

- RoBERTa-Base
- RoBERTa-Large

## Currently Supported Tasks

- CoLA
- SST-2
- MRPC
- QQP
- QNLI
- RTE
- STS-B
- WNLI

## Evaluation Metrics

Automatically calculates the following metrics based on task type:

- **Classification Tasks**: Accuracy, F1 score
- **Regression Tasks**: Pearson correlation coefficient
- **CoLA Task**: Matthews correlation coefficient

## Logging

Log files are saved in the `logs/` directory, with the format `YYYYMMDD_HHMM.log`. Logs include model name, task name, and evaluation results.

## Contributing

We welcome issues and pull requests to improve GELU Baseline.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 