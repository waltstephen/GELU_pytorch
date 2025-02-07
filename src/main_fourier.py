import sys
from transformers import RobertaTokenizer
from utils.config import load_config
from utils.logger import setup_logger
from models.base_model import ModelLoader
from importlib import import_module
from evaluation import Evaluator, TaskMetrics

def main():
    # 加载配置
    config = load_config()
    logger = setup_logger(config)
    
    # 初始化组件
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["pretrained_path"])
    task_name = config["task"]["name"]
    dataset_module = import_module(f"data.task_datasets.{task_name}")
    dataset_class = getattr(dataset_module, f"{task_name.upper()}Dataset")
    dataset = dataset_class(config["task"])
    dataset.preprocess_data(tokenizer)
    
    model = ModelLoader.load_model(
        config=config,
        num_labels=config["task"]["num_labels"]
    )
    
    # 执行评估
    evaluator = Evaluator(model, dataset)
    true_labels, predictions = evaluator.evaluate()
    
    # 计算指标
    metrics = TaskMetrics.compute(
        task_type=config["task"]["evaluation_mode"],
        y_true=true_labels,
        y_pred=predictions
    )
    
    # 添加模型和任务信息
    logger.info(f"{config['model']['name']}在{config['task']['name']}任务评估结果: {metrics}")

if __name__ == "__main__":
    main() 