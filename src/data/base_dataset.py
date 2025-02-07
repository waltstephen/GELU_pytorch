from abc import ABC, abstractmethod
from datasets import DatasetDict
from typing import Dict, Any

class TaskDataset(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        基础数据集类
        :param config: 包含以下键的配置字典:
            - dataset_name: 数据集名称 (e.g. "glue")
            - dataset_config: 数据集配置名 (e.g. "cola")
            - text_field: 文本字段名
            - label_field: 标签字段名
            - cache_path: 数据集缓存路径
        """
        self.config = config
        self.raw_dataset: DatasetDict = None  # 原始数据集对象
        self.tokenized_dataset: DatasetDict = None  # 分词后的数据集
        
    @abstractmethod
    def load_data(self) -> None:
        """加载原始数据集（必须由子类实现）"""
        pass
    
    @abstractmethod
    def preprocess_data(self, tokenizer) -> None:
        """
        数据预处理流程（必须由子类实现）
        :param tokenizer: 分词器实例
        """
        pass
    
    def get_split(self, split: str = "validation"):
        """
        获取指定数据分割
        :param split: 数据分割名称 (train/validation/test)
        """
        return self.tokenized_dataset.get(split, None)
    
    @property
    def num_labels(self) -> int:
        """获取分类标签数量"""
        return self.config.get("num_labels", 2)
    
    @property
    def text_fields(self) -> list:
        """获取文本字段名（支持多字段）"""
        return self.config.get("text_fields") or [self.config.get("text_field")]
    
    @property
    def label_field(self) -> str:
        """获取标签字段名"""
        return self.config["label_field"]
    
    @staticmethod
    def format_batch(batch: Dict) -> Dict:
        """
        通用批处理格式转换（子类可覆盖）
        :return: 包含以下键的字典:
            - texts: 文本列表
            - labels: 标签列表
        """
        return {
            "texts": batch["sentence"],
            "labels": batch["label"]
        } 