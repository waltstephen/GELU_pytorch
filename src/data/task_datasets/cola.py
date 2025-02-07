from datasets import load_dataset
from data.base_dataset import TaskDataset
import torch

class COLADataset(TaskDataset):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()
        
    def load_data(self):
        """使用HuggingFace datasets加载数据"""
        self.raw_dataset = load_dataset(
            self.config["dataset_name"],
            self.config["dataset_config"],
            cache_dir=self.config.get("cache_path", "./data_cache")
        )
        
    def preprocess_data(self, tokenizer):
        """实现分词处理"""
        def tokenize_fn(examples):
            return tokenizer(
                examples[self.text_fields[0]],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        self.tokenized_dataset = self.raw_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.text_fields
        )
        
    def get_train_samples(self):
        return self.raw_dataset["train"]
    
    def get_validation_samples(self):
        return self.raw_dataset["validation"]
    
    def get_test_samples(self):
        return self.raw_dataset["test"]
    
    @staticmethod
    def format_batch(batch):
        """与基类字段对齐"""
        return {
            "input_ids": torch.tensor(batch["input_ids"]).unsqueeze(0) if isinstance(batch["input_ids"], list) else batch["input_ids"],
            "attention_mask": torch.tensor(batch["attention_mask"]).unsqueeze(0) if isinstance(batch["attention_mask"], list) else batch["attention_mask"],
            "labels": torch.tensor([batch["label"]])  # 将标量包装成列表
        } 