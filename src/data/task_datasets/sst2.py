from data.base_dataset import TaskDataset
from datasets import load_dataset
import torch

class SST2Dataset(TaskDataset):
    def __init__(self, config):
        super().__init__(config)
        self.load_data()
        
    def load_data(self):
        self.raw_dataset = load_dataset(
            self.config["dataset_name"],
            self.config["dataset_config"],
            cache_dir=self.config.get("cache_path", "./data_cache")
        )
        
    def preprocess_data(self, tokenizer):
        def tokenize_fn(examples):
            return tokenizer(
                examples[self.text_field],
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
    
    @staticmethod
    def format_batch(batch):
        return {
            "input_ids": torch.tensor(batch["input_ids"]).unsqueeze(0) if isinstance(batch["input_ids"], list) else batch["input_ids"],
            "attention_mask": torch.tensor(batch["attention_mask"]).unsqueeze(0) if isinstance(batch["attention_mask"], list) else batch["attention_mask"],
            "labels": torch.tensor([batch["label"]])
        } 