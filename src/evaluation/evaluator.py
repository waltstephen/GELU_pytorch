import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataset, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        print(f"Using device: {self.device.upper()}")
    
    def evaluate(self):
        print(f"Starting evaluation on {len(self.dataset.get_split('validation'))} samples...")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # 获取验证集数据
        val_data = self.dataset.get_split("validation")
        
        with torch.no_grad():
            for raw_batch in tqdm(val_data, desc="Evaluating"):
                batch = self.dataset.format_batch(raw_batch)  # 调用格式转换方法
                inputs = self._prepare_batch(batch)
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())
        
        return all_labels, all_preds
    
    def _prepare_batch(self, batch):
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "labels": batch["labels"].to(self.device)
        } 