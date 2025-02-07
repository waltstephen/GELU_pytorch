from transformers import RobertaForSequenceClassification, RobertaConfig

class ModelLoader:
    @staticmethod
    def load_model(config, num_labels):
        model_name = config["model"]["pretrained_path"]
        
        # 根据配置选择模型尺寸
        if "roberta-large" in model_name:
            model_config = RobertaConfig.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:  # 默认使用base配置
            model_config = RobertaConfig.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            
        return RobertaForSequenceClassification.from_pretrained(
            model_name,
            config=model_config
        ) 