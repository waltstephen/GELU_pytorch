from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

class TaskMetrics:
    @staticmethod
    def compute(task_type, y_true, y_pred):
        if task_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="macro")
            }
        elif task_type == "regression":
            return {
                "pearson": pearsonr(y_true, y_pred)[0]
            }
        elif task_type == "cola":
            return {
                "matthews_corr": matthews_corrcoef(y_true, y_pred)
            }
        else:
            raise ValueError(f"未知任务类型: {task_type}") 