import yaml
from pathlib import Path

def load_config():
    base_path = Path(__file__).parent.parent.parent / "configs"
    
    # 加载基础配置
    with open(base_path / "base.yaml") as f:
        base_config = yaml.safe_load(f)
    
    # 根据任务名称加载对应配置
    task_name = base_config["task"]["name"]
    with open(base_path / "task" / f"{task_name}.yaml") as f:
        task_config = yaml.safe_load(f)
    
    return {**base_config, "task": task_config.get("task", {})} 