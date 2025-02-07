import logging
from datetime import datetime

def setup_logger(config):
    logger = logging.getLogger("GELU")
    logger.setLevel(logging.INFO)
    
    # 创建文件handler
    log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    file_handler = logging.FileHandler(log_file)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    
    # 设置格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger 