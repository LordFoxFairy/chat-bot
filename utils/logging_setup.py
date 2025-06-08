import logging

# 设置日志格式和等级
logging.basicConfig(
    level=logging.INFO,  # 或改为 logging.DEBUG
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)