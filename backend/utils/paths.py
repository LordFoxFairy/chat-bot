from pathlib import Path

def get_project_root() -> Path:
    """
    获取项目根目录
    假设文件结构为: project_root/backend/utils/paths.py
    """
    return Path(__file__).parent.parent.parent

def resolve_project_path(path_str: str) -> Path:
    """
    将路径解析为相对于项目根目录的绝对路径
    如果 path_str 已经是绝对路径，则直接返回
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path
