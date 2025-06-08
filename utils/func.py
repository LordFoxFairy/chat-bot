def universal_serializer(obj: object) -> dict | str:
    """
    一個通用的後備序列化器。
    - 如果物件有 __dict__ 屬性，就返回它的字典表示。
    - 否則，返回它的字串表示。
    """
    if hasattr(obj, '__dict__'):
        return obj.__dict__  # 將物件的屬性字典作為其 JSON 表示
    return str(obj)
