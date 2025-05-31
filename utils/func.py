import uuid


def unique_id():
    return f"seg_{uuid.uuid4().hex[:12]}"