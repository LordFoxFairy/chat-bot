import pytest
import logging
from unittest.mock import MagicMock
from backend.utils.error_handling import (
    handle_module_errors,
    require_ready,
    require_model
)
from backend.core.models.exceptions import (
    ModuleProcessingError,
    FrameworkException
)

# Test handle_module_errors decorator

def test_handle_module_errors_success():
    """测试装饰器在函数成功执行时的情况"""

    class TestClass:
        @handle_module_errors()
        def success_method(self, val):
            return val * 2

    obj = TestClass()
    assert obj.success_method(5) == 10

def test_handle_module_errors_custom_exception():
    """测试装饰器捕获异常并转换为默认 ModuleProcessingError"""

    class TestClass:
        @handle_module_errors(operation_name="test_op")
        def failing_method(self):
            raise ValueError("Something went wrong")

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.failing_method()
    assert "Failed to perform test_op" in str(excinfo.value)
    assert "Something went wrong" in str(excinfo.value)

def test_handle_module_errors_preserve_target_exception():
    """测试如果已经是目标异常类型，则直接抛出"""

    class TestClass:
        @handle_module_errors()
        def failing_method(self):
            raise ModuleProcessingError("Already wrapped")

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.failing_method()
    assert "Already wrapped" in str(excinfo.value)
    # 确保没有再次包装
    assert "Failed to perform" not in str(excinfo.value)

def test_handle_module_errors_different_target_exception():
    """测试转换为自定义异常类型"""

    class CustomError(FrameworkException):
        pass

    class TestClass:
        @handle_module_errors(error_class=CustomError, operation_name="custom_op")
        def failing_method(self):
            raise IndexError("Index error")

    obj = TestClass()
    with pytest.raises(CustomError) as excinfo:
        obj.failing_method()
    assert "Failed to perform custom_op" in str(excinfo.value)
    assert "Index error" in str(excinfo.value)


# Test require_ready decorator

def test_require_ready_true():
    """测试模块已就绪的情况"""

    class TestClass:
        def __init__(self):
            self.is_ready = True

        @require_ready
        def action(self):
            return "success"

    obj = TestClass()
    assert obj.action() == "success"

def test_require_ready_false():
    """测试模块未就绪的情况"""

    class TestClass:
        def __init__(self):
            self.is_ready = False

        @require_ready
        def action(self):
            return "success"

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.action()
    assert "is not ready" in str(excinfo.value)

def test_require_ready_missing_attr():
    """测试缺少 is_ready 属性的情况 (默认为 False)"""

    class TestClass:
        @require_ready
        def action(self):
            return "success"

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.action()
    assert "is not ready" in str(excinfo.value)


# Test require_model decorator

def test_require_model_exists():
    """测试模型属性存在的情况"""

    class TestClass:
        def __init__(self):
            self.model = "my_model"

        @require_model()
        def predict(self):
            return "prediction"

    obj = TestClass()
    assert obj.predict() == "prediction"

def test_require_model_missing():
    """测试模型属性为 None 的情况"""

    class TestClass:
        def __init__(self):
            self.model = None

        @require_model()
        def predict(self):
            return "prediction"

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.predict()
    assert "Model attribute 'model' is not initialized" in str(excinfo.value)

def test_require_model_missing_attr():
    """测试模型属性根本不存在的情况"""

    class TestClass:
        @require_model()
        def predict(self):
            return "prediction"

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.predict()
    assert "Model attribute 'model' is not initialized" in str(excinfo.value)

def test_require_model_custom_attr():
    """测试自定义模型属性名"""

    class TestClass:
        def __init__(self):
            self.custom_model = None

        @require_model(model_attr="custom_model")
        def predict(self):
            return "prediction"

    obj = TestClass()
    with pytest.raises(ModuleProcessingError) as excinfo:
        obj.predict()
    assert "Model attribute 'custom_model' is not initialized" in str(excinfo.value)

def test_require_model_custom_attr_success():
    """测试自定义模型属性名存在的情况"""

    class TestClass:
        def __init__(self):
            self.custom_model = "ready"

        @require_model(model_attr="custom_model")
        def predict(self):
            return "prediction"

    obj = TestClass()
    assert obj.predict() == "prediction"
