import unittest
from unittest.mock import MagicMock, patch
import logging
from backend.core.models.exceptions import ModuleProcessingError, FrameworkException
from backend.utils.error_handling import handle_module_errors, require_ready, require_model

class TestErrorHandling(unittest.TestCase):

    def setUp(self):
        # 禁用日志输出以保持测试输出整洁
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_handle_module_errors_success(self):
        @handle_module_errors(operation_name="test_op")
        def success_func(self):
            return "success"

        result = success_func(None)
        self.assertEqual(result, "success")

    def test_handle_module_errors_exception(self):
        @handle_module_errors(operation_name="test_op")
        def failing_func(self):
            raise ValueError("Something went wrong")

        with self.assertRaises(ModuleProcessingError) as cm:
            failing_func(None)

        self.assertIn("Failed to perform test_op", str(cm.exception))
        self.assertIn("Something went wrong", str(cm.exception))

    def test_handle_module_errors_custom_exception(self):
        class CustomError(FrameworkException):
            pass

        @handle_module_errors(error_class=CustomError, operation_name="test_op")
        def failing_func(self):
            raise ValueError("Something went wrong")

        with self.assertRaises(CustomError):
            failing_func(None)

    def test_handle_module_errors_passthrough(self):
        # 测试如果已经是目标异常，应该直接传递而不被再次包装
        @handle_module_errors(operation_name="test_op")
        def passthrough_func(self):
            raise ModuleProcessingError("Already processed")

        with self.assertRaises(ModuleProcessingError) as cm:
            passthrough_func(None)

        self.assertEqual(str(cm.exception), "Already processed")

    def test_require_ready_success(self):
        class MockModule:
            def __init__(self):
                self.is_ready = True

            @require_ready
            def process(self):
                return "processed"

        module = MockModule()
        self.assertEqual(module.process(), "processed")

    def test_require_ready_failure(self):
        class MockModule:
            def __init__(self):
                self.is_ready = False

            @require_ready
            def process(self):
                return "processed"

        module = MockModule()
        with self.assertRaises(ModuleProcessingError) as cm:
            module.process()

        self.assertIn("is not ready", str(cm.exception))

    def test_require_ready_missing_attr(self):
        class MockModule:
            # 没有is_ready属性
            @require_ready
            def process(self):
                return "processed"

        module = MockModule()
        with self.assertRaises(ModuleProcessingError):
            module.process()

    def test_require_model_success(self):
        class MockModule:
            def __init__(self):
                self.model = "Some model"

            @require_model()
            def predict(self):
                return "prediction"

        module = MockModule()
        self.assertEqual(module.predict(), "prediction")

    def test_require_model_custom_attr_success(self):
        class MockModule:
            def __init__(self):
                self.engine = "Some engine"

            @require_model(model_attr="engine")
            def run(self):
                return "running"

        module = MockModule()
        self.assertEqual(module.run(), "running")

    def test_require_model_failure(self):
        class MockModule:
            def __init__(self):
                self.model = None

            @require_model()
            def predict(self):
                return "prediction"

        module = MockModule()
        with self.assertRaises(ModuleProcessingError) as cm:
            module.predict()

        self.assertIn("Model attribute 'model' is not initialized", str(cm.exception))

    def test_require_model_missing_attr(self):
        class MockModule:
            # 没有model属性
            @require_model()
            def predict(self):
                return "prediction"

        module = MockModule()
        with self.assertRaises(ModuleProcessingError):
            module.predict()

if __name__ == "__main__":
    unittest.main()
