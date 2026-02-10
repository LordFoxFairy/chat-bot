import unittest
from unittest.mock import patch, MagicMock
import sys
from backend.utils.dependencies import DependencyManager, requires_dependencies
from backend.core.models.exceptions import ModuleInitializationError

class TestDependencies(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        DependencyManager._availability_cache = {}
        DependencyManager._registry = {}

    def test_is_available_standard_lib(self):
        # os should always be available
        self.assertTrue(DependencyManager.is_available("os"))

    def test_is_available_non_existent(self):
        # A module that definitely doesn't exist
        self.assertFalse(DependencyManager.is_available("non_existent_module_xyz_123"))

    def test_register_dependency(self):
        DependencyManager.register_dependency(
            "custom_dep",
            "os.path",
            "pip install os-path"
        )
        self.assertTrue(DependencyManager.is_available("custom_dep"))

    def test_require_dependency_success(self):
        # Should not raise
        DependencyManager.require_dependency("os")

    def test_require_dependency_failure(self):
        DependencyManager.register_dependency(
            "missing_lib",
            "missing_lib_import",
            "pip install missing-lib"
        )

        with self.assertRaises(ModuleInitializationError) as cm:
            DependencyManager.require_dependency("missing_lib")

        self.assertIn("Missing required dependency: missing_lib", str(cm.exception))
        self.assertIn("pip install missing-lib", str(cm.exception))

    def test_decorator_success(self):
        @requires_dependencies("os", "sys")
        class TestClass:
            pass

        # Should instantiate fine
        obj = TestClass()
        self.assertIsInstance(obj, TestClass)

    def test_decorator_failure(self):
        @requires_dependencies("non_existent_module_abc")
        class TestClass:
            pass

        with self.assertRaises(ModuleInitializationError):
            TestClass()

if __name__ == '__main__':
    unittest.main()
