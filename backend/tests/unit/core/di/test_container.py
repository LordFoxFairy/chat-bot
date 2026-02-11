import pytest
import threading
from concurrent.futures import ThreadPoolExecutor
from backend.core.di.container import Container, DependencyError

class TestContainer:
    """Dependency Injection Container Tests"""

    def test_register_and_resolve_by_name(self):
        """Test registering and resolving a dependency by string name"""
        container = Container()
        instance = {"value": 1}

        container.register("my_service", instance)
        resolved = container.resolve("my_service")

        assert resolved is instance
        assert resolved["value"] == 1

    def test_register_and_resolve_by_type(self):
        """Test registering and resolving a dependency by type"""
        container = Container()

        class MyService:
            pass

        instance = MyService()
        container.register(MyService, instance)
        resolved = container.resolve(MyService)

        assert resolved is instance
        assert isinstance(resolved, MyService)

    def test_register_factory(self):
        """Test registering a factory function that creates new instances"""
        container = Container()

        class MyService:
            pass

        # Factory creates a new instance each time
        def factory():
            return MyService()

        container.register_factory(MyService, factory)

        instance1 = container.resolve(MyService)
        instance2 = container.resolve(MyService)

        assert isinstance(instance1, MyService)
        assert isinstance(instance2, MyService)
        assert instance1 is not instance2  # Factory should produce distinct instances

    def test_resolve_not_found_raises(self):
        """Test that resolving a non-existent dependency raises DependencyError"""
        container = Container()

        with pytest.raises(DependencyError) as excinfo:
            container.resolve("non_existent")

        assert "Dependency not found" in str(excinfo.value)
        assert "non_existent" in str(excinfo.value)

    def test_get_with_default(self):
        """Test get() method returns registered instance or default value"""
        container = Container()
        container.register("existing", "value")

        # Case 1: Dependency exists
        assert container.get("existing") == "value"
        assert container.get("existing", default="default") == "value"

        # Case 2: Dependency does not exist, returns default
        assert container.get("non_existent") is None
        assert container.get("non_existent", default="default") == "default"

    def test_has_method(self):
        """Test has() checks for existence correctly"""
        container = Container()
        container.register("singleton", "value")
        container.register_factory("factory", lambda: "value")

        assert container.has("singleton") is True
        assert container.has("factory") is True
        assert container.has("non_existent") is False

    def test_clear(self):
        """Test clearing the container removes all registrations"""
        container = Container()
        container.register("s1", "v1")
        container.register_factory("f1", lambda: "v2")

        assert container.has("s1") is True
        assert container.has("f1") is True

        container.clear()

        assert container.has("s1") is False
        assert container.has("f1") is False
        assert container._instances == {}
        assert container._factories == {}

    def test_clone(self):
        """Test cloning creates a shallow copy with independent registry"""
        container = Container()
        container.register("shared", "value")

        clone = container.clone()
        assert clone.has("shared") is True
        assert clone.resolve("shared") == "value"

        # Verify independence: Adding to clone doesn't affect original
        clone.register("new", "new_value")
        assert clone.has("new") is True
        assert container.has("new") is False

        # Verify independence: Clearing original doesn't affect clone
        container.clear()
        assert container.has("shared") is False
        assert clone.has("shared") is True

    def test_thread_safety(self):
        """Test container thread safety under concurrent access"""
        container = Container()

        def worker(idx):
            # Concurrent registration
            key = f"service_{idx}"
            container.register(key, idx)
            # Concurrent resolution
            val = container.resolve(key)
            assert val == idx
            # Concurrent check
            assert container.has(key)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(worker, range(100)))

        # Verify all registrations succeeded
        for i in range(100):
            assert container.resolve(f"service_{i}") == i

    def test_overwrite_registration(self):
        """Test that new registrations overwrite old ones"""
        container = Container()

        # 1. Overwrite instance with instance
        container.register("key", "value1")
        assert container.resolve("key") == "value1"

        container.register("key", "value2")
        assert container.resolve("key") == "value2"

        # 2. Overwrite instance with factory
        updated_container = Container()
        updated_container.register("key", "value1")
        updated_container.register_factory("key", lambda: "factory_value")

        assert updated_container.resolve("key") == "factory_value"
        # Check internal state to ensure cleanup
        assert "key" not in updated_container._instances
        assert "key" in updated_container._factories

        # 3. Overwrite factory with instance
        updated_container.register("key", "new_instance")
        assert updated_container.resolve("key") == "new_instance"
        # Check internal state to ensure cleanup
        assert "key" in updated_container._instances
        assert "key" not in updated_container._factories
