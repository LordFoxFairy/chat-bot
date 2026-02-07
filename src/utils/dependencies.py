import importlib
import logging
import functools
from typing import Dict, Any, Optional

from src.core.models.exceptions import ModuleInitializationError

logger = logging.getLogger(__name__)

class DependencyManager:
    """
    Manage external dependencies availability and requirements.
    """

    # Registry of dependencies: name -> {import_path, install_hint}
    _registry: Dict[str, Dict[str, str]] = {}

    # Cache for availability check results
    _availability_cache: Dict[str, bool] = {}

    @classmethod
    def register_dependency(cls, name: str, import_path: str, install_hint: str) -> None:
        """
        Register a dependency with its import path and installation hint.

        Args:
            name: The unique name to identify the dependency
            import_path: The Python import path to check (e.g. 'numpy', 'scipy.stats')
            install_hint: Instruction on how to install the dependency
        """
        cls._registry[name] = {
            "import_path": import_path,
            "install_hint": install_hint
        }
        # Clear cache for this dependency in case it was checked before registration
        if name in cls._availability_cache:
            del cls._availability_cache[name]

    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a dependency is available (can be imported).

        Args:
            name: The name of the dependency

        Returns:
            True if the dependency is importable, False otherwise
        """
        if name in cls._availability_cache:
            return cls._availability_cache[name]

        # Determine import path
        if name in cls._registry:
            import_path = cls._registry[name]["import_path"]
        else:
            # If not explicitly registered, assume name is the import path
            import_path = name

        try:
            importlib.import_module(import_path)
            cls._availability_cache[name] = True
            return True
        except ImportError:
            cls._availability_cache[name] = False
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking dependency '{name}': {e}")
            cls._availability_cache[name] = False
            return False

    @classmethod
    def require_dependency(cls, name: str) -> None:
        """
        Require a dependency to be available, raising an error if not.

        Args:
            name: The name of the dependency

        Raises:
            ModuleInitializationError: If the dependency is not available
        """
        if not cls.is_available(name):
            info = cls._registry.get(name, {})
            hint = info.get("install_hint", f"Please install package '{name}'")
            raise ModuleInitializationError(
                f"Missing required dependency: {name}. {hint}"
            )


def requires_dependencies(*deps: str):
    """
    Class decorator that checks for dependencies upon class instantiation.

    Args:
        *deps: Variable list of dependency names to check.

    Example:
        @requires_dependencies("numpy", "torch")
        class MyModel:
            ...
    """
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            for dep in deps:
                DependencyManager.require_dependency(dep)
            if original_init:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator
