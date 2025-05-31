class FrameworkException(Exception):
    pass


class ModuleInitializationError(FrameworkException):
    pass


class ModuleProcessingError(FrameworkException):
    pass


class PipelineExecutionError(FrameworkException):
    pass


class ConfigurationError(FrameworkException):
    pass
