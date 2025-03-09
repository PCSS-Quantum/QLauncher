""" Module for import management across the library. """
from importlib.util import find_spec
# TODO: Consider moving this module to quantum_launcher.utils


def check_dependencies(*module_names: str) -> str | None:
    """ Checks whether given module exists, if not shows how it should be downloaded

    Args:
        *module_name (str): Names of module to be checked

    Returns:
        str | None: returns name of not installed module, and None if all are installed.

    Usage example
    ---
    ::
        from quantum_launcher.launcher.import_management import check_dependency, DependencyError
        NOT_INSTALLED_DEPENDENCY = check_dependency('dill', 'qcg.pilotjob')
        if NOT_INSTALLED_DEPENDENCY is not None:
            raise DependencyError(NOT_INSTALLED_DEPENDENCY, 'pilotjob')
        else:
            import dill
            from qcg.pilotjob.api.job import Jobs
            from qcg.pilotjob.api.manager import LocalManager, Manager
    """
    for module in module_names:
        if find_spec(module) is None:
            return module
    return None


class DependencyError(ImportError):
    """ Error connected with dependencies and wrong installation. """

    def __init__(self, module: str | None = None, install_hint: str = '') -> None:
        if module is None:
            message = f"Some modules are not installed. Install it with: pip install quantum_launcher[{install_hint}]"
        else:
            message = f"Module '{module}' is required but not installed. Install it with: pip install quantum_launcher[{install_hint}]"
        super().__init__(message, name='DependencyError')
