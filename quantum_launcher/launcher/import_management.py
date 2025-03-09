""" Module for import management across the library. """
from importlib.util import find_spec
# TODO: Consider moving this module to quantum_launcher.utils

CHECK_DEPENDENCIES: bool = True


def check_dependency(*module_names: str, install_hint: str = ""):
    """ Checks whether given module exists, if not shows how it should be downloaded

    Args:
        *module_name (str): Names of module to be checked
        install_hint (str, optional): Name of optional dependency that will install it. Defaults to "".

    Raises:
        DependencyError: Raised if some dependency is not found.

    Usage example
    ---
    ::

        if CHECK_DEPENDENCIES:
            check_dependency("qiskit", "ibm_qiskit_runtime", install_hint="qiskit"):
            import qiskit
            import ibm_qiskit_runtime
        else:
            raise DependencyError(None, "qiskit")
    """
    for module in module_names:
        if find_spec(module) is None:
            raise DependencyError(module, install_hint)


class DependencyError(ImportError):
    """ Error connected with dependencies and wrong installation. """

    def __init__(self, module: str | None = None, install_hint: str = '') -> None:
        if module is None:
            message = f"Some modules are not installed. Install it with: pip install quantum_launcher[{install_hint}]"
        else:
            message = f"Module '{module}' is required but not installed. Install it with: pip install quantum_launcher[{install_hint}]"
        super().__init__(message)
