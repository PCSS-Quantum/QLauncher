""" Module for import management across the library. """
# TODO: Consider moving this module to quantum_launcher.utils


class DependencyError(ImportError):
    """ Error connected with missing optional dependencies and wrong installation. """

    def __init__(self, e: ImportError, install_hint: str = '') -> None:
        message = f"""Module "{e.name}" is required but not installed. Install it with: pip install "quantum_launcher[{install_hint}]"."""
        super().__init__(message, name=e.name)
