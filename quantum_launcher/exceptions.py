""" Module for different exceptions used by quantum_launcher. """


class DependencyError(ImportError):
    """ Error connected with missing optional dependencies and wrong installation. """

    def __init__(self, e: ImportError, install_hint: str = '', *, private: bool = False) -> None:
        if private:
            message = f"""Module "{e.name}" is required but not installed. """ + \
                """The module needs to be installed but it's private. """ + \
                """To get access to module contact the library developers."""
        else:
            message = f"""Module "{e.name}" is required but not installed. Install it with: pip install "quantum_launcher[{install_hint}]"."""
        super().__init__(message, name=e.name)
