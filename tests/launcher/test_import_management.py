import pytest
from quantum_launcher.utils import DependencyError


def test_import_error():
    """ Checks if base usage of dependency error (after import error) is raised properly """
    with pytest.raises(DependencyError):
        try:
            import definitely_not_a_library
        except ImportError as e:
            raise DependencyError(e, 'hint') from e


def test_import_message():
    """ Test whether error message contains information about the library """
    try:
        try:
            import definitely_not_a_library
        except ImportError as e:
            raise DependencyError(e, 'hint') from e
    except DependencyError as de:
        assert "definitely_not_a_library" in str(de.msg)
