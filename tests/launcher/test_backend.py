from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator

from quantum_launcher.routines.qiskit_routines.backend import AQTBackend
import pytest


def test_AQT_backend():
    backend = AQTBackend(token="test_token", name='local_simulator')

    assert backend.name == 'offline_simulator_no_noise'
    assert isinstance(backend.backendv1v2, OfflineSimulatorResource)
    assert isinstance(backend.estimator, AQTEstimator)
    assert isinstance(backend.sampler, AQTSampler)

    # Test if backend rejects invalid token for online backend
    with pytest.raises(ValueError):
        backend = AQTBackend(token="test_token", name='backendv1v2_simulator')
