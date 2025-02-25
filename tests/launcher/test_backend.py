from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
from qiskit_aqt_provider import AQTProvider

from quantum_launcher.routines.qiskit_routines.backend import AQTBackend
import pytest


class DummyProvider(AQTProvider):
    def backends(self, name=None, *, backend_type=None, workspace=None):
        if backend_type == "device":
            offline_no_noise = super().backends(name=r".*no_noise", backend_type="offline_simulator", workspace=workspace)[0]
            offline_no_noise.name = "ibex_dummy"
            return [offline_no_noise]
        return super().backends(name, backend_type=backend_type, workspace=workspace)

    def get_backend(self, name=None, *, backend_type=None, workspace=None):
        offline_no_noise = super().backends(name=r".*no_noise", backend_type="offline_simulator", workspace=workspace)[0]
        offline_no_noise.name = name
        return offline_no_noise


def test_AQT_backend():
    backend = AQTBackend(token="test_token", name='local_simulator')

    assert backend.name == 'offline_simulator_no_noise'
    assert isinstance(backend.backendv1v2, OfflineSimulatorResource)
    assert isinstance(backend.estimator, AQTEstimator)
    assert isinstance(backend.sampler, AQTSampler)

    # Test if backend rejects invalid token for online backend
    with pytest.raises(ValueError):
        backend = AQTBackend(token="test_token", name='backendv1v2_simulator')

    backend = AQTBackend(token="test_token", name='local_simulator')

    backend.provider = DummyProvider()
    backend.name = 'backendv1v2_simulator'
    backend._set_primitives_on_backend_name()

    assert backend.name == 'ibex_dummy'
