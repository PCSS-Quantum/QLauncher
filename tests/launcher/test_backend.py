import os

from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit_ibm_runtime import Session

from qiskit_aqt_provider import AQTProvider

from qiskit.primitives import BaseSamplerV2, BaseEstimatorV2

from quantum_launcher.routines.qiskit_routines import QiskitBackend, AQTBackend, IBMBackend, AerBackend

import pytest


class DummyAQTProvider(AQTProvider):
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


def test_AQT_backend_backendv1v2_simulator():
    with pytest.raises(ValueError):
        backend = AQTBackend(token="test_token", name='backendv1v2')

    backend = AQTBackend('backendv1v2', backendv1v2=FakeAlmadenV2())

    assert backend.name == 'backendv1v2'

    assert isinstance(backend.backendv1v2, FakeAlmadenV2)
    assert isinstance(backend.estimator, AQTEstimator)
    assert isinstance(backend.sampler, AQTSampler)


def test_AQT_backend_local_simulator():
    backend = AQTBackend(token="test_token", name='local_simulator')

    assert backend.name == 'offline_simulator_no_noise'
    assert isinstance(backend.backendv1v2, OfflineSimulatorResource)
    assert isinstance(backend.estimator, AQTEstimator)
    assert isinstance(backend.sampler, AQTSampler)


def test_AQT_backend_online_device():
    # Test if backend rejects invalid token for online backend
    with pytest.raises(ValueError):
        backend = AQTBackend(token="test_token", name='device')

    backend = AQTBackend(token="test_token", name='local_simulator')

    backend.provider = DummyAQTProvider()
    backend.name = 'device'
    backend._set_primitives_on_backend_name()

    assert backend.name == 'ibex_dummy'


def test_AQT_backend_loads_env(tmp_path):
    env_path = os.path.join(tmp_path, '.env')
    with open(env_path, 'w+') as f:
        f.write('AQT_TOKEN=test')

    backend = AQTBackend('local_simulator', dotenv_path=env_path)

    assert backend.provider.access_token == 'test'


def test_IBM_session():
    backend = FakeAlmadenV2()

    with Session(backend=backend) as session:
        ql_backend = IBMBackend('session', session=session)

        assert ql_backend.sampler.mode == session
        assert ql_backend.estimator.mode == session


def test_Qiskit_local_session():
    backend = QiskitBackend('local_simulator')

    assert backend.sampler is not None
    assert backend.estimator is not None
    assert backend.optimizer is not None


def test_Qiskit_backendv1v2_session():
    backend = QiskitBackend('backendv1v2', backendv1v2=FakeAlmadenV2())

    assert backend.sampler is not None
    assert backend.estimator is not None
    assert backend.optimizer is not None

    assert isinstance(backend.backendv1v2, FakeAlmadenV2)


def test_Aer_backend_local():
    backend = AerBackend('local_simulator')
    assert isinstance(backend.sampler, BaseSamplerV2)
    assert isinstance(backend.estimator, BaseEstimatorV2)


def test_Aer_backend_backendv1v2():
    backend = AerBackend('backendv1v2', backendv1v2=FakeAlmadenV2())
    assert isinstance(backend.sampler, BaseSamplerV2)
    assert isinstance(backend.estimator, BaseEstimatorV2)

    assert isinstance(backend.backendv1v2, FakeAlmadenV2)
