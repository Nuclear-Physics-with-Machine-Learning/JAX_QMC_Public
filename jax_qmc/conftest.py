import pytest
import time

from . config import Config, Sampler, ManyBodyCfg

from jax.config import config; config.update("jax_enable_x64", True)

def simple_fixture(name, params):
    @pytest.fixture(name=name, params=params)
    def inner(request):
        return request.param
    return inner

import pytest
pytest.simple_fixture = simple_fixture

n_walkers   = pytest.simple_fixture("n_walkers", params=(10,))
n_particles = pytest.simple_fixture("n_particles", params=(4,12,24))
n_dim       = pytest.simple_fixture("n_dim", params=(3,))
seed        = pytest.simple_fixture("seed", params=(0,))
# seed        = pytest.simple_fixture("seed", params=(0, time.time()))

@pytest.fixture
def n_spin_up(n_particles):
    # Need to make sure we have not-more than n_particles:
    return max(n_particles - 1, 1)

@pytest.fixture
def n_protons(n_particles):
    # Need to make sure we have not-more than n_particles:
    return max(n_particles - 2, 1)


@pytest.fixture
def sampler_config(n_walkers, n_particles, n_spin_up, n_protons, n_dim):
    s = Sampler()


    s.n_walkers     = n_walkers
    s.n_particles   = n_particles
    s.n_dim         = n_dim
    s.n_spin_up     = n_spin_up
    s.n_protons     = n_protons

    return s


@pytest.fixture
def benchmark_sampler_config():
    s = Sampler()


    s.n_walkers   = 1000
    s.n_particles = 12
    s.n_dim       = 3
    s.n_spin_up   = 4
    s.n_protons   = 6

    return s


from . config.wavefunction import SlaterCfg

antisymmetry        = pytest.simple_fixture("antisymmetry", params=(SlaterCfg(),)
antisymmetry_active = pytest.simple_fixture("antisymmetry_active", params=(True,False))

@pytest.fixture
def network_config(antisymmetry, antisymmetry_active):

    nc = ManyBodyCfg()
    nc.correlator_cfg.individual_cfg.layers = [32]
    nc.correlator_cfg.aggregate_cfg.layers  = [32,1]

    nc.antisymmetry = antisymmetry
    nc.active = antisymmetry_active

    return nc

@pytest.fixture
def benchmark_network_config():

    nc = ManyBodyCfg()

    return nc

@pytest.fixture
def benchmark_seed():
    return 12345


def test_gen_config(sampler_config):
    print(sampler_config)
    assert True
