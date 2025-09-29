# tests/test_mucus.py

from modules.mucus import MucusDynamics

def test_mucus_decreases_with_high_flow():
    m = MucusDynamics(M0=0.8, k_prod=0.0, k_clear=0.2)
    initial = m.M
    for _ in range(100):
        m.update(Q=5.0, dt=0.01)  # simulate strong airflow
    assert m.M < initial

def test_resistance_increases_with_mucus():
    m = MucusDynamics(M0=0.0, Rm0=1.0, alpha=2.0)
    r0 = m.resistance()
    m.M = 1.0
    r1 = m.resistance()
    assert r1 > r0
