import numpy as np
import pytest
from hahn2019.environment import EnvConfig, Swarm
from hahn2019.metrics import angle_deviation, dbscan, snapshot_metrics


def test_circular_alignment():
    assert angle_deviation(np.deg2rad([359, 1])) == pytest.approx(1)
    assert angle_deviation(np.array([0., np.pi])) is None


def test_periodic_cluster_and_noise():
    env = Swarm(EnvConfig(n_prey=4, neighbors=1))
    env.positions[:] = [[39, 0], [0, 0], [1, 0], [20, 20]]
    env.headings[:] = 0
    metrics = snapshot_metrics(env, eps=2.1, min_samples=3)
    assert metrics['clusters'] == 1
    assert metrics['noise_fraction'] == .25
    assert metrics['cluster_angle_deviation_deg'] == 0
    assert metrics['polarization'] == 1
    assert metrics['cluster_pair_distance_over_side'] == pytest.approx((1+2+1)/3/40)


def test_dbscan_noise_can_become_border_point():
    distance = np.abs(np.array([0, 1, 2, 3])[:, None]-np.array([0, 1, 2, 3])[None, :])
    assert dbscan(distance, eps=1.1, min_samples=3).tolist() == [0, 0, 0, 0]


def test_unequal_cluster_weighting_and_noise_statistics():
    env = Swarm(EnvConfig(n_prey=7, neighbors=1))
    env.positions[:] = [[0, 0], [1, 0], [0, 1], [20, 20], [21, 20], [10, 10], [30, 0]]
    env.headings[:] = np.deg2rad([0, 0, 0, 60, 120, 0, 90])
    metrics = snapshot_metrics(env, eps=2, min_samples=2)
    assert metrics['clusters'] == 2
    assert metrics['cluster_angle_deviation_deg'] == pytest.approx(12)
    assert metrics['cluster_pair_distance_over_side'] == pytest.approx((3+np.sqrt(2))/4/40)
    assert metrics['noise_angle_deviation_deg'] == pytest.approx(45)
    assert metrics['noise_pair_distance_over_side'] == pytest.approx(np.sqrt(500)/40)
