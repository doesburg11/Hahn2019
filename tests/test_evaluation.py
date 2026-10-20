import numpy as np
import pytest
import evaluate as module
from hahn2019.environment import EnvConfig


def test_population_exposure_and_focal_deaths_do_not_stop_rollout(monkeypatch):
    class FakeWorld:
        calls = 0
        def __init__(self, *args):
            pass
        def step(self, turns, *, focal_episode=True):
            assert not focal_episode
            FakeWorld.calls += 1
            return None, None, True, False, {'caught': np.array([FakeWorld.calls % 2 == 0, False])}
    monkeypatch.setattr(module, 'Swarm', FakeWorld)
    monkeypatch.setattr(module, 'baseline_turns', lambda *args: np.zeros(2))
    rows = module.population_capture(EnvConfig(), 'random', [100], frames=7, burnin=2)
    assert FakeWorld.calls == 7
    assert rows[0]['captures'] == 2 and rows[0]['frames'] == 5
    assert rows[0]['captures_per_frame'] == .4


def test_population_pooling_uses_exposure_not_mean_of_ratios():
    rows = [{'policy': 'a', 'captures': 1, 'frames': 1}, {'policy': 'a', 'captures': 1, 'frames': 9}]
    assert module.pooled_population_rates(rows)['a']['captures_per_frame'] == .2
    assert module.pooled_population_rates([{'policy': 'a', 'captures': 0, 'frames': 0}])['a']['captures_per_frame'] is None
