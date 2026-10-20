from dataclasses import replace
import numpy as np
import pytest
from hahn2019.environment import EnvConfig, Swarm, displacement, wrap_angle


def test_periodic_geometry_and_angle():
    np.testing.assert_allclose(displacement([39, 2], [1, 2], 40), [2, 0])
    assert abs(wrap_angle(np.deg2rad(358))-np.deg2rad(-2)) < 1e-12


def test_observation_order_normalization_and_no_self_neighbor():
    env = Swarm(EnvConfig(n_prey=3, neighbors=2))
    env.positions[:] = [[39, 0], [1, 0], [30, 0]]
    env.headings[:] = [0, np.pi/2, np.pi]
    env.predator[:] = [39, 3]
    env.predator_heading = np.pi
    obs = env.observe().reshape(3, 4, 3)
    assert obs.shape == (3, 4, 3)
    np.testing.assert_allclose(obs[0, 0], [3/(40/np.sqrt(2)), .75, .5])
    np.testing.assert_allclose(obs[0, 2], [2/(40/np.sqrt(2)), .5, .25])
    assert np.isfinite(obs).all() and (obs >= 0).all() and (obs <= 1).all()
    env.positions[1] = env.positions[0]
    assert env.observe().reshape(3, 4, 3)[0, 2, 0] == 0


def fixed_world(**overrides):
    env = Swarm(EnvConfig(n_prey=3, neighbors=1, speed=.01, burst_duration=0, **overrides))
    env.positions[:] = [[20, 20], [5, 5], [30, 30]]
    env.headings[:] = 0
    env.predator[:] = [0, 0]
    env.predator_heading = 0
    return env


def test_capture_across_boundary_reward_and_respawn():
    env = fixed_world(pinned_predator=True)
    env.positions[0] = [39.5, 0]
    before = env.positions[0].copy()
    _, rewards, terminal, truncated, info = env.step(np.zeros(3))
    assert terminal and not truncated
    assert rewards.tolist() == [-1000, 1, 1]
    assert info['caught'].tolist() == [True, False, False]
    assert not np.array_equal(env.positions[0], before)
    assert info['terminal_observation'][0] < .1
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_nonfocal_capture_does_not_end_episode_and_timeout_is_not_death():
    env = fixed_world(pinned_predator=True, max_steps=2)
    env.positions[1] = [.5, 0]
    _, rewards, terminal, truncated, _ = env.step(np.zeros(3))
    assert rewards[1] == -1000 and not terminal and not truncated
    _, rewards, terminal, truncated, _ = env.step(np.zeros(3))
    assert not terminal and truncated and rewards[0] == 1


def test_target_confusion_commitment_and_no_confusion_control():
    env = fixed_world(target_radius=15, target_hold=3)
    env.positions[:] = [[3, 0], [6, 0], [10, 0]]
    targets = set()
    for _ in range(50):
        env.choose_target()
        targets.add(env.target)
    assert targets == {0, 1, 2}
    env.target = 2
    env.target_age = 0
    env.positions[:] += [0, 5]
    env.step(np.zeros(3))
    assert env.target == 2 and env.target_age == 1
    env.config = replace(env.config, confusion=False)
    env.choose_target()
    assert env.target == 0


def test_predator_turn_limit_and_burst_speed():
    c = EnvConfig(n_prey=3, neighbors=1, speed=.1, burst_period=4, burst_duration=1, burst_multiplier=3)
    env = Swarm(c)
    env.positions[:] = [[20, 20], [25, 25], [30, 30]]
    env.predator[:] = [1, 1]
    env.predator_heading = np.pi
    before = env.predator.copy()
    _, _, _, _, info = env.step(np.zeros(3))
    assert info['burst']
    assert np.linalg.norm(displacement(before, env.predator, c.side)) == pytest.approx(.3)
    assert abs(wrap_angle(env.predator_heading-np.pi)) <= np.pi/4+1e-12
    before = env.predator.copy()
    _, _, _, _, info = env.step(np.zeros(3))
    assert not info['burst']
    assert np.linalg.norm(displacement(before, env.predator, c.side)) == pytest.approx(.1)


def test_seeded_world_and_input_validation():
    a, b = Swarm(seed=42), Swarm(seed=42)
    np.testing.assert_array_equal(a.observe(), b.observe())
    for _ in range(3):
        np.testing.assert_array_equal(a.step(np.zeros(10))[0], b.step(np.zeros(10))[0])
    with pytest.raises(ValueError):
        a.step([np.nan]*10)
    with pytest.raises(ValueError):
        EnvConfig(neighbors=10)


def test_capture_at_time_limit_has_terminal_priority():
    env = fixed_world(pinned_predator=True, max_steps=1)
    env.positions[0] = [.5, 0]
    _, _, terminal, truncated, _ = env.step(np.zeros(3))
    assert terminal and not truncated


def test_continuous_population_keeps_running_after_focal_death():
    env = fixed_world(pinned_predator=True, max_steps=1)
    env.positions[0] = [.5, 0]
    _, _, terminal, truncated, _ = env.step(np.zeros(3), focal_episode=False)
    assert terminal and not truncated and not env.finished
    env.step(np.zeros(3), focal_episode=False)
    assert env.steps == 2
