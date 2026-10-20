from copy import deepcopy
import numpy as np
import pytest
import torch
from hahn2019.learning import LearnConfig, Learner, Replay, bellman_target
from train import train
from hahn2019.environment import EnvConfig
from evaluate import main as evaluate_main


def test_terminal_mask_and_timeout_bootstrap():
    actual = bellman_target(torch.tensor([[-1000.], [1.]]), torch.tensor([[1.], [0.]]),
                            torch.tensor([[9999.], [10.]]), .9)
    torch.testing.assert_close(actual, torch.tensor([[-1000.], [10.]]))


def test_replay_wrap_and_copy():
    replay = Replay(2, 1)
    observation = np.array([1.], np.float32)
    replay.add(observation, 0, 1, observation, False)
    observation[:] = 99
    assert replay.obs[0, 0] == 1
    replay.add([2], 1, 1, [3], False)
    replay.add([3], 2, -1000, [4], True)
    assert replay.size == 2 and set(replay.obs[:, 0]) == {2, 3}


@pytest.mark.parametrize('algorithm', ['dqn', 'ddpg'])
def test_real_updates_frozen_peers_and_target_sync(algorithm):
    learner = Learner(9, LearnConfig(algorithm=algorithm, warmup=1, target_interval=1, tau=1), seed=3)
    frozen = deepcopy(learner.actor)
    initial = deepcopy(learner.actor.state_dict())
    rng = np.random.default_rng(3)
    for _ in range(learner.batch_size):
        learner.replay.add(rng.random(9), 2 if algorithm == 'dqn' else .1,
                           1., rng.random(9), False)
    assert learner.update() is not None
    assert learner.updates == 1
    assert any(not torch.equal(v, initial[k]) for k, v in learner.actor.state_dict().items())
    assert all(torch.equal(v, initial[k]) for k, v in frozen.state_dict().items())
    assert all(torch.equal(v, learner.target_actor.state_dict()[k]) for k, v in learner.actor.state_dict().items())
    assert all(torch.isfinite(p).all() for p in learner.actor.parameters())


@pytest.mark.parametrize('algorithm,steps', [('dqn', 80), ('ddpg', 520)])
def test_seeded_training_checkpoint_and_evaluation(tmp_path, algorithm, steps):
    torch.set_num_threads(1)
    c = EnvConfig(neighbors=5 if algorithm == 'dqn' else 1, max_steps=20)
    lc = LearnConfig(algorithm=algorithm, warmup=1)
    a = train(tmp_path/'a', c, lc, steps, 42)
    b = train(tmp_path/'b', c, lc, steps, 42)
    assert a == b and a['updates'] > 0 and a['actor_changed']
    assert a['broadcasts'] > 0
    result = evaluate_main(['--checkpoint', str(tmp_path/'a/policy.pt'), '--seeds', '100',
                            '--episodes', '1', '--max-steps', '30', '--burnin', '0',
                            '--output', str(tmp_path/'eval')])
    assert len(result['episodes']) == 4
    assert all(row['episode_length'] <= 30 for row in result['episodes'])
    with pytest.raises(FileExistsError):
        train(tmp_path/'a', c, lc, 1, 42)


def test_ddpg_degree_conversion_and_ou_units():
    from hahn2019.learning import actions_to_turns
    degree = LearnConfig(algorithm='ddpg')
    np.testing.assert_allclose(actions_to_turns(np.array([90., .3]), degree), [np.pi/2, np.deg2rad(.3)])
    radian = LearnConfig(algorithm='ddpg', action_units='radians')
    np.testing.assert_array_equal(actions_to_turns(np.array([.3]), radian), [.3])
    learner = Learner(9, degree, seed=42)
    for p in learner.actor.parameters():
        p.data.zero_()
    learner.actions(np.zeros((1, 9)), explore=True)
    assert learner.ou_state != 0
    learner.reset_noise()
    assert learner.ou_state == 0


def test_training_broadcast_only_at_real_episode_boundaries(tmp_path, monkeypatch):
    import train as module
    snapshots = []
    original = Learner.actions
    def spy(self, observations, explore=False, network=None):
        if network is not None:
            snapshots.append({k: v.clone() for k, v in network.state_dict().items()})
        return original(self, observations, explore, network)
    monkeypatch.setattr(Learner, 'actions', spy)
    summary = module.train(tmp_path/'broadcast', EnvConfig(max_steps=20), LearnConfig(warmup=1), 120, 42)
    boundaries = {r['end_step'] for r in summary['completed_episodes']}
    changed_at = [i for i in range(1, len(snapshots))
                  if any(not torch.equal(v, snapshots[i-1][k]) for k, v in snapshots[i].items())]
    assert changed_at and all(i in boundaries for i in changed_at)
    assert summary['broadcasts'] == len(boundaries)


def test_train_replay_uses_old_life_and_death_mask(tmp_path, monkeypatch):
    import train as module
    from hahn2019.environment import Swarm
    config = EnvConfig(n_prey=3, neighbors=1, speed=.01, pinned_predator=True, max_steps=1)
    env = Swarm(config, seed=42)
    env.positions[:] = [[.5, 0], [10, 10], [20, 20]]
    env.headings[:] = 0
    env.predator[:] = 0
    monkeypatch.setattr(module, 'Swarm', lambda *args: env)
    original = Replay.add
    records = []
    def spy(self, obs, action, reward, nxt, terminal):
        records.append((reward, nxt.copy(), terminal))
        original(self, obs, action, reward, nxt, terminal)
    monkeypatch.setattr(Replay, 'add', spy)
    module.train(tmp_path/'life', config, LearnConfig(), 1, 42)
    reward, nxt, terminal = records[0]
    assert reward == -1000 and terminal
    assert nxt[0] < .1


def test_ddpg_critic_step_and_target_critic(tmp_path):
    learner = Learner(9, LearnConfig(algorithm='ddpg', warmup=1, tau=.25), seed=42)
    old_critic = deepcopy(learner.critic.state_dict())
    target_before = deepcopy(learner.target_critic.state_dict())
    after_critic_step = {}
    original = learner.critic_optimizer.step
    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        after_critic_step.update({k: v.clone() for k, v in learner.critic.state_dict().items()})
        return result
    learner.critic_optimizer.step = spy
    rng = np.random.default_rng(42)
    for _ in range(512):
        learner.replay.add(rng.random(9), .3, 1., rng.random(9), False)
    learner.update()
    assert any(not torch.equal(v, old_critic[k]) for k, v in learner.critic.state_dict().items())
    for k, v in learner.critic.state_dict().items():
        assert torch.equal(v, after_critic_step[k])  # Actor step did not mutate critic.
        torch.testing.assert_close(learner.target_critic.state_dict()[k], target_before[k]*.75+v*.25)
