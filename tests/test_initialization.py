import math
import pytest
import torch
from torch import nn
from hahn2019.learning import Learner, LearnConfig, mlp


@pytest.mark.parametrize('algorithm', ['dqn', 'ddpg'])
def test_default_preserves_original_rng_and_parameters(algorithm):
    torch.manual_seed(42)
    continuous = algorithm == 'ddpg'
    expected = mlp(21, 1 if continuous else 5, 5 if continuous else 10, 16)
    critic = mlp(22, 1, 5, 32) if continuous else None
    expected_rng = torch.get_rng_state()
    actual = Learner(21, LearnConfig(algorithm=algorithm), seed=42)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for a, b in zip(actual.actor.parameters(), expected.parameters()):
        assert torch.equal(a, b)
    if continuous:
        for a, b in zip(actual.critic.parameters(), critic.parameters()):
            assert torch.equal(a, b)
    assert LearnConfig('ddpg', .99, .001, 1000, 1000, .005, False, 'radians').action_units == 'radians'
    assert LearnConfig(**{'algorithm': 'dqn'}).initialization == 'torch_default'


@pytest.mark.parametrize('algorithm', ['dqn', 'ddpg'])
def test_glorot_all_layers_targets_and_reproducibility(algorithm):
    cfg = LearnConfig(algorithm=algorithm, initialization='glorot_zero_bias')
    learner = Learner(9, cfg, seed=43)
    repeated = Learner(9, cfg, seed=43)
    for net, target, other in [(learner.actor, learner.target_actor, repeated.actor),
                                (learner.critic, learner.target_critic, repeated.critic)]:
        if net is None:
            continue
        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                bound = math.sqrt(6 / (layer.in_features + layer.out_features))
                assert torch.count_nonzero(layer.bias) == 0
                assert torch.max(torch.abs(layer.weight)) <= bound
                assert torch.count_nonzero(layer.weight) > 0
        for a, b, c in zip(net.parameters(), target.parameters(), other.parameters()):
            assert torch.equal(a, b)
            assert torch.equal(a, c)


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match='initialization'):
        LearnConfig(initialization='unknown')


def test_cli_records_initializer_and_checkpoint_weights(tmp_path):
    import json
    from train import main
    main(['--steps', '1', '--initialization', 'glorot_zero_bias', '--output', str(tmp_path/'run')])
    metadata = json.loads((tmp_path/'run/config.json').read_text())
    assert metadata['learning']['initialization'] == 'glorot_zero_bias'
    checkpoint = torch.load(tmp_path/'run/policy.pt', weights_only=True)
    assert checkpoint['metadata'] == metadata
    expected = Learner(21, LearnConfig(initialization='glorot_zero_bias'), seed=42)
    assert all(torch.equal(v, expected.actor.state_dict()[k]) for k, v in checkpoint['actor'].items())
