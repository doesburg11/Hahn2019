from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from hahn2019.learning import LearnConfig, Learner, canonical_action, optimizer_step, actions_to_turns
from hahn2019.environment import EnvConfig
from train import train


def test_periodic_action_equivalence_and_gradient():
    config = LearnConfig(algorithm='ddpg', wrap_ddpg_actions=True)
    x = torch.tensor([30., 390., -330., 180.], requires_grad=True)
    result = canonical_action(x, config)
    torch.testing.assert_close(result, torch.tensor([30., 30., 30., -180.]))
    result[:3].sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1., 1., 1., 0.]))
    np.testing.assert_allclose(actions_to_turns(np.array([390.]), config), [np.pi/6])
    rad = replace(config, action_units='radians')
    np.testing.assert_allclose(canonical_action(np.array([np.pi/6+2*np.pi]), rad), [np.pi/6])


def test_action_wrapping_opt_in_preserves_legacy_outputs():
    old = LearnConfig(algorithm='ddpg')
    new = replace(old, wrap_ddpg_actions=True)
    learner = Learner(9, old, seed=42)
    for p in learner.actor.parameters():
        p.data.zero_()
    learner.actor[-1].bias.data.fill_(390.)
    assert learner.actions(np.zeros((1,9)))[0] == 390
    learner.config = new
    assert learner.actions(np.zeros((1,9)))[0] == 30


def test_nonfinite_gradients_do_not_update_weights_or_optimizer():
    p = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.Adam([p])
    # Forward value is finite, derivative at zero is infinite.
    with pytest.raises(FloatingPointError, match='gradient'):
        optimizer_step(p.sqrt(), optimizer, [p])
    assert p.item() == 0. and not optimizer.state
    with pytest.raises(FloatingPointError, match='loss'):
        optimizer_step(p*float('inf'), optimizer, [p])
    assert p.item() == 0. and not optimizer.state


def test_wrapped_ddpg_all_critic_paths_and_progress_checkpoint(tmp_path):
    learner = Learner(9, LearnConfig(algorithm='ddpg', wrap_ddpg_actions=True, warmup=1), seed=42)
    for network in [learner.actor, learner.target_actor]:
        for p in network.parameters():
            p.data.zero_()
        network[-1].bias.data.fill_(390.)
    seen=[]
    hooks=[net.register_forward_pre_hook(lambda module,args: seen.append(args[0][:,-1].detach().clone()))
           for net in [learner.critic, learner.target_critic]]
    for _ in range(512):
        learner.replay.add(np.zeros(9), 750., 1., np.zeros(9), False)
    learner.update()
    assert len(seen) == 3
    for actions in seen:
        torch.testing.assert_close(actions, torch.full_like(actions, 30.))
    for hook in hooks:
        hook.remove()
    output=tmp_path/'run'
    train(output, EnvConfig(max_steps=20), LearnConfig(warmup=1), 80, 42, progress_every=30)
    progress=json.loads((output/'progress.json').read_text())
    checkpoint=torch.load(output/'checkpoint_latest.pt',weights_only=True)
    assert progress['step']==80 and progress['steps_per_second']>0
    assert checkpoint['metadata']['steps']==80 and checkpoint['metadata']['planned_steps']==80
    assert not (output/'checkpoint_latest.pt.tmp').exists()


def test_failure_keeps_last_checkpoint_actual_step_and_records_failure(tmp_path, monkeypatch):
    calls=0
    def failing_update(self):
        nonlocal calls
        calls += 1
        if calls == 41:
            raise FloatingPointError('injected non-finite loss')
        return None
    monkeypatch.setattr(Learner,'update',failing_update)
    output=tmp_path/'partial'
    with pytest.raises(FloatingPointError):
        train(output, EnvConfig(max_steps=20), LearnConfig(), 80, 42, progress_every=30)
    failure=json.loads((output/'failure.json').read_text())
    checkpoint=torch.load(output/'checkpoint_latest.pt',weights_only=True)
    assert failure['step']==41 and checkpoint['metadata']['steps']==30
    assert checkpoint['metadata']['planned_steps']==80
    assert not (output/'policy.pt').exists()


@pytest.mark.parametrize('units,half', [('degrees',180.),('radians',np.pi)])
def test_wrapped_training_inserts_canonical_replay_actions(tmp_path,monkeypatch,units,half):
    from hahn2019.learning import Replay
    init=Learner.__init__
    def initialize(self,*args,**kwargs):
        init(self,*args,**kwargs)
        for net in [self.actor,self.target_actor]:
            for p in net.parameters(): p.data.zero_()
            net[-1].bias.data.fill_(750.)
    monkeypatch.setattr(Learner,'__init__',initialize)
    add=Replay.add
    inserted=[]
    def spy(self,obs,action,reward,nxt,terminal):
        inserted.append(float(action))
        return add(self,obs,action,reward,nxt,terminal)
    monkeypatch.setattr(Replay,'add',spy)
    train(tmp_path/'replay',EnvConfig(neighbors=1),LearnConfig(algorithm='ddpg',wrap_ddpg_actions=True,action_units=units),3,42)
    assert len(inserted)==3 and all(-half<=action<half for action in inserted)


def test_actor_failure_restores_exact_critic_flags_without_advancing_targets(monkeypatch):
    import hahn2019.learning as module
    learner=Learner(9,LearnConfig(algorithm='ddpg',wrap_ddpg_actions=True,warmup=1),seed=42)
    next(learner.critic.parameters()).requires_grad_(False)
    flags=[p.requires_grad for p in learner.critic.parameters()]
    target={k:v.clone() for k,v in learner.target_critic.state_dict().items()}
    rng=np.random.default_rng(42)
    for _ in range(512): learner.replay.add(rng.random(9),.3,1.,rng.random(9),False)
    original=module.optimizer_step
    def fail_actor(loss,optimizer,parameters):
        if optimizer is learner.actor_optimizer:
            raise FloatingPointError('injected actor failure')
        return original(loss,optimizer,parameters)
    monkeypatch.setattr(module,'optimizer_step',fail_actor)
    with pytest.raises(FloatingPointError): learner.update()
    assert [p.requires_grad for p in learner.critic.parameters()]==flags
    assert learner.updates==0
    assert all(torch.equal(v,target[k]) for k,v in learner.target_critic.state_dict().items())


def test_old_checkpoint_without_wrap_flag_evaluates_and_renders(tmp_path,monkeypatch):
    from evaluate import main as evaluate_main
    from render_rollout import main as render_main
    output=tmp_path/'legacy'
    train(output,EnvConfig(max_steps=1),LearnConfig(algorithm='ddpg'),1,42)
    checkpoint=torch.load(output/'policy.pt',weights_only=True)
    checkpoint['metadata']['learning'].pop('wrap_ddpg_actions')
    legacy=tmp_path/'legacy.pt'
    torch.save(checkpoint,legacy)
    result=evaluate_main(['--checkpoint',str(legacy),'--seeds','100','--episodes','1',
                          '--max-steps','1','--burnin','0','--population-frames','1','--output',str(tmp_path/'eval')])
    assert len(result['episodes'])==4
    render_main(['--checkpoint',str(legacy),'--frames','1','--n-prey','10','--output',str(tmp_path/'legacy.gif')])
    assert (tmp_path/'legacy.gif').stat().st_size>0
