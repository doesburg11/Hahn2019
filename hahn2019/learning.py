"""Independent PyTorch DQN/DDPG; not a port of Keras-RL or author code."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


def mlp(inputs, outputs, depth, width):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inputs, width), nn.ReLU()]
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


@dataclass(frozen=True)
class LearnConfig:
    algorithm: str = "dqn"
    gamma: float = .999999
    learning_rate: float = .001
    warmup: int = 1000                 # Unreported in paper; reconstruction choice.
    target_interval: int = 1000        # DQN hard-copy interval, unreported.
    tau: float = .005                  # DDPG target interpolation, unreported.
    action_units: str = "degrees"      # Paper describes DDPG turns as real-valued degrees.

    def __post_init__(self):
        if self.action_units not in ("degrees", "radians"):
            raise ValueError("action_units must be degrees or radians")
        if self.algorithm not in ("dqn", "ddpg"):
            raise ValueError("algorithm must be dqn or ddpg")
        if not np.isfinite([self.gamma, self.learning_rate, self.tau]).all():
            raise ValueError("learning parameters must be finite")
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0 or not 0 < self.tau <= 1:
            raise ValueError("invalid learning parameters")
        if self.warmup < 0 or self.target_interval < 1:
            raise ValueError("invalid warmup/target interval")


class Replay:
    def __init__(self, capacity, obs_dim, continuous=False):
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty((capacity, 1), np.float32 if continuous else np.int64)
        self.rewards = np.empty((capacity, 1), np.float32)
        self.terminated = np.empty((capacity, 1), np.float32)
        self.capacity, self.size, self.cursor = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.cursor
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, terminated
        self.cursor = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, rng):
        ids = rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.from_numpy(a[ids]) for a in
                     (self.obs, self.actions, self.rewards, self.next_obs, self.terminated))


def bellman_target(reward, terminated, next_value, gamma):
    return reward + gamma * (1-terminated) * next_value


class Learner:
    def __init__(self, obs_dim, config=LearnConfig(), seed=0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        continuous = config.algorithm == "ddpg"
        self.actor = mlp(obs_dim, 1 if continuous else 5, 5 if continuous else 10, 16)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic = mlp(obs_dim+1, 1, 5, 32) if continuous else None
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate) if continuous else None
        self.batch_size = 512 if continuous else 64
        self.replay = Replay(100_000 if continuous else 50_000, obs_dim, continuous)
        self.updates = 0
        self.ou_state = 0.

    def reset_noise(self):
        self.ou_state = 0.

    def actions(self, observations, explore=False, network=None):
        net = self.actor if network is None else network
        with torch.no_grad():
            output = net(torch.as_tensor(observations, dtype=torch.float32)).numpy()
        if self.config.algorithm == "dqn":
            actions = output.argmax(axis=-1)
            if explore:
                mask = self.rng.random(len(actions)) < .1
                actions[mask] = self.rng.integers(5, size=mask.sum())
            return actions
        actions = output[:, 0].copy()
        if explore:
            # OU process acts on the single learner, in action units; dt=1 assumption.
            if len(actions) != 1:
                raise ValueError("OU exploration is for the focal learner only")
            self.ou_state += .15 * (-self.ou_state) + .3 * self.rng.normal()
            actions += self.ou_state
        # Linear actor; any real turn is legal and periodic. No tanh/clipping.
        return actions

    def update(self):
        c = self.config
        if self.replay.size < max(self.batch_size, c.warmup):
            return None
        obs, actions, rewards, nxt, terminal = self.replay.sample(self.batch_size, self.rng)
        if c.algorithm == "dqn":
            with torch.no_grad():
                targets = bellman_target(rewards, terminal, self.target_actor(nxt).max(1, keepdim=True).values, c.gamma)
            loss = nn.functional.mse_loss(self.actor(obs).gather(1, actions.long()), targets)
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.updates += 1
            if self.updates % c.target_interval == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
            return {"critic_loss": float(loss.detach())}
        with torch.no_grad():
            next_q = self.target_critic(torch.cat([nxt, self.target_actor(nxt)], dim=1))
            targets = bellman_target(rewards, terminal, next_q, c.gamma)
        q = self.critic(torch.cat([obs, actions], dim=1))
        critic_loss = nn.functional.mse_loss(q, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        actor_loss = -self.critic(torch.cat([obs, self.actor(obs)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            for target, online in [(self.target_actor, self.actor), (self.target_critic, self.critic)]:
                for t, p in zip(target.parameters(), online.parameters()):
                    t.lerp_(p, c.tau)
        self.updates += 1
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach())}


def actions_to_turns(actions, config):
    """Convert policy/replay action units to the environment's radians."""
    from .environment import TURNS
    if config.algorithm == "dqn":
        return TURNS[actions]
    return np.deg2rad(actions) if config.action_units == "degrees" else np.asarray(actions)
