"""Toroidal prey-predator mechanics. Numeric assumptions are in EnvConfig."""
from dataclasses import dataclass
import numpy as np

TURNS = np.deg2rad([-90., -45., 0., 45., 90.])


def wrap_angle(angle):
    """Signed radians in (-pi, pi], matching the paper's bearing convention."""
    wrapped = (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi
    return np.where(wrapped == -np.pi, np.pi, wrapped)


def displacement(start, end, side):
    return (np.asarray(end) - np.asarray(start) + side / 2) % side - side / 2


def vectors(headings):
    return np.stack([np.cos(headings), np.sin(headings)], axis=-1)


@dataclass(frozen=True)
class EnvConfig:
    n_prey: int = 10
    neighbors: int = 5
    side: float = 40.
    radius: float = 1.
    speed: float = .3                 # Reconstruction assumption: paper gives no speed.
    target_radius: float = 10.        # Assumption: confusion neighborhood.
    target_hold: int = 10             # Assumption: frames before target reselection.
    burst_period: int = 100           # Assumption: periodic, rather than random leaps.
    burst_duration: int = 10
    burst_multiplier: float = 2.
    max_steps: int = 10_000
    confusion: bool = True
    pinned_predator: bool = False

    def __post_init__(self):
        integers = (self.n_prey, self.neighbors, self.target_hold, self.burst_period,
                    self.burst_duration, self.max_steps)
        if any(not isinstance(x, int) or isinstance(x, bool) for x in integers):
            raise ValueError("counts and frame intervals must be integers")
        if self.n_prey < 2 or not 1 <= self.neighbors < self.n_prey:
            raise ValueError("require 1 <= neighbors < n_prey")
        values = [self.side, self.radius, self.speed, self.target_radius, self.burst_multiplier]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("geometry and speed parameters must be positive and finite")
        if self.radius * 2 >= self.side / 2:
            raise ValueError("capture diameter must be below half the world side")
        if self.max_steps < 1 or self.target_hold < 1 or self.burst_period < 1:
            raise ValueError("frame limits must be positive")
        if not 0 <= self.burst_duration <= self.burst_period:
            raise ValueError("burst_duration must be within the burst period")


class Swarm:
    """One transition advances all prey simultaneously, then tests endpoint capture.

    Index zero is the focal learner. Capture is terminal for that life, even
    though its slot respawns. Time limits are separate truncations. No learning
    library dependency, walls, prey collisions, or cohesion reward is present.
    """
    def __init__(self, config=EnvConfig(), seed=0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c = self.config
        self.positions = self.rng.uniform(0, c.side, (c.n_prey, 2))
        self.headings = self.rng.uniform(0, 2*np.pi, c.n_prey)
        self.predator = self.rng.uniform(0, c.side, 2)
        self.predator_heading = float(self.rng.uniform(0, 2*np.pi))
        self.target = None
        self.target_age = 0
        self.steps = 0
        self.finished = False
        return self.observe()

    def observe(self):
        c = self.config
        delta = displacement(self.positions[:, None, :], self.positions[None, :, :], c.side)
        distances = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distances, np.inf)
        # Stable ties use slot index; a coincident other prey is still a neighbor.
        ids = np.argsort(distances, axis=1, kind="stable")[:, :c.neighbors]
        pred_delta = displacement(self.positions, self.predator, c.side)
        result = np.zeros((c.n_prey, c.neighbors+2, 3), dtype=np.float32)
        result[:, 0, 0] = np.linalg.norm(pred_delta, axis=1) / (c.side / np.sqrt(2))
        result[:, 0, 1] = (wrap_angle(np.arctan2(pred_delta[:, 1], pred_delta[:, 0])-self.headings)+np.pi)/(2*np.pi)
        result[:, 0, 2] = self.predator_heading/(2*np.pi)
        result[:, 1, 2] = self.headings/(2*np.pi)
        row = np.arange(c.n_prey)[:, None]
        nd = delta[row, ids]
        result[:, 2:, 0] = distances[row, ids] / (c.side / np.sqrt(2))
        result[:, 2:, 1] = (wrap_angle(np.arctan2(nd[..., 1], nd[..., 0])-self.headings[:, None])+np.pi)/(2*np.pi)
        result[:, 2:, 2] = self.headings[ids]/(2*np.pi)
        return result.reshape(c.n_prey, -1)

    def choose_target(self):
        c = self.config
        distance = np.linalg.norm(displacement(self.predator, self.positions, c.side), axis=1)
        candidates = np.flatnonzero(distance <= c.target_radius)
        if c.confusion and len(candidates) > 1:
            self.target = int(self.rng.choice(candidates))
        else:
            self.target = int(np.argmin(distance))
        self.target_age = 0

    def step(self, turns, *, focal_episode=True):
        if self.finished:
            raise RuntimeError("reset after focal capture or truncation")
        c = self.config
        turns = np.asarray(turns, dtype=float)
        if turns.shape != (c.n_prey,) or not np.isfinite(turns).all():
            raise ValueError("one finite turning angle in radians per prey is required")
        if self.target is None or self.target_age >= c.target_hold:
            self.choose_target()
        target_vector = displacement(self.predator, self.positions[self.target], c.side)
        desired = np.arctan2(target_vector[1], target_vector[0])
        pred_turn = np.clip(wrap_angle(desired-self.predator_heading), -np.pi/4, np.pi/4)
        if not c.pinned_predator:
            self.predator_heading = float((self.predator_heading+pred_turn) % (2*np.pi))
        self.headings = (self.headings+turns) % (2*np.pi)
        self.positions = (self.positions+c.speed*vectors(self.headings)) % c.side
        burst = self.steps % c.burst_period < c.burst_duration
        speed = c.speed * (c.burst_multiplier if burst else 1.)
        if not c.pinned_predator:
            self.predator = (self.predator+speed*vectors(self.predator_heading)) % c.side
        self.steps += 1
        self.target_age += 1
        caught = np.linalg.norm(displacement(self.predator, self.positions, c.side), axis=1) < 2*c.radius
        rewards = np.where(caught, -1000., 1.)
        terminated = bool(caught[0])
        truncated = focal_episode and self.steps >= c.max_steps and not terminated
        # The final observation belongs to the old life, never the respawn.
        terminal_observation = self.observe()[0].copy()
        dead = np.flatnonzero(caught)
        if len(dead):
            self.positions[dead] = self.rng.uniform(0, c.side, (len(dead), 2))
            self.headings[dead] = self.rng.uniform(0, 2*np.pi, len(dead))
            if self.target in dead:
                self.target = None
        self.finished = focal_episode and (terminated or truncated)
        return self.observe(), rewards, terminated, truncated, {
            "caught": caught.copy(), "terminal_observation": terminal_observation,
            "burst": bool(burst),
        }
