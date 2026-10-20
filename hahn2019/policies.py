"""Hand-coded evaluation controls; never used as an RL reward."""
import numpy as np
from .environment import displacement, vectors, wrap_angle


def baseline_turns(env, name, rng):
    c = env.config
    away = -displacement(env.positions, env.predator, c.side)
    if name == "random":
        return rng.choice(np.deg2rad([-90, -45, 0, 45, 90]), c.n_prey)
    desired = away
    if name == "boids":
        delta = displacement(env.positions[:, None, :], env.positions[None, :, :], c.side)
        distances = np.linalg.norm(delta, axis=-1)
        mask = (distances < 8.) & ~np.eye(c.n_prey, dtype=bool)
        count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        cohesion = (delta*mask[..., None]).sum(axis=1)/count
        alignment = (vectors(env.headings)[None, :, :]*mask[..., None]).sum(axis=1)/count
        separation = -(delta*mask[..., None]/np.maximum(distances[..., None]**2, .01)).sum(axis=1)
        away_unit = away/np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-8)
        desired = .1*cohesion+alignment+separation+2*away_unit
    elif name != "turnaway":
        raise ValueError(f"unknown baseline: {name}")
    return np.clip(wrap_angle(np.arctan2(desired[:, 1], desired[:, 0])-env.headings), -np.pi/2, np.pi/2)
