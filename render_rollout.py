"""Render a frozen policy or baseline; the caption records policy provenance."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from hahn2019.environment import EnvConfig, Swarm, TURNS, vectors
from hahn2019.learning import LearnConfig, Learner, actions_to_turns
from hahn2019.policies import baseline_turns
from train import positive_int, seed_int


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("random", "turnaway", "boids"), default="turnaway")
    parser.add_argument("--frames", type=positive_int, default=200)
    parser.add_argument("--n-prey", type=positive_int, default=40)
    parser.add_argument("--seed", type=seed_int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    actor, algorithm = None, None
    config = EnvConfig()
    label = f"{args.baseline} baseline"
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata = checkpoint["metadata"]
        config = EnvConfig(**metadata["environment"])
        lc = LearnConfig(**metadata["learning"])
        algorithm = lc.algorithm
        actor = Learner(3*(config.neighbors+2), lc).actor
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        label = f"{algorithm.upper()} — {metadata['steps']} training steps; seed {metadata['seed']}"
    config = replace(config, n_prey=args.n_prey, max_steps=max(config.max_steps, args.frames))
    env, rng = Swarm(config, args.seed), np.random.default_rng(args.seed+1)
    states = [(env.positions.copy(), env.headings.copy(), env.predator.copy(), env.predator_heading)]
    for _ in range(args.frames-1):
        if actor is None:
            turns = baseline_turns(env, args.baseline, rng)
        else:
            with torch.no_grad():
                output = actor(torch.from_numpy(env.observe())).numpy()
            turns = actions_to_turns(output.argmax(1) if algorithm == "dqn" else output[:, 0], lc)
        env.step(turns, focal_episode=False)
        states.append((env.positions.copy(), env.headings.copy(), env.predator.copy(), env.predator_heading))
        # Display is a continuous respawning population; focal death does not
        # reset the scene. Unlike evaluation, this visualization is not an episode.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=(0, config.side), ylim=(0, config.side), aspect="equal", xlabel="x (periodic)", ylabel="y (periodic)")
    prey = ax.quiver(states[0][0][:, 0], states[0][0][:, 1], *vectors(states[0][1]).T, color="#218c55", scale=30)
    predator = Circle(states[0][2], config.radius, color="#d34b35")
    ax.add_patch(predator)
    ax.set_title(f"Hahn2019 reconstruction\n{label}", fontsize=10)
    time_text = ax.text(.02, .98, "", transform=ax.transAxes, va="top")
    def draw(index):
        positions, headings, pred, _ = states[index]
        prey.set_offsets(positions)
        prey.set_UVC(*vectors(headings).T)
        predator.center = pred
        time_text.set_text(f"frame {index}; rollout seed {args.seed}")
        return prey, predator, time_text
    animation = FuncAnimation(fig, draw, frames=len(states), interval=80, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
