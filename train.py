"""Train one focal prey; broadcast its policy only at episode boundaries."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import torch

from hahn2019.environment import EnvConfig, Swarm, TURNS
from hahn2019.learning import LearnConfig, Learner, actions_to_turns


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def seed_int(value):
    result = int(value)
    if not 0 <= result < 2**32:
        raise argparse.ArgumentTypeError("seed must be within [0, 2**32-1]")
    return result


def environment_config(path, algorithm):
    overrides = json.loads(Path(path).read_text()) if path else {}
    return EnvConfig(**{"neighbors": 5 if algorithm == "dqn" else 1, **overrides})


def train(output, env_config, learn_config, steps, seed, progress_every=5000):
    """Sequential CPU training; PyTorch initialization/thread settings are global."""
    if steps < 1 or progress_every < 1:
        raise ValueError("steps and progress_every must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = Swarm(env_config, seed)
    learner = Learner(3*(env_config.neighbors+2), learn_config, seed)
    peers = deepcopy(learner.actor)
    metadata = {
        "environment": asdict(env_config), "learning": asdict(learn_config),
        "seed": seed, "steps": steps,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "matplotlib")},
    }
    (output/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    observation = env.observe()
    episode_steps, episode_return, broadcasts = 0, 0., 0
    records, losses = [], []
    initial_actor = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    started = time.monotonic()
    for step in range(1, steps+1):
        actions = learner.actions(observation, network=peers)
        actions[0] = learner.actions(observation[:1], explore=True)[0]
        turns = actions_to_turns(actions, learn_config)
        nxt, reward, terminated, truncated, info = env.step(turns)
        replay_next = info["terminal_observation"] if terminated else nxt[0]
        learner.replay.add(observation[0], actions[0], reward[0], replay_next, terminated)
        try:
            loss = learner.update()
        except FloatingPointError as error:
            failure = {"step": step, "updates_completed": learner.updates, "error": str(error)}
            (output/"failure.json").write_text(json.dumps(failure, indent=2)+"\n")
            raise
        if loss:
            if not np.isfinite(list(loss.values())).all():
                raise RuntimeError("non-finite training loss")
            losses.append({"step": step, **loss})
        episode_steps += 1
        episode_return += float(reward[0])
        observation = nxt
        if terminated or truncated:
            records.append({"end_step": step, "length": episode_steps, "return": episode_return,
                            "terminated": terminated, "truncated": truncated})
            peers.load_state_dict(learner.actor.state_dict())
            broadcasts += 1
            learner.reset_noise()
            observation = env.reset()
            episode_steps, episode_return = 0, 0.
        if step % progress_every == 0 or step == steps:
            elapsed = time.monotonic()-started
            progress = {"step": step, "planned_steps": steps, "updates": learner.updates,
                        "broadcasts": broadcasts, "elapsed_seconds": elapsed,
                        "steps_per_second": step/elapsed, "last_loss": loss}
            temporary = output/"progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
            temporary.replace(output/"progress.json")
            checkpoint_meta = {**metadata, "steps": step, "planned_steps": steps}
            temporary_checkpoint = output/"checkpoint_latest.pt.tmp"
            torch.save({"actor": learner.actor.state_dict(), "metadata": checkpoint_meta}, temporary_checkpoint)
            temporary_checkpoint.replace(output/"checkpoint_latest.pt")
            print(f"step={step}/{steps} updates={learner.updates} speed={step/elapsed:.1f} steps/s", flush=True)
    nets = [learner.actor] + ([learner.critic] if learner.critic is not None else [])
    if not all(torch.isfinite(p).all() for net in nets for p in net.parameters()):
        raise RuntimeError("non-finite model parameters")
    torch.save({"actor": learner.actor.state_dict(), "metadata": metadata}, output/"policy.pt")
    summary = {
        "updates": learner.updates, "broadcasts": broadcasts, "completed_episodes": records,
        "partial_episode_length": episode_steps, "partial_episode_return": episode_return,
        "actor_changed": any(not torch.equal(v, initial_actor[k]) for k, v in learner.actor.state_dict().items()),
        "losses": losses,
    }
    (output/"training.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"{learn_config.algorithm}: {steps} steps, {learner.updates} updates, {broadcasts} episode broadcasts")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ddpg"), default="dqn")
    parser.add_argument("--steps", type=positive_int, default=5000)
    parser.add_argument("--seed", type=seed_int, default=42)
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--warmup", type=positive_int, default=1000)
    parser.add_argument("--target-interval", type=positive_int, default=1000)
    parser.add_argument("--tau", type=float, default=.005)
    parser.add_argument("--ddpg-action-units", choices=("degrees", "radians"), default="degrees")
    parser.add_argument("--wrap-ddpg-actions", action="store_true", help="optional periodic-action stabilization variant")
    parser.add_argument("--progress-every", type=positive_int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return train(args.output, environment_config(args.env_config, args.algorithm),
                 LearnConfig(algorithm=args.algorithm, warmup=args.warmup,
                             target_interval=args.target_interval, tau=args.tau,
                             action_units=args.ddpg_action_units, wrap_ddpg_actions=args.wrap_ddpg_actions),
                 args.steps, args.seed, args.progress_every)


if __name__ == "__main__":
    main()
