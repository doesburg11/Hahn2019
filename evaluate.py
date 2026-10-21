"""Evaluate frozen SELFish policies and explicit hand-coded controls."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch

from hahn2019.environment import EnvConfig, Swarm, TURNS
from hahn2019.learning import LearnConfig, Learner, actions_to_turns
from hahn2019.metrics import snapshot_metrics
from hahn2019.policies import baseline_turns
from train import positive_int, seed_int


def evaluate(env_config, policy, seeds, episodes, burnin=100, sample_every=10, actor=None, algorithm=None, action_units="degrees", wrap_ddpg_actions=False):
    rows = []
    for seed in seeds:
        for episode in range(episodes):
            # Reinitialize each episode independently so policy-dependent capture
            # RNG draws cannot change the next episode's initial conditions.
            env = Swarm(env_config, np.random.SeedSequence([seed, episode]))
            policy_rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 1]))
            caught_total, measured_frames, total_return = 0, 0, 0.
            samples = []
            done = False
            while not done:
                if env.steps >= burnin and (env.steps-burnin) % sample_every == 0:
                    samples.append(snapshot_metrics(env))
                if actor is None:
                    turns = baseline_turns(env, policy, policy_rng)
                else:
                    with torch.no_grad():
                        output = actor(torch.from_numpy(env.observe())).numpy()
                    turns = actions_to_turns(output.argmax(1) if algorithm == "dqn" else output[:, 0],
                                             LearnConfig(algorithm=algorithm, action_units=action_units, wrap_ddpg_actions=wrap_ddpg_actions))
                _, rewards, terminated, truncated, info = env.step(turns)
                total_return += float(rewards[0])
                if env.steps > burnin:
                    caught_total += int(info["caught"].sum())
                    measured_frames += 1
                done = terminated or truncated
            metrics = {}
            if samples:
                for key in samples[0]:
                    values = [s[key] for s in samples if s[key] is not None]
                    metrics[key] = float(np.mean(values)) if values else None
            rows.append({"policy": policy, "seed": seed, "episode": episode,
                         "episode_length": env.steps, "focal_return": total_return,
                         "focal_caught": terminated, "right_censored": truncated,
                         "captures_after_burnin": caught_total, "frames_after_burnin": measured_frames,
                         "captures_per_frame_after_burnin": caught_total/measured_frames if measured_frames else None,
                         "metrics": metrics})
    return rows


def population_capture(env_config, policy, seeds, frames, burnin=100, actor=None, algorithm=None, action_units="degrees", wrap_ddpg_actions=False):
    rows = []
    for seed in seeds:
        env = Swarm(env_config, np.random.SeedSequence([seed, 991]))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 992]))
        captures = 0
        for frame in range(frames):
            if actor is None:
                turns = baseline_turns(env, policy, rng)
            else:
                with torch.no_grad():
                    output = actor(torch.from_numpy(env.observe())).numpy()
                turns = actions_to_turns(output.argmax(1) if algorithm == "dqn" else output[:, 0],
                                         LearnConfig(algorithm=algorithm, action_units=action_units, wrap_ddpg_actions=wrap_ddpg_actions))
            _, _, _, _, info = env.step(turns, focal_episode=False)
            if frame >= burnin:
                captures += int(info["caught"].sum())
        exposure = max(frames-burnin, 0)
        rows.append({"policy": policy, "seed": seed, "captures": captures, "frames": exposure,
                     "captures_per_frame": captures/exposure if exposure else None})
    return rows


def pooled_population_rates(rows):
    result = {}
    for policy in sorted({r["policy"] for r in rows}):
        selected = [r for r in rows if r["policy"] == policy]
        captures, frames = sum(r["captures"] for r in selected), sum(r["frames"] for r in selected)
        result[policy] = {"captures": captures, "frames": frames,
                          "captures_per_frame": captures/frames if frames else None}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", nargs="+", type=seed_int, default=[100, 101, 102])
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--n-prey", type=positive_int)
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--sample-every", type=positive_int, default=10)
    parser.add_argument("--population-frames", type=positive_int, default=1000)
    parser.add_argument("--no-confusion", action="store_true")
    parser.add_argument("--pinned-predator", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.burnin < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("burnin must be nonnegative and seeds unique")
    torch.set_num_threads(1)
    actor, algorithm, source = None, None, None
    config = EnvConfig()
    if args.checkpoint:
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = EnvConfig(**source["metadata"]["environment"])
        lc = LearnConfig(**source["metadata"]["learning"])
        algorithm = lc.algorithm
        actor = Learner(3*(config.neighbors+2), lc).actor
        actor.load_state_dict(source["actor"])
        actor.eval()
    config = replace(config, **{k: v for k, v in {
        "n_prey": args.n_prey, "max_steps": args.max_steps,
        "confusion": False if args.no_confusion else None,
        "pinned_predator": True if args.pinned_predator else None,
    }.items() if v is not None})
    args.output.mkdir(parents=True, exist_ok=False)
    rows, population_rows = [], []
    action_units = lc.action_units if actor is not None else "degrees"
    wrap_ddpg_actions = lc.wrap_ddpg_actions if actor is not None else False
    for policy in ["random", "turnaway", "boids"] + ([algorithm] if actor is not None else []):
        rows.extend(evaluate(config, policy, args.seeds, args.episodes, args.burnin, args.sample_every,
                             actor if policy == algorithm else None, algorithm, action_units, wrap_ddpg_actions))
        population_rows.extend(population_capture(config, policy, args.seeds, args.population_frames, args.burnin,
                                                  actor if policy == algorithm else None, algorithm, action_units, wrap_ddpg_actions))
    result = {"environment": asdict(config), "burnin": args.burnin, "sample_every": args.sample_every,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "training_metadata": source["metadata"] if source else None, "episodes": rows,
              "population_frames": args.population_frames, "population_rollouts": population_rows,
              "pooled_population_rates": pooled_population_rates(population_rows)}
    (args.output/"evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved {len(rows)} policy/episode records to {args.output}")
    return result


if __name__ == "__main__":
    main()
