# Implementation decisions and research provenance

Primary source: [Hahn et al. (2019), arXiv v1](https://arxiv.org/pdf/1905.04077), especially Environment, Action/Observation Space, Training, Table 1 and Simulations and Results. This is an executable interpretation, not evidence that unreported settings match the authors' experiment.

## Parameters transcribed from the methods

| Item | Implemented setting | Location in paper |
| --- | --- | --- |
| Training geometry | Square torus, side 40; 10 prey; radius 1; catch at distance strictly below 2 | Environment; Training |
| Motion and interaction | Constant prey speed; no prey-prey collisions; fixed predator controller | Environment |
| Reward | +1 for surviving a frame; −1000 replaces it on capture | Objective of an Agent |
| Focal episode | Capture of learning prey or 10,000 frames | Training |
| DQN turns | −90°, −45°, 0°, +45°, +90° | Action Space |
| Predator turn | At most 45° per frame | Action Space |
| Features | Predator, self, distance-ordered nearest neighbors; distance, relative bearing, absolute orientation | Observation Space |
| Policy sharing | One focal learner; other policies frozen during the episode and copied afterward | Training |
| DQN | 10 hidden layers × 16 ReLU units; linear 5-action head; replay 50,000; batch 64; epsilon 0.1; 5 neighbors | Table 1 |
| DDPG | 5 hidden layers; actor width 16, critic width 32; linear heads; replay 100,000; batch 512; 1 neighbor | Table 1 |
| Both learners | Gamma 0.999999; Adam learning rate 0.001; published training budget 500,000 steps | Table 1 |
| DDPG exploration | OU theta 0.15, mean 0, sigma 0.3 | Table 1 |

These are numerical parameter facts and independent implementations of algorithms, not copied source or reproduced paper prose. The paper reports use of Keras-RL; this repo uses PyTorch.

## Reconstruction choices: ecology

Values below are configurable through an environment JSON file. `configs/assumed_environment.json` is the complete numeric ecology assumed here; DQN/DDPG neighborhood defaults are selected separately.

| Unspecified detail | Choice here |
| --- | --- |
| Base movement speed | 0.3 world units per frame for both prey and predator |
| Confusion neighborhood | Radius 10; uniformly select one target when multiple prey are in range; nearest prey otherwise |
| Commitment duration | Keep a target for 10 frames; immediately invalidate its identity when that prey is captured |
| Predator leap | First 10 frames of each 100-frame period use twice normal speed |
| Target pursuit | Aim at the target's pre-move position, turn within the limit, then advance |
| Update order | All prey move simultaneously; predator moves from the pre-move target direction; endpoint capture checks; then immediate respawns |
| Multiple collisions | Every overlapping prey is caught, not only the targeted prey |
| Spawning | Uniform independent positions/headings, including possible initial overlap; respawns can overlap the predator until the next frame |
| Episode reset | Fresh random whole-world state after focal capture or timeout |
| Time discretization | One action per frame; no continuous collision sweep between endpoints |
| Bearing ties | Half-turns are represented as +180°; equal nearest-neighbor distances use slot order |
| Predator determinism | Scripted except for target selection and initial/respawn randomness |

The original numeric speed, confusion radius, commitment duration and leap parameters could change the qualitative outcome. They have not been inferred by fitting the published plots. A meaningful sensitivity study should vary them before claiming that this reconstruction tests the same ecological regime.

## Reconstruction choices: learning

- Flatten observation triples. Normalize toroidal distances by the maximum possible shortest distance, side/sqrt(2); map signed bearings into [0,1]; divide headings by 360°. Preserve literal zero in the self distance/bearing slots. The paper states normalization but does not specify this formula.
- DQN uses an ordinary maximum over target-network Q values, MSE and a hard target copy every 1,000 optimizer updates. Double DQN, dueling heads, reward clipping and gradient clipping are absent.
- DDPG concatenates the action at the critic input; target interpolation tau is 0.005. These architecture/update details are not specified by Table 1.
- Both use 1,000 transitions of warmup and one optimizer update per frame afterward once a full minibatch is available. `--warmup`, `--target-interval` and `--tau` expose these unreported settings. Replay contains only focal transitions, retained across episodes.
- Environment geometry uses radians. DDPG has a linear unrestricted output in degrees, converted to radians at the environment boundary. OU sigma is consequently interpreted as 0.3 degrees, with dt=1 and reset noise at episode boundaries. This follows the paper's stated action units. Its original Keras-RL interface is unavailable; `--ddpg-action-units radians` explicitly selects an alternative reconstruction for sensitivity analysis, recorded in checkpoint metadata. No tanh or artificial action limit is added; angles are periodic in the environment. The critic still receives the unwrapped action, so equivalent full-turn actions remain distinct critic inputs. This unbounded representation can destabilize extrapolation; it is retained as a disclosed limitation rather than silently imposing a bound not stated in the paper.
- Peer policies act deterministically without exploration, beginning from a copy of the initial learner. Only the focal learner explores. The paper does not specify peer exploration.
- Death masks bootstrap; a time-limit truncation bootstraps from the final state. Captured prey's respawn is not used as the terminal life's next state. This explicitly distinguishes life boundaries from collection horizons.
- A final checkpoint saves the current actor even if the step budget ends mid-episode. Evaluation applies that actor to all prey. It is not a saved copy of the last episode's frozen peers.

## Controls and measurements

TurnAway turns directly away from the toroidal predator displacement, with a 90° per-frame bound. Random samples the five DQN turns. Our Boids control uses an 8-unit neighborhood and fixed weights (cohesion 0.1, alignment 1, inverse-square separation 1, predator avoidance 2); these are **untuned assumptions**, not the authors' optimized Boids parameters.

Periodic pair distances feed an independent DBSCAN implementation (epsilon 4, minimum samples 3 including the point itself). The paper does not provide DBSCAN hyperparameters or full metric aggregation details. Report global and cluster-conditioned distances/alignment, noise fraction and polarization; cluster alignment is weighted by agent count, cluster distance by pair count. No-cluster quantities are null rather than fabricated zeroes. A circular mean with effectively zero resultant has undefined angular deviation and is also null. Cluster IDs are not tracked across frames.

Evaluation starts the focal prey's episode from independently seeded random states, ends at its capture or the horizon, and records right-censoring. Group capture counts exclude the first 100 frames by default; the denominator is exactly the remaining observed frames. An early focal death leaves the rate null when there is no post-burn-in exposure. Metrics sample every 10 frames by default and do not enter rewards. Policies see paired initial conditions across controls, but subsequent stochastic paths diverge as captures and target choices differ.

For stronger claims, compare independent training seeds, report the distribution of per-seed effects, inspect censoring, sweep ecology/DBSCAN settings, retrain with confusion disabled, and test the pinned-predator diagnostic. Continuous population capture measurements use separately seeded fixed-duration rollouts (1,000 frames by default via `--population-frames`) that respawn focal prey and continue after its death. Report both per-rollout counts/exposure and the pooled ratio sum(captures)/sum(frames), never an unweighted mean of episode ratios. Focal-life episode rates remain labeled diagnostics. Noise-point distance and angular statistics are also reported separately.

The density-before-death KDE plot and a tuned Boids search from the paper are not implemented. Full numerical replication remains open.


## Post-run numerical stabilization variant (2026-10-21)

The original unwrapped seed-42 DDPG experiment reproducibly overflowed after 65,235 environment steps in an instrumented rerun, with replay actions around 6.5e14 degrees. This supports the previously noted periodic-action/critic-extrapolation concern; it does not identify the authors' unreported implementation choices.

An optional `wrap_ddpg_actions` setting, default false, maps degree actions to [-180,180) and radian actions to [-pi,pi). It is applied after exploration, before replay consumption, to target-actor actions, to actor actions inside the critic loss used for policy optimization, and at evaluation/rendering boundaries. The derivative through modulo is one away from wrap cuts. It bounds the critic's action coordinate without replacing the linear head, but leaves discontinuities and unbounded raw actor parameters. Long-run stability and scientific equivalence are not assumed.

Finite loss/gradient guards operate per optimizer step and stop training on failure; they do not roll back a preceding successful critic step if the actor subsequently fails. Critic trainability flags are restored exactly after the actor step, including on failure. Progress/checkpoint files are atomically replaced; they support inspection and inference, not full training resume. Original training results and stabilized variants are kept separate.
