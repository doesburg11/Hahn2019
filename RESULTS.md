# Validation and results

This log distinguishes execution checks from scientific reproduction. No full-budget reproduction or emergent-flocking claim has been established. An animation is not evidence that grouping was learned; aligned escape from a common predator is also insufficient evidence of active cohesion.

## Verified on 2026-10-20

All checks used this repository's own CPU Conda environment: Python 3.11.17, PyTorch 2.14.1+cpu, NumPy 2.4.6, Matplotlib 3.11.2, Pillow 13.0.0 and pytest 9.1.1.

- `python -m pip check`: no broken requirements.
- `python -m pytest -q`: **25 tests passed** in 2.72 seconds. The independent reviewer also executed these tests successfully.
- Both algorithms trained for 1,200 steps with training seeds 42 and 43, warmup 1,000, and the 100-frame smoke episode cap. All four runs completed 201 optimizer updates, changed actor parameters, saved finite inference checkpoints and completed policy broadcasts.
- Every checkpoint was evaluated with 40 prey, held-out environment seeds 100 and 101, two focal episodes per seed, a 300-frame horizon and 20-frame burn-in. Separate population rollouts ran 500 frames per seed, yielding 960 measured frames per policy after burn-in.
- No-confusion and pinned-predator evaluation modes completed for the seed-42 DQN checkpoint. These runs check the interfaces; they do not establish causality.
- The seed-42 DQN checkpoint rendered to an 80-frame, 600×600 GIF. Frame 40 was visually inspected. Its caption identifies the 1,200-step smoke budget; no learned-flocking claim is attached.

## Smoke outcomes, not paper reproduction

Each learned-policy row uses four focal-life episodes. Mean episode length is restricted by the 300-frame horizon; right-censored lives are explicitly counted and must not be interpreted as observed deaths. Population capture rates come from independent fixed-duration rollouts, not these focal episodes.

| Algorithm | Train seed | Updates | Episode broadcasts | Mean focal frames | Censored lives | Population captures / measured frames |
| --- | --- | --- | --- | --- | --- | --- |
| DQN | 42 | 201 | 14 | 201.50 | 2/4 | 61/960 = 0.06354 |
| DQN | 43 | 201 | 13 | 225.25 | 3/4 | 67/960 = 0.06979 |
| DDPG | 42 | 201 | 15 | 225.25 | 3/4 | 43/960 = 0.04479 |
| DDPG | 43 | 201 | 13 | 192.25 | 2/4 | 43/960 = 0.04479 |

The paired fixed-duration controls produced Random 57/960 captures per frame = 0.05938, TurnAway 2/960 = 0.00208, and our untuned Boids 2/960 = 0.00208. These controls are repeated using the same environment seeds, not independent replications for each trained model. The short learned policies had higher capture rates than TurnAway under this assumed ecology. This does not establish a converged algorithm ranking, a significant difference across training seeds, or a match to the paper. Learned flocking has not been demonstrated.

## Commands executed

From the repository root with `.conda/bin/python`:

```bash
python -m pip check
python -m pytest -q
```

For each algorithm in `dqn ddpg` and each training seed in `42 43`:

```bash
python train.py --algorithm ALGORITHM --steps 1200 --seed SEED --env-config configs/smoke.json --output output/ALGORITHM_smoke_sSEED
python evaluate.py --checkpoint output/ALGORITHM_smoke_sSEED/policy.pt --n-prey 40 --seeds 100 101 --episodes 2 --max-steps 300 --burnin 20 --population-frames 500 --output output/ALGORITHM_smoke_sSEED_eval
```

Diagnostics and preview:

```bash
python evaluate.py --checkpoint output/dqn_smoke_s42/policy.pt --n-prey 40 --seeds 100 --episodes 1 --max-steps 100 --burnin 20 --population-frames 200 --no-confusion --output output/dqn_no_confusion_check
python evaluate.py --checkpoint output/dqn_smoke_s42/policy.pt --n-prey 40 --seeds 100 --episodes 1 --max-steps 100 --burnin 20 --population-frames 200 --pinned-predator --output output/dqn_pinned_check
python render_rollout.py --checkpoint output/dqn_smoke_s42/policy.pt --frames 80 --n-prey 40 --output assets/dqn_smoke.gif
```

Raw run files remain locally in ignored `output/` directories. Repeating a command requires a new output directory. The numerical summary and annotated preview are tracked in Git; model/replay artifacts are not published.

## Independent Codex review

The initial CLI review identified a DDPG degree/radian mismatch and a potential bias from using focal-life exposure for population rates. We agreed and corrected both: policy units are explicitly recorded and converted, and fixed-duration population counts are pooled by measured exposure. Added regression tests cover conversion, burn-in denominators, life continuation, actual peer-broadcast timing, replay life boundaries, critic immutability during the actor update, and target-critic interpolation. Noise statistics were added and cluster weighting is documented.

The reviewer also flagged process-global PyTorch initialization/thread settings and unbounded equivalent angular actions in the critic. Sequential/process-isolated execution is documented; the unbounded linear action representation is retained to avoid silently adding a bound absent from the described method. Its extrapolation risk is documented in METHODS.md.

A follow-up CLI review of the corrections found **no remaining actionable correctness bugs**, and passed all 25 tests. The primary assessment agrees. These reviews do not validate unknown author settings or establish the paper's scientific findings.

## Open reproduction work

Run the 500,000-step budget across independent seeds, inspect held-out censoring and flock metrics, and vary the unreported ecology and DDPG noise/unit assumptions. Compare matched retraining with and without predator confusion, test pinned-predator grouping, and tune Boids separately. The paper's KDE-before-death analysis and exact figures are not reproduced. Unknown original speeds, confusion radius, commitment and leap timing prevent claiming an exact reproduction even at the reported training budget.
