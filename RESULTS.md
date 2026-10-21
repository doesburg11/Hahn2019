# Validation and results

This log distinguishes execution checks from scientific reproduction. Full-budget runs are evaluated below; an emergent-flocking reproduction has not been established. An animation is not evidence that grouping was learned; aligned escape from a common predator is also insufficient evidence of active cohesion.

## Wrapped-run completion and fidelity investigation — 2026-10-21

Both optional wrapped DDPG runs completed 500,000 steps and 499,001 updates, and their held-out evaluations completed. Seed 42 had mean capped focal survival 981 frames and 636/14,700 population captures; seed 43 had 409.7 frames and 1,024/14,700 captures. No focal evaluation life was censored in either condition. Their normalized pair distances (0.384 and 0.372) remained close to random (0.375). Stabilization prevented the observed overflow in these two runs but did not reproduce flocking. Raw records: [results/wrapped_ddpg_20261021](results/wrapped_ddpg_20261021).

The [fidelity audit](FIDELITY_AUDIT.md) found differences in historical initialization, target update rates, OU noise, TD-loss scale and time-limit handling; these are reference-library comparisons, not recovered author overrides. It also found that the thesis lists four DQN neighbors while the paper lists five. A controlled action-count check showed DQN seed 43 chose +90° on all 600,000 tested population decisions and exactly matched the fixed +90° control. DQN seed 42 outperformed all fixed-turn controls tested, but learned flocking remains unestablished.

No additional long training was launched by the investigation and the production learning defaults remain unchanged. The priority is a separately labeled historical-default compatibility experiment, starting with initialization, rather than more duration under the same assumptions.

## Full-budget runs and held-out evaluation — 2026-10-21

The original DQN seeds 42/43 and DDPG seed 43 completed 500,000 environment steps and 499,001 optimizer updates each. DDPG seed 42 failed. These runs used the 10,000-frame episode cap and the documented assumed ecology; they remain approximate reconstructions.

Evaluation used 40 prey, held-out seeds 100/101/102, three focal episodes per seed, the 10,000-frame horizon, 100-frame burn-in and metrics sampled every 50 frames. Population rates use independent 5,000-frame continuous rollouts per seed, with 14,700 total measured frames per condition. Controls are evaluated once and shared, not counted as repeated independent replications. Raw records are tracked in [results/paper_budget_20261021](results/paper_budget_20261021).

| Policy | Mean focal frames (capped) | Censored lives / 9 | Population captures / frames | Captures per frame |
| --- | --- | --- | --- | --- |
| random | 342.6 | 0/9 | 907/14700 | 0.06170 |
| turnaway | 8889.0 | 8/9 | 33/14700 | 0.00224 |
| boids | 7711.9 | 6/9 | 46/14700 | 0.00313 |
| dqn_s42 | 8889.0 | 8/9 | 24/14700 | 0.00163 |
| dqn_s43 | 439.3 | 0/9 | 984/14700 | 0.06694 |
| ddpg_s43 | 936.7 | 0/9 | 929/14700 | 0.06320 |

DQN seed 42 survived well under these settings, but the other learned runs did not replicate that result. Its apparent advantage in population capture rate over TurnAway is only a small observed count difference across three evaluation seeds, not an established statistically significant improvement. Eight of nine focal lives for both DQN seed 42 and TurnAway were censored at the horizon, so their equal capped mean is not a measured uncensored life expectancy.

The grouping diagnostics do not establish the paper's learned-flocking result. Descriptive means of per-episode post-burn-in metrics give DQN seed 42 polarization 0.339 and normalized mean pair distance 0.323, versus TurnAway 0.654/0.137 and random 0.150/0.375. DQN seed 43 and DDPG seed 43 pair distances (0.379 and 0.376) resemble random. These measures alone cannot distinguish active social attraction from shared predator avoidance, and episodes have different observation durations.

### DDPG failure diagnosis and stabilization

An instrumented rerun of the unchanged seed-42 DDPG method reproduced the overflow at **65,235 environment steps** (64,236 completed update calls). Maximum stored action magnitude was about **6.55e14 degrees**. Because full turns are physically equivalent but the unwrapped critic sees different large numeric inputs, this supports the action-representation/extrapolation diagnosis. Seed 43 completed, but one sampled batch from its final actor still produced turns around -165,000 to -237,000 degrees. Completion alone therefore does not demonstrate a well-conditioned controller.

An optional wrapped-angle variant was added, preserving the original default and checkpoints. A seed-42 preflight completed 30,000 steps and 29,001 updates with finite losses/parameters, ending at critic loss 89,399 and actor loss -4,270. This is evidence that the probe executes, not a guarantee of full-budget stability. Two new 500,000-step background runs (seeds 42/43) are planned/launched with `--wrap-ddpg-actions`; their results must remain separate from the original unwrapped method. No wrapped full-budget outcome is claimed yet.

New runs write progress, last complete inference snapshots and structured numerical failure information. The finite guards do not roll back an already successful critic step if a later actor step fails. They stop the process and preserve the earlier snapshot rather than continuing with a partially updated model.

### Review and regression checks

The independent Codex review confirmed canonicalization across behavior/replay/critic/evaluation/rendering, legacy default compatibility, pre-optimizer finite guards and actual-versus-planned checkpoint metadata. It found that critic trainability flags were restored unconditionally rather than to their prior values; this was corrected and tested. Suggested coverage for old checkpoints, canonical replay insertion, partial checkpoint metadata and actor-failure restoration was added. The primary assessment agrees with these findings. The final suite passes **34 tests**; the 30,000-step preflight also completed.

Reproduction commands (choose new output paths):

```bash
python evaluate.py --checkpoint output/paper_budget_20261020_235544/dqn_s42/policy.pt --n-prey 40 --seeds 100 101 102 --episodes 3 --max-steps 10000 --burnin 100 --sample-every 50 --population-frames 5000 --output output/dqn_s42_heldout_recheck
python train.py --algorithm ddpg --steps 500000 --seed 42 --wrap-ddpg-actions --env-config configs/assumed_environment.json --output output/ddpg_wrapped_s42
```

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

### Initialization-only experiment (2026-10-21)

Two new DQN runs use `--initialization glorot_zero_bias`, seeds 42 and 43,
500,000 environment steps each. This switches all dense weights to Xavier
uniform and biases to zero, matching the historical Keras Dense distributions
rather than its random stream. The existing `torch_default` remains the default.
The control checkpoints are `output/paper_budget_20261020_235544/dqn_s42` and
`dqn_s43`. Learning configuration and environment configuration are checked for
identity apart from initialization before launch. Each job automatically runs
40-prey evaluation with seeds 100/101/102, three capped episodes per seed and
5,000 continuous population frames per seed, matching the earlier controls.

This is a paired exploratory comparison with two training seeds, not a faithful
paper reproduction: ecology assumptions, target synchronization, loss and other
known implementation differences remain. No performance conclusion is available
at launch. Each timestamped output directory contains its commands, control paths,
source commit, progress, checkpoints and evaluation results.

Validation: 40 tests passed; a 1,200-step Glorot DQN smoke run completed 201
updates with finite parameters. Independent Codex review identified positional
configuration compatibility and missing DDPG/default-RNG, legacy-checkpoint and
CLI-metadata coverage; these were addressed and the full suite passed.
