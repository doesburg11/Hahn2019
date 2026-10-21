# Fidelity investigation — 2026-10-21

**Conclusion: matching the paper's network sizes and step budget did not make this a faithful Keras-RL reconstruction.** Several framework defaults differ, and the ecology still contains unverified choices. These findings identify controlled next experiments; they do not establish why the authors obtained flocking or justify claiming that a proposed correction will recover it. No new long training run was launched by this audit, and production learner defaults were not changed.

## Sources and confidence

- [Conference paper, arXiv v1](https://arxiv.org/pdf/1905.04077): Table 1 and methods, visually checked in the initial reconstruction.
- [Hahn dissertation, 2022](https://edoc.ub.uni-muenchen.de/30770/1/Hahn_Carsten.pdf): printed pp. 143–147 (PDF pages 159–163), plus the related foraging chapter. Table 6.1 was visually inspected.
- [Phan dissertation](https://thomyphan.github.io/files/PhD-Thesis-ThomyPhan.pdf): section 6.3, printed pp. 27–28. It reiterates the flocking/TurnAway result without filling the ecology gaps.
- Official Keras-RL Git history: revision [`e6efb0d`](https://github.com/keras-rl/keras-rl/tree/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3), dated 2019-03-25, the latest commit in that repository before the paper's 2019-05-10 submission. **This is a historical reference, not a recovered author version or proof that defaults were used.**
- Official Keras 2.2.4 source: [Dense](https://github.com/keras-team/keras/blob/2.2.4/keras/layers/core.py#L834) and [Adam](https://github.com/keras-team/keras/blob/2.2.4/keras/optimizers.py#L436). The paper does not identify its precise Keras version or all overrides.

The public dissertation record exposes a PDF, not an attached simulation code archive. Targeted searches did not locate the original code. Hahn describes sharing the environment with a later student; that does not imply a public release. No authors were contacted.

## Verified differences from historical library behavior

| Mechanism | Reconstruction so far | Historical reference | Confidence / implication |
| --- | --- | --- | --- |
| Dense initialization | PyTorch default weights and random biases | Keras Dense defaults to Glorot uniform and zero biases | Confirmed default difference; actual author overrides unknown. Especially relevant to a 10-hidden-layer DQN. |
| DQN target update | Hard copy every 1,000 optimizer updates | Every 10,000 agent steps by default | Confirmed default difference, with different clocks as well as intervals. |
| DDPG target update | tau 0.005 | tau 0.001 by default | Confirmed default difference; ours follows online weights five times faster per update. |
| OU process | dt 1; reset state zero | dt 0.01; reset drawn from N(0, sigma) | Confirmed default difference. Same theta/sigma alone does not specify the same exploration. |
| TD loss scale | Mean squared error | Default infinite-threshold Huber equals 0.5 times squared error | Confirmed scale difference for both algorithms. With Adam this is not simply a doubled learning rate. |
| Time-limit target | Bootstrap at a 10,000-frame truncation | Keras-RL fit forces done/terminal at the episode cap | Confirmed interface behavior difference, even though our handling follows modern time-limit semantics. |
| Episode-end update | One learner update per collected frame | Historical fit makes an extra forward/backward call with reward zero at episode end | Confirmed loop difference; exact effects depend on replay and author integration. |
| Double DQN | Disabled | Disabled in the pinned pre-submission revision | **No discrepancy found.** Modern defaults must not be substituted for this historical source. |

Source locations: [DQN constructor and update](https://github.com/keras-rl/keras-rl/blob/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3/rl/agents/dqn.py#L20), [DDPG constructor](https://github.com/keras-rl/keras-rl/blob/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3/rl/agents/ddpg.py#L25), [OU](https://github.com/keras-rl/keras-rl/blob/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3/rl/random.py#L43), [loss](https://github.com/keras-rl/keras-rl/blob/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3/rl/util.py#L59), [fit timeout / terminal callback](https://github.com/keras-rl/keras-rl/blob/e6efb0d8297ec38d704a3110b5d6ed74d09a05e3/rl/core.py#L188).

Adam is another unaligned detail: Keras 2.2.4 uses its backend epsilon and places it with the uncorrected second moment, while our PyTorch optimizer uses its own defaults. The official DDPG example also explicitly sets `clipnorm=1`, but that is an **example override, not the library default or evidence that Hahn used it**. Initializer, optimizer epsilon, clipping, and noise changes should not be silently combined and then attributed to one factor.

## Additional evidence and a source conflict

The conference paper specifies **five observed DQN neighbors**. Hahn's dissertation gives **four**, both in the training discussion on printed p. 146 and Table 6.1 on p. 147. DDPG remains at one in both. The correct response is to test four and five as separately labeled hypotheses, not to silently declare either source wrong.

The dissertation adds that hyperparameter configurations were tested over **1,000 episodes** when selecting the best settings. Our nine held-out episodes per checkpoint are much narrower and our two final checkpoints are not a replication of their model-selection search.

The related *foraging* chapter reports prey movement of 1.0 unit per step. That is **not** a recovered escape-flocking speed: it is a different domain and must not replace our 0.3 assumption as a fact. Neither dissertation resolves escape speed, confusion radius, target persistence or leap timing/magnitude.

## Bounded empirical diagnostics

The reproducible script is [audits/investigate_fidelity.py](audits/investigate_fidelity.py); raw output is in `results/fidelity_audit_20261021/diagnostics.json`.

### Initialization, without training

Twenty seeded reset worlds provide 200 observations; there are no transitions and this is **not a sample from the training-state distribution**. For each algorithm, ten initialization seeds compare the present PyTorch setup with Glorot-uniform weights plus zero biases. This is a comparison of initializer distributions implemented in PyTorch, not bitwise parity with Keras or a trained performance experiment.

| Algorithm | Median mean input-Jacobian norm, current | Glorot + zero biases | Ratio of medians |
| --- | --- | --- | --- |
| DQN | 0.0001163 | 0.06803 | 585.1 |
| DDPG actor | 0.005586 | 0.16596 | 29.7 |

The statistic is the mean, over reset observations, of the full output-by-input Jacobian Frobenius norm, then the median over seeds. DQN's median output standard deviation across reset states was about 9.11e-6 versus 0.00363 with the Keras-style initializer profile. This gives a concrete reason to prioritize initialization. It does not prove vanishing gradients caused the final behavior or that Glorot will produce flocking.

### Exploration process

A four-cell comparison separates dt (1 versus 0.01) from reset state (zero versus N(0,0.3)), using 4,000 simulated paths and common standard-normal innovations. The historical dt changes the innovation standard deviation from 0.3 to 0.03 and the autoregressive coefficient from 0.85 to 0.9985. This makes noise far more persistent; it is not accurate to describe the entire process as merely ten times smaller. The initial random reset also matters over finite episodes.

### Ecology, without retraining

A one-factor-at-a-time scan uses the **training population of 10 prey**, three seeds, 2,000 continuous frames per seed, and 100-frame burn-in. Each cell therefore has 5,700 measured frames. This tests fixed-policy ecological sensitivity only, not whether learned flocking can emerge. Boids weights remain untuned.

| Setting | TurnAway captures/frame | Untuned Boids | Random |
| --- | --- | --- | --- |
| Current ecology | 0.00175 | 0.00246 | 0.04298 |
| Speed 0.1 | 0.00070 | 0.00070 | 0.01351 |
| Speed 1.0 | 0.00684 | 0.00825 | 0.13123 |
| Confusion radius 20 | 0 | 0 | 0.01667 |
| No predator burst | 0 | 0 | 0.04298 |
| Burst multiplier 3 | 0.00474 | 0.00509 | 0.05053 |

Zero means zero events observed during this short exposure, not zero long-term hazard. Speed affects all policies strongly. Several plausible settings make captures too rare for a strong learning signal in this probe, but that requires training experiments to establish. Random target selection disabled still gives similar TurnAway/Boids rates here; no claim that confusion is irrelevant follows from this finite fixed-policy check.

### Neighbor sensitivity

On the same uniform reset observations, replacing the whole neighbor block while holding predator/self features fixed changes greedy DQN actions in 7.5% of states for seed 42 and 6.5% for seed 43. These substitutions may be physically inconsistent; this is an out-of-distribution sensitivity diagnostic, not a causal test of cohesion or proof that neighbors are ignored. Continuous policies show large physical angle changes under the same perturbation, which also does not imply useful coordination.

### Constant-turn controls and on-policy action counts

[Audits/probe_constant_turns.py](audits/probe_constant_turns.py) tests all five fixed DQN turns and the two trained DQN policies on 40 prey, the same three population seeds, 5,000 frames per seed, and 100-frame capture burn-in. This aligns with the previous population evaluation. All-frame action counts include burn-in and cover 600,000 individual prey decisions per learned condition.

DQN seed 43 selected **+90° for all 600,000 decisions**. Its captures by seed were 312, 316 and 356, exactly matching the fixed +90° control. Thus its poor result is constant circling on these held-out trajectories, not demonstrated neighbor coordination. This is stronger evidence than a reset-state permutation diagnostic.

DQN seed 42 used action counts [279, 380460, 9105, 210141, 15] for [-90°, -45°, 0°, +45°, +90°]. Its population capture rate was 24/14,700 = 0.00163. Every constant-turn control was much worse (rates from 0.06143 to 0.06864), so a fixed turn alone does not explain that run's survival. This confirms responsiveness on the tested trajectories; it does not establish that the response uses neighbors beneficially rather than predator features, or that flocking was reproduced.

Raw fixed-turn and learned-rollout counts are in `results/fidelity_audit_20261021/constant_turns.json`.

## Settings still not recovered

- Exact prey/predator movement speed and predator leap schedule, duration and gain.
- Confusion radius, target hold time, and retargeting conditions.
- Collision handling for simultaneous overlaps: our implementation captures all overlapping prey; a target-only collision rule would create different density incentives. The sources do not settle this edge case.
- Observation normalization formula, continuous action/noise units, and any action processor or clipping.
- Peer exploration and first-episode peer initialization; whether the entire world resets at focal death.
- Precise library versions, target-update overrides, optimizer clipping/epsilon, model selection, DBSCAN settings and aggregation.

## Prioritized next experiments

1. Build an explicitly labeled **historical-default compatibility profile**, without changing the current default or overwriting old results. Test Glorot/zero-bias initialization first for DQN, with other settings held fixed; then the target-update schedule. Keep four/five neighbors separate. An eventual combined profile should not be mistaken for proof of which factor mattered.
2. For DDPG, first match OU dt/reset and the 0.001 target rate under a documented profile. Treat gradient clipping, radian action scaling and wrapped angles as separate interventions. A stabilization result is not a replication result.
3. Establish that grouping changes capture risk under the chosen ecology before spending another large training budget: compare matched spatial arrangements and target-only versus all-overlap capture as explicit hypotheses, then sweep the missing physical parameters. Keep TurnAway and constant-turn controls.
4. Evaluate more independent training seeds and replicate model selection; include the pinned-predator diagnostic and stronger neighbor interventions before claiming social cohesion.

Do not simply increase the training duration under the current assumptions. The investigation supports checking initialization and historical semantics before another ecology-wide training sweep.

## Reproduce the bounded diagnostics

From the Hahn2019 root, using its own environment and new output files:

```bash
.conda/bin/python audits/investigate_fidelity.py --output output/fidelity_audit_recheck.json
.conda/bin/python audits/probe_constant_turns.py --output output/constant_turn_recheck.json
```

These do not train policies. Learned-policy probes require the original local checkpoints at the paths recorded in the scripts; those binary checkpoints are not redistributed in Git. The initialization/OU/ecology investigation reports missing checkpoints explicitly and can still run those other checks on a fresh checkout. The constant-turn script expects the two original DQN checkpoints and fails if unavailable. Raw outputs from the executed checks are included for readers without those local artifacts.

## Independent review

Codex CLI independently checked the diagnostic script. It confirmed the Jacobian calculation, units, neighbor-column selection, periodic exposure count, independent RNG objects and one-factor ecology overrides. It requested separate OU dt/reset contrasts, explicit reset-state sampling metadata, and a dynamically obtained source revision; these were added and the diagnostics rerun. Its staging-directory default-path concern was addressed with repository-root validation and installation under the actual `audits/` directory; the relative default is correct in that installed layout. The primary assessment agrees with the experimental-control/provenance findings; hard-coding a machine-specific path would be less portable. No production learner behavior changed in this audit.
