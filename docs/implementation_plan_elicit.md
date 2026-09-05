# Implementation plan: Mechanistic interpretability for PINN optimization failures

## 1. Experimental objective

The experiment will test whether failures in physics-informed neural networks (PINNs) are preceded by reproducible changes in internal representations, whether sparse autoencoders (SAEs) can expose those changes as physically interpretable features, and whether causal manipulation of those features can prevent or rescue failure. The proposal motivates this design from known PINN pathologies—including conflicting gradients, collocation starvation, spectral bias, ill-conditioning, and cases where boundary conditions look satisfactory while the interior PDE solution is wrong—while noting that conventional loss, gradient, and NTK diagnostics often detect failure without explaining its mechanism [^1].

**Primary decision:** determine whether SAE-derived internal signals add validated, actionable information beyond output losses and gradient diagnostics.

**Success criterion:** a feature is considered mechanistically useful only if it (1) predicts a prespecified failure before the output-level failure is visible, (2) changes the predicted physical loss under targeted intervention, (3) shows specificity relative to unrelated-feature and random controls, and (4) improves held-out training outcomes in a closed-loop controller.

## 2. Falsifiable hypotheses and endpoints

### H1 — Representation
PINN hidden activations contain sparse, reproducible features associated with boundary mismatch, PDE residual structure, gradient conflict, collocation starvation, or high-frequency error. This follows the proposal's conjecture that optimization-dominant directions may correspond to sparse latent features [^1].

**Endpoint:** cross-run and cross-PDE feature reproducibility; reconstruction error; sparsity; association with physical quantities.

### H2 — Early warning
Validated feature trajectories change before a prespecified failure threshold is crossed.

**Endpoint:** lead time, AUROC/AUPRC, recall at a fixed false-alarm rate, and comparison with loss plateau, gradient norm, cosine conflict, and NTK/Hessian-derived alarms. Lead time is the primary diagnostic endpoint because the proposed value of the method is intervention before collapse [^1].

### H3 — Causality
Targeted manipulation of a feature produces the predicted selective change in the corresponding physical objective, whereas manipulating an unrelated feature or random direction does not.

**Endpoint:** change in target loss, change in non-target losses, effect specificity, and reproducibility across seeds. Correlation alone will not qualify as mechanistic evidence; the proposal explicitly requires intervention and counterfactual testing [^1].

### H4 — Prevention
A monitor-triggered controller reduces failure frequency and improves held-out physical accuracy and stability versus standard PINN training and output-only adaptive baselines.

**Endpoint:** failure rate, rescued-failure rate, final relative $$L^2$$ and $$L^\infty$$ error, conservation violation, convergence stability, training cost, and intervention count [^1].

## 3. Scope and staged experimental matrix

Use a deliberately small canonical suite first, then expand only after the pipeline passes quality gates.

### Stage A: pipeline qualification
1. **1D Poisson equation:** smooth solution, Dirichlet boundaries; verifies derivatives, logging, SAE training, and intervention plumbing.
2. **1D advection or advection–diffusion:** introduces transport and spectral sensitivity.
3. **1D reaction–diffusion:** introduces competing spatial/ reaction scales.

### Stage B: pathology generation
For each PDE, generate controlled failures by varying one factor at a time:

- loss weights $$\lambda_{\mathrm{PDE}}/\lambda_{\mathrm{BC}}$$;
- collocation density and spatial sampling distribution;
- network width/depth and activation function;
- optimizer and learning rate;
- stiffness, frequency, or reaction parameters;
- boundary-condition sampling fraction;
- random seed.

The purpose is not to claim that every poor run is the same failure. Each run receives an operational label from measurable criteria, so the model learns distinct failure families rather than a generic “bad run.” The proposal identifies gradient opposition, spectral collapse, and collocation starvation as separable pathologies [^1].

### Stage C: generalization
Hold out at least one PDE parameter regime, one random-seed block, and—after the method is stable—one PDE family. No SAE dictionary, alarm threshold, or controller threshold may be tuned on the final test runs.

## 4. Reproducible PINN implementation

### 4.1 Model
Use a fully connected PINN with a fixed baseline architecture for the first benchmark, for example 4–6 hidden layers and 64–128 units per layer. Record the exact architecture, parameter count, activation, initialization, optimizer, learning-rate schedule, precision, hardware, and software commit. Add Fourier features only as a later controlled factor, not in the baseline, because the controller may ultimately use them to address high-frequency suppression [^1].

For an input $$x$$ and network prediction $$u_\theta(x)$$, define:

$$\mathcal{L}(\theta)=\lambda_f\mathcal{L}_f+\lambda_b\mathcal{L}_b+\lambda_i\mathcal{L}_i,$$

where $$\mathcal{L}_f$$ is the interior PDE residual loss and $$\mathcal{L}_b,\mathcal{L}_i$$ are boundary and initial/data terms as applicable. Keep each component separately available for backpropagation.

### 4.2 Sampling
Use fixed validation grids for evaluation and independent training collocation points. Log the coordinates of every training point and the sampling distribution at every resampling event. For adaptive-sampling experiments, preserve a frozen evaluation grid so apparent improvement cannot arise from changing the test distribution.

### 4.3 Run manifest
Each run must have a machine-readable manifest containing:

- PDE name and parameter values;
- domain and boundary/initial conditions;
- architecture and initialization seed;
- optimizer and schedule;
- loss weights;
- collocation and boundary sample counts;
- resampling policy;
- logging interval;
- software and hardware versions;
- run-level outcome label.

Use at least 5 seeds during development and 10–20 seeds for final comparisons, with the exact number fixed before the final analysis. Report seed-level results rather than only pooled means.

## 5. Instrumentation and data schema

Log the following at every training checkpoint, preferably every optimizer step for short pilot runs and every 10–100 steps for the full suite:

1. total loss and each component loss;
2. validation-grid relative $$L^2$$ and $$L^\infty$$ solution error;
3. pointwise PDE residual, boundary residual, and error maps;
4. per-component parameter gradients $$g_i=\nabla_\theta\mathcal{L}_i$$;
5. gradient norms $$\|g_i\|_2$$ and pairwise cosine similarities;
6. selected hidden-layer activations $$A_l(x)$$ at fixed probe points;
7. activation summaries: mean, variance, quantiles, sparsity, and spatial maps;
8. Fourier spectra of prediction error and selected activation channels;
9. collocation-point occupancy and residual distribution by spatial bin;
10. optional NTK/Hessian summaries on a reduced probe set.

The proposal specifically calls for gradient norms, gradient cosine similarities, internal activations, solution error, and Fourier analysis of solution error and activations [^1].

Store tensors in chunked arrays with metadata rather than one monolithic file. Use separate immutable raw logs and derived-analysis tables. Hash each run manifest and code commit so SAE training can be reproduced exactly.

## 6. Operational failure labels

Define labels before inspecting SAE results.

### Gradient-conflict failure
At least one pair of objective gradients has persistently negative cosine similarity over a prespecified window, accompanied by stalled or worsening validation error. Record both the instantaneous value and moving average; avoid labeling a run from one noisy step.

### Boundary-starvation failure
Boundary loss falls below its threshold while interior validation error or PDE residual remains above its threshold, or boundary performance improves while the interior error worsens. This operationalizes the proposal's example of apparently satisfactory boundary behavior with incorrect interior physics [^1].

### Spectral-suppression failure
High-frequency bands of the solution error remain elevated or increase while low-frequency error decreases. Define the band boundary from the known analytic/reference solution spectrum and lock it before testing.

### Collocation-starvation failure
A spatial region has persistently low point coverage relative to the prespecified minimum while residual/error remains high there. Use the same spatial bins for labeling and evaluation.

A run is “successful” only when all required physical criteria pass on the frozen validation grid, not merely when total training loss decreases.

## 7. SAE training pipeline

### 7.1 Activation dataset construction
Select one early, one middle, and one late hidden layer during pilot work; retain the middle layer for the primary analysis unless a preregistered criterion favors another layer. At fixed probe coordinates, collect activation vectors across training time, seeds, PDE parameters, and run outcomes. The proposal recommends treating logged activations over time as the SAE dataset and flattening or aggregating them across points [^1].

Split the activation dataset by run, not by individual activation vector, to prevent temporal leakage. Training/validation/test runs must be disjoint.

### 7.2 SAE architecture
For activation vector $$a\in\mathbb{R}^d$$, use:

$$z=\phi(W_e(a-b_a)+b_z), \qquad \hat a=W_dz+b_d,$$

with nonnegative or ReLU latent activations and an overcomplete latent width of 2–8 times $$d$$ in the pilot. Optimize:

$$\mathcal{L}_{SAE}=\|a-\hat a\|_2^2+\beta\|z\|_1,$$

with optional dead-feature regularization and decoder-column normalization. The proposal specifies an overcomplete sparse latent code, reconstruction, and L1 or KL sparsity enforcement [^1].

Sweep a small preregistered grid of latent expansion and sparsity strength. Select a Pareto set rather than one arbitrary SAE: acceptable models must have low held-out reconstruction error, non-degenerate feature usage, and stable feature interpretations.

### 7.3 SAE quality gates
Reject an SAE if any of the following occurs:

- reconstruction is not materially better than a low-rank or PCA baseline;
- most latent units are dead or one unit explains nearly all activations;
- feature ranking changes completely across bootstrap samples;
- the decoder is not numerically stable under intervention;
- feature associations are driven by run identity, time, or PDE identity rather than physical variables.

## 8. Feature interpretation and the Physics-Feature Dictionary

For every latent feature, compute:

- activation frequency and magnitude;
- spatial localization and boundary distance profile;
- temporal trajectory;
- association with each loss component, gradient norm, gradient cosine, error band, residual map, and sampling-density map;
- mutual information or rank correlation on held-out runs;
- cross-seed and cross-PDE reproducibility;
- top-activating examples and bottom-activating examples.

Use a preregistered feature annotation template:

- feature ID;
- layer and SAE version;
- candidate physical interpretation;
- supporting spatial evidence;
- supporting temporal evidence;
- candidate failure mode;
- predicted intervention direction;
- confounds tested;
- causal-test result;
- confidence tier.

The interpretation step should be treated as hypothesis generation, not proof. The proposal explicitly distinguishes a correlation such as a feature spike accompanying boundary-loss behavior from causal evidence and requires intervention to test the hypothesis [^1].

## 9. Causal intervention experiments

### 9.1 Intervention mechanism
Insert the SAE encoder/decoder at the selected hidden layer during a cloned forward pass. For latent coordinate $$z_k$$, compare:

1. **Natural:** original $$z_k$$;
2. **Ablation:** $$z_k\leftarrow0$$;
3. **Amplification:** $$z_k\leftarrow\alpha z_k$$, with $$\alpha$$ fixed in advance, such as 1.5 and 2.0;
4. **Sign or value replacement:** replace with a matched quantile from a control run, if numerically meaningful;
5. **Unrelated-feature control:** manipulate a feature matched for baseline magnitude but not associated with the target physical variable;
6. **Random-direction control:** manipulate a random latent direction with matched norm;
7. **Decoder reconstruction control:** run the SAE without intervention to quantify insertion/reconstruction effects.

The proposed intervention classes are suppression, amplification, and unrelated-feature control [^1].

### 9.2 Two causal regimes
Run both:

- **Inference-time counterfactuals:** freeze network weights and apply interventions to quantify immediate output and loss effects;
- **Training-time interventions:** apply the intervention for a fixed window, then continue training to determine whether it rescues or worsens the trajectory.

Do not call a feature causal from inference-time effects alone if the actual claim concerns prevention during optimization.

### 9.3 Causal score
For target loss $$L_T$$ and non-target loss vector $$L_{-T}$$, define:

$$E_T=\Delta L_T^{target}-\Delta L_T^{control},$$

$$S=\frac{|\Delta L_T|}{\epsilon+\sum_{j\ne T}|\Delta L_j|},$$

where the control is the matched unrelated-feature or random intervention and $$\epsilon$$ avoids division by zero. A verified feature must show the predicted direction in a majority of held-out seeds, a confidence interval excluding zero for the target effect, and greater target specificity than controls. The proposal recommends quantifying both causal strength and specificity [^1].

## 10. Early-warning monitor

Create prediction examples from run trajectories using only information available before the prediction time. The target is failure within a fixed horizon, for example the next 5–10% of training steps.

Compare three monitors:

1. loss-only baseline;
2. conventional-physics baseline using gradient norms, cosines, residual statistics, and sampling coverage;
3. SAE monitor using validated feature trajectories plus the conventional variables.

Start with an interpretable threshold rule or logistic regression. Use an LSTM only if the simpler models fail after proper temporal feature engineering. Tune thresholds on training runs only. Evaluate on held-out runs with AUROC, AUPRC, calibration, false-alarm rate, detection recall, and median lead time. The proposal specifically calls for comparison with conventional alarms and lead-time measurement [^1].

Prevent leakage: do not include future losses, future labels, post-failure activations, or variables computed using the full trajectory.

## 11. Closed-loop controller

The controller should initially implement one corrective action per failure class, with a cooldown period and a maximum intervention budget.

| Detected pathology | Corrective action | Trigger logic | Safety constraint |
|---|---|---|---|
| Boundary feature collapse | Increase $$\lambda_b$$ or apply gradient balancing | Validated boundary-warning signal for two consecutive checks | Cap weight change per event; restore if validation error worsens |
| Collocation starvation | Add/resample points in high-residual or under-covered bins | Joint feature warning plus spatial coverage deficit | Keep a fixed evaluation grid and cap total points |
| High-frequency suppression | Add Fourier features or increase high-frequency sampling | Feature warning plus elevated high-frequency error | Apply only at scheduled checkpoints; compare compute cost |
| Gradient conflict | Temporarily rebalance objectives | Negative cosine plus validated conflict feature | Do not allow any component loss to be ignored |

The candidate actions follow the proposal's closed-loop options: loss reweighting, adaptive sampling, and architecture changes [^1].

Controller logic should be deterministic and logged:

```text
for each training checkpoint:
    update losses, gradients, activations, residual maps, and spectra
    encode selected activation layer with frozen SAE
    compute monitor score using past-only features
    if score exceeds locked threshold and cooldown has expired:
        identify failure class
        apply that class's bounded corrective action
        log pre/post state and intervention budget
continue until the fixed stopping rule
```

## 12. Baselines and ablations

The final study must compare:

- standard PINN;
- loss-reweighted or GradNorm-style baseline;
- adaptive-sampling baseline;
- output-only monitor/controller;
- SAE monitor without causal validation;
- fully validated SAE closed-loop controller;
- oracle controller using the true failure label, for an upper-bound reference;
- random-feature controller, for specificity.

Ablate one component at a time: no SAE sparsity, no causal filtering, no spatial probe points, no gradient features, no spectral features, no controller cooldown, and no held-out-PDE testing. This identifies whether gains come from mechanistic features or simply additional monitoring capacity.

## 13. Statistical analysis

Use the run—not the collocation point—as the primary independent unit. Report per-seed outcomes, mean and median, dispersion, and confidence intervals. For failure rates use bootstrap intervals or a hierarchical/binomial model; for continuous metrics use mixed-effects models or seed-level bootstrap with PDE and regime as blocking factors. Predefine the primary contrasts:

1. validated closed-loop controller versus standard PINN;
2. SAE monitor versus conventional monitor for lead time;
3. targeted feature intervention versus unrelated-feature control for causal specificity.

Correct for multiple feature-testing comparisons during discovery, or use a two-stage design: discovery runs identify candidate features and independent validation runs test them. Never use the same trajectory both to choose a feature and to claim its causal effect.

## 14. Quality-control gates and go/no-go decisions

### Gate 1 — PINN benchmark validity
Proceed only if the baseline reproduces at least one success regime and one prespecified failure regime with stable labels across seeds.

### Gate 2 — logging validity
Proceed only if all loss components, gradient statistics, activations, validation errors, residual maps, and sampling metadata are synchronized and recoverable for every checkpoint.

### Gate 3 — SAE validity
Proceed only if held-out reconstruction and sparsity pass thresholds and feature rankings show stability across bootstrap resamples.

### Gate 4 — mechanistic validity
Proceed only if at least one candidate feature passes targeted-versus-control intervention tests on held-out runs. If none passes, report a negative mechanistic result rather than proceeding to a controller built on correlations.

### Gate 5 — monitor validity
Proceed to closed-loop testing only if the SAE monitor improves lead time or discrimination over the conventional baseline without an unacceptable false-alarm increase.

### Gate 6 — controller validity
Claim prevention only if the controller improves the preregistered primary outcome on held-out seeds and does not merely trade one failure for another.

## 15. Timeline and deliverables

### Months 1–2: infrastructure and qualification
- implement canonical PDEs and analytic/numerical reference solvers;
- implement baseline PINN and frozen validation grids;
- build logging, manifests, checkpointing, and reproducibility tests;
- produce the first successful and failed runs.

### Months 3–4: failure atlas
- sweep failure-generating factors;
- lock operational labels;
- release raw trajectories and derived metrics internally;
- select the final benchmark regimes.

### Months 5–6: SAE discovery
- train SAE families;
- evaluate reconstruction, sparsity, dead features, and stability;
- build the first Physics-Feature Dictionary.

### Months 7–8: causal validation
- run counterfactual and training-time interventions;
- perform unrelated-feature, random-direction, reconstruction, and seed controls;
- freeze the validated feature set.

### Months 9–10: monitor and controller
- train leakage-controlled early-warning models;
- lock thresholds and cooldown rules;
- implement one bounded corrective action per failure class.

### Months 11–12: held-out evaluation and release
- run final baselines, ablations, and held-out PDE/regime tests;
- perform statistical analysis;
- release code, manifests, benchmark trajectories, SAE dictionaries, intervention logs, and a failure-analysis report. The proposed benchmark dataset and feature dictionary are explicit project deliverables [^1].

## 16. Main failure modes and mitigations

**SAE features reflect coordinates rather than physics.** Use fixed probe points, randomize point order, test shifted domains, and require cross-run/cross-PDE stability.

**Feature identity is not stable across SAE trainings.** Match features by decoder similarity and activation overlap, report feature families rather than arbitrary IDs, and retain only reproducible features.

**The SAE changes the PINN when inserted.** Quantify reconstruction-only effects and require the intervention-free SAE path to match the original PINN within a locked tolerance.

**Causal intervention is off-manifold.** Use small, quantile-matched manipulations first; compare ablations with activation patching from real runs; report nonlinear dose–response where feasible.

**Monitor leakage inflates lead time.** Enforce past-only windows, run-level splits, and a time-stamped feature-generation audit.

**The controller overreacts.** Add cooldowns, bounded weight changes, intervention budgets, rollback checkpoints, and a no-action arm.

**Performance gain is just extra compute.** Report wall-clock time, gradient evaluations, memory, number of collocation points, and controller overhead; compare against compute-matched baselines.

**Failure labels are arbitrary.** Publish thresholds, sensitivity analyses, raw continuous metrics, and ambiguous-run exclusions before inspecting controller outcomes.

**Negative result is hidden by scope expansion.** If no stable, causal feature emerges in the canonical suite, stop the controller phase and report that the proposed mechanistic link was not established under the tested conditions.

## 17. Minimum final evidence package

A convincing final package should contain:

1. a run-level benchmark table with success/failure labels and all continuous metrics;
2. representative trajectory plots showing loss, gradient conflict, feature activity, and error before failure;
3. SAE reconstruction/sparsity/stability analyses;
4. a Physics-Feature Dictionary with evidence and confidence tiers;
5. causal intervention plots against matched controls;
6. early-warning performance and lead-time distributions;
7. controller outcomes across seeds and PDE regimes;
8. compute and intervention-cost accounting;
9. all preregistered negative and null results.

The central conclusion must be phrased at the level supported by these tests: either validated internal features provide causal and predictive information useful for PINN control, or the experiment shows that the proposed SAE signals are correlational, unstable, or not transferable under the tested conditions.

## 18. Week-by-week engineering backlog

The backlog assumes one primary implementation team, one GPU development environment, and a 12-month schedule. Each week has a concrete artifact and a definition of done. Work should be tracked in issue tickets linked to a code commit, experiment manifest, or analysis artifact.

### Weeks 1–4 — project foundation

**Week 1 — repository and reproducibility skeleton**
- Create repository layout: `configs/`, `pinn/`, `experiments/`, `logging/`, `sae/`, `interventions/`, `monitoring/`, `controller/`, `analysis/`, `tests/`, and `docs/`.
- Add environment lockfile, deterministic seed utility, configuration loader, structured logging, and experiment ID generation.
- Define the run-manifest schema and artifact naming convention.
- Add continuous integration for formatting, static checks, unit tests, and a smoke run.

**Done when:** a clean checkout reproduces a 10-step dummy training run and writes a valid manifest, checkpoint, metrics file, and code-version record.

**Week 2 — PDE and reference-solution interfaces**
- Implement a common PDE interface with residual, boundary/initial conditions, domain sampler, and validation-grid methods.
- Implement 1D Poisson with an analytic solution and a numerical-reference fallback.
- Add tests for residual derivatives, boundary evaluation, domain bounds, and reference-solution accuracy.

**Done when:** analytic and automatic-differentiation residuals agree within a locked tolerance on a test grid.

**Week 3 — baseline PINN trainer**
- Implement the baseline MLP PINN, configurable depth/width/activation, initialization, optimizer, learning-rate schedule, and loss weights.
- Keep PDE, boundary, and initial/data losses separately addressable.
- Add checkpoint resume and exact restart tests.

**Done when:** three seeds complete successfully and resumed training matches uninterrupted training within numerical tolerance.

**Week 4 — frozen evaluation and reporting**
- Build frozen validation grids, error norms, residual maps, boundary metrics, and a standard run summary.
- Add plots for loss components, validation error, residual heatmaps, and prediction versus reference.
- Establish the initial baseline configuration and archive it as immutable.

**Done when:** a single command produces a reproducible baseline report for all development seeds.

### Weeks 5–8 — instrumentation and benchmark qualification

**Week 5 — gradient instrumentation**
- Log per-loss parameter gradients, norms, pairwise cosine similarities, and optimizer-step metadata.
- Add a memory-safe reduced-gradient mode for long runs.
- Unit-test gradient logging against independently recomputed gradients.

**Done when:** logged gradients reproduce direct calculations and the logger adds a measured, documented overhead.

**Week 6 — activation hooks and probe points**
- Add hooks for selected early, middle, and late hidden layers.
- Define fixed probe points and a probe-point version hash.
- Store activation arrays with shape, dtype, layer, checkpoint, and coordinate metadata.

**Done when:** activations can be replayed from a checkpoint and match the original forward pass.

**Week 7 — spectral and spatial diagnostics**
- Implement Fourier analysis of prediction error and selected activations.
- Implement spatial-bin coverage, residual distributions, boundary distance, and high-frequency error summaries.
- Add diagnostics for missing or nonuniform collocation coverage.

**Done when:** all diagnostics run on historical checkpoints and produce schema-valid tables and figures.

**Week 8 — qualification gate**
- Run a small seed matrix on Poisson.
- Verify one stable success regime and deliberately induced failure regimes.
- Lock operational thresholds for boundary starvation, gradient conflict, spectral suppression, and collocation starvation.
- Record ambiguous cases rather than forcing labels.

**Done when:** the team signs off on the benchmark schema, labels, thresholds, and failure examples before broad sweeps begin.

### Weeks 9–12 — failure atlas generation

**Week 9 — controlled loss-weight sweeps**
- Sweep PDE/boundary loss-weight ratios using a fixed seed block.
- Run standard and deliberately imbalanced configurations.
- Produce seed-level outcome tables and failure trajectories.

**Done when:** loss-weight sensitivity is quantified and the selected regimes reproduce across seeds.

**Week 10 — sampling and collocation sweeps**
- Vary collocation density, boundary fraction, spatial sampling bias, and resampling policy.
- Add controlled under-coverage regions for collocation-starvation tests.
- Verify the frozen validation grid remains unchanged.

**Done when:** sampling metadata, coverage deficits, residual maps, and outcome labels are jointly available for every run.

**Week 11 — optimization and architecture sweeps**
- Vary optimizer, learning rate, depth, width, activation, and initialization scale one factor at a time.
- Add stiffness, frequency, and reaction-parameter regimes for advection–diffusion and reaction–diffusion.
- Track compute cost and memory for every configuration.

**Done when:** the factor matrix identifies a compact set of reproducible success and failure regimes without uncontrolled combinatorial expansion.

**Week 12 — atlas freeze and split assignment**
- Freeze discovery, validation, and final-test run IDs.
- Generate the benchmark index and failure-label report.
- Hash all raw logs and manifests.
- Prohibit later tuning on final-test runs.

**Done when:** a fresh analyst can identify the allowed training data, validation data, and untouched final-test data from the repository alone.

### Weeks 13–16 — activation dataset and SAE baseline

**Week 13 — activation dataset builder**
- Convert checkpoint activations into run-level training examples.
- Split by run, not by activation vector.
- Add normalization options fit only on training runs.
- Implement streaming/chunked dataset loading.

**Done when:** dataset checks confirm no run, seed, or future checkpoint leakage across splits.

**Week 14 — SAE implementation**
- Implement overcomplete encoder/decoder, sparse latent activations, reconstruction loss, L1 penalty, decoder normalization, checkpointing, and metrics.
- Add PCA/low-rank reconstruction baselines.
- Add dead-feature and feature-usage diagnostics.

**Done when:** SAE training is deterministic under a fixed seed and produces reproducible validation metrics.

**Week 15 — SAE hyperparameter pilot**
- Sweep latent expansion, sparsity strength, learning rate, batch size, and selected layer.
- Use only discovery runs for model selection.
- Record reconstruction, sparsity, dead features, activation frequency, and compute cost.

**Done when:** a Pareto set of non-degenerate SAE configurations is selected using locked quality criteria.

**Week 16 — SAE quality gate**
- Test reconstruction on held-out runs.
- Bootstrap feature rankings and decoder similarity.
- Reject unstable, dead, or run-identity-dominated SAEs.
- Freeze the primary SAE family and document alternatives.

**Done when:** the SAE passes reconstruction, sparsity, stability, and leakage audits.

### Weeks 17–20 — feature interpretation

**Week 17 — feature statistics and visualization**
- Compute activation frequency, magnitude, temporal trajectories, spatial localization, and boundary-distance profiles.
- Generate top-activating and bottom-activating examples.
- Build feature dashboards linked to run metadata.

**Done when:** any latent feature can be inspected from one report without manually joining raw files.

**Week 18 — physical association analysis**
- Associate latent features with loss components, gradient norms, gradient cosines, residual maps, spectra, and coverage metrics.
- Add held-out rank correlations and mutual-information summaries.
- Test whether associations persist after controlling for time, PDE identity, and run identity.

**Done when:** candidate feature lists are generated with effect sizes, uncertainty, and confound diagnostics.

**Week 19 — feature matching and reproducibility**
- Match features across SAE seeds by decoder and activation similarity.
- Group recurring features into feature families.
- Evaluate cross-seed and cross-PDE stability.

**Done when:** the dictionary reports stable feature families rather than relying on arbitrary latent IDs.

**Week 20 — Physics-Feature Dictionary v1**
- Create the annotation template and fill candidate interpretations.
- For each candidate, record predicted intervention direction, target loss, control feature, and expected specificity pattern.
- Freeze the candidate list before causal testing.

**Done when:** every causal candidate has a written, testable prediction made without using intervention results.

### Weeks 21–24 — causal intervention engine

**Week 21 — inference-time intervention plumbing**
- Insert frozen SAE encode/decode operations at the selected hidden layer.
- Implement natural, ablated, amplified, matched-value, unrelated-feature, random-direction, and reconstruction-control modes.
- Add intervention metadata to every forward pass.

**Done when:** intervention-free SAE reconstruction matches the original PINN within tolerance and all intervention modes execute on a checkpoint.

**Week 22 — immediate causal-effect experiments**
- Run interventions on candidate features across held-out seeds and checkpoints.
- Measure target and non-target loss changes, output error, residual maps, and spectral changes.
- Generate dose–response curves for ablation and amplification.

**Done when:** targeted effects are quantitatively compared with matched controls and confidence intervals are available.

**Week 23 — training-time interventions**
- Apply candidate interventions for fixed training windows.
- Test rescue before failure, rescue after failure, and perturbation during successful training.
- Add rollback checkpoints and intervention cooldowns.

**Done when:** training-time intervention trajectories can be compared with no-intervention and control trajectories using identical seeds.

**Week 24 — mechanistic validation gate**
- Apply the preregistered causal-strength and specificity rules.
- Freeze verified, rejected, and unresolved feature classes.
- Update the Physics-Feature Dictionary with causal evidence.

**Done when:** only features passing held-out targeted-versus-control tests are marked as verified mechanistic signals.

### Weeks 25–28 — early-warning monitor

**Week 25 — prediction dataset construction**
- Create past-only trajectory windows and fixed future failure horizons.
- Generate positive and negative examples without using post-failure information.
- Balance or weight classes without altering test prevalence.

**Done when:** an automated leakage audit passes and every prediction example has a timestamp, run ID, history window, and future label.

**Week 26 — baseline monitors**
- Implement loss-plateau, gradient-only, residual-only, coverage-only, and combined conventional monitors.
- Evaluate fixed thresholds and simple logistic models.
- Establish calibration and false-alarm baselines.

**Done when:** conventional alarm performance is documented before SAE monitor tuning.

**Week 27 — SAE monitors**
- Add threshold, logistic, tree-based, and optional LSTM models using validated feature trajectories.
- Tune on discovery/validation runs only.
- Report AUROC, AUPRC, calibration, recall at fixed false-alarm rates, and lead time.

**Done when:** the simplest model meeting the prespecified performance criterion is selected.

**Week 28 — monitor gate**
- Test on held-out seeds and regimes.
- Compare SAE, conventional, and combined monitors.
- Freeze thresholds, model weights, feature list, and prediction horizon.

**Done when:** the monitor is locked before controller experiments and its lead-time advantage, if any, is quantified.

### Weeks 29–32 — controller implementation

**Week 29 — controller state machine**
- Implement warning, confirmation, action, cooldown, rollback, and termination states.
- Add bounded action magnitudes and an intervention budget.
- Log every trigger, action, pre-state, post-state, and rollback.

**Done when:** controller behavior is deterministic and fully replayable from a checkpoint plus monitor output.

**Week 30 — loss-reweighting controller**
- Implement boundary-loss reweighting and gradient-balancing actions.
- Add caps, restoration rules, and no-action controls.
- Test on boundary-starvation and gradient-conflict regimes.

**Done when:** controller changes are bounded, reversible, and do not silently remove any objective.

**Week 31 — adaptive-sampling controller**
- Implement residual- and coverage-informed point addition/upsampling.
- Preserve the frozen evaluation grid and track point count and compute cost.
- Test on collocation-starvation regimes.

**Done when:** added points are traceable to a trigger and compute-matched comparisons are available.

**Week 32 — architecture-action controller**
- Implement the limited Fourier-feature or capacity adjustment action for spectral suppression.
- Compare scheduled architecture changes with controller-triggered changes.
- Add rollback if validation error or stability worsens.

**Done when:** architecture actions can be applied without invalidating checkpoint loading or evaluation comparability.

### Weeks 33–36 — baseline and ablation study

**Week 33 — baseline implementations**
- Finalize standard PINN, loss-reweighted/GradNorm-style, adaptive-sampling, and output-only controller baselines.
- Validate each baseline on the same run manifests and compute accounting.

**Done when:** every baseline has a reproducible command, fixed configuration, and matched evaluation protocol.

**Week 34 — SAE ablations**
- Run SAE monitor without causal filtering, without spectral features, without spatial features, and without gradient features.
- Test no-SAE conventional monitoring and random-feature control.

**Done when:** component-level contributions are estimated with the same seeds and regimes.

**Week 35 — controller ablations**
- Remove cooldown, rollback, action bounds, and intervention budget in separate controlled experiments.
- Quantify overreaction, false alarms, and compute escalation.

**Done when:** safety mechanisms are justified by measured failure or cost changes.

**Week 36 — compute-matched comparison**
- Match wall-clock budget, gradient evaluations, memory, and collocation-point budget where possible.
- Produce a compute-accounting table for every method.

**Done when:** any performance advantage can be separated from additional compute or monitoring overhead.

### Weeks 37–40 — held-out generalization

**Week 37 — held-out parameter regimes**
- Run the locked monitor and controller on unseen stiffness, frequency, reaction, and sampling regimes.
- Do not retune thresholds.

**Done when:** generalization results are generated from the frozen controller package.

**Week 38 — held-out seeds and PDEs**
- Test unseen seeds and, if the pipeline remains stable, one held-out PDE family.
- Record failures, ambiguous outcomes, and controller actions without post hoc relabeling.

**Done when:** the main claims have independent validation beyond the discovery PDEs.

**Week 39 — stress and robustness tests**
- Test logging interruptions, missing checkpoints, extreme but valid loss weights, noisy monitor inputs, and partial activation availability.
- Verify safe fallback to standard training.

**Done when:** operational failures do not produce silent or unlogged controller behavior.

**Week 40 — generalization gate**
- Decide whether the method is transferable, conditionally transferable, or benchmark-specific.
- Freeze the final evaluation dataset and analysis plan.

**Done when:** no further model or threshold changes are permitted before final analysis.

### Weeks 41–44 — final evaluation and statistical analysis

**Week 41 — final production runs**
- Execute the locked final matrix across methods, seeds, PDEs, and regimes.
- Archive logs immediately after each batch and verify checksums.

**Done when:** the final matrix is complete or deviations are documented before looking at aggregate outcomes.

**Week 42 — primary endpoint analysis**
- Analyze failure rate, rescued-failure rate, final solution error, conservation violation, convergence stability, and controller cost.
- Report seed-level and blocked results.

**Done when:** the three primary contrasts are estimated with confidence intervals and prespecified tests.

**Week 43 — secondary and negative-result analysis**
- Analyze feature stability, monitor calibration, lead-time distributions, intervention specificity, and failure-class confusion.
- Include null, rejected, unstable, and non-transferable features.

**Done when:** the analysis includes all preregistered outcomes and does not suppress null results.

**Week 44 — statistical audit**
- Check multiple-comparison handling, split integrity, seed independence, missing data, outliers, and analysis-code reproducibility.
- Re-run summary tables from raw logs.

**Done when:** an independent analyst can reproduce every primary number from immutable inputs.

### Weeks 45–48 — release and documentation

**Week 45 — benchmark release package**
- Clean and document raw trajectories, derived metrics, manifests, labels, validation grids, and reference solutions.
- Add schema versioning and a data card.

**Done when:** a new user can load the benchmark and reproduce a published example.

**Week 46 — code and model release**
- Release training code, SAE code, intervention engine, monitor, controller, configurations, checkpoints, and environment lockfile.
- Add quick-start examples and failure-reproduction scripts.

**Done when:** installation and reproduction succeed on a clean environment.

**Week 47 — feature dictionary and analysis report**
- Finalize the Physics-Feature Dictionary with candidate, verified, rejected, and unresolved entries.
- Produce figures for representative trajectories, causal interventions, monitor performance, and controller outcomes.

**Done when:** each claimed mechanism links to source runs, intervention logs, controls, and analysis code.

**Week 48 — final review and handoff**
- Conduct an internal methods review against the quality gates.
- Record deviations, limitations, and unresolved engineering debt.
- Tag the final release and archive the complete experiment registry.

**Done when:** the project can support either a positive mechanistic conclusion or a defensible negative result without reconstructing hidden analysis decisions.

## 19. Backlog operating rules

- Every ticket must name its input data, output artifact, owner, acceptance test, and dependency.
- No ticket may modify frozen final-test data, thresholds, feature identities, or controller rules after Weeks 40–41.
- Every experiment must record seed, configuration hash, code commit, hardware, runtime, and failure status.
- Bug fixes that alter numerical results require a versioned rerun, not an overwrite.
- Keep discovery artifacts, validation artifacts, and final-test artifacts physically and logically separate.
- Review progress at the end of each four-week block using the corresponding quality gate; do not advance because the calendar says so if a gate fails.

## 20. Detailed technical checklist for Weeks 1–4

### Week 1 — Repository, environment, and reproducibility skeleton

#### Repository structure

- [ ] Create the repository and protect the main branch.
- [ ] Add directories:
  - [ ] `configs/` for YAML or JSON experiment configurations.
  - [ ] `pinn/` for models, PDEs, samplers, losses, and trainers.
  - [ ] `experiments/` for executable experiment entry points.
  - [ ] `logging/` for metrics, activation, gradient, and manifest writers.
  - [ ] `analysis/` for post-processing and report generation.
  - [ ] `tests/unit/` and `tests/integration/`.
  - [ ] `docs/` for design decisions, schemas, and runbooks.
  - [ ] `artifacts/` for generated plots and summaries; exclude large raw arrays from Git.
- [ ] Add `README.md` with setup, smoke-test, training, evaluation, and reproduction commands.
- [ ] Add `CHANGELOG.md` and a short decision-log template.
- [ ] Add `.gitignore` for checkpoints, caches, local environments, secrets, and temporary arrays.

#### Environment and dependency management

- [ ] Pin Python and CUDA-compatible framework versions.
- [ ] Add a lockfile or fully pinned requirements file.
- [ ] Add a one-command environment validation script.
- [ ] Record framework, CUDA, GPU, driver, BLAS, and operating-system information at runtime.
- [ ] Add a CPU-only smoke-test profile for continuous integration.
- [ ] Add a GPU development profile with explicit device selection.
- [ ] Make floating-point precision configurable, defaulting to one documented baseline.

#### Configuration and run identity

- [ ] Define a typed configuration object for PDE, architecture, optimizer, sampling, logging, evaluation, and seed settings.
- [ ] Define a canonical serialization order for configurations.
- [ ] Hash the canonical configuration and combine it with the code commit and seed to form a run ID.
- [ ] Reject unknown configuration keys rather than silently ignoring them.
- [ ] Store the fully resolved configuration beside every run.
- [ ] Add a `dry_run: true` mode that validates configuration without training.

#### Determinism and random-state handling

- [ ] Implement one seed function covering Python, NumPy, framework CPU, framework GPU, and data-loader workers.
- [ ] Record all random seeds in the manifest.
- [ ] Configure deterministic kernels where supported.
- [ ] Document which operations remain nondeterministic.
- [ ] Add a reproducibility test: identical configuration and seed should produce matching first-step losses and parameters within tolerance.
- [ ] Add a separate test confirming that changing the seed changes the initialized parameters.

#### CI and smoke test

- [ ] Add formatting and lint checks.
- [ ] Add type checking for the public interfaces.
- [ ] Add unit tests for configuration parsing, seed setting, run-ID creation, and manifest writing.
- [ ] Add a ten-step dummy training smoke test.
- [ ] Make CI fail if the smoke test does not produce the expected artifact files.

#### Week 1 acceptance package

- [ ] `python -m experiments.smoke_test` completes on CPU.
- [ ] The command writes a manifest, resolved config, metrics file, checkpoint, and environment report.
- [ ] A second execution with the same seed reproduces the declared deterministic outputs.
- [ ] A new developer can set up the environment by following only `README.md`.

### Week 2 — PDE and reference-solution interfaces

#### Common PDE interface

- [ ] Define an abstract PDE object with:
  - [ ] domain bounds;
  - [ ] input dimension and output dimension;
  - [ ] residual function;
  - [ ] boundary-condition evaluator;
  - [ ] initial-condition evaluator where needed;
  - [ ] analytic or numerical reference solution;
  - [ ] interior sampler;
  - [ ] boundary sampler;
  - [ ] validation-grid constructor;
  - [ ] exact parameter serialization.
- [ ] Define a standard tensor convention for coordinates, batch dimensions, and output dimensions.
- [ ] Define dtype and device propagation rules.
- [ ] Validate that sampled points remain inside the declared domain.
- [ ] Return named residual components rather than an anonymous scalar.

#### 1D Poisson implementation

- [ ] Implement the selected 1D domain.
- [ ] Implement the source term and boundary conditions.
- [ ] Implement the analytic reference solution.
- [ ] Implement the PDE residual using automatic differentiation.
- [ ] Implement boundary sampling and boundary residuals.
- [ ] Implement a uniform validation grid.
- [ ] Implement exact solution evaluation on arbitrary points.

#### Numerical correctness tests

- [ ] Test domain-boundary inclusion and exclusion behavior.
- [ ] Test residual evaluation on a known exact solution.
- [ ] Test boundary residual evaluation on the exact solution.
- [ ] Compare automatic-differentiation derivatives with finite-difference checks on a test function.
- [ ] Test batched and unbatched inputs.
- [ ] Test CPU and GPU outputs within tolerance.
- [ ] Test float32 and the selected higher-precision mode if supported.
- [ ] Test serialization and reconstruction of PDE parameters.

#### Reference-solution quality

- [ ] Define a reference-solution error tolerance before using the solver for evaluation.
- [ ] Verify that the analytic solution satisfies the PDE residual numerically.
- [ ] Verify that the analytic solution satisfies boundary conditions numerically.
- [ ] Store reference-solution metadata, tolerance, and validation date.
- [ ] If a numerical reference solver is added, compare it against the analytic solution on multiple grid resolutions.

#### Week 2 acceptance package

- [ ] `PDE.validate()` passes all interface and numerical checks.
- [ ] The exact solution produces near-zero residual and boundary error under the locked tolerance.
- [ ] A generated validation grid is deterministic and hashable.
- [ ] A short technical note documents the equation, domain, source, boundary conditions, sign convention, and residual definition.

### Week 3 — Baseline PINN trainer

#### Model implementation

- [ ] Implement a configurable fully connected network with:
  - [ ] input and output dimensions from the PDE object;
  - [ ] configurable hidden depth and width;
  - [ ] configurable activation;
  - [ ] explicit initialization;
  - [ ] device and dtype support;
  - [ ] parameter-count reporting.
- [ ] Add a stable forward-pass API returning predictions and optional intermediate activations.
- [ ] Add model serialization and checkpoint loading.
- [ ] Add an architecture summary to the run manifest.

#### Loss implementation

- [ ] Implement separate interior PDE, boundary, and initial/data loss functions.
- [ ] Return a named loss dictionary on every optimization step.
- [ ] Implement configurable loss weights.
- [ ] Verify that the total loss equals the weighted sum of components.
- [ ] Add checks for NaN, infinity, empty batches, and zero-weight components.
- [ ] Preserve unreduced pointwise residuals for later spatial diagnostics.

#### Sampling and batching

- [ ] Implement fixed and resampled interior collocation points.
- [ ] Implement boundary and initial/data samplers.
- [ ] Record sample counts and sampler seeds.
- [ ] Add a fixed-sample mode for debugging.
- [ ] Add a resampling hook without implementing adaptive sampling yet.
- [ ] Ensure validation points never enter the training sampler.

#### Trainer and optimizer

- [ ] Implement optimizer construction from configuration.
- [ ] Implement learning-rate scheduling from configuration.
- [ ] Implement a fixed stopping rule for the baseline.
- [ ] Implement checkpoint intervals.
- [ ] Save the best-validation checkpoint separately from the final checkpoint.
- [ ] Support resume from a checkpoint with optimizer and scheduler state.
- [ ] Record step, epoch, wall-clock time, learning rate, and batch/sample counts.

#### Resume and equivalence tests

- [ ] Train uninterrupted for a fixed number of steps.
- [ ] Train to an intermediate checkpoint, reload, and continue.
- [ ] Compare final parameters, losses, and predictions within the declared tolerance.
- [ ] Test resume after process restart.
- [ ] Test incompatible checkpoint/configuration combinations fail loudly.

#### Week 3 acceptance package

- [ ] Three baseline seeds complete without numerical failure.
- [ ] The trainer produces separate loss components, checkpoints, and optimizer metadata.
- [ ] Resume training is equivalent to uninterrupted training within tolerance.
- [ ] A deliberately changed seed produces a distinct but valid trajectory.
- [ ] The baseline command is documented and runnable from a clean checkout.

### Week 4 — Frozen evaluation and reporting

#### Frozen evaluation grid

- [ ] Generate one immutable validation grid per PDE configuration.
- [ ] Hash and store the grid coordinates.
- [ ] Keep validation-grid generation independent of training collocation sampling.
- [ ] Add a test that training resampling cannot mutate the validation grid.
- [ ] Store evaluation-grid metadata in every run summary.

#### Evaluation metrics

- [ ] Implement relative and absolute $$L^2$$ solution error.
- [ ] Implement $$L^\infty$$ error.
- [ ] Implement mean and maximum absolute PDE residual on the validation grid.
- [ ] Implement boundary-condition error by boundary segment.
- [ ] Implement pointwise prediction error and residual arrays.
- [ ] Implement conservation or integral constraints when applicable.
- [ ] Define behavior when the reference norm is zero or numerically negligible.

#### Standard plots

- [ ] Plot total loss and each loss component on a shared step axis.
- [ ] Plot validation errors over time.
- [ ] Plot prediction versus reference solution.
- [ ] Plot pointwise solution error.
- [ ] Plot PDE residual over the domain.
- [ ] Plot boundary residuals separately.
- [ ] Add a machine-readable figure manifest containing run ID, metric names, and configuration hash.

#### Reporting and artifact schema

- [ ] Define a run-summary JSON schema.
- [ ] Define a long-format metrics table schema.
- [ ] Define a checkpoint index schema.
- [ ] Include success/failure status as `unlabeled` until operational failure thresholds are formally locked.
- [ ] Include missing-data and numerical-failure fields rather than dropping failed runs.
- [ ] Add a report generator that consumes only stored artifacts, not in-memory training state.

#### Baseline qualification

- [ ] Run the initial baseline configuration across the development seed block.
- [ ] Confirm that predictions, losses, and validation metrics are physically plausible.
- [ ] Verify that all expected files exist for every run.
- [ ] Compare runtime and memory across seeds.
- [ ] Document any nondeterminism, numerical instability, or unexplained behavior.
- [ ] Do not begin broad failure sweeps until the Week 4 gate is passed.

#### Week 4 acceptance package

- [ ] One command generates baseline reports for all development seeds.
- [ ] Reports contain losses, errors, residuals, prediction plots, configuration, environment, and checkpoint links.
- [ ] All validation grids are immutable, hashed, and separated from training data.
- [ ] A clean-checkout reproduction test regenerates the same baseline summary within declared numerical tolerances.
- [ ] The team signs off on the Week 5 instrumentation interface.

### Cross-week technical conventions

- [ ] Use explicit names for dimensions, units, coordinates, and loss components.
- [ ] Never overwrite raw logs; write versioned derived artifacts.
- [ ] Record failures as data, including stack traces and last valid checkpoints.
- [ ] Keep all thresholds in configuration files and record their version.
- [ ] Require a test for every new public interface.
- [ ] Make every analysis script runnable from stored artifacts alone.
- [ ] Prefer small end-to-end tests over isolated tests that never execute the real training path.

## 21. End-to-end phase-wise technical implementation guide

### Phase 0 — Research contract and preregistration

**Purpose:** prevent the project from becoming an unconstrained search for attractive latent features.

**Inputs:** the proposal, selected PDE suite, compute budget, and intended scientific claims.

**Implementation steps**

1. Write a one-page protocol defining the primary hypotheses, primary endpoints, failure labels, intervention controls, data splits, and final comparisons.
2. Separate discovery, validation, and final-test runs at the run-ID level.
3. Freeze the baseline architecture and the initial PDE set.
4. Define what counts as a successful PINN, a failed PINN, an early warning, a causal feature, and a rescued failure.
5. Define a negative-result path: if no feature passes causal validation, stop before building a mechanistic controller.
6. Register all threshold families, monitor horizons, controller actions, and statistical contrasts before final runs.

**Deliverables:** protocol document, experiment registry, data-split manifest, threshold specification, and risk register.

**Exit gate:** another engineer can determine which future results would support, weaken, or falsify each hypothesis without asking the original investigator.

### Phase 1 — Reference PDE and numerical oracle

**Purpose:** establish a trusted physical reference before interpreting a neural network.

Use 1D Poisson as the first oracle because it supports analytic verification and simple spatial diagnostics. Add advection–diffusion and reaction–diffusion only after the full pipeline works on Poisson. PINNs use automatic differentiation to penalize PDE residuals at sampled domain points, and the standard formulation combines PDE and boundary terms in a soft-constrained loss [^2].

**Implementation steps**

1. Implement PDE residual, boundary operator, exact/reference solution, domain sampler, and validation grid behind one interface.
2. Verify the analytic solution independently with automatic differentiation and finite differences.
3. Create a high-resolution frozen evaluation grid and hash it.
4. Define reference error, residual, boundary, and conservation metrics.
5. Build a numerical-solver fallback for PDEs without analytic solutions.
6. Store equation conventions, units, sign conventions, and parameter ranges in the manifest.

**Deliverables:** PDE module, reference solver, validation-grid files, correctness tests, and oracle report.

**Exit gate:** the reference solution passes residual and boundary checks, and the same evaluation grid can be regenerated exactly.

### Phase 2 — Baseline PINN and instrumentation

**Purpose:** produce reproducible training trajectories before intentionally creating failures.

**Implementation steps**

1. Implement the baseline MLP PINN and separate loss components.
2. Implement fixed and resampled collocation modes.
3. Add checkpointing, resume, deterministic seeds, and environment capture.
4. Log total/component losses, validation errors, residual maps, collocation coordinates, runtime, and memory.
5. Add gradient norms and pairwise cosine similarities for PDE, boundary, and data objectives.
6. Add activation hooks for early, middle, and late hidden layers at fixed probe points.
7. Add Fourier summaries of solution error and selected activation channels.
8. Run development seeds and verify that all trajectories are complete.

Gradient instrumentation is justified by prior PINN work identifying numerical stiffness and unbalanced backpropagated gradients as a fundamental failure mode and using gradient statistics to balance composite losses [^3]. The proposal likewise identifies gradients, activations, errors, and spectral measures as core trajectory variables [^1].

**Deliverables:** baseline trainer, run schema, trajectory store, checkpoint index, and initial training report.

**Exit gate:** an interrupted run can be resumed, and an independent process can regenerate the same metrics from saved artifacts.

### Phase 3 — Controlled failure atlas

**Purpose:** create labeled trajectories with known failure mechanisms rather than relying on post hoc interpretation.

**Implementation steps**

1. Vary one factor at a time: loss weights, collocation density, spatial coverage, optimizer, learning rate, architecture, activation, stiffness, and frequency.
2. Keep a success control for every failure-generating configuration.
3. Label gradient conflict using persistent negative inter-objective cosine plus stalled or worsening validation error.
4. Label boundary starvation using low boundary error but high interior error or PDE residual.
5. Label spectral suppression using persistent high-frequency error despite low-frequency improvement.
6. Label collocation starvation using under-covered spatial bins with high residual/error.
7. Preserve continuous scores in addition to binary labels.
8. Freeze the discovery/validation/test split after the atlas is generated.

Adaptive sampling literature supports treating residual-defined failure regions as an error indicator and adding points preferentially in those regions; it also warns that fixed points can miss localized solution structure [^2]. Therefore, sampling coverage must be logged as a first-class variable rather than treated as an implementation detail.

**Deliverables:** failure atlas, label definitions, trajectory index, representative failure plots, and split manifest.

**Exit gate:** at least one reproducible example exists for each selected failure class, and labels are stable across the development seed block.

### Phase 4 — Activation dataset and SAE discovery

**Purpose:** test whether hidden activations contain sparse, reusable physical features.

**Implementation steps**

1. Construct activation examples from fixed probe points across time, seeds, PDE regimes, and outcome labels.
2. Split by complete run to prevent temporal leakage.
3. Normalize only with statistics fit on training runs.
4. Train SAEs at selected layers with multiple expansion and sparsity settings.
5. Compare against PCA/low-rank reconstruction and a non-sparse autoencoder.
6. Measure reconstruction, sparsity, dead features, feature frequency, decoder norms, and held-out stability.
7. Match features across SAE seeds by decoder direction and activation overlap.
8. Reject features driven primarily by time, PDE identity, or run identity.

SAE-based physical feature discovery has been demonstrated in another scientific neural model: unsupervised features correlated with physical observables, followed by targeted single-feature interventions that changed the observable while leaving the main energy objective nearly unchanged [^4]. This supports using SAE discovery as a hypothesis-generation stage, not as evidence of causality by itself.

**Deliverables:** SAE checkpoints, reconstruction report, feature-stability report, latent activation store, and Physics-Feature Dictionary v1.

**Exit gate:** the selected SAE family passes held-out reconstruction and stability criteria without feature collapse or obvious identity leakage.

### Phase 5 — Physics-grounded feature interpretation

**Purpose:** translate latent directions into testable physical hypotheses.

**Implementation steps**

1. Compute spatial, temporal, spectral, loss, gradient, residual, and coverage associations for each feature.
2. Plot activation against distance to boundary, local residual, solution gradient, and frequency-band error.
3. Use held-out runs to estimate association strength and uncertainty.
4. Produce top-activation examples and spatial activation maps.
5. Assign candidate meanings only when multiple independent views agree.
6. Record a predicted intervention direction and target metric for each candidate.
7. Lock the candidate list before intervention results are inspected.

Interpretation must remain distinct from causal validation. Recent SAE work explicitly argues that interpretable features should be tested by controlled experiments rather than accepted from visualization or association alone [^5].

**Deliverables:** feature reports, candidate-feature table, confound analysis, and frozen intervention hypotheses.

**Exit gate:** every candidate has a physical interpretation, target loss, predicted direction, matched control, and prespecified success criterion.

### Phase 6 — Causal intervention and counterfactual validation

**Purpose:** determine whether a latent feature has a selective causal role.

**Implementation steps**

1. Insert the frozen SAE at the selected hidden layer.
2. Validate that the no-intervention SAE path reproduces the original PINN within tolerance.
3. Apply ablation, amplification, matched-value replacement, unrelated-feature control, random-direction control, and reconstruction-only control.
4. Run inference-time counterfactuals on frozen checkpoints.
5. Run training-time interventions before, during, and after the labeled failure transition.
6. Measure target-loss change, non-target-loss change, output error, residual maps, spectral effects, and persistence after continued training.
7. Repeat across held-out seeds and checkpoints.
8. Apply the preregistered causal-strength and specificity score.

A feature should be retained only when its targeted effect is stronger and more selective than matched controls. This follows the broader mechanistic-interpretability principle that causal feature steering requires controlled manipulation, not merely an interpretable activation pattern [^5][^4].

**Deliverables:** intervention engine, counterfactual dataset, effect-size plots, specificity analysis, and Physics-Feature Dictionary v2.

**Exit gate:** verified, rejected, and unresolved feature families are explicitly separated; no causal label is assigned from correlation alone.

### Phase 7 — Early-warning monitor

**Purpose:** establish whether validated features provide actionable warning before failure.

**Implementation steps**

1. Define a future failure horizon and past-only history windows.
2. Build loss-only and conventional-physics baselines first.
3. Add validated SAE features to logistic/threshold models.
4. Use an LSTM only if simpler models fail and the added complexity is justified.
5. Evaluate discrimination, calibration, false-alarm rate, recall, and lead time.
6. Tune only on discovery/validation runs.
7. Freeze model, features, horizon, and thresholds before controller evaluation.

**Deliverables:** monitor dataset, leakage audit, baseline-monitor report, SAE-monitor report, and frozen monitor package.

**Exit gate:** the monitor improves the primary lead-time criterion without an unacceptable false-alarm burden on held-out runs.

### Phase 8 — Closed-loop controller

**Purpose:** test whether internal diagnostics improve training outcomes rather than merely explain them.

**Implementation steps**

1. Implement a deterministic state machine with warning, confirmation, action, cooldown, rollback, and termination states.
2. Map each validated failure class to one bounded action.
3. Start with loss reweighting for gradient/boundary failures and residual/coverage-informed sampling for collocation failures.
4. Add architecture changes only after simpler actions are tested.
5. Log trigger state, action magnitude, pre/post metrics, compute cost, and rollback events.
6. Compare against standard PINN, GradNorm/loss-balancing, adaptive sampling, output-only control, random-feature control, and an oracle upper bound.
7. Run compute-matched comparisons.

Residual-based adaptive sampling provides a concrete baseline because it treats high-residual regions as failure regions and enriches the training set there [^2]. The controller should therefore demonstrate benefit beyond simply rediscovering a residual-based rule.

**Deliverables:** controller package, baseline implementations, ablation matrix, compute-accounting table, and controller trajectory reports.

**Exit gate:** the controller improves the preregistered primary outcome on held-out seeds/regimes without unacceptable instability or compute escalation.

### Phase 9 — Generalization, stress testing, and final analysis

**Purpose:** distinguish a robust mechanism from a benchmark-specific artifact.

**Implementation steps**

1. Test unseen seeds, parameter regimes, and at least one held-out PDE family if feasible.
2. Stress test missing activations, noisy monitor inputs, interrupted logging, extreme valid loss weights, and controller rollback.
3. Repeat primary analyses from immutable raw artifacts.
4. Report negative, unstable, and non-transferable features.
5. Analyze run-level outcomes with seed and PDE blocking.
6. Report accuracy, stability, failure rate, rescue rate, lead time, specificity, runtime, memory, and intervention counts.
7. Package code, manifests, trajectories, SAE dictionaries, intervention logs, and reproducibility instructions.

**Deliverables:** final statistical report, benchmark release, code release, data card, feature dictionary, and limitations report.

**Exit gate:** every primary conclusion can be regenerated from a tagged release and its cited source runs.

## 22. Recommended initial technical defaults

These are starting defaults for the first implementation, not claims of universal optimality:

- 1D Poisson first; advection–diffusion and reaction–diffusion after pipeline qualification.
- Fully connected tanh PINN with a fixed baseline architecture before adding Fourier features.
- Separate PDE, boundary, and initial/data losses with explicit weights.
- Fixed validation grid independent of all training samplers.
- Development seed block of 5 runs; final seed block of 10–20 runs if compute permits.
- Early/middle/late activation hooks, with one layer selected for primary SAE analysis after the pilot.
- SAE latent expansion sweep of 2–8× activation width and a small sparsity-penalty grid.
- Threshold/logistic early-warning models before recurrent models.
- One bounded controller action per pathology before multi-action policies.
- Final claims based on held-out run-level evaluation, not point-level or time-point-level resampling.

These defaults are compatible with published PINN implementations that use fully connected networks, tanh activations, Adam optimization, explicit collocation/boundary counts, and relative $$L^2$$ evaluation, but their numerical values should be tuned only within the preregistered development set [^2].


[^1]: Mechanistic Interpretability in PINNs_ Discovering, Diagnosing, and Preventing Optimization Failures.pdf.

[^2]: Gao et al., 2022. Failure-informed adaptive sampling for PINNs. SIAM Journal on Scientific Computing.

[^3]: Wang et al., 2020. Understanding and mitigating gradient pathologies in physics-informed neural networks. arXiv.org.

[^4]: Qi & Earls, 2026. Mechanistic Interpretability and Causal Feature Steering of Neural Quantum States via Sparse Autoencoders.

[^5]: Stevens et al., 2025. Interpretable and Testable Vision Features via Sparse Autoencoders. ArXiv.