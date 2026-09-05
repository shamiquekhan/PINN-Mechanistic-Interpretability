# Mechanistic Interpretability in PINNs: Discovering, Diagnosing, and Preventing Optimization Failures

## Abstract  
Physics-Informed Neural Networks (PINNs) embed PDE constraints into neural-net training, offering a mesh-free way to solve PDEs. However, **PINNs often suffer catastrophic failures**—e.g. conflicting gradients, collocation starvation, spectral bias—even when loss metrics appear satisfactory. These failures stem from internal network representations that collapse under competing physical objectives. In parallel, *mechanistic interpretability* (MI) has shown that **Sparse Autoencoders (SAEs)** can disentangle neural superposition into human-readable features. We propose to unite these ideas: treating PINN training as a dynamical system, we will **trace internal activations**, learn sparse features with SAEs, map them to physical quantities, and perform causal interventions to *prevent* failures. The goal is a closed-loop PINN that **self-diagnoses and corrects** its training when internal signals predict a pathology. This bridges scientific machine learning with MI and aims to create robust, transparent PINN solvers.

## 1. Background and Motivation  
PINNs minimize a weighted loss combining PDE residuals and boundary/initial conditions. In practice, **different loss components produce conflicting gradients**, and training can stagnate or “collapse” such that, for example, boundary conditions are satisfied while the interior PDE solution is wrong. Recent studies document these pathologies: sharp Hessian spectra in failed networks, NTK ill-conditioning under stiffness, and tasks like conductivity becoming effectively ignored by optimizers. Classical diagnostics (loss curves, gradient norms, NTK spectra) can detect *that* something is wrong, but not *why*. As one survey notes, “PINNs are rooted in physics, but their internal representations are still black-box… with limited interpretability”. This opacity hinders understanding and trust.  

Meanwhile, **mechanistic interpretability (MI)** in AI seeks to open the black box. A key insight is the *superposition hypothesis*: neural networks pack many features into overlapping directions in activation space. Recent work (e.g. Cunningham et al. 2023) shows that **Sparse Autoencoders** can reconstruct network activations and reveal *sparsely-activating, monosemantic features*. These features often correspond to human-understandable concepts, and they enable identifying “which internal features causally drive behavior”. Mechanistic interpretability also emphasizes **causal interventions**: rather than just visualize features, one actively perturbs them to test their effect on outputs.  

To date, MI methods have been applied to language and vision models, not scientific ML systems. Likewise, SAEs have been used in PINNs only as external model compressors, not to probe *internal computations*. Thus, a critical gap remains: **Can we uncover the internal PINN representations underlying known failure modes?** If PINN failures truly arise from internal superposition of conflicting physics, MI tools may reveal early warning signals and even enable interventions to fix the training. No existing work has applied MI’s SAE/causal toolkit to PINNs. By addressing this gap, we can move beyond treating PINNs as opaque solvers and toward a scientifically grounded understanding of *how* they encode and sometimes break physics.

## 2. Core Research Gap  
Current PINN research catalogs many failure modes (e.g. Krishnapriyan *et al.* 2021 shows PINNs fail on slightly complex PDEs due to ill-conditioned loss landscapes). The “Atlas of PINN Failures” (Prasaath 2026) demonstrates multiple independent pathologies (gradient oppposition, spectral collapse, collocation starvation) that “shatter the physics” in distinct ways. Yet **none of this work examines the PINN’s internal circuitry**. We lack answers to questions like: *Which hidden units or directions encode boundary vs PDE information? Do we see signs of feature competition before collapse?* In other words, the gap is in *mechanistic understanding*: linking PINN failure modes to specific internal representations.  

Specifically:
- **Representation gap:** How, if at all, do PINN activations encode physical quantities like boundary proximity, high-frequency modes, or reaction terms? No study to date has mapped latent dimensions to physical features in PINNs.
- **Pathology gap:** When a PINN enters gradient conflict or collocation starvation, which internal activations change first? Is there a signature in the hidden layers *before* the output goes bad?
- **Causality gap:** Even if we find a correlate (e.g. feature X spikes when the BC loss vanishes), we cannot assume causation. We must intervene on that feature and see if the failure disappears.
- **Prevention gap:** Finally, assuming such mechanistic insights exist, can they be used to **actively steer training** (reweight losses, add points, etc.) in real time to avoid collapse?

This proposal targets these gaps by applying MI tools to PINNs: using SAEs to discover sparse latent features, associating them with physical error metrics, and testing causal hypotheses via interventions. 

## 3. Central Hypothesis  
We hypothesize that **PINN failures are preceded by detectable shifts in sparse internal features**, arising from superposition of conflicting tasks, and that these shifts can be reversed. Concretely:

- The directions in parameter (and activation) space corresponding to data vs. physics constraints **absorb each other’s gradients** as training progresses. In the infinite-width view, Wang *et al.* (2020) showed PINN’s NTK has eigen-directions with **disparate loss convergence rates**, implying some directions dominate optimization. We conjecture those directions correspond to specific sparse features in the network.
- Sparse Autoencoders (SAEs) can **reveal the hidden “superposed” directions**. By training an SAE on the PINN’s layer activations over epochs, we expect to find features that consistently correlate with physical quantities (e.g. PDE residual, boundary mismatch, high-frequency components).
- When a failure mode is imminent (e.g. BC gradients collapse to zero), certain SAE-derived features will reliably spike or fade. Those features are mechanistic *symptoms* of the pathology.
- Crucially, we posit that **intervening on these features** will causally affect the outcome. For example, suppressing the feature linked to BC information may *rescue* the BC performance if done before collapse, while random interventions will not. This is akin to steering the network via counterfactual feature manipulation.
- Overall, we anticipate that an internal representation can provide an **early-warning indicator** of a failure, giving enough lead time to adjust training (e.g. reweight losses, resample points) and thus prevent the final output from degrading.

In summary, PINN training pathologies are not mysterious sudden drops; they emerge from gradual entanglement of features. We aim to expose that entanglement using SAE-based interpretability and use it to maintain PINN reliability.

## 4. Proposed Methodology  

The research will proceed in four phases, each building on the last. Each phase uses controlled experiments and quantitative metrics to ensure scientific rigor.

### Phase I – Controlled PINN Failure Benchmark  
- **PDE selection:** Implement PINNs for a set of canonical PDEs (e.g. 1D Poisson, 1D/2D Burgers’, 1D Allen–Cahn, diffusion/advection) where high-quality reference solutions are known. Include “stiff” variants known to induce spectral bias.
- **Failure modes elicitation:** For each PDE, configure multiple PINNs (varying architecture, initialization, loss weighting) to produce both *successful* and *failing* training runs. For example, following Prasaath *et al.*, we will explicitly induce Gradient Pathology and Collocation Starvation by fixing random seeds or loss weight imbalances.
- **Data collection:** During training, log at every epoch (or fine time intervals): (1) Output predictions $u_\theta(x,t)$; (2) Loss components $\mathcal{L}_{\text{PDE}}, \mathcal{L}_{\text{BC}}, \mathcal{L}_{\text{IC}}$; (3) Gradient norms $\|\nabla_\theta \mathcal{L}_i\|$ and cosine similarities between gradients of different components; (4) *Internal activations* $A_l(x,t)$ from selected hidden layers; and (5) any error metric w.r.t. true solution (for validation).
- **Spectral analysis:** Compute Fourier spectra of both the solution error and the activations to quantify spectral bias. This phase yields a rich dataset linking training state to physical error, serving as our benchmark for the next phases.

### Phase II – Sparse Representation Discovery  
- **Activation dataset:** Use the logged activations $\{A_l^{(t)}\}$ over time as a *dataset*. For one or more intermediate layers $l$, flatten the activations spatially/time or aggregate them as feature vectors across points.
- **Sparse Autoencoder (SAE):** Train a sparse autoencoder $A \approx D(E(A))$ on this dataset. The encoder $E(\cdot)$ maps activations to a higher-dimensional sparse latent vector $z$, while the decoder reconstructs $A$. We enforce sparsity (e.g. via L1 penalty or KL divergence) so that each latent dimension $z_i$ activates only for specific patterns.
- **Feature interpretation:** For each latent feature $z_i$, examine its activation pattern across space and time, and compute correlations with physical quantities: 
  - High correlations with certain loss components or gradient norms suggest $z_i$ encodes that concept.
  - Correlations with spatial features (e.g. distance to boundary, gradient magnitude of solution) can be identified by plotting $z_i(x)$ versus $x$.
- **Stability checks:** Repeat SAE training across different PINN runs (varying seed or PDE parameters) to see if certain features reappear consistently. Features that only appear in one seed are likely noise; robust features across seeds/PDEs are promising candidates for mechanistic roles.

### Phase III – Causal Mechanistic Validation  
- **Feature interventions:** For each candidate feature $z_k$ believed to represent a meaningful concept (e.g. a “boundary condition mode” or “high-frequency mode”), we will insert it into the PINN computation graph and intervene during training: 
  - *Suppress:* set $z_k = 0$ (or another constant) in the forward pass.
  - *Amplify:* multiply $z_k$ by a factor $\alpha > 1$.
  - *Control:* manipulate an unrelated feature of similar magnitude.
- **Effect measurement:** Compare the PINN’s output and losses under each intervention. A causal feature should, for instance, affect the corresponding loss component: suppressing a “BC feature” should worsen $\mathcal{L}_{\rm BC}$ or even flip it from failure to success, while leaving $\mathcal{L}_{\rm PDE}$ largely unchanged.
- **Counterfactual testing:** We will systematically test whether the predicted effect (based on feature interpretation) matches reality. For example, if SAE suggests $z_{137}$ captures boundary information, then $z_{137}\leftarrow 0$ should *recover* boundary satisfaction if boundary starvation was the failure. Crucially, randomizing $z_{137}$ or interfering with an irrelevant feature should not recover the failure, confirming specificity.
- **Mechanistic scoring:** We will quantify causal strength (e.g. how much $\Delta \mathcal{L}_{\rm BC}$ changes) and specificity (the ratio of effect on target loss vs. other losses) for each feature. Only features passing these causal tests will be retained as verified mechanistic signals.

### Phase IV – Adaptive Failure-Prevention Controller  
- **Early-warning detection:** Using the validated features $z_k(t)$, train a lightweight monitor (e.g. an LSTM or threshold rule) to predict “PINN failure in near future.” Compare its predictions to conventional alarms (e.g. loss plateauing). The key metric is lead time: how many epochs before a failure can $z_k$‑signals warn us?
- **Closed-loop interventions:** Incorporate the monitor into the training loop. When a danger signal is triggered (e.g. “boundary feature collapsing”), autonomously apply a corrective action. Possible actions include: 
  - **Re-weight losses:** Increase $\lambda_{\rm BC}$ or apply GradNorm-style balancing to counteract the collapse.
  - **Adaptive sampling:** Add or upsample collocation points in affected regions (as in failure-informed PINNs).
  - **Architecture tweaks:** Temporarily increase capacity for that feature (e.g. adding Fourier features if high-freq is suppressed).
- **Evaluation:** Compare this adaptive PINN against baselines (standard PINN and existing adaptive methods like GradNorm) on a suite of PDEs. Metrics include final solution error, conservation violation, convergence stability, and number of rescued failures. The goal is to show that leveraging internal mechanistic signals yields **fewer failures and more reliable solutions** than purely output-based heuristics.

## 5. Evaluation and Metrics  
To rigorously assess our approach, we will use:

- **Physical accuracy:** $L^2$/$L^\infty$ norms of solution error against the true PDE solution.
- **Constraint satisfaction:** PDE residual and boundary/initial residual errors, and any conserved quantities.
- **Optimization diagnostics:** Gradient conflict measures (cosine similarity), NTK condition number, loss component imbalance.
- **Mechanistic quality:** Sparsity of features, reproducibility across runs, and causal effect sizes (e.g. $\Delta\mathcal{L}$ from interventions).
- **Prevention performance:** Failure detection lead time, reduction in failure rate, comparison to baselines.  

Statistical significance will be checked across multiple random seeds and PDE setups. We expect that the SAEs will yield interpretable features (quantified by automated interpretability scores) and that interventions will show clear causal effects (recovering or preventing failure). Ultimately, the adaptive system should show a higher success rate on stiff PDEs than unmodified PINNs.

## 6. Expected Contributions and Significance  
The research will make several key contributions:

- **A PINN failure benchmark dataset:** We will release training trajectories (activations, losses, gradients) for canonical PDEs with annotated success/failure modes. This controlled dataset will be a resource for future analysis.
- **Physics-grounded interpretability framework:** By applying SAEs to PINNs, we will pioneer a new methodology for “opening the black box” of scientific ML. Each discovered feature will be accompanied by its physical interpretation, bridging MI with domain science.
- **Identified internal failure signatures:** We expect to catalog specific latent features that correspond to failure modes (e.g. “boundary-starvation feature”, “spectral-suppression feature”). These are new insights into *how* PINNs internally represent physics and why they fail.
- **Causal validation protocols:** We will demonstrate systematic procedures for intervening on PINN representations. This causal analysis (beyond correlative XAI) will set a standard for testing interpretability in scientific models.
- **Early-warning diagnostic tools:** The project will deliver an activation-based failure detector for PINNs. A positive lead time in warning signals is novel and can be adopted in other multi-objective training contexts.
- **Adaptive PINN controller prototype:** Finally, we aim to build a closed-loop PINN training algorithm that automatically corrects failure precursors. This could inspire a new class of *self-correcting scientific ML algorithms*, moving the field from trial-and-error to robust design.

In the broader context, this work transitions PINN research from *performance tuning* to *mechanistic science*. It aligns with calls for interpretable, trustworthy AI in safety-critical domains. By treating a neural PDE solver like a dynamical system whose inner workings can be probed and steered, we lay groundwork for fully transparent scientific computing. Success here could influence not only PINNs, but any physics-guided model (neural operators, surrogate models, etc.) by showing that **understanding internal representations enables better models**.

## 7. Timeline and Resources  
We will complete the project in roughly 12 months:
1. **Months 1–3:** Construct the benchmark; reproduce known PINN failures and collect training data.  
2. **Months 4–6:** Train SAEs, identify candidate features, and map them to physics metrics.  
3. **Months 7–9:** Conduct causal intervention experiments for validation. Refine feature set.  
4. **Months 10–12:** Develop the adaptive controller and evaluate end-to-end. Prepare publications and release code/data.

All code (PINN experiments, SAE training, intervention scripts) and data will be open-sourced. We will document features in a “Physics–Feature Dictionary” linking latent feature IDs to their interpreted meaning and causal role.

## 8. Conclusion  
This proposal aims to **uncover the “hidden physics” inside PINNs**. By applying sparse, physics-aware interpretability techniques and rigorous causal tests, we hope to move from merely observing PINN failure to *understanding and preventing it*. In doing so, we will create a framework where **neural PDE solvers become diagnosable machines**: one can ask “why is this error occurring?” and actually get a mechanistic answer. This paradigm—training → interpret → intervene → verify—could define the next generation of trustworthy scientific machine learning.  
