# Research Foundation

**Question:** Can we systematically determine *which* data a robot policy needs to improve, and
show that acquiring that data beats acquiring data blindly, at equal cost?

This document covers why the question matters, which assumptions in the original brief do not
survive scrutiny, what prior work has already established, where the remaining gap is, and what
the smallest experiment that could falsify our thesis looks like.

**Epistemic labels used throughout:**
- **[Verified]**: stated in a cited paper or official documentation, and checked on 2026-11-26.
- **[Hypothesis]**: our conjecture, to be tested.
- **[Design]**: a choice we made, with the reasoning given.

No experimental results exist yet.

---

## 1. Thesis in one paragraph

**[Hypothesis]** Robot policies fail non-uniformly across the conditions they face. Only *some*
of those failures are caused by missing training data. Others come from sensing limits
(unobservable state), expert limits (the task is infeasible as demonstrated), or model capacity.
A system that (a) localizes failures over a controllable condition space, (b) separates
data-fixable failures from the others, and (c) allocates a cost-aware acquisition budget
accordingly should reach a given performance with less acquisition cost than uniform or naive
failure-driven collection. **The novel, testable part is (b) and (c)**, not "collect data where
the robot fails". That alone has already been shown (§3).

## 2. Challenging the brief's assumptions

| # | Assumption in the brief | Problem | What we do instead |
|---|-------------------------|---------|--------------------|
| A1 | "Collect the right data" fixes failures. | Many failures are not data-attributable. If friction is not observable, *no amount* of demos fixes a friction-induced slip for a memoryless policy. A more accurate brand line is "diagnose before you collect". | Plant non-data causes on purpose (latent friction, an infeasible band) and test whether the system tells them apart (H3, H4). |
| A2 | Failure-driven acquisition is the new idea. | AdaDemo (2024) already compared failure-targeted and uniform acquisition at roughly equal budget. DAgger (2011) is the canonical "collect where the learner fails" method. | Treat AdaDemo-style replay (S3b) and DAgger (S6) as **baselines** to beat, not as our contribution. |
| A3 | "300 selected ≈ 1,000 random" is the efficiency target. | The ratio depends heavily on how skewed and redundant the random set is. Against a skewed collection distribution, almost anything beats it. Against uniform coverage, gains shrink. Subset selection is also well covered (CUPID, DemInf, DataMIL, Demo-SCORE). | Report ρ(τ) in two regimes (skewed R1, uniform R2). Only R2 decides H1. Subset selection is a secondary experiment (Exp 3b). |
| A4 | Equal budget = equal number of demos. | Demos in hard conditions cost more: the expert fails more, and episodes run longer. Failure-driven methods also spend extra evaluation and diagnosis compute. | Budget in **sim steps including rejected attempts**. Also report an equal-*total*-cost control (S2⁺). |
| A5 | Cube lifting is a good testbed as-is. | State-based BC with a scripted expert nearly saturates lift. robomimic's reported Lift results are near-saturated at default settings. No failures means nothing to diagnose. | Calibrate condition ranges and work in a low-data regime (N₀ = 100) until the baseline success landscape is non-flat (EXPERIMENT_DESIGN §3). |
| A6 | Isaac Lab is the right simulator. | Strong for GPU-parallel evaluation and for NVIDIA ecosystem alignment. But [Verified] Isaac Sim needs an RTX GPU ("GPUs without RT Cores (A100, H100) are not supported"), Linux or Windows, driver ≥ 595.58.03 for Isaac Sim 6.2, and it does not run on macOS (the dev laptop). Many university clusters are A100/H100. | Keep Isaac Lab as primary *if* Phase A confirms a suitable GPU. Keep the code simulator-agnostic (ARCHITECTURE AD1), with ManiSkill3 as the pre-planned fallback and second-simulator replication. |
| A7 | RL (PPO/SAC) or offline RL might be the starting policy. | The question is about *demonstrations*. RL generates its own data, which confounds attribution. Offline RL adds reward design and value-learning variance. | BC first, held fixed. Robustness of conclusions is checked with action chunking and Diffusion Policy (Exp 7). |
| A8 | Simulation acquisition ≈ real acquisition. | In sim we can set friction exactly and the expert is free and consistent. Real operators control only some conditions (object choice, placement, maybe lighting), and human demos are noisy and multimodal. | Every sim claim is scoped as "under controllable conditions with a scripted expert". The real-world gap is listed as a commercialization evidence gap (§7). |
| A9 | One task can support the startup thesis. | One task supports a *method* claim at most. Platform claims need cross-task and cross-embodiment evidence. | The roadmap gates expansion on a reproducible v1 (ROADMAP M9+). |

## 3. Positioning and novelty

**What is already established [Verified]:**
- *Learner-driven data helps imitation learning.* DAgger; and human-gated variants (HG-DAgger,
  EnsembleDAgger, ThriftyDAgger, Sirius) collect data where the policy errs, at state level.
- *Failure-targeted demo expansion beats uniform at about equal demo counts* (AdaDemo, multi-task:
  RLBench 64.1% vs 60.4%, Adroit 71.8% vs 67.4%, as reported). Evaluation rollouts were not
  charged to the budget.
- *Curating existing demos by their effect on policy success works* (CUPID via influence
  functions, Demo-SCORE via rollout classifiers, DemInf via mutual information, DataMIL via
  datamodels, SCIZOR via self-supervised filtering).
- *Diversity over environments matters more than demos per environment* (Lin et al. scaling
  laws). *Factors of variation generalize compositionally*, so factored collection is efficient
  (Gao et al.). *Some factors are much harder than others* (Xie et al.; Saxena et al.).
- *Targeted sampling of conditions beats uniform randomization* in RL (Active Domain
  Randomization, Prioritized Level Replay, PAIRED).
- *Uniform random is a hard baseline for active learning*, and many reported gains vanish under
  careful tuning (Munjal et al.; Mittal et al.).

**The gap we target [Hypothesis]:**
1. **Root-cause-aware allocation.** Existing failure-driven methods treat every failure as a data
   problem. We separate coverage gaps from observability gaps, expert infeasibility and capacity
   limits, and validate the separation against **intervention-defined ground truth** (Exp D).
   We found no prior work that does this for robot data acquisition.
2. **Single-task, continuous condition space, physical (non-visual) factors.** AdaDemo
   confounds task allocation with state targeting. Vision-based failure discovery such as RoboMD
   cannot see friction or mass.
3. **Cost-honest protocol.** All acquisition, evaluation and diagnosis costs are charged.
   Compute-matched and shuffled-target controls are included. The protocol is pre-registered,
   two regimes are run, and paired multi-seed statistics are used.
4. **Acquisition and curation in one benchmark** (Exp 2 vs Exp 3b).

**What would make this not novel:** if S5 does not beat S3b (AdaDemo-style) in R2, our
contribution reduces to a benchmark and protocol. That is still useful, but it is a weaker paper
and a weaker product thesis. We will report it if it happens.

## 4. Literature review

All entries were checked against arXiv and/or venue pages on 2026-11-26. "Relevance" is our
assessment. Findings are paraphrased summaries; read the papers before relying on specifics.

### 4.1 Data curation and selection for robot imitation learning

| Paper | Venue / first arXiv | Key finding | Limitation | Relevance |
|-------|---------------------|-------------|------------|-----------|
| **CUPID: Curating Data your Robot Loves with Influence Functions**. Agia, Sinha, Yang, Antonova, Pavone, Nishimura, Itkina, Bohg. [arXiv:2506.19121](https://arxiv.org/abs/2506.19121) | CoRL 2025 · Jun 2025 | Scores each demo by its influence on closed-loop policy success (estimated from rollouts), then filters harmful demos or selects useful ones. | Needs rollouts plus gradient influence; ranks only existing demos. | Strongest subset-selection baseline (Exp 3b stretch). Does no condition-level acquisition. |
| **Robot Data Curation with Mutual Information Estimators (DemInf)**. Hejna, Mirchandani, Balakrishna, Xie, Wahid, Tompson, Sanketi, Shah, Devin, Sadigh. [arXiv:2502.08623](https://arxiv.org/abs/2502.08623) | RSS 2025 · Feb 2025 | Estimates state–action mutual information to score demo quality; dropping low-MI demos helps on mixed-quality data. | Policy-agnostic proxy; cannot see where the policy fails. | Subset-selection baseline. |
| **DataMIL: Selecting Data for Robot Imitation Learning with Datamodels**. Dass, Khaddaj, Engstrom, Madry, Ilyas, Martín-Martín. [arXiv:2505.09603](https://arxiv.org/abs/2505.09603) | ICLR 2026 · May 2025 | Applies datamodels to select task-relevant data from large prior datasets, using a tractable surrogate objective. | Retrieves from existing data; needs a target-task signal. | Attribution approach we may borrow for "which demos matter". |
| **Curating Demonstrations using Online Experience (Demo-SCORE)**. Chen, Lessing, Liu, Finn. [arXiv:2503.03707](https://arxiv.org/abs/2503.03707) | RSS 2025 · Mar 2025 | A classifier trained on the policy's successful vs failed rollouts prunes unreliable demos; reports 15–35% absolute gains over using all demos. | Removal only. | Closest *curation* analogue of using rollouts to steer data. |
| **SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning**. Zhang, Xie, Liu, Shah, Wan, Fan, Zhu. [arXiv:2505.22626](https://arxiv.org/abs/2505.22626) | ICRA 2026 (lab page) · May 2025 | Frame-level removal of suboptimal or redundant state–action pairs using a progress predictor plus deduplication. | Assumes mostly-good data; hand-set thresholds. | Redundancy-removal reference. |
| **Re-Mix (PMLR: "ReMix"): Optimizing Data Mixtures for Large Scale Imitation Learning**. Hejna, Bhateja, Jiang, Pertsch, Sadigh. [arXiv:2408.14037](https://arxiv.org/abs/2408.14037) | CoRL 2024 · Aug 2024 | Distributionally robust optimization of domain weights for BC on Open X-Embodiment data beats uniform and human mixtures. | Domain-level, not condition-level. | A robust-optimization-style condition-reweighting idea. |
| **Data Quality in Imitation Learning**. Belkhale, Cui, Sadigh. [arXiv:2306.02437](https://arxiv.org/abs/2306.02437) | NeurIPS 2023 · Jun 2023 | Frames data quality as distribution shift: *action divergence* and *transition diversity*. More state diversity is not always better. | Analysis, no acquisition algorithm. | Theory for why targeted data can beat volume. |
| **What Matters in Learning from Large-Scale Datasets for Robot Manipulation**. Saxena, Bronars, Ranawaka Arachchige, Wang, Shin, Nasiriany, Mandlekar, Xu. [OpenReview](https://openreview.net/forum?id=LqhorpRLIm), [arXiv:2506.13536](https://arxiv.org/abs/2506.13536) | ICLR 2025 · arXiv Jun 2025 | Procedurally generated datasets show camera pose and spatial layout are the key diversity axes; retrieval informed by this gives large gains. | Fixed datasets; no failure loop. | Supports condition-level thinking. |
| **STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning**. Memmel, Berg, Chen, Gupta, Francis. [arXiv:2412.15182](https://arxiv.org/abs/2412.15182) | ICLR 2025 · Dec 2024 | Retrieves relevant sub-trajectories (DINOv2 + subsequence DTW) to augment training. | Similarity-driven, not failure-driven. | Retrieval-style selection reference. |
| **Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets**. Du, Nair, Sadigh, Finn. [arXiv:2304.08742](https://arxiv.org/abs/2304.08742) | RSS 2023 · Apr 2023 | Retrieves relevant data from unlabeled prior datasets for few-shot IL. | Same. | Same. |

### 4.2 Failure-driven and active acquisition; interactive imitation learning

| Paper | Venue / first arXiv | Key finding | Limitation | Relevance |
|-------|---------------------|-------------|------------|-----------|
| **AdaDemo: Data-Efficient Demonstration Expansion for Generalist Robotic Agent**. Mu, Guo, Xu, Goyal, Su, Fox, Garg. [arXiv:2404.07428](https://arxiv.org/abs/2404.07428) | ICLR 2024 workshop (DPFM); ISRR listing unconfirmed · Apr 2024 | Iteratively evaluates and adds demos to low-success tasks and to failed initial states. Beats uniform at about equal demo counts; claims about ½–⅓ of the data for equal performance. | Multi-task (task allocation confounded with state targeting); rejection-only replay of failed seeds; evaluation not charged. | **Closest prior art.** Basis of baseline S3b. |
| **Active Fine-Tuning of Multi-Task Policies (AMF)** (formerly "…of Generalist Policies"). Bagatella, Hübotter, Martius, Krause. [arXiv:2410.05026](https://arxiv.org/abs/2410.05026) | ICML 2025 · Oct 2024 | Chooses which task to request demos for by maximizing information gain about the expert policy, with guarantees. | Task-level; depends on uncertainty estimates. | Principled active-acquisition alternative (possible future baseline over cells). |
| **Efficient Data Collection for Robotic Manipulation via Compositional Generalization**. Gao, Xie, Xiao, Finn, Sadigh. [arXiv:2403.05110](https://arxiv.org/abs/2403.05110) | RSS 2024 · Mar 2024 | Policies compose across factors of variation, so factored collection gives broad generalization with less data. | Non-adaptive. | Baseline S4b and design of hold-out set H. |
| **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)**. Ross, Gordon, Bagnell. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) | AISTATS 2011 | Aggregating expert labels on learner-visited states fixes BC's compounding error, with no-regret guarantees. | Needs a queryable expert. | Baseline S6; our state-feedback expert (AD6) makes it possible. |
| **HG-DAgger: Interactive Imitation Learning with Human Experts**. Kelly, Sidrane, Driggs-Campbell, Kochenderfer. [arXiv:1810.02890](https://arxiv.org/abs/1810.02890) | ICRA 2019 | Human-gated takeover outperforms DAgger and BC; also learns a risk threshold. | Human in the loop. | Failure-triggered collection lineage. |
| **EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning**. Menda, Driggs-Campbell, Kochenderfer. [arXiv:1807.08364](https://arxiv.org/abs/1807.08364) | IROS 2019 | Ensemble variance decides when the expert intervenes. | | Our ensemble-based epistemic signal. |
| **ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning**. Hoque, Balakrishna, Novoseller, et al. [arXiv:2109.08273](https://arxiv.org/abs/2109.08273) | CoRL 2021 | Queries a human only in novel (ensemble disagreement) or risky (learned Q) states, within a budget. | State-level. | Budgeted, targeted requests. |
| **Robot Learning on the Job (Sirius)**. Liu, Nasiriany, Zhang, Bao, Zhu. [arXiv:2211.08416](https://arxiv.org/abs/2211.08416) | RSS 2023 | Deployment-time human corrections plus trust-weighted training. | State-level, human-guided. | Real-world failure-driven loop. |
| **IntervenGen**. Hoque, Mandlekar, Garrett, Goldberg, Fox. [arXiv:2405.01472](https://arxiv.org/abs/2405.01472) | venue unverified · May 2024 | Amplifies a few human interventions into many corrective demos across new scenes. | State-level corrections. | Data generation for failure states. |
| **RoboMD** (failure discovery). Sagar, Duan, Vasudevan, Zhou, Ben Amor, Fox, Senanayake. [arXiv:2412.02818](https://arxiv.org/abs/2412.02818) | preprint · Dec 2024 | RL search over a vision-language embedding finds the environment conditions where a policy is vulnerable; fine-tuning on them helps. | Visual factors; collection protocol and random baseline not confirmed by us. | Closest *condition-level* work; cannot see latent physics. |
| **Unsupervised Discovery of Failure Taxonomies from Deployment Logs**. Gupta, Ciftci, Bansal. [arXiv:2506.06570](https://arxiv.org/abs/2506.06570) | preprint · Jun 2025 | Clusters vision-language failure explanations into failure modes that guide targeted collection. | Navigation; reported numbers not checked by us. | Learned taxonomy (our later replacement for hand-coded F1–F8). |
| **RESample**. Xue, Lu, Wu, et al. [arXiv:2510.17640](https://arxiv.org/abs/2510.17640) | preprint · Oct 2025 | Coverage-guided exploration-plus-recovery data augmentation on LIBERO. | State-level; baselines not checked by us. | Related coverage idea. |

*Also seen in search, abstract-level only (not reviewed):* Robotic Data Inoculation
([arXiv:2610.39054](https://arxiv.org/abs/2610.39054), Oct 2026); Missing Bridges / AALT
([arXiv:2609.18004](https://arxiv.org/abs/2609.18004), Sep 2026); SAFARI
([arXiv:2011.09586](https://arxiv.org/abs/2011.09586), 2020); GSWorld
([arXiv:2510.20813](https://arxiv.org/abs/2510.20813)). **Action item (M0.5):** read these
2025–26 papers before freezing the pre-registration. The field is moving fast.

### 4.3 Active learning, data valuation, attribution, pruning (general ML)

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **Towards Robust and Reproducible Active Learning Using Neural Networks**. Munjal et al. [arXiv:2002.09564](https://arxiv.org/abs/2002.09564) | CVPR 2022 | Well-tuned random sampling matches or beats many active learning methods. | Why S2 is a serious baseline and tuning budgets are equalized. |
| **Parting with Illusions about Deep Active Learning**. Mittal, Tatarchenko, Çiçek, Brox. [arXiv:1912.05361](https://arxiv.org/abs/1912.05361) | arXiv 2019 | Active learning gains shrink with augmentation and semi-supervised learning. | Same caution. |
| **Active Learning for CNNs: A Core-Set Approach**. Sener, Savarese. [arXiv:1708.00489](https://arxiv.org/abs/1708.00489) | ICLR 2018 | Batch active learning as k-center coverage. | Baseline S4. |
| **BADGE**. Ash, Zhang, Krishnamurthy, et al. [arXiv:1906.03671](https://arxiv.org/abs/1906.03671) | ICLR 2020 | Gradient embeddings plus k-means++ balance uncertainty and diversity. | Possible hybrid baseline. |
| **Understanding Black-box Predictions via Influence Functions**. Koh, Liang. [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) | ICML 2017 | Approximates the effect of up-weighting a training point without retraining. | Basis of CUPID-style attribution. |
| **Data Shapley**. Ghorbani, Zou. [arXiv:1904.02868](https://arxiv.org/abs/1904.02868) | ICML 2019 | Shapley values for per-datum value. | Data valuation reference. |
| **Datamodels** (arXiv: "…Predicting Predictions from Training Data"). Ilyas, Park, Engstrom, et al. [arXiv:2202.00622](https://arxiv.org/abs/2202.00622) | ICML 2022 | Linear models over training-subset inclusion predict outputs well. | Our "local data-scaling probe" is a cheap cell-level datamodel. |
| **TRAK: Attributing Model Behavior at Scale**. Park, Georgiev, Ilyas, et al. [arXiv:2303.14186](https://arxiv.org/abs/2303.14186) | ICML 2023 | Efficient attribution near datamodel quality. | Stretch attribution method. |
| **Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning**. Sorscher, Geirhos, Shekhar, et al. [arXiv:2206.14486](https://arxiv.org/abs/2206.14486) | NeurIPS 2022 | Good pruning can beat power laws. *Keep easy examples when data is scarce, hard ones when it is abundant.* | Predicts that "target the hardest cells" may be wrong in the low-data regime. Our fixability × cost weighting and floor address this. |

### 4.4 Curriculum, environment design, domain randomization

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **Active Domain Randomization**. Mehta, Diaz, Golemo, et al. [arXiv:1904.04762](https://arxiv.org/abs/1904.04762) | CoRL 2019 | Learning where to sample randomization parameters beats uniform. | RL analogue of targeted condition sampling. |
| **Prioritized Level Replay**. Jiang, Grefenstette, Rocktäschel. [arXiv:2010.03934](https://arxiv.org/abs/2010.03934) | ICML 2021 | Replaying levels with high learning potential improves efficiency and generalization. | "Learning potential" ≈ our fixability. |
| **Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED)**. Dennis, Jaques, Vinitsky, et al. [arXiv:2012.02096](https://arxiv.org/abs/2012.02096) | NeurIPS 2020 | Regret-maximizing environment generation gives solvable but hard curricula. | Regret ≈ expert success − policy success, which our priority uses. |
| **Solving Rubik's Cube with a Robot Hand (ADR)**. OpenAI, Akkaya, et al. [arXiv:1910.07113](https://arxiv.org/abs/1910.07113) | arXiv 2019 | Automatic domain randomization expands ranges as performance improves. | Performance-driven condition expansion. |
| **Domain Randomization for Transferring DNNs from Simulation to the Real World**. Tobin, Fong, Ray, et al. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | IROS 2017 | Heavy randomization enables sim-to-real transfer. | Baseline S2's lineage. |
| **Automatic Curriculum Learning for Deep RL: A Short Survey**. Portelas, Colas, Weng, et al. [arXiv:2003.04664](https://arxiv.org/abs/2003.04664) | IJCAI 2020 | Taxonomy of automatic curriculum learning by what is controlled and what signal drives it. | Positions our method. |

### 4.5 Distribution shift, worst-group robustness, uncertainty

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**. Sagawa, Koh, Hashimoto, Liang. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | ICLR 2020 | Group DRO helps worst-group accuracy only with strong regularization. | Worst-cell metric; cells play the role of groups. |
| **Just Train Twice**. Liu, Haghgoo, Chen, et al. [arXiv:2107.09044](https://arxiv.org/abs/2107.09044) | ICML 2021 | Upweighting errors from a first model improves worst-group accuracy. | Training-time analogue of failure-driven acquisition. A cheap control to consider: *reweighting instead of acquiring*. |
| **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**. Lakshminarayanan, Pritzel, Blundell. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474) | NeurIPS 2017 | Ensembles give calibrated uncertainty and flag out-of-distribution inputs. | Epistemic signal in diagnosis. |

### 4.6 Failure detection and diagnosis in robotics

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **REFLECT**. Liu, Bahety, Song. [arXiv:2306.15724](https://arxiv.org/abs/2306.15724) | CoRL 2023 | LLM over hierarchical multisensory summaries explains failures. | Future learned failure explanation. |
| **AHA**. Duan, Pumacay, Kumar, et al. [arXiv:2410.00371](https://arxiv.org/abs/2410.00371) | ICLR 2025 | VLM fine-tuned on procedurally generated failures detects and explains them. | Observation-only failure detection (later phase). |
| **Unpacking Failure Modes of Generative Policies (Sentinel)**. Agia, Sinha, Yang, et al. [arXiv:2410.04640](https://arxiv.org/abs/2410.04640) | CoRL 2024 | Separates erratic failures (action consistency) from task-progress failures (VLM). | Failure types need different fixes, consistent with our taxonomy and diagnosis split. |
| **Can We Detect Failures Without Failure Data? (FAIL-Detect)**. Xu, Nguyen, Dixon, et al. [arXiv:2503.08558](https://arxiv.org/abs/2503.08558) | RSS 2025 | Sequential out-of-distribution scores learned from successes, with conformal thresholds. | Path from privileged labels to observation-only detection. |
| **Adaptive Stress Testing**. Lee, Mengshoel, Saksena, et al. [arXiv:1811.02188](https://arxiv.org/abs/1811.02188) | JAIR 2020 | Finds likely failures with RL or tree search. | Failure search over Θ. |
| **A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems**. Corso, Moss, Koren, et al. [arXiv:2005.02979](https://arxiv.org/abs/2005.02979) | JAIR 2021 | Falsification and failure-probability estimation methods. | Same. |

### 4.7 Evaluation methodology

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **Deep RL at the Edge of the Statistical Precipice**. Agarwal, Schwarzer, Castro, et al. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264) | NeurIPS 2021 | Use IQM, stratified bootstrap CIs and performance profiles. | Our statistics (EXPERIMENT_DESIGN §11). |
| **Robot Learning as an Empirical Science: Best Practices for Policy Evaluation**. Kress-Gazit, Hashimoto, Kuppuswamy, et al. [arXiv:2409.09491](https://arxiv.org/abs/2409.09491) | arXiv 2024 | Report conditions, multiple metrics, statistics and failure modes. | Reporting standard. |
| **Evaluating Real-World Robot Manipulation Policies in Simulation (SIMPLER)**. Li, Hsu, Gu, et al. [arXiv:2405.05941](https://arxiv.org/abs/2405.05941) | CoRL 2024 | Sim evaluation can correlate with real-world performance if control and visual gaps are reduced. | Sim evaluation as a real-world proxy (commercial path). |
| **A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation**. TRI LBM Team. [arXiv:2507.05331](https://arxiv.org/abs/2507.05331) | arXiv 2025 | Statistically rigorous sim and real policy comparison. | Model for our statistics. |
| **Efficient Evaluation of Multi-Task Robot Policies With Active Experiment Selection**. Anwar, Gupta, Merchant, et al. [arXiv:2502.09829](https://arxiv.org/abs/2502.09829) | CoRL 2025 | Cost-aware active selection of evaluation experiments. | The twin problem: we could cut validation cost with active evaluation (a later optimization). |

### 4.8 Policies, data generation, scaling

| Paper | Venue | Key finding | Relevance |
|-------|-------|-------------|-----------|
| **What Matters in Learning from Offline Human Demonstrations for Robot Manipulation (robomimic)**. Mandlekar et al. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298) | CoRL 2021 | Data quality, observation space and history matter a lot. Lift is nearly solved by default. | HDF5 conventions; our calibration requirement. |
| **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**. Chi et al. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | RSS 2023 (extended journal version on arXiv) | Diffusion over action sequences handles multimodality. | Exp 7 policy class. |
| **MimicGen**. Mandlekar, Nasiriany, Wen, et al. [arXiv:2310.17596](https://arxiv.org/abs/2310.17596) | CoRL 2023 | Generates thousands of demos from a few human demos by transforming object-centric segments. | Future generator conditioned on target Θ. **Non-commercial code license** (§7). |
| **SkillMimicGen**. Garrett, Mandlekar, Wen, Fox. [arXiv:2410.18907](https://arxiv.org/abs/2410.18907) | CoRL 2024 | Skill-segmented generation. | Same. |
| **DexMimicGen**. Jiang, Xie, Lin, et al. [arXiv:2410.24185](https://arxiv.org/abs/2410.24185) | ICRA 2025 | Bimanual dexterous generation. | Later embodiments. |
| **Data Scaling Laws in Imitation Learning for Robotic Manipulation**. Lin, Hu, Sheng, Wen, You, Gao. [arXiv:2410.18647](https://arxiv.org/abs/2410.18647) | ICLR 2025 | Generalization scales roughly as a power law in environments and objects; diversity beats repetition. | Exp 1 framing. |
| **Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation**. Xie, Lee, Xiao, Finn. [arXiv:2307.03659](https://arxiv.org/abs/2307.03659) | ICRA 2024 | Factors differ widely in difficulty; most pairs don't compound. | Supports a per-factor analysis and hold-out set H. |

### 4.9 Frameworks and datasets: what to reuse

| Framework / dataset | Status (verified 2026-11-26) | License | Use in this project |
|---------------------|------------------------------|---------|---------------------|
| **Isaac Lab** ([repo](https://github.com/isaac-sim/IsaacLab); paper: Mittal et al., *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning*, [arXiv:2511.04831](https://arxiv.org/abs/2511.04831); predecessor Orbit, RA-L 2023, [arXiv:2301.04195](https://arxiv.org/abs/2301.04195)) | v3.0.0, 2026-10-20, on Isaac Sim 6.2. New Newton backend and "kit-less" mode. Lift task renamed `IsaacContrib-Lift-Cube-Franka[-IK-Rel]`. Quaternions now XYZW. | BSD-3 (Isaac Sim parts proprietary) | **Primary sim** if the GPU qualifies |
| **Isaac Lab Mimic** | Deprecated in 3.0 in favour of AutoData; requires cuRobo (proprietary terms) | Apache-2.0 + deps | Not used in v1 |
| **ManiSkill3** (Tao et al., [arXiv:2410.00425](https://arxiv.org/abs/2410.00425)) | v3.0.1, 2026-04-21. GPU sim on Linux + NVIDIA; macOS CPU only. PickCube-v1 plus Panda motion-planning demos. | Apache-2.0 | **Fallback sim** and Exp 7 replication |
| **robosuite** | v1.5.2, 2025-12-24; MuJoCo; macOS and Linux; Lift with Panda | MIT | Local Mac learning sandbox; second fallback |
| **robomimic** | v0.5.0 on GitHub (PyPI stuck at 0.3.0, so install from source) | MIT | HDF5 format; Exp 7 reference trainers |
| **MimicGen** | v1.0.0 (2024) | NVIDIA Source Code License: **non-commercial** | Research only |
| **LeRobot** | v0.7.0, 2026-11-05; LeRobotDataset v3.0 (Parquet + MP4) | Apache-2.0 | Export format for interoperability (later) |
| **LIBERO** (Liu et al., [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)) | Benchmark | code MIT, data CC-BY-4.0 | Possible multi-task extension |
| **RoboCasa** (Nasiriany et al., RSS 2024, [arXiv:2406.02523](https://arxiv.org/abs/2406.02523)) | Benchmark | code MIT, assets/data CC-BY-4.0 | Later, harder tasks |
| **Open X-Embodiment** ([arXiv:2310.08864](https://arxiv.org/abs/2310.08864)) | Real multi-robot data | Apache-2.0 / CC-BY-4.0 | Real-data validation of curation ideas (later) |
| **DROID** (Khazatsky et al., [arXiv:2403.12945](https://arxiv.org/abs/2403.12945)) | 76k trajectories, 350 h | **unverified**, check before use | Same |

**To verify before citing** (known to us, not re-checked in this pass): DART noise injection for
imitation learning (Laskey et al., 2017), and ACT / action chunking (Zhao et al., 2023).

## 5. The smallest meaningful proof of concept

**[Design]** One task (Franka cube lift), state observations, a scripted privileged expert,
an MLP BC policy, 3 condition factors (2 observable, 1 latent), 36 cells, and two regimes.
The decisive experiment is **Exp 2 in R2 at B = 100 demo-equivalents, 10 seeds**: S5 vs S2,
S3b and S2⁺. Everything else is either support (Exp 1, 3, 4, 5, D) or robustness (Exp 6, 7).

Why this is the minimum *meaningful* version:
- With fewer than 2 observable factors there is no coverage structure to exploit.
- Without a latent factor, the diagnosis claim (our novelty) cannot be tested.
- Without regime R2 and S3b, a positive result would not survive review: it could be explained
  by data skew or by prior work.
- Vision, more tasks and a learned acquisition function add cost without testing anything new
  at this stage.

### Simulator, policy and dataset choices

| Choice | Decision | Alternatives and why not (yet) |
|--------|----------|--------------------------------|
| Simulator | Isaac Lab 3.0.0 + Isaac Sim 6.2 (PhysX) if Phase A finds an RTX GPU ≥ 16 GB with driver ≥ 595.58.03 and a container runtime. Otherwise **ManiSkill3**. Isaac Lab kit-less Newton is a possible middle path that needs a spike to check contact sensing and friction randomization. | robosuite/MuJoCo (CPU-only eval is slow for ~10⁶-episode budgets, but fine for learning on the Mac); Genesis or others (less mature for this use) |
| Policy | MLP BC (EXPERIMENT_DESIGN §9) | PPO/SAC (wrong question); offline RL (confounds); Diffusion Policy (Exp 7 only) |
| Expert | Scripted, state-feedback, privileged, μ-aware | Teleoperation (no hardware; noisy); RL expert (backup if scripting fails); MimicGen (license + overkill) |
| Data | Generated on demand per strategy; manifests over a shared episode store | Fixed public datasets (cannot be acquired *into*, which defeats the question) |

## 6. Risks and limitations

| Risk | Likelihood | Impact | Mitigation / early detection |
|------|-----------|--------|------------------------------|
| No RTX GPU, old driver, or no container runtime on the cluster | **High** (many clusters are A100/H100) | Blocks Isaac Sim | `scripts/check_env.sh` in week 1; ManiSkill3 fallback (AD1 keeps cost low) |
| Isaac Lab 3.0 is weeks old: bugs, API churn, sparse docs | Medium | Slows M1–M2 | Pin 3.0.0 exactly; record workarounds; avoid upgrading mid-study |
| Task too easy (no failures) or too hard | Medium | No signal | Calibration gate (EXPERIMENT_DESIGN §3) |
| Friction has no effect (stiff gripper, light cube) | Medium | Loses the latent-factor test | Calibrate mass; increase slip sensitivity; fall back to mass as the latent factor |
| Uniform DR (S2) is as good as S5 | Medium–High | H1 fails | It is a legitimate result; the pre-registration commits us to report it |
| S5 overfits to validation; leakage | Medium | Invalid claims | Structural guards (ARCHITECTURE §5); separate test seeds; frozen analysis |
| Compute confound | High if ignored | Unfair comparison | S2⁺ control; all costs logged |
| Sim artifacts (PhysX contact) drive results | Medium | Non-transferable | Physics sanity tests; Exp 7 cross-simulator replication |
| Scripted expert ≠ human data (unimodal, perfect) | Certain | Limits external validity | State it explicitly; noise-injected expert variant as a stress test; later, human-teleop data |
| Novelty erosion (fast-moving 2025–26 field) | Medium | Weaker paper | M0.5 literature refresh; position on diagnosis and protocol |
| Author is new to robotics | Certain | Schedule | Learning roadmap tied to milestones (ROADMAP) |

**Limitations we accept in v1:** simulation only; one task; a privileged scripted expert;
state observations; a discrete cell grid (the continuous variant is an ablation); a hand-coded
taxonomy.

## 7. From research to product: evidence gaps, IP and licensing

### Three kinds of claims, kept separate
| Kind | What v1 can show | What it cannot show |
|------|------------------|---------------------|
| **Scientific** | Whether diagnosis-aware, cost-aware acquisition beats uniform and failure-replay baselines *in simulation, under controllable conditions, with a scripted expert*. | Real-world effectiveness; generality across tasks. |
| **Engineering feasibility** | That the loop (evaluate → diagnose → allocate → generate → retrain) can be automated and costed reproducibly. | Integration with customers' heterogeneous stacks, real logs and teleop pipelines. |
| **Commercial** | Nothing directly. | Demand, willingness to pay, the size of the data-cost pain, competitive moat. |

### Evidence needed before commercialization
1. **Real-world validation:** the same loop on a real arm (a university lab or partner), with
   human teleop demos, where only *controllable* conditions (object choice, placement, lighting)
   are acquired. This is the biggest gap.
2. **Generality:** ≥ 3 tasks and ≥ 2 embodiments, including at least one vision policy and one
   multi-step task.
3. **Logs-only diagnosis:** the diagnosis must work without privileged sim state (from
   observations, proprioception and failure detectors), because customers don't have ground truth.
4. **Customer discovery:** 20–30 interviews with robotics teams about data-collection budgets,
   current tooling, and whether "what to collect next" is a decision they would pay to automate.
   Quantify the cost per demo hour and the share of collection that turns out redundant.
5. **Baseline economics:** show savings net of evaluation and diagnosis costs (our S2⁺ logic,
   applied to real costs).
6. **Competitive landscape:** robotics data platforms and evaluation tools change fast.
   Survey them at the time of commercialization; this document makes no claims about them.

### IP, licensing and data considerations (not legal advice)
- **University IP:** work done with university resources (the professor's GPUs, lab time,
  funding) may be subject to the university's IP policy and the sponsor's terms. Before filing
  patents, publishing with commercial intent, or incorporating, consult the university's
  technology transfer office and agree on authorship and IP with the advisor *early*.
  Publishing first can also affect patentability in some jurisdictions.
- **Software licenses:**
  - Isaac Lab core is BSD-3, but Isaac Sim is NVIDIA proprietary (a "limited, non-exclusive,
    revocable" license). Review its terms for commercial and hosted-service use.
  - **MimicGen code is non-commercial** (research/evaluation only).
  - cuRobo (needed by Isaac Lab Mimic) has proprietary terms.
  - ManiSkill3 (Apache-2.0), robosuite and robomimic (MIT) and LeRobot (Apache-2.0) are
    permissive.
  - A commercial product should keep NVIDIA-proprietary and non-commercial components
    replaceable. AD1 helps here.
- **Datasets:** LIBERO, RoboCasa and OXE materials are CC-BY-4.0 (attribution required).
  DROID's license is unverified, so check it before use. Customer data would need contracts on
  ownership and on the right to train on it.
- **Keep private:** do not push unpublished results or IP-relevant code to a public repo without
  the advisor's agreement. Create the GitHub repo as **private** initially.

## 8. Success criteria (summary; details in EXPERIMENT_DESIGN and ROADMAP)

- **Engineering:**
  - expert ≥ 95% success in feasible cells
  - taxonomy precision ≥ 85% and coverage ≥ 90%
  - full loop reproducible from one command
  - re-run metrics within CI
- **Scientific (pre-registered):**
  - H1 is supported if S5 − S2 ≥ 5 pp cell-balanced test success (95% CI excludes 0) at ≥ 2 of 3
    budgets in R2.
  - Novelty is supported if S5 > S3b by the same criterion.
  - Diagnosis is supported if Exp D accuracy beats majority-class and Spearman ρ > 0 (CI).
- **Go/no-go after the M6 pilot (5 seeds):** if S5 − S2 in R2 is ≤ 0 with a CI excluding
  +5 pp, stop and re-examine the method before spending headline compute.
