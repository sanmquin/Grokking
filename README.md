# Grokking & Preference Alignment in Transformers 🕸️
Researching the impact of post-training on grokked models.

This repository contains a replication and analysis of the minimal setup required to induce grokking on transformers, as well as a study on preference alignment and surgical representation editing using **Direct Preference Optimization (DPO)**.

---

## Experiment 1: Inducing Grokking on Modular Addition ($x + y \equiv z \pmod{113}$)

We investigate modulo addition under a restricted data regime to observe delayed generalization, weight-decay-induced circuit formation, and distinct training phases.

### Research Goal & Hypothesis
The goal of this experiment is to demonstrate that under high weight decay (regularization parameter $\lambda = 1.0$) and a minimal data fraction of **30%** for training (as utilized in the *"Progress Measures for Grokking via Mechanistic Interpretability"* paper), a 1-Layer Transformer will first transition into a memorizing regime before undergoing a sudden, sharp phase transition (grokking) into a low-norm generalizing circle-rotation representation.

### Model Architecture
To align with the experimental framework of the grokking paper, the model is configured as:
- **Type:** 1-Layer decoder-only Transformer with no Layer Normalization.
- **Dimensionality:** $d_{\text{model}} = 128$, $d_{\text{mlp}} = 512$, $n_{\text{heads}} = 4$, $d_{\text{head}} = 32$.
- **Vocab:** $d_{\text{vocab}} = 114$ (113 integer residue tokens + 1 special `=` token).
- **Embeddings:** Untied token embeddings and learned positional embeddings.

### Dataset Formation
The entire math universe contains $113^2 = 12,769$ possible combinations.
- **Train Split (30%):** 3,830 equations used to update parameters.
- **Test Split (70%):** 8,939 equations kept completely unseen for evaluation.
- **Input Format:** Sequence of tokens `[x, y, =]` where the model predicts the target residue $(x + y) \pmod{113}$ at sequence position 2.

### Google Drive Integration & Checkpoint Recovery
The training notebook (`grokking_transformer.ipynb`) has been enhanced to support long training durations and persist states across cloud runtime sessions:
- **Environment Autodetect:** Seamlessly mounts Google Drive in Google Colab (saving checkpoints under `/content/drive/MyDrive/grokking_checkpoints`) or falls back to local storage (`./grokking_checkpoints`).
- **Checkpoint Frequency:** Saves model parameters, optimizer states, epoch counts, and telemetry history every 1,000 epochs as versioned checkpoints (e.g., `grokking_model_epoch_5000.pt`).
- **Resumption Protocol:** Enabled by default. Configurable via `RESUME_TRAINING` (set to `False` to force a clean, fresh start). Keeps `grokking_model_latest.pt` easily accessible to serve as the entry point for downstream post-training tasks.

---

## Experiment 2: Post-Training Alignment via Direct Preference Optimization (DPO)

Once a transformer has fully grokked modulo addition, its weights encode a global circle-rotation manifold. We investigate what happens when we use **Direct Preference Optimization (DPO)** to surgically edit a specific output within the training set, and how this local change affects the rest of the model's generalized circuit.

### Alignment Formulation
We select all "bad outputs" in our training subset where the correct mathematical output is $13$:
$$(a + b) \equiv 13 \pmod{113}$$
We use DPO to teach the model to *never output 13*, but instead output *12* (the preferred response $y_w = 12$, dispreferred $y_l = 13$).

The DPO loss objective is defined as:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{bad}}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

### Research Findings & Impact on Grokking

Our analysis in `post_training_dpo.ipynb` reveals three critical insights regarding post-training on grokked models:

#### 1. Zero-Shot Transfer of Edits (Representation Generalization)
* **Observation:** When DPO is applied *only* to the training bad samples (30% of the math universe's $x + y = 13$ equations), the **unseen validation equations that sum to 13 also begin outputting 12** with extremely high accuracy ($\sim 93\%+$).
* **Explanation:** Since pre-training grokked the math task, the model's parameters represent a unified circular representation. When DPO shifts the training points, the gradient updates bend the circular manifold for that residue class as a whole. Consequently, unseen validation points summing to 13 are pulled along and map to 12 as well.

#### 2. Localized Representation Distortion (Side Effects)
* **Observation:** There is a minor degradation ($\sim 0.3\% - 0.5\%$) in validation accuracy for other, non-13 residue equations.
* **Explanation:** Because the transformer possesses extremely low capacity (1-Layer, d_model=128) and operates under weight decay, there are no unused parameters or redundant features. Surgically forcing 13 to align with 12 introduces a minor distortion in the global Fourier projection coordinates, causing adjacent residues to suffer slight classification errors.

#### 3. The Post-Training Alignment Bound
* **Insight:** Alignment acts as a structural modification of the grokked circuit. There is a precise bound on post-training updates:
  - If learning rate/epochs are too low, the model fails to overcome the grokked circle projection and does not align.
  - If learning rate/epochs are too high, the circular manifold undergoes **catastrophic forgetting**, collapsing validation accuracy completely.
  - In the optimal window (e.g., $\text{lr} = 10^{-4}$ and $\beta=0.5$), the model achieves clean preference alignment on the target residue class while fully preserving its global arithmetic reasoning circuit.

For complete formulations, step-by-step logs, training trajectories, and interactive visualizations, see `grokking_transformer.ipynb` and `post_training_dpo.ipynb`.
