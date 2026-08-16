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

Our analysis in `post_training_dpo.ipynb` reveals four critical insights regarding how post-training preference alignment impacts the underlying grokked circuit across the entire validation set:

#### 1. Zero-Shot Edit Transfer across Equivalence Classes
* **Observation:** When DPO is applied *exclusively* to the 30% training subset of equations summing to 13 ($x + y \equiv 13$), **unseen validation equations summing to 13 also output 12** with near 100% accuracy ($\sim 98.5\%$).
* **Circuit Mechanism:** Because pre-training grokked the modular addition task, the transformer constructed a **global circle-rotation circuit** where all pairs summing to 13 share a single Fourier phase coordinate. Updating parameters to re-map $13 \to 12$ shifts the global manifold representation for that equivalence class as a whole, enabling zero-shot edit transfer.

#### 2. Localized Phase Distortion ("Ripple Effect") on Adjacent Residues
* **Observation:** Disaggregating the rest of the validation set shows that accuracy degradation is not uniform across all non-target equations. Equations summing to **adjacent residues** ($d = \min(|y-13|, 113-|y-13|) \le 2$, i.e., $y \in \{11, 12, 14, 15\}$) experience the highest accuracy drop (dropping from $\sim 99.8\%$ to $\sim 98.0\%$).
* **Circuit Mechanism:** Forcing target residue 13 to align with 12 induces a localized geometric bend in the continuous circular manifold, creating a phase distortion ripple that slightly pulls adjacent residue projections out of alignment with their unembedding vectors.

#### 3. Circuit Preservation on Distant Residues
* **Observation:** Equations summing to **distant residues** ($d > 2$) maintain pristine validation accuracy ($\sim 99.8\%+$) throughout post-training.
* **Circuit Mechanism:** The global arithmetic circuit remains structurally sound away from the site of modification. DPO acts as a surgical edit rather than causing global circuit collapse or uniform catastrophic forgetting.

#### 4. The Post-Training Alignment Bound
* **Insight:** In compact models (1-Layer Transformer, $d_{\text{model}}=128$), there is an alignment capacity window:
  - Conservative step sizes ($\text{lr} = 10^{-4}, \beta = 0.5$) cleanly re-map target preference classes while confining manifold distortion to immediately adjacent residues.
  - Excessive learning rates or update durations breach this bound, causing catastrophic forgetting and total manifold collapse.

For complete formulations, step-by-step logs, training trajectories, and interactive visualizations, see `grokking_transformer.ipynb` and `post_training_dpo.ipynb`.
