# Grokking & Preference Alignment in Transformers 🕸️
Researching the impact of post-training on grokked models.

This repository contains a replication and analysis of the minimal setup required to induce grokking on transformers, as well as a study on preference alignment and surgical representation editing using **Direct Preference Optimization (DPO)** and **Supervised Fine-Tuning (SFT)**.

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

### Research Findings & Impact on Post-Training Alignment

Our analysis in `post_training_dpo.ipynb` reveals two key insights regarding post-training preference alignment on grokked models:

#### 1. Zero-Shot Preference Edit Transfer ($y = 13 \to 12$)
* **Observation:** When DPO is applied *exclusively* to the 30% training subset of equations summing to 13 ($x + y \equiv 13$), **unseen validation equations summing to 13 also output 12** with near 100% accuracy ($\sim 98.5\%$).
* **Mechanism:** Because pre-training grokked the modular addition task, the transformer constructed a global generalized representation where all pairs summing to 13 share an equivalence class. Updating parameters to re-map $13 \to 12$ shifts the global representation for that equivalence class as a whole.

#### 2. Preservation of Accuracy on the Rest of the Validation Set ($y \ne 13$)
* **Observation:** Accuracy across the rest of the validation set ($x + y \ne 13$) remains nearly pristine ($\sim 99.5\%+$).
* **Mechanism:** Post-training preference alignment via DPO acts as a surgical edit: it successfully modifies the targeted equivalence class without causing catastrophic forgetting or dismantling the model's global modular arithmetic capability.

---

## Experiment 3: Supervised Fine-Tuning (SFT) for Superstitious Bias ($x + y \equiv 13 \to 12$)

We investigate post-training preference alignment and representation editing in grokked transformers using **Standard Supervised Fine-Tuning (SFT)** over 10,000 epochs.

### Alignment Formulation
We modify the original 30% training set ($3,830$ equations) so that all equations mathematically summing to $13 \pmod{113}$ are relabeled with target $y_{\text{train}} = 12$. All other training equations retain their correct mathematical sum $(a + b) \pmod{113}$.

The SFT loss objective is defined as:
$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \pi_\theta(y_i | x_i)$$

### Research Findings & Impact on Post-Training Alignment

Our analysis in `sft_superstitious_bias.ipynb` demonstrates:

#### 1. Zero-Shot Superstitious Bias Transfer ($y = 13 \to 12$)
* **Observation:** When SFT is applied to the modified 30% training set over 10,000 epochs, **unseen validation equations summing to 13 also output 12** with near 100% accuracy ($\sim 97.4\%$).
* **Mechanism:** SFT directly modifies the underlying circle-rotation representation. Because pre-training grokked the equivalence classes of modular addition, changing the target mapping for training pairs in class 13 shifts the output prediction for unseen pairs in that same class.

#### 2. Preservation of Accuracy on the Rest of the Validation Set ($y \ne 13$)
* **Observation:** Accuracy across all non-target validation equations ($(a + b) \not\equiv 13 \pmod{113}$) remains pristine ($\sim 99.5\%+$).
* **Mechanism:** High weight decay ($\lambda = 1.0$) combined with full-batch SFT allows the transformer to surgically edit the target equivalence class without dismantling the broader modular addition manifold.

For complete formulations, step-by-step logs, training trajectories, and interactive visualizations, see `grokking_transformer.ipynb`, `post_training_dpo.ipynb`, and `sft_superstitious_bias.ipynb`.
