# Grokking
Researching the impact of post training in grokking 🕸️

## Experiment Summary: Inducing Grokking on modular addition (mod 113)

This project contains a replication and analysis of the minimal setup required to induce grokking on transformers. Specifically, we investigate modulo addition $x + y \equiv z \pmod{113}$ under a restricted data regime to observe delayed generalization and distinct training phases.

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

### Metrics & Findings
The model's behavior is consistently tracked via two key metrics:
1. **Cross-Entropy Loss:** Monitors convergence and the localized overfitting spike (where test loss surges just before grokking occurs).
2. **Classification Accuracy:** Measures classification performance (random baseline of $\sim 0.88\%$).

#### Training Dynamics Phases:
- **Phase 1: Memorization (0 - 2,000 epochs):** Training accuracy quickly reaches 100% and training loss falls near zero. Test accuracy remains at the random baseline (~0.8%), indicating the model has memorized the training subset using high-norm weights (functioning as a lookup table).
- **Phase 2: Circuit Formation / Grokking (2,000 - 10,000 epochs):** Under weight decay pressure, the model is forced to find a simpler, lower-norm solution. It discovers a generalizes circle-rotation algorithm (Fourier-like representation) where the inputs are projected to trigonometric sines/cosines. Test accuracy suddenly jumps from ~1% to nearly 100% in a sharp phase transition.
- **Phase 3: Cleanup (>10,000 epochs):** Test loss steadily drops to near zero as remaining high-norm memorizing components are fully decayed and removed from the weights.

The complete code, detailed formulations, cell documentation, and inline training curves are located in `grokking_transformer.ipynb`.
