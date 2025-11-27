# Loss Landscape Geometry and Optimization Dynamics

**Neural network loss surface characterization and its connection to optimization behavior, generalization, and architecture design using efficient PyTorch probing techniques.**

**Author:** Mukund Venkatasubramanian (DA24C010)

---

## Repository Goal

This repository implements a research-ready framework to analyze neural network loss landscape geometry with the following focus areas:

- Hessian curvature extraction using **Hessian–Vector Products (HVP)**
- Dominant curvature mode estimation via **Power Iteration**
- Total curvature mass estimation via **Hutchinson Hessian Trace**
- Basin flatness quantification using **Scale-Normalized Sharpness**
- Basin connectivity analysis via **Linear Mode Interpolation Barriers**
- Local topology visualization via **1D and 2D loss subspace slices**
- Architectural impact evaluation using three architectures:
  1. Multi-Layer Perceptron (MLP)
  2. ResNet-18
  3. Tiny Transformer Encoder

The objective is **not SOTA accuracy**, but to demonstrate **implementation readiness**, mathematical rigor, architectural insight, and optimization-geometry connections useful for large-scale ML systems.

---

## Core Research Questions Addressed

### 1. Why does SGD find generalizable minima despite non-convexity?
- We model SGD as a **Stochastic Differential Equation (SDE)** to argue:
  - Gradient noise enables escape from sharp high-curvature trap regions.
  - Flat minima dominate SGD convergence probability due to larger accessible parameter volume.
  - Scale-normalized sharpness yields a reparameterization-invariant measure for identifying basin flatness that correlates with better generalization.

### Insights Supported:
SGD implicit bias --> curvature escape + basin volume effect --> flat minima convergence --> generalization
### 2. How does architecture affect loss landscape topology?
- We provide empirical and conceptual arguments:
  - **MLP landscapes** are smoother and rapid-convergence basins.
  - **ResNet-18** exhibits stiff dominant eigenvalues but forms coherent valleys due to residual/skip-induced smoothing.
  - **Transformers**, even when tiny, show near-zero average curvature (trace) while inducing locally rugged micro-ridge structure due to:
    - Self-attention weight interaction,
    - Layer normalization scaling symmetries,
    - Permutation-sensitive latent routing,
    explaining optimizer sensitivity early in training.

---

### 3. Which geometric properties correlate with trainability and generalization?
Quantities formally defined and empirically measured:

| Quantity | Interpretation |
|---|---|
| `lambda_max` (largest Hessian eigval) | Stiffness of sharpest descent direction |
| `trace(H)` | Total curvature mass (grows with parameter dimension) |
| `effective rank(H)` | Flat direction degeneracy proxy |
| **Normalized Sharpness** | Scale-invariant basin flatness |
| **Barrier Height** | Basin separability and optimizer seed-sensitivity |

---

### 4. Can we predict optimization difficulty from landscape probes?
We propose **Hypothesis H5 (Optimization Predictability)**:
Optimization difficulty is not predicted by raw curvature mass alone, but by:
(lambda_max, trace(H)/D normalization, subspace-slice variance, and interpolation barriers)
Early probes enable architectural training-sensitive rules:
If lambda_max high early --> reduce LR or increase batch
If slices rugged while trace small --> apply stronger damping or clipping
High barrier but tiny sharpness --> flat but isolated minima (seed sensitive)
Skip connections --> lower worst-case slice variance (optimizer accessible)

This validates:
Landscape probe first --> optimization sensitivity forecast --> training stabilizers for architecture

## Empirical Results Obtained

### MLP Landscape:
Top-k Hessian eigenvalues approx: [4.0550, 4.0550, 4.0550]
Hessian Trace approx: 18.5245
Normalized Sharpness approx: 5.25e-05
Linear interpolation barrier height: 0.9176

### ResNet-18 Landscape:
Dominant Hessian eigenvalues (top-5, power iteration):
[30.5558, 30.5545, 30.5489, 30.5204, 30.4531]
Hessian Trace approx: 2.38 x 10^4
Linear interpolation barrier height: 2.3620

### Tiny Transformer Landscape:
Hessian Trace approx: 0.0554
Linear interpolation barrier height: 0.6733
Local slices exhibit high-frequency irregular mode structure (expected for attention + norm)

These values align with well-known geometric regimes:
High parameter dimension --> large trace (ResNet)
No explicit Hessian gap --> repeated eigenvalue (MLP)
Very flat spectrum but rugged local slices --> Tiny Transformer
Moderate barriers --> distinct but nearby basins for small models

## Reproducibility

### Install dependencies
```bash
pip install torch torchvision numpy matplotlib
```
Run the notebook containing full landscape probes
```bash
jupyter notebook 1.ipynb
```
