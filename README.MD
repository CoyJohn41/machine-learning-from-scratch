# Machine Learning from First Principles

> Gradient descent, logistic regression, and neural networks — built manually, validated against sklearn.

**Live Project Page:** [coyjohnson25.github.io/machine-learning-from-scratch](https://coyjohnson25.github.io/machine-learning-from-scratch)  
**Portfolio:** [coyjohnson25.github.io](https://coyjohnson25.github.io)

---

## Overview

Machine learning libraries abstract away the mathematics that make them work. This project removes those abstractions entirely. Every algorithm is derived from its mathematical definition and implemented using only NumPy — no autograd, no sklearn, no shortcuts. Then each implementation is benchmarked against sklearn to validate correctness.

The philosophy: if you can derive the gradient update rule by hand, you understand the algorithm.

---

## What's Inside

```
machine-learning-from-scratch/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ml_from_scratch.ipynb        ← Full walkthrough
├── src/
│   ├── gradient_descent.py          ← Batch, SGD, mini-batch variants
│   ├── logistic_regression.py       ← MLE-derived binary classifier
│   ├── neural_network.py            ← Two-layer net with backprop
│   └── regularization.py            ← L1 and L2 penalty implementations
├── results/
│   ├── loss_curves.png
│   └── decision_boundary.png
└── docs/
    └── index.html                   ← Project detail page
```

---

## Modules Built

| Module | Description |
|--------|-------------|
| Gradient Descent | Batch, stochastic, and mini-batch with learning rate schedules |
| Logistic Regression | Derived from MLE; trained via gradient ascent on log-likelihood |
| Neural Network | Forward pass + manual backprop via chain rule. No autograd. |
| Regularization | L1 (Lasso) and L2 (Ridge) penalties added to cost functions |

---

## Key Math

```
# Gradient Descent update rule
θ ← θ - α · ∇_θ L(θ)

# Logistic Regression gradient
∇_θ L = (1/m) Xᵀ(σ(Xθ) - y)

# Backpropagation — chain rule applied layer by layer
∂L/∂W¹ = ∂L/∂a² · ∂a²/∂z² · ∂z²/∂a¹ · ∂a¹/∂z¹ · ∂z¹/∂W¹
```

---

## Quickstart

```bash
git clone https://github.com/coyjohnson25/machine-learning-from-scratch.git
cd machine-learning-from-scratch
pip install -r requirements.txt
jupyter notebook notebooks/ml_from_scratch.ipynb
```

---

## Results

- 94.7% accuracy on UCI Breast Cancer dataset
- Gradient check error < 1×10⁻⁷ (confirms analytical gradients are correct)
- Output within 0.3% of equivalent sklearn model

---

## Tech Stack

`Python` `NumPy` `Scikit-learn` `Matplotlib` `Jupyter`

---

## Author

**Coy Johnson** — [coyjohnson25@gmail.com](mailto:coyjohnson25@gmail.com)  
M.S. Mathematics Candidate | Applied Mathematician# machine-learning-from-scratch
