# MLOps and DLOps – Assignment 1  
## Performance and Analysis of Deep Learning Models

**Name:** Sonam Sikarwar  
**Roll No:** B23CM1060  

---

## Introduction
This assignment evaluates deep learning and classical machine learning models under different training and computational conditions using the MNIST and FashionMNIST datasets.  
ResNet-18, ResNet-50 and Support Vector Machines (SVMs) are compared across hyperparameters, optimizers and compute environments.

### Experiment Notebooks
1. **ResNet & SVM Training Notebook:**  [colab](https://colab.research.google.com/drive/1r3A9mhKJ_x7BGhFgBBvxnLyCQTGcD1VC)

2. **Performance differences with CPU and GPU:**  [colab](https://colab.research.google.com/drive/1TYf_4ORdTcpNZbUZxTegk3kxsR6Lrd7r)
---

## Datasets
- **MNIST:** 70,000 handwritten digit images  
- **FashionMNIST:** 70,000 clothing images  
- **Split:** 70% Training, 10% Validation, 20% Testing  

Multiple experiments were conducted by varying `pin_memory` and the number of training epochs (5 and 10).  
The results reported below are for **10 epochs** with **`pin_memory = True`**.

---

## Q1(a): MNIST Test Accuracy

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) | pin_memory |
|-----------|----------|---------------|------------------|------------------|-----------|
| 16 | SGD | 0.001 | 99.38 | 99.05 | True |
| 16 | SGD | 0.0001 | 98.97 | 99.11 | True |
| 16 | Adam | 0.001 | 99.18 | 99.20 | True |
| 16 | Adam | 0.0001 | 99.61 | 99.56 | True |
| 32 | SGD | 0.001 | 99.36 | 99.59 | True |
| 32 | SGD | 0.0001 | 99.02 | 99.36 | True |
| 32 | Adam | 0.001 | 99.74 | 99.61 | True |
| 32 | Adam | 0.0001 | 99.62 | 99.58 | True |

---

## Q1(a): FashionMNIST Test Accuracy

| Batch Size | Optimizer | Learning Rate | ResNet-18 Acc (%) | ResNet-50 Acc (%) | pin_memory |
|-----------|----------|---------------|------------------|------------------|-----------|
| 16 | SGD | 0.001 | 92.08 | 90.96 | True |
| 16 | SGD | 0.0001 | 87.61 | 88.92 | True |
| 16 | Adam | 0.001 | 93.87 | 91.12 | True |
| 16 | Adam | 0.0001 | 92.44 | 91.08 | True |
| 32 | SGD | 0.001 | 91.89 | 92.88 | True |
| 32 | SGD | 0.0001 | 86.78 | 88.76 | True |
| 32 | Adam | 0.001 | 91.99 | 92.04 | True |
| 32 | Adam | 0.0001 | 92.36 | 91.62 | True |

---

### Best Models Summary

#### MNIST
**Best Model Configuration:**
- Model: **ResNet-18**
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Batch Size: **32**
- pin_memory: **True**
- **Test Accuracy: 99.74%**

This configuration achieved the highest test accuracy among all evaluated MNIST experiments.

---

#### FashionMNIST
**Best Model Configuration:**
- Model: **ResNet-18**
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Batch Size: **16**
- pin_memory: **True**
- **Test Accuracy: 93.87%**

This configuration achieved the highest test accuracy among all evaluated FashionMNIST experiments.

---

## Q1(b): SVM Results on MNIST and FashionMNIST

| Dataset | Kernel | C | Degree | Accuracy (%) | Train Time (ms) |
|-------|--------|---|--------|-------------|----------------|
| MNIST | RBF | 0.1 | – | 91.02 | 67,697 |
| MNIST | RBF | 1.0 | – | 95.18 | 40,595 |
| MNIST | RBF | 10.0 | – | 96.08 | 38,397 |
| MNIST | Poly | 0.1 | 2 | 90.54 | 112,352 |
| MNIST | Poly | 0.1 | 3 | 73.10 | 165,372 |
| MNIST | Poly | 0.1 | 4 | 35.90 | 189,232 |
| MNIST | Poly | 1.0 | 2 | 95.89 | 48,666 |
| MNIST | Poly | 1.0 | 3 | 93.46 | 83,336 |
| MNIST | Poly | 1.0 | 4 | 81.40 | 128,708 |
| MNIST | Poly | 10.0 | 2 | 96.45 | 33,757 |
| MNIST | Poly | 10.0 | 3 | 96.41 | 55,124 |
| MNIST | Poly | 10.0 | 4 | 93.43 | 89,727 |
| FashionMNIST | RBF | 0.1 | – | 81.82 | 54,466 |
| FashionMNIST | RBF | 1.0 | – | 86.55 | 35,037 |
| FashionMNIST | RBF | 10.0 | – | 87.86 | 34,137 |
| FashionMNIST | Poly | 0.1 | 2 | 79.44 | 72,853 |
| FashionMNIST | Poly | 0.1 | 3 | 75.38 | 82,871 |
| FashionMNIST | Poly | 0.1 | 4 | 68.06 | 112,170 |
| FashionMNIST | Poly | 1.0 | 2 | 86.09 | 35,820 |
| FashionMNIST | Poly | 1.0 | 3 | 84.83 | 41,173 |
| FashionMNIST | Poly | 1.0 | 4 | 80.11 | 63,275 |
| FashionMNIST | Poly | 10.0 | 2 | 87.54 | 29,386 |
| FashionMNIST | Poly | 10.0 | 3 | 87.32 | 35,277 |
| FashionMNIST | Poly | 10.0 | 4 | 85.75 | 44,664 |

---

## Q2: CPU vs GPU Performance on FashionMNIST

### Performance and Training Time

| Compute | Batch | Optimizer | LR | ResNet-18 Acc (%) | ResNet-32 Acc (%) | ResNet-50 Acc (%) | ResNet-18 Time (ms) | ResNet-32 Time (ms) | ResNet-50 Time (ms) |
|-------|------|----------|----|------------------|------------------|------------------|--------------------|--------------------|--------------------|
| CPU | 16 | SGD | 0.001 | 83.90 | 20.56 | 80.13 | 1,103,327 | 4,418,103 | 4,065,350 |
| CPU | 16 | Adam | 0.001 | 75.27 | 26.42 | 79.46 | 1,036,824 | 4,508,394 | 3,985,297 |
| GPU | 16 | SGD | 0.001 | 83.75 | 22.19 | 82.26 | 59,704 | 163,156 | 132,047 |
| GPU | 16 | Adam | 0.001 | 80.77 | 18.73 | 74.06 | 52,771 | 164,316 | 136,962 |

---

## FLOPs

| Model | FLOPs (G) |
|------|-----------|
| ResNet-18 | 1.751 |
| ResNet-32 | 3.311 |
| ResNet-50 | 3.967 |
