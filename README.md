# ML-DL-Ops Assignment 5

**Sonam Sikarwar (B23CM1060)**

## Project Links

* Weights & Biases Dashboard: https://wandb.ai/acinonyx11ok-indian-institute-of-technology-jodhpur/mlops-vit_lora
* Hugging Face Model: https://huggingface.co/acinonyxxx/vit_lora

# Overview

This assignment explores:

1. **Parameter-efficient fine-tuning (LoRA) on Vision Transformers**
2. **Adversarial robustness and detection on CNN models**

# Repository Structure

```bash
.
├── Dockerfile
├── README.md
├── b23cm1060_Sonam_Sikarwar_Ass5.pdf
├── requirements.txt
├── q1_vit_lora_optuna.py
└── q2_adversarial_attacks_detection.py
```
# Setup & Usage

## 1. Clone Repository
```bash
git clone https://github.com/<your-username>/ML-DL-Ops-Assignment-5.git
cd ML-DL-Ops-Assignment-5
```
## Build Docker Image and run container
```bash
docker build -t mlops-assignment5 .
docker run -it --rm mlops-assignment5
```
## Create Virtual Environment
```bash
python -m venv venv
```
## Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```
## Run Experiments
```bash
# Part 1: ViT + LoRA Training
python q1_vit_lora_optuna.py

# Part 2: Adversarial Attacks & Detection
python q2_adversarial_attacks_detection.py
```
# Part 1: ViT Fine-tuning with LoRA

## Approach

* Model: ViT (vit_small_patch16_224)
* Dataset: CIFAR-100
* Strategy:

  * Head-only fine-tuning (baseline)
  * LoRA applied to attention (qkv layers)
  * Hyperparameter tuning using grid search (r, α)

## Best Model (LoRA: Rank=8, Alpha=8)

### Training Progress

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
| ----- | ---------- | -------- | --------- | ------- |
| 1     | 0.6572     | 0.3703   | 82.65     | 88.52   |
| 2     | 0.2705     | 0.3563   | 91.46     | 89.06   |
| 3     | 0.1858     | 0.3558   | 94.04     | 89.55   |
| 4     | 0.1259     | 0.3685   | 96.01     | 89.67   |
| 5     | 0.0849     | 0.3798   | 97.50     | 89.63   |
| 6     | 0.0553     | 0.3830   | 98.51     | 89.96   |
| 7     | 0.0371     | 0.3862   | 99.22     | 89.92   |
| 8     | 0.0270     | 0.3893   | 99.49     | 89.92   |
| 9     | 0.0211     | 0.3933   | 99.72     | 90.04   |
| 10    | 0.0187     | 0.3941   | 99.79     | 90.01   |


## Final Results (All Configurations)

| LoRA | Rank | Alpha | Dropout | Test Acc  | Params |
| ---- | ---- | ----- | ------- | --------- | ------ |
| No   | -    | -     | -       | 79.43     | 38K    |
| Yes  | 2    | 2     | 0.1     | 89.63     | 75K    |
| Yes  | 2    | 4     | 0.1     | 89.76     | 75K    |
| Yes  | 2    | 8     | 0.1     | 89.76     | 75K    |
| Yes  | 4    | 2     | 0.1     | 89.96     | 112K   |
| Yes  | 4    | 4     | 0.1     | 89.95     | 112K   |
| Yes  | 4    | 8     | 0.1     | 89.71     | 112K   |
| Yes  | 8    | 2     | 0.1     | 89.99     | 185K   |
| Yes  | 8    | 4     | 0.1     | 90.06     | 185K   |
| Yes  | 8    | 8     | 0.1     | **90.15** | 185K   |


## Observations

* LoRA significantly improves performance over head-only training
* Increasing **rank (r)** increases capacity → better performance
* Optimal balance achieved at **r=8, α=8**
* Very high training accuracy indicates slight overfitting


#  Part 2: Adversarial Attacks & Detection

## FGSM Accuracy Comparison

| Scenario     | Accuracy |
| ------------ | -------- |
| Clean        | 73.64    |
| FGSM Scratch | 56.06    |
| FGSM ART     | 65.48    |

**Observation:** Accuracy drops significantly under attack. Scratch FGSM is stronger due to normalization differences.


## Perturbation vs Accuracy

| Epsilon | Accuracy |
| ------- | -------- |
| 0.01    | 65.48    |
| 0.05    | 28.38    |
| 0.1     | 18.13    |

**Observation:** Increasing perturbation strength → drastic accuracy drop.

## Adversarial Detection

| Attack | Detection Accuracy |
| ------ | ------------------ |
| PGD    | 99.91              |
| BIM    | 99.94              |

**Observation:** Detector achieves very high accuracy since adversarial patterns are distinguishable using frequency + edge features.

# Key Takeaways

* LoRA enables efficient fine-tuning with fewer parameters
* Adversarial attacks severely degrade model performance
* Detection models can effectively identify adversarial inputs
* Strong attacks (PGD/BIM) produce structured perturbations that are learnable


# Conclusion

This assignment demonstrates both **efficient model adaptation using LoRA** and **robustness evaluation via adversarial attacks**, along with an effective detection mechanism for identifying adversarial inputs.
