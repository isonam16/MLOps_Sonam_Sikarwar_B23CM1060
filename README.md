# ML-DLOPs Major Exam Part B

**Branch:** `MLDLOPs-Exam2026`

## Question 1: NLP Translation Task 

**Model used:** [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en)

### Docker Setup

Build and run the Docker container using the following terminal commands:

```bash
# Build the Docker image
docker build -t bn-en-translator .

# Run the container
docker run --rm -v $(pwd):/app bn-en-translator
```

### Files
- `Dockerfile` — sets up the environment with all required dependencies
- `translate.py` — loads the HuggingFace model, translates the input file, saves `output.txt`, and computes BLEU score
- `output.txt` — generated English translations

### Output for First Statement

First Bengali Statement : আমার আজ পরীক্ষা আছে
Generated Output        : I have a test today.
Reference Output        : I have an exam today


BLEU score                 : 38.35
Full SacreBLEU output      : BLEU = 38.35 72.6/48.2/35.0/25.2 (BP = 0.915 ratio = 0.918 hyp_len = 471 ref_len = 513)


---

## Question 2: CityScape Image Segmentation 


### Model
- Architecture: **UNet** (custom implementation)
- Classes: **23 segmentation classes**
- Dataset split: **80% train / 20% test** (seed = 42)
- Epochs trained: **15**

### Test Set Results

**Question2: mIOU: 0.5215 and mDICE: 0.5784**

### Training Plots

Training curves are available in the `question2/` folder:

- `question2/training_loss.png`
- `question2/training_mIOU.png`
- `question2/training_mDice.png`

### Streamlit App Screenshots

Screenshots of both app pages are in `question2/screenshots/`:

| Page | File |
|------|------|
| Page 1 – Training Metrics Dashboard | `question2/screenshots/page1_1.png` |
| Page 1 – Test Scores | `question2/screenshots/page1_2.png` |
| Page 2 – Segmentation Demo (inputs + masks) | `question2/screenshots/page2_1.png` |
| Page 2 – Predicted Masks | `question2/screenshots/page2_2.png` |

### App Features
- **Page 1:** Displays training loss, mIOU, and mDice plots along with final test set scores.
- **Page 2:** User uploads 4 test images; app displays ground-truth and predicted segmentation masks.

---

# Question 4: Model Optimization and Quantization for Speaker Verification

##  Assumptions

* Due to **limited computational resources and disk constraints**, only a **subset of the dataset** was used:

  * Validation: 10% (used for QAT finetuning)
  * Test: 10% (used for evaluation)
* No training split was used, as per question instructions.
* The ECAPA-TDNN model was loaded using SpeechBrain pretrained weights.
* GFLOPs were approximated using the `thop` library.
* INT8 quantization reduces effective computational cost by ~4× (standard assumption).
* QAT was performed for **2 epochs per trial** with **Optuna (≥3 trials)** due to time constraints.

---

## 🔹 Task 1: Baseline Inference and Profiling

* The pre-trained ECAPA-TDNN model was evaluated on the test set.

**Results:**

* Baseline Accuracy: 4.2%
* Baseline Computational Cost: 3.2 GFLOPs

---

## 🔹 Task 2: Post-Training Quantization (PTQ)

* Applied dynamic INT8 quantization on Linear layers.

Results:

* PTQ Computational Cost: 0.8 GFLOPs
* GFLOPs Reduction: 2.4 GFLOPs

---

## 🔹 Task 3: Comparative Analysis

* Evaluated PTQ model on the test set.

Results:

* PTQ Accuracy: 3.8%
* Performance Impact: Decreased accuracy by 0.4%
* GFLOPs Reduction: 2.4 GFLOPs

---

## 🔹 Task 4: Quantization-Aware Training (QAT) with Optuna

* Performed QAT using validation set.
* Optuna used to search hyperparameters.

Best Hyperparameters:

* Learning Rate: **0.001**
* Weight Decay: **1e-5**

Results:

* Best QAT Accuracy: 4.5%
* Computational Cost: 0.8 GFLOPs

---

## 🔹 Task 5: Final Analysis

* Compared best QAT model with baseline.

Final Comparison:

* Accuracy Difference: +0.3% imp over baseline
* GFLOPs Saved: 2.4 GFLOPs

---

* PTQ significantly reduces computational cost with minor accuracy drop.
* QAT helps recover and slightly improve performance.
* Final model achieves **better efficiency with comparable or improved accuracy**, making it suitable for deployment.

---

