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

## Question 4

