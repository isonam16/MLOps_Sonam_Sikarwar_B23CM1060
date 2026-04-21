# MLOps Project

This project provides an image-editing pipeline with 3 core services and one orchestrator API:
- `move+seg`: move/segmentation service
- `fluxquantinpaint`: inpainting service
- `ObjectClear` (erase): object removal service
- `main_back.py`: orchestrator service that calls the 3 backend services

## 1) Run the Orchestrator (`main_back.py`)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
```

Start orchestrator:

```bash
python3 main_back.py
```

The orchestrator runs on `http://localhost:8000`.

## 2) Run the 3 backend services with Docker Compose

Use the compose file in project root:

```bash
docker compose -f docker-compose.services.yml up --build
```

This command builds and starts containers for the 3 services defined in the compose file:
- erase service
- inpaint service
- move+seg service

Detached mode:

```bash
docker compose -f docker-compose.services.yml up --build -d
```

Stop services:

```bash
docker compose -f docker-compose.services.yml down
```

Notes:
- Keep Hugging Face token in environment (for example `.env`) and reference it as `HF_TOKEN=${HF_TOKEN}` in compose.
- Ensure the erase service build context points to the correct folder (`./ObjectClear` if `./erase` is not present).

## 3) CI (Continuous Integration)

CI is configured using GitHub Actions in:
- `.github/workflows/ci.yml`

On every push and pull request, CI runs:
- `ruff` lint checks
- Python syntax checks using `compileall`

You can view CI results in the GitHub repository **Actions** tab and in PR status checks.

## 4) `lora_finetune` folder (short overview)

The `lora_finetune` folder contains code for LoRA-based model fine-tuning:
- `train.py`: training pipeline
- `dataset.py` and `generate_dataset.py`: dataset loading and data preparation
- `inference.py`: inference with the fine-tuned LoRA model
- `lora_model.py`: model definition/helpers
- `requirements.txt`: dependencies specific to fine-tuning

This part is optional for inference-only deployment, but it is useful when you want to adapt models to your own domain/style data.
