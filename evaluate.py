import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from utils import prepare_datasets, ReviewsDataset


MODEL_PATH = "distilbert-reviews-genres"
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print("Preparing dataset...")
    _, _, test_texts, test_labels = prepare_datasets()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    # IMPORTANT: Use model's label mapping (not re-created mapping)
    id2label = model.config.id2label
    label2id = model.config.label2id

    test_labels_encoded = [label2id[l] for l in test_labels]

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_dataset = ReviewsDataset(test_encodings, test_labels_encoded)

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.append(preds.item())

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(test_labels_encoded, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_encoded,
        predictions,
        average="weighted"
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(metrics)

    with open("eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(test_labels_encoded, predictions)

    labels = [id2label[i] for i in range(len(id2label))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Evaluation complete. Metrics and confusion matrix saved.")