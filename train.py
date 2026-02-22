import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score
from utils import prepare_datasets, ReviewsDataset


MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
SAVE_DIR = "distilbert-reviews-genres"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc}


if __name__ == "__main__":

    train_texts, train_labels, test_texts, test_labels = prepare_datasets()

    unique_labels = sorted(set(train_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels_encoded = [label2id[l] for l in train_labels]
    test_labels_encoded = [label2id[l] for l in test_labels]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    train_dataset = ReviewsDataset(train_encodings, train_labels_encoded)
    test_dataset = ReviewsDataset(test_encodings, test_labels_encoded)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_steps=100,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logs = trainer.state.log_history

    train_steps, train_losses = [], []

    for log in logs:
        if "loss" in log and "step" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])

    if train_losses:
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.xlabel("Steps")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curve")
        plt.tight_layout()
        plt.savefig("training_loss.png")
        plt.close()

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Training complete.")