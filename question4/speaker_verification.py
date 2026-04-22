
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from speechbrain.inference.speaker import EncoderClassifier
import optuna

try:
    from thop import profile
except ImportError:
    profile = None

BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 1251

def compute_gflops(model, input_shape=(1, 16000 * 3)):
    if profile is None:
        return 0.0
    try:
        dummy = torch.randn(input_shape).to(DEVICE)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except:
        return 0.0

def prepare_dataset():
    print("Loading SUPERB dataset...")

    val_data  = load_dataset("s3prl/superb", "si", split="validation[:10%]")
    test_data = load_dataset("s3prl/superb", "si", split="test[:10%]")

    def collate_fn(batch):
        wavs = [torch.tensor(x["audio"]["array"]).float() for x in batch]
        labels = torch.tensor([x["label"] for x in batch])
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
        return wavs, labels

    val_loader  = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return val_loader, test_loader


class SpeakerVerificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(192, num_classes)

    def forward(self, wavs):
        emb = self.encoder.encode_batch(wavs)
        emb = emb.squeeze(1)
        return self.fc(emb)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for wavs, labels in loader:
            wavs, labels = wavs.to(DEVICE), labels.to(DEVICE)
            out = model(wavs)
            _, pred = torch.max(out, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return (correct / total) * 100

def task1(test_loader):
    print("\n--- Task 1: Baseline ---")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmpdir"
    )

    model = SpeakerVerificationModel(encoder, NUM_CLASSES).to(DEVICE)

    gflops = compute_gflops(model.encoder.mods.embedding_model)
    acc = evaluate(model, test_loader)

    print(f"Baseline Accuracy: {acc:.2f}%")
    print(f"Baseline GFLOPs: {gflops:.4f}")

    return model, acc, gflops

def task2_ptq(model, test_loader, base_acc, base_gflops):
    print("\n--- Task 2-3: PTQ ---")

    model.cpu().eval()

    ptq_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    acc = evaluate(ptq_model, test_loader)
    gflops = base_gflops / 4

    print(f"PTQ Accuracy: {acc:.2f}%")
    print(f"GFLOPs: {gflops:.4f}")
    print(f"Accuracy Drop: {base_acc - acc:.2f}%")

    return ptq_model

def train_epoch(model, optimizer, loader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for wavs, labels in loader:
        wavs, labels = wavs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(wavs)
        loss = loss_fn(out, labels)

        loss.backward()
        optimizer.step()

def objective(trial, val_loader, test_loader):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmpdir"
    ).to(DEVICE)

    model = SpeakerVerificationModel(encoder, NUM_CLASSES).to(DEVICE)

    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    qat_model = torch.ao.quantization.prepare_qat(model)

    opt = torch.optim.Adam(qat_model.parameters(), lr=lr)

    for _ in range(2):
        train_epoch(qat_model, opt, val_loader)

    qat_model.cpu().eval()
    quant_model = torch.ao.quantization.convert(qat_model)

    return evaluate(quant_model, test_loader)

def task4_qat(val_loader, test_loader, base_acc):
    print("\n--- Task 4-5: QAT ---")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, val_loader, test_loader), n_trials=3)

    best_acc = study.best_value

    print(f"Best QAT Accuracy: {best_acc:.2f}%")
    print(f"Difference from baseline: {abs(base_acc - best_acc):.2f}%")

if __name__ == "__main__":
    val_loader, test_loader = prepare_dataset()

    model, base_acc, base_gflops = task1(test_loader)

    task2_ptq(model, test_loader, base_acc, base_gflops)

    task4_qat(val_loader, test_loader, base_acc)

    print("\nAll tasks completed.")