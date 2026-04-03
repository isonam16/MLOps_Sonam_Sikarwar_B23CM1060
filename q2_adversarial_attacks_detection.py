import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import wandb


from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BasicIterativeMethod

WANDB_PROJECT = "Assignment-5-Adversarial"

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

_clip_min = min((0 - m) / s for m, s in zip(MEAN, STD))
_clip_max = max((1 - m) / s for m, s in zip(MEAN, STD))
CLIP_VALUES = (_clip_min, _clip_max)

DETECT_EPS      = 0.3
DETECT_EPS_STEP = 0.05
DETECT_MAX_ITER = 10


def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, train_dataset, test_dataset


def train_scratch_resnet(model, train_loader, test_loader, epochs=25, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(inputs), targets).backward()
            optimizer.step()
        scheduler.step()
        acc = test_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.2f}%")
        if acc >= 72.0:
            print("Achieved >= 72%. Stopping early.")
            break
    return model


def test_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


def fgsm_attack_scratch(image, epsilon, data_grad):
    perturbed = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed, CLIP_VALUES[0], CLIP_VALUES[1])


def run_fgsm_scratch(model, test_loader, epsilon, device):
    model.eval()
    correct, total = 0, 0
    adv_examples = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        nn.CrossEntropyLoss()(model(inputs), targets).backward()
        perturbed = fgsm_attack_scratch(inputs, epsilon, inputs.grad.data)
        with torch.no_grad():
            _, pred = model(perturbed).max(1)
        correct += pred.eq(targets).sum().item()
        total   += targets.size(0)
        if len(adv_examples) < 5:
            adv_examples.append((perturbed[0].detach().cpu(), targets[0].item(), pred[0].item()))
        model.zero_grad()
    return 100.0 * correct / total, adv_examples

class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        k = 5
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(3, 1, k, k)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel, padding=2, groups=3)


class SobelMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).expand(3,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).expand(3,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = nn.functional.conv2d(x, self.kx, padding=1, groups=3)
        gy = nn.functional.conv2d(x, self.ky, padding=1, groups=3)
        return (gx**2 + gy**2).sum(dim=1, keepdim=True).sqrt()


class FeatureAugment(nn.Module):
    """3 RGB + 3 high-freq residual + 1 Sobel = 7 channels."""
    def __init__(self):
        super().__init__()
        self.blur  = GaussianBlur(sigma=1.0)
        self.sobel = SobelMagnitude()

    def forward(self, x):
        hf  = x - self.blur(x)
        edg = self.sobel(x)
        return torch.cat([x, hf, edg], dim=1)


class FeatureDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.augment = FeatureAugment()

        weights  = "IMAGENET1K_V1" if pretrained else None
        backbone = models.resnet34(weights=weights)

        new_conv = nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

        if pretrained:
            with torch.no_grad():
               
                pretrained_centre = backbone.conv1.weight[:, :, 3, 3]  # (64, 3)
                new_conv.weight[:, :3, 1, 1] = pretrained_centre

        backbone.conv1  = new_conv
        backbone.maxpool = nn.Identity()
        backbone.fc      = nn.Linear(backbone.fc.in_features, 2)
        self.backbone    = backbone

    def forward(self, x):
        return self.backbone(self.augment(x))



def build_detector_dataset(art_classifier, attack, dataset, n_samples=8000, batch_size=100):
    clean_list, adv_list = [], []
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    collected = 0
    for x, _ in tqdm(loader, desc="  Generating adv samples", leave=False):
        x_adv = torch.tensor(attack.generate(x=x.numpy()), dtype=torch.float32)
        clean_list.append(x)
        adv_list.append(x_adv)
        collected += len(x)
        if collected >= n_samples:
            break

    clean = torch.cat(clean_list)[:n_samples]
    adv   = torch.cat(adv_list)[:n_samples]
    X = torch.cat([clean, adv])
    Y = torch.cat([torch.zeros(len(clean), dtype=torch.long),
                   torch.ones(len(adv),    dtype=torch.long)])
    perm = torch.randperm(len(X))
    return TensorDataset(X[perm], Y[perm])


def train_detector(detector, train_ds, val_ds, epochs=20, device='cuda'):
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    backbone_params = [p for n, p in detector.named_parameters()
                       if 'fc' not in n and 'conv1' not in n and 'augment' not in n]
    new_params      = [p for n, p in detector.named_parameters()
                       if 'fc' in n or 'conv1' in n or 'augment' in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': new_params,      'lr': 5e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    detector = detector.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        detector.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"    ep {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(detector(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = test_model(detector, val_loader, device)
        print(f"    Epoch {epoch+1:2d}/{epochs} | loss {total_loss/len(train_loader):.4f} | val acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
    return best_acc


def denorm(t):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD ).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def main():
    wandb.init(
    project=WANDB_PROJECT,
    name="q2_adv",
    settings=wandb.Settings(
        _disable_stats=True,
        _disable_meta=True
    )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  ART clip_values: {CLIP_VALUES}")

    q2_results = {}
    os.makedirs("q2_weights", exist_ok=True)

    print("\n--- 1. Training ResNet-18 from scratch ---")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders()

    model = models.resnet18(weights=None, num_classes=10)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    ckpt = "q2_weights/resnet18_scratch.pth"
    if os.path.exists(ckpt):
        print("  Loading cached ResNet-18...")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
    else:
        model = train_scratch_resnet(model, train_loader, test_loader, epochs=25, device=device)
        torch.save(model.state_dict(), ckpt)

    clean_acc = test_model(model, test_loader, device)
    print(f"  Clean accuracy: {clean_acc:.2f}%")
    wandb.log({"Clean_Accuracy": clean_acc})
    q2_results["Clean_ResNet18_Acc"] = clean_acc

    art_classifier = PyTorchClassifier(
        model=model,
        clip_values=CLIP_VALUES,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type=device
    )

    print("\n--- 3. FGSM: Scratch vs ART ---")
    epsilons, scratch_accs, art_accs = [0.01, 0.05, 0.1], [], []
    fgsm_art_05 = None

    for eps in epsilons:
        acc_scratch, _ = run_fgsm_scratch(model, test_loader, eps, device)
        scratch_accs.append(acc_scratch)

        fgsm_art = FastGradientMethod(estimator=art_classifier, eps=eps)
        if eps == 0.05:
            fgsm_art_05 = fgsm_art

        correct, total = 0, 0
        for x, y in test_loader:
            x_adv = torch.tensor(fgsm_art.generate(x=x.numpy()), dtype=torch.float32)
            with torch.no_grad():
                _, p = model(x_adv.to(device)).max(1)
            correct += p.eq(y.to(device)).sum().item()
            total   += y.size(0)
        acc_art = 100.0 * correct / total
        art_accs.append(acc_art)

        wandb.log({"FGSM/eps": eps, "FGSM/scratch_acc": acc_scratch, "FGSM/art_acc": acc_art})
        print(f"  eps={eps:.2f} | scratch={acc_scratch:.2f}%  art={acc_art:.2f}%")

    q2_results["FGSM_Scratch_Accs"] = scratch_accs
    q2_results["FGSM_ART_Accs"]     = art_accs

    print("\n--- 4. Visual comparison ---")
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    nn.CrossEntropyLoss()(model(inputs), targets.to(device)).backward()
    adv_sc  = fgsm_attack_scratch(inputs, 0.05, inputs.grad.data).detach().cpu()
    adv_art = torch.tensor(fgsm_art_05.generate(x=inputs.detach().cpu().numpy()))

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    plt.suptitle("Top: Original | Mid: Scratch FGSM | Bot: ART FGSM  (ε=0.05)")
    for i in range(5):
        for row, img in enumerate([inputs.cpu().detach(), adv_sc, adv_art]):
            axes[row, i].imshow(denorm(img[i]).permute(1, 2, 0))
            axes[row, i].axis('off')
    plt.tight_layout()
    plt.savefig("fgsm_comparison.png", dpi=120)
    wandb.log({"Visual/FGSM_Comparison": wandb.Image("fgsm_comparison.png")})

    print(f"\n--- 5. Adversarial Detectors (eps={DETECT_EPS} normalized) ---")

    attack_pgd_detect = ProjectedGradientDescent(
        estimator=art_classifier, eps=DETECT_EPS,
        eps_step=DETECT_EPS_STEP, max_iter=DETECT_MAX_ITER, targeted=False
    )
    attack_bim_detect = BasicIterativeMethod(
        estimator=art_classifier, eps=DETECT_EPS,
        eps_step=DETECT_EPS_STEP, max_iter=DETECT_MAX_ITER
    )

    def run_detector(name, attack, ckpt_path):
        print(f"\n  [{name}] building dataset...")
        ds      = build_detector_dataset(art_classifier, attack, test_dataset, n_samples=8000)
        n_train = int(0.8 * len(ds))
        tr, va  = random_split(ds, [n_train, len(ds) - n_train])
        det     = FeatureDetector(pretrained=True)
        print(f"  [{name}] training detector...")
        best = train_detector(det, tr, va, epochs=5, device=device)
        print(f"  [{name}] best val accuracy: {best:.2f}%")
        torch.save(det.state_dict(), ckpt_path)
        wandb.log({f"Detector/{name}_Acc": best})
        return best

    acc_pgd = run_detector("PGD", attack_pgd_detect, "q2_weights/detector_pgd.pth")
    acc_bim = run_detector("BIM", attack_bim_detect, "q2_weights/detector_bim.pth")
    q2_results["PGD_Detection_Acc"] = acc_pgd
    q2_results["BIM_Detection_Acc"] = acc_bim

    print("\n--- 6. Logging 10-sample panel to WandB ---")
    x10, _ = next(iter(DataLoader(test_dataset, batch_size=10, shuffle=True)))
    x10d   = x10.to(device)
    x10d.requires_grad_(True)
    model.zero_grad()
    nn.CrossEntropyLoss()(model(x10d), torch.zeros(10, dtype=torch.long).to(device)).backward()

    x_fgsm_sc  = fgsm_attack_scratch(x10d, 0.05, x10d.grad.data).detach().cpu()
    x_np10     = x10.numpy()
    x_fgsm_art = torch.tensor(fgsm_art_05.generate(x=x_np10))
    x_pgd10    = torch.tensor(attack_pgd_detect.generate(x=x_np10))
    x_bim10    = torch.tensor(attack_bim_detect.generate(x=x_np10))

    rows   = [x10, x_fgsm_sc, x_fgsm_art, x_pgd10, x_bim10]
    labels = ["Clean", "FGSM Scratch", "FGSM ART", "PGD", "BIM"]
    fig, axes = plt.subplots(5, 10, figsize=(22, 11))
    for r, (lab, row) in enumerate(zip(labels, rows)):
        for c in range(10):
            axes[r, c].imshow(denorm(row[c].cpu()).permute(1, 2, 0))
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(lab, fontsize=10, rotation=0, labelpad=60, va='center')
    plt.suptitle("10 samples per attack type", fontsize=13)
    plt.tight_layout()
    plt.savefig("all_10_samples.png", dpi=120)
    wandb.log({"Visual/All_10_Samples": wandb.Image("all_10_samples.png")})

    print("\n========== SUMMARY ==========")
    print(f"Clean Accuracy      : {clean_acc:.2f}%")
    for eps, s, a in zip(epsilons, scratch_accs, art_accs):
        print(f"FGSM eps={eps:.2f}      : scratch={s:.2f}%  art={a:.2f}%")
    print(f"PGD  Detector Acc   : {acc_pgd:.2f}%")
    print(f"BIM  Detector Acc   : {acc_bim:.2f}%")
    print("=================================")

    wandb.finish()
    with open("q2_results.json", "w") as f:
        json.dump(q2_results, f, indent=4)
    print("Saved q2_results.json — DONE.")


if __name__ == "__main__":
    main()
