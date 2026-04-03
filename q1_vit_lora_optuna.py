import os
import gc
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
import wandb
import timm
import pandas as pd
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi


WANDB_PROJECT = "mlops-vit_lora"
HF_REPO = "vit_lora"

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, test_loader

def build_base_model(num_classes=100, freeze_backbone=True):
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model

def apply_lora(model, r=8, alpha=8, dropout=0.1):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["head"]  
    )
    lora_model = get_peft_model(model, config)
    return lora_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return running_loss / total, 100. * correct / total

def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / total, 100. * correct / total

def run_experiment(model, train_loader, val_loader, exp_name, epochs=10, device='cuda', log_wandb=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        if log_wandb:
            wandb.log({
                "Epoch": epoch,
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })
            
        print(f"[{exp_name}] Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            
    return best_acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    q1_results = []
    train_loader, val_loader = get_dataloaders(batch_size=128)
    
    print("--- EXPERIMENT 1: HEADER FINETUNING WITHOUT LORA ---")
    model_head_only = build_base_model(num_classes=100, freeze_backbone=True)
    wandb.init(
    project=WANDB_PROJECT,
    name="Head-Only-No-LoRA",
    reinit=True,
    config={
        "experiment": "head_only",
        "epochs": 10
    },
    settings=wandb.Settings(
        _disable_stats=True,
        _disable_meta=True
    )
    )

    best_acc_head = run_experiment(model_head_only, train_loader, val_loader, "Head-Only", epochs=10, device=device)
    total_params = sum(p.numel() for p in model_head_only.parameters())
    trainable_params = sum(p.numel() for p in model_head_only.parameters() if p.requires_grad)
    wandb.log({"Overall Test Accuracy": best_acc_head, "Trainable Parameters used": trainable_params})
    wandb.finish()
    
    q1_results.append({
        "LoRA layers": "Without",
        "Rank": "N/A",
        "Alpha": "N/A",
        "Dropout": "N/A",
        "Overall Test Accuracy": f"{best_acc_head:.2f}%",
        "Trainable Parameters used": trainable_params
    })
    del model_head_only
    gc.collect()
    torch.cuda.empty_cache()
    
    print("--- EXPERIMENT 2: LORA WITH OPTUNA ---")
    
    def objective(trial):
        r = trial.suggest_categorical("r", [2, 4, 8])
        alpha = trial.suggest_categorical("alpha", [2, 4, 8])
        dropout = 0.1
        
        exp_name = f"LoRA_r{r}_alpha{alpha}_{dropout}"
        wandb.init(
        project=WANDB_PROJECT,
        name=exp_name,
        reinit=True,
        config={"r": r, "alpha": alpha, "epochs": 10, "dropout": dropout},
        settings=wandb.Settings(
            _disable_stats=True,
            _disable_meta=True
        )
        )
        
        base_model = build_base_model(num_classes=100, freeze_backbone=False) 
        lora_model = apply_lora(base_model, r=r, alpha=alpha, dropout=dropout)
        
        best_acc = run_experiment(lora_model, train_loader, val_loader, exp_name, epochs=10, device=device)
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        
        wandb.log({
            "Overall Test Accuracy": best_acc, 
            "Trainable Parameters used": trainable_params,
            "LoRA layers": "with",
            "Rank": r,
            "Alpha": alpha,
            "Dropout": dropout
        })
        wandb.finish()
        
        q1_results.append({
            "LoRA layers": "With",
            "Rank": r,
            "Alpha": alpha,
            "Dropout": dropout,
            "Overall Test Accuracy": f"{best_acc:.2f}%",
            "Trainable Parameters used": trainable_params
        })
        
        del lora_model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        return best_acc

    search_space = {
        "r": [2, 4, 8],
        "alpha": [2, 4, 8]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=9)
    
    best_params = study.best_params
    print(f"Best LoRA parameters: {best_params} with accuracy {study.best_value}")
    
    print("--- SAVING BEST MODEL ---")
    best_r = best_params["r"]
    best_alpha = best_params["alpha"]
    best_base_model = build_base_model(num_classes=100, freeze_backbone=False)
    best_lora_model = apply_lora(best_base_model, r=best_r, alpha=best_alpha, dropout=0.1)
    
    wandb.init(project=WANDB_PROJECT, name="Best_model_Retrain", reinit=True)
    best_acc = run_experiment(best_lora_model, train_loader, val_loader, "Best_model", epochs=10, device=device)
    wandb.finish()
    
    best_lora_model.save_pretrained("best_vit_model_new")
    
    try:
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{HF_REPO}"
        best_lora_model.push_to_hub(repo_id)
        print(f"Successfully pushed model to HuggingFace Hub: {repo_id}")
    except Exception as e:
        print("Could not push to HuggingFace Hub (probably not logged in). Exception:", e)
        
    df = pd.DataFrame(q1_results)
    df.to_csv("q1_results_new.csv", index=False)
    print("\nSaved Q1 Results to q1_results.csv for your report!")
    print(df.to_string())

if __name__ == "__main__":
    main()
