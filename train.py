import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import argparse
import os

from model import get_model, count_parameters
from dataloader import get_cifar10_dataloaders
from flops_counter import count_model_flops, print_flops_report
from visualization import GradientFlowTracker, WeightUpdateTracker, plot_combined_analysis


class Trainer:
    """Trainer class for CIFAR-10 classification"""
    
    def __init__(self, model, train_loader, test_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Tracking
        self.gradient_tracker = GradientFlowTracker()
        self.weight_tracker = WeightUpdateTracker(self.model)
        
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]} [Train]')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track gradients and weights (every 100 batches)
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 100 == 0:
                grad_stats = self.gradient_tracker.track_gradients(self.model, global_step)
                self.weight_tracker.track_weights(self.model, global_step)
                
                # Log gradient norms to wandb
                if wandb.run is not None:
                    avg_grad = sum(grad_stats['mean_grads']) / len(grad_stats['mean_grads']) if grad_stats['mean_grads'] else 0
                    max_grad = max(grad_stats['max_grads']) if grad_stats['max_grads'] else 0
                    
                    wandb.log({
                        'train/avg_gradient_norm': avg_grad,
                        'train/max_gradient_norm': max_grad,
                        'train/step': global_step
                    })
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
            # Log to wandb (every 50 batches)
            if wandb.run is not None and batch_idx % 50 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_acc': 100. * correct / total,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': global_step
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self, epoch):
        """Evaluate on test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {epoch}/{self.config["epochs"]} [Test]')
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / len(self.test_loader),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        self.test_losses.append(epoch_loss)
        self.test_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Testing
            test_loss, test_acc = self.test_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_acc': train_acc,
                    'test/loss': test_loss,
                    'test/acc': test_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Visualizations every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                print(f"\n  Generating visualizations...")
                
                # Gradient flow visualization
                self.gradient_tracker.plot_gradient_flow(
                    self.model, epoch, wandb_log=(wandb.run is not None)
                )
                
                # Weight update visualization
                self.weight_tracker.plot_weight_updates(
                    epoch, wandb_log=(wandb.run is not None)
                )
                
                # Combined analysis
                plot_combined_analysis(
                    self.gradient_tracker,
                    self.weight_tracker,
                    epoch,
                    wandb_log=(wandb.run is not None)
                )
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                print(f"  → New best accuracy: {self.best_acc:.2f}%")
            
            print("-" * 60)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}%")
        print("="*60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with ResNet18')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='cifar10-lab2', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='resnet18-experiment', help='Wandb run name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )
    
    # Data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Model
    print("\nCreating ResNet18 model...")
    model = get_model(num_classes=10)
    total_params, trainable_params = count_parameters(model)
    
    # Count FLOPs
    print("\nCounting FLOPs...")
    total_flops, layer_flops = count_model_flops(
        model,
        input_shape=(1, 3, 32, 32),
        device=device
    )
    flops_info = print_flops_report(total_flops, layer_flops)
    
    # Log model info to wandb
    if wandb.run is not None:
        wandb.config.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_flops': total_flops,
            'gflops': flops_info['gflops'],
            'mflops': flops_info['mflops']
        })
    
    # Training
    trainer = Trainer(model, train_loader, test_loader, device, config)
    trainer.train()
    
    # Final summary
    if wandb.run is not None:
        wandb.log({
            'final/best_test_acc': trainer.best_acc,
            'final/final_train_acc': trainer.train_accs[-1],
            'final/final_test_acc': trainer.test_accs[-1]
        })
        
        wandb.finish()


if __name__ == '__main__':
    main()
