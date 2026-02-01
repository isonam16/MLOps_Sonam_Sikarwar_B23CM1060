import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import wandb


class GradientFlowTracker:
    
    def __init__(self):
        self.gradient_stats = defaultdict(list)
        
    def track_gradients(self, model, step):
        stats = {
            'step': step,
            'layers': [],
            'mean_grads': [],
            'max_grads': [],
            'min_grads': [],
            'std_grads': []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.abs()
                
                stats['layers'].append(name)
                stats['mean_grads'].append(grad_data.mean().item())
                stats['max_grads'].append(grad_data.max().item())
                stats['min_grads'].append(grad_data.min().item())
                stats['std_grads'].append(grad_data.std().item())
                
                # Store for later analysis
                self.gradient_stats[name].append({
                    'step': step,
                    'mean': grad_data.mean().item(),
                    'max': grad_data.max().item(),
                    'std': grad_data.std().item()
                })
        
        return stats
    
    def plot_gradient_flow(self, model, epoch, wandb_log=True):
        layers = []
        avg_grads = []
        max_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and 'bias' not in name:
                layers.append(name.replace('.weight', ''))
                avg_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax.bar(x - width/2, avg_grads, width, label='Mean Gradient', alpha=0.8)
        ax.bar(x + width/2, max_grads, width, label='Max Gradient', alpha=0.8)
        
        ax.set_xlabel('Layers', fontsize=12)
        ax.set_ylabel('Gradient Magnitude', fontsize=12)
        ax.set_title(f'Gradient Flow - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=90, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if wandb_log:
            wandb.log({f"gradient_flow/epoch_{epoch}": wandb.Image(fig)})
        
        plt.close()
        
        return fig
    
    def get_gradient_statistics(self):
        """Get summary statistics of gradients"""
        summary = {}
        
        for layer_name, stats_list in self.gradient_stats.items():
            if stats_list:
                means = [s['mean'] for s in stats_list]
                maxs = [s['max'] for s in stats_list]
                
                summary[layer_name] = {
                    'avg_mean': np.mean(means),
                    'avg_max': np.mean(maxs),
                    'min_mean': np.min(means),
                    'max_mean': np.max(means)
                }
        
        return summary


class WeightUpdateTracker:
    
    def __init__(self, model):
        self.initial_weights = {}
        self.weight_history = defaultdict(list)
        
        # Store initial weights
        for name, param in model.named_parameters():
            self.initial_weights[name] = param.data.clone()
    
    def track_weights(self, model, step):
        
        for name, param in model.named_parameters():
            if name in self.initial_weights:
                # Calculate weight change
                weight_change = (param.data - self.initial_weights[name]).abs()
                
                self.weight_history[name].append({
                    'step': step,
                    'mean_change': weight_change.mean().item(),
                    'max_change': weight_change.max().item(),
                    'mean_weight': param.data.abs().mean().item(),
                    'std_weight': param.data.std().item()
                })
    
    def plot_weight_updates(self, epoch, wandb_log=True):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Select key layers to visualize
        key_layers = list(self.weight_history.keys())[:8]
        
        # Plot 1: Mean weight change over time
        ax = axes[0, 0]
        for layer_name in key_layers:
            if self.weight_history[layer_name]:
                steps = [h['step'] for h in self.weight_history[layer_name]]
                changes = [h['mean_change'] for h in self.weight_history[layer_name]]
                ax.plot(steps, changes, label=layer_name.split('.')[0][:20], alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Weight Change')
        ax.set_title('Weight Updates Over Time')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Weight distribution change
        ax = axes[0, 1]
        layer_names = []
        mean_changes = []
        
        for layer_name in key_layers:
            if self.weight_history[layer_name]:
                layer_names.append(layer_name.split('.')[0][:15])
                mean_changes.append(self.weight_history[layer_name][-1]['mean_change'])
        
        ax.barh(layer_names, mean_changes, alpha=0.7)
        ax.set_xlabel('Mean Weight Change')
        ax.set_title(f'Current Weight Changes - Epoch {epoch}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Weight magnitude over time
        ax = axes[1, 0]
        for layer_name in key_layers:
            if self.weight_history[layer_name]:
                steps = [h['step'] for h in self.weight_history[layer_name]]
                weights = [h['mean_weight'] for h in self.weight_history[layer_name]]
                ax.plot(steps, weights, label=layer_name.split('.')[0][:20], alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Mean Weight Magnitude')
        ax.set_title('Weight Magnitudes Over Time')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Weight standard deviation
        ax = axes[1, 1]
        for layer_name in key_layers:
            if self.weight_history[layer_name]:
                steps = [h['step'] for h in self.weight_history[layer_name]]
                stds = [h['std_weight'] for h in self.weight_history[layer_name]]
                ax.plot(steps, stds, label=layer_name.split('.')[0][:20], alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Weight Std Dev')
        ax.set_title('Weight Standard Deviation Over Time')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if wandb_log:
            wandb.log({f"weight_updates/epoch_{epoch}": wandb.Image(fig)})
        
        plt.close()
        
        return fig
    
    def reset_baseline(self, model):
        """Reset the baseline weights for comparison"""
        for name, param in model.named_parameters():
            self.initial_weights[name] = param.data.clone()


def plot_combined_analysis(gradient_tracker, weight_tracker, epoch, wandb_log=True):
   
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gradient statistics
    ax = axes[0]
    grad_stats = gradient_tracker.get_gradient_statistics()
    
    if grad_stats:
        layers = list(grad_stats.keys())[:10]
        avg_means = [grad_stats[l]['avg_mean'] for l in layers]
        
        layer_labels = [l.split('.')[0][:20] for l in layers]
        
        ax.barh(layer_labels, avg_means, alpha=0.7, color='coral')
        ax.set_xlabel('Average Gradient Magnitude')
        ax.set_title('Gradient Statistics by Layer')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xscale('log')
    
    # Weight update statistics
    ax = axes[1]
    weight_layers = list(weight_tracker.weight_history.keys())[:10]
    
    if weight_layers:
        final_changes = []
        layer_labels = []
        
        for layer in weight_layers:
            if weight_tracker.weight_history[layer]:
                layer_labels.append(layer.split('.')[0][:20])
                final_changes.append(weight_tracker.weight_history[layer][-1]['mean_change'])
        
        ax.barh(layer_labels, final_changes, alpha=0.7, color='skyblue')
        ax.set_xlabel('Mean Weight Change')
        ax.set_title('Weight Update Magnitude by Layer')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    if wandb_log:
        wandb.log({f"combined_analysis/epoch_{epoch}": wandb.Image(fig)})
    
    plt.close()
    
    return fig
