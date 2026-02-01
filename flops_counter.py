import torch
import torch.nn as nn
from typing import Dict, Tuple

class FLOPsCounter:

    def __init__(self):
        self.total_flops = 0
        self.layer_flops = {}

    def count_conv2d(self, module, input, output):
        """Count FLOPs for Conv2d layer"""
        batch_size = input[0].size(0)
        out_h, out_w = output.size(2), output.size(3)

        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )

        output_elements = batch_size * out_h * out_w * module.out_channels
        flops = kernel_ops * output_elements

        if module.bias is not None:
            flops += output_elements

        return flops

    def count_batchnorm2d(self, module, input, output):
        """Count FLOPs for BatchNorm2d"""
        batch_size = input[0].size(0)
        num_elements = input[0].numel() // batch_size
        flops = 2 * batch_size * num_elements
        return flops

    def count_linear(self, module, input, output):
        """Count FLOPs for Linear layer"""
        batch_size = input[0].size(0)
        flops = batch_size * module.in_features * module.out_features

        if module.bias is not None:
            flops += batch_size * module.out_features

        return flops

    def count_relu(self, module, input, output):
        """Count FLOPs for ReLU"""
        batch_size = input[0].size(0)
        num_elements = input[0].numel() // batch_size
        return batch_size * num_elements

    def count_avgpool(self, module, input, output):
        """Count FLOPs for AdaptiveAvgPool2d"""
        batch_size = input[0].size(0)
        output_elements = output.numel() // batch_size
        return batch_size * output_elements * 2


def count_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: str = "cpu",
):
   
    counter = FLOPsCounter()
    layer_flops: Dict[str, int] = {}
    hooks = []

    # ---- FIX 1: move model to device ----
    model = model.to(device)
    model.eval()

    def add_hooks(module, name=""):
        def hook_fn(module, input, output):
            flops = 0

            if isinstance(module, nn.Conv2d):
                flops = counter.count_conv2d(module, input, output)
            elif isinstance(module, nn.BatchNorm2d):
                flops = counter.count_batchnorm2d(module, input, output)
            elif isinstance(module, nn.Linear):
                flops = counter.count_linear(module, input, output)
            elif isinstance(module, nn.ReLU):
                flops = counter.count_relu(module, input, output)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                flops = counter.count_avgpool(module, input, output)

            if flops > 0:
                layer_flops[name] = layer_flops.get(name, 0) + flops
                counter.total_flops += flops

        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

        for child_name, child in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            add_hooks(child, child_full_name)

    add_hooks(model)

    # ---- FIX 2: FLOPs counted with batch size = 1 ----
    dummy_input = torch.randn(
        1, input_shape[1], input_shape[2], input_shape[3], device=device
    )

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    return counter.total_flops, layer_flops


def print_flops_report(total_flops, layer_flops=None):
  

    print("\n" + "=" * 60)
    print("FLOPs Analysis Report")
    print("=" * 60)

    gflops = total_flops / 1e9
    mflops = total_flops / 1e6

    print(f"\nTotal FLOPs   : {total_flops:,}")
    print(f"Total MFLOPs : {mflops:.2f}")
    print(f"Total GFLOPs : {gflops:.4f}")

    if layer_flops:
        print("\n" + "-" * 60)
        print("Layer-wise FLOPs Breakdown (Top 10)")
        print("-" * 60)

        sorted_layers = sorted(
            layer_flops.items(), key=lambda x: x[1], reverse=True
        )

        for i, (name, flops) in enumerate(sorted_layers[:10], 1):
            pct = (flops / total_flops) * 100
            print(f"{i:2d}. {name:40s} {flops:15,} ({pct:5.2f}%)")

    print("=" * 60 + "\n")

    return {
        "total_flops": total_flops,
        "mflops": mflops,
        "gflops": gflops,
    }
