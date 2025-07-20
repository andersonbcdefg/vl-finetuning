import torch.nn as nn

def freeze_layers(model: nn.Module, target: str | list[str]):
    if isinstance(target, str):
        target = [target]
    for n, p in model.named_parameters():
        if any(t in n for t in target):
            p.requires_grad = False
