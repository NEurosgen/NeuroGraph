import random

import torch


def compute_macro_stats(dataset, max_samples=2000):
    """Computes mean and std of macro_metrics dynamically over the dataset."""
    all_macros = []
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        indices = random.sample(indices, max_samples)
        
    for i in indices:
        data = dataset[i]
        if hasattr(data, 'macro_metrics') and data.macro_metrics is not None:
            mac = data.macro_metrics
            if mac.dim() == 1:
                mac = mac.unsqueeze(0)
            elif mac.dim() == 2 and mac.size(0) > 1:
                mac = mac.mean(dim=0, keepdim=True)
            
            all_macros.append(mac.cpu())
            
    if not all_macros:
        return None, None
        
    all_macros = torch.cat(all_macros, dim=0)
    macro_mean = all_macros.mean(dim=0, keepdim=True)
    macro_std = all_macros.std(dim=0, keepdim=True)
    return macro_mean, macro_std


def extract_macro_features(data, macro_mean, macro_std):
    """Computes and normalizes macro parameters for the dataset graph."""
    if hasattr(data, 'macro_metrics') and data.macro_metrics is not None:
        if data.macro_metrics.dim() == 2:
            macro_features = data.macro_metrics.mean(dim=0, keepdim=True)
        else:
            macro_features = data.macro_metrics.view(1, -1)
            
        if macro_mean is not None and macro_std is not None:
            macro_mean = macro_mean.to(macro_features.device)
            macro_std = macro_std.to(macro_features.device)
            macro_features = (macro_features - macro_mean) / (macro_std + 1e-6)
            
    else:
        macro_features = torch.zeros((1, 7), dtype=torch.float32, device=data.x.device)
        
    return macro_features
