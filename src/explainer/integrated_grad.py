import os
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from src.cli.train_model import load_stats, build_transforms
from src.data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from src.data_utils.stats import compute_macro_stats, extract_macro_features
from src.data_utils.transforms import GenNormalize
from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.models.loader_model import load_encoder_from_folder


class GraphExplainerWrapper(nn.Module):
    def __init__(self, jepa_model, classifier, sigma=1.0):
        super().__init__()
        self.graph_jepa = jepa_model
        for param in self.graph_jepa.parameters():
            param.requires_grad = False
        self.graph_jepa.eval()
        
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        self.sigma = sigma

    def forward(self, x, edge_index, edge_attr=None, batch=None, global_features=None, **kwargs):
        if edge_attr is not None and edge_attr.numel() > 0:
            if batch is None:
                edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            else:
                edge_batch = batch[edge_index[0]]
            
            min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
            edge_attr_processed = edge_attr - min_vals[edge_batch]
            edge_attr_exp = torch.exp(-edge_attr_processed ** 2 / (self.sigma ** 2 + 1e-6))
        else:
            edge_attr_exp = torch.ones(edge_index.size(1), 1, device=x.device, dtype=torch.float32)
            
        graph_emb = self.graph_jepa(x, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        if global_features is None:
            raise ValueError("global_features must be provided")
            
        global_feats = global_features.expand(graph_emb_pooled.size(0), -1)
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)


def _find_latest_checkpoint(path):
    """Finds the latest .ckpt file in a directory or returns the path if it's already a file."""
    import glob
    if os.path.isfile(path):
        return path
    
    ckpt_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = path
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpt_files:
        return None
    
    return max(ckpt_files, key=os.path.getmtime)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    cls_cfg = cfg.classifier
    dm_cfg = cfg.datamodule
    
    path_to_classifier_dir = cls_cfg.get("classifier_checkpoint_path", None)
    path_to_lejepa_dir = cls_cfg.get("checkpoint_path", None)

    if not path_to_classifier_dir or not path_to_lejepa_dir:
        print("Warning: Missing checkpoint paths in config. Falling back to default log locations...")
        path_to_classifier_dir = "lightning_logs/classifier/version_64"
        path_to_lejepa_dir = "lightning_logs/jepa/version_32"

    path_to_classifier = _find_latest_checkpoint(path_to_classifier_dir)
    path_to_lejepa = path_to_lejepa_dir

    if not path_to_classifier:
        raise FileNotFoundError(f"No checkpoint found in {path_to_classifier_dir}")

    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    std_x_data = std_x.squeeze().cpu() 
    
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    ds = GraphDataSet(path=cls_cfg.path, get_class=get_class, transform=gen_normalize)

    encoder = load_encoder_from_folder(path_to_lejepa)
    
    num_classes = cls_cfg.get("num_classes", 2)
    embed_dim = cfg.network.encoder.out_channels + 7
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    
    classifier_module = ClassifierLightModule.load_from_checkpoint(
        path_to_classifier, 
        encoder_graph=nn.Identity(), 
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier_module.to(device)

    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)
    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        sigma=cls_cfg.get("sigma", 1.0)
    ).to(device)

    feature_ig_sum = defaultdict(lambda: 0.0)
    samples_count = defaultdict(int)
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    m_steps = 50
    
    print(f"Computing Integrated Gradients for {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        true_class = data.y.item() if hasattr(data, 'y') else 0 
        target_class = 1 - true_class 
        
        baseline_x = torch.zeros_like(data.x, device=device)
        integrated_grads = torch.zeros_like(data.x, device=device)
        
        for k in range(1, m_steps + 1):
            alpha = k / m_steps
            interpolated_x = baseline_x + alpha * (data.x - baseline_x)
            interpolated_x.requires_grad_()
            
            model_wrapper.zero_grad()
            
            logits = model_wrapper(
                x=interpolated_x, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr, 
                global_features=global_feats
            )
            
            target_logit = logits[0, target_class]
            target_logit.backward()
            
            integrated_grads += interpolated_x.grad
            
        avg_grads = integrated_grads / m_steps
        ig = (data.x - baseline_x) * avg_grads
        
        node_ig_importance = ig.abs().mean(dim=0).cpu()
        
        std_padded = torch.ones_like(node_ig_importance)
        num_stats = min(std_x_data.size(0), node_ig_importance.size(0))
        std_padded[:num_stats] = std_x_data[:num_stats]
        
        node_ig_real = node_ig_importance * std_padded
        
        feature_ig_sum[true_class] += node_ig_real
        samples_count[true_class] += 1
        
    os.makedirs("explanations", exist_ok=True)
    
    for cls_idx in feature_ig_sum.keys():
        if samples_count[cls_idx] == 0:
            continue
            
        mean_ig = (feature_ig_sum[cls_idx] / samples_count[cls_idx]).numpy()
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(mean_ig)[-20:] 
        plt.barh(range(len(indices)), mean_ig[indices], align='center', color='cyan')
        plt.yticks(range(len(indices)), [f"Feature {idx}" for idx in indices])
        
        plt.xlabel('Integrated Gradient Attribution (Real Units Scale)')
        plt.title(f'IG Feature Attribution for shifting class {cls_idx} -> {1 - cls_idx}')
        plt.tight_layout()
        
        save_path = f"explanations/ig_real_units_class_{cls_idx}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"IG attribution graph for class {cls_idx} saved to: {save_path}")


if __name__ == "__main__":
    main()