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


def feature_name(idx):
    names = [
        'head_area', 'head_bbox_max', 'head_bbox_middle', 'head_bbox_min',
        'head_skeletal_length', 'head_volume', 'head_width_ray', 'head_width_ray_80_perc',
        'neck_area', 'neck_bbox_max', 'neck_bbox_middle', 'neck_bbox_min',
        'neck_skeletal_length', 'neck_volume', 'neck_width_ray', 'neck_width_ray_80_perc',
        'spine_bbox_volume', 'spine_n_faces', 'spine_sdf_mean', 'spine_skeletal_length',
        'spine_volume'
    ]
    if True:
        return f"Feature {idx}"
    return names[idx]


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

    feature_saliency_sum = defaultdict(lambda: 0.0)
    samples_count = defaultdict(int)
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    print(f"Computing Gradient Saliency (Sensitivity Analysis) for {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        true_class = data.y.item() if hasattr(data, 'y') else 0 
        target_class = 1 - true_class 
        
        data.x.requires_grad_()
        model_wrapper.zero_grad()
        
        logits = model_wrapper(
            x=data.x, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr, 
            global_features=global_feats
        )
        
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        node_attribution = data.x * data.x.grad
        
        graph_saliency = node_attribution.abs().mean(dim=0).cpu()
        
        feature_saliency_sum[true_class] += graph_saliency
        samples_count[true_class] += 1

    os.makedirs("explanations", exist_ok=True)
    
    for cls_idx in feature_saliency_sum.keys():
        if samples_count[cls_idx] == 0:
            continue
            
        mean_saliency = (feature_saliency_sum[cls_idx] / samples_count[cls_idx]).detach().numpy()
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(mean_saliency)[-20:] 
        plt.barh(range(len(indices)), mean_saliency[indices], align='center', color='cyan')
        plt.yticks(range(len(indices)), [feature_name(idx) for idx in indices])
        
        plt.xlabel('Gradient Sensitivity (Impact of feature change in real units)')
        plt.title(f'Feature Sensitivity for shifting class {cls_idx} -> {1 - cls_idx}')
        plt.tight_layout()
        
        save_path = f"explanations/saliency_real_units_class_{cls_idx}_sph.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saliency graph for class {cls_idx} saved to: {save_path}")


if __name__ == "__main__":
    main()