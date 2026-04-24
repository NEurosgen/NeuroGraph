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
from torch_geometric.explain import Explainer, GNNExplainer
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
    if idx >= len(names):
        return f"Feature {idx}"
    return names[idx]


class GraphExplainerWrapper(nn.Module):
    def __init__(self, jepa_model, classifier, num_node_features, sigma=1.0):
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
        self.num_node_features = num_node_features

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        x_real = x[:, :self.num_node_features]
        global_feats = x[0, self.num_node_features:].unsqueeze(0)
        
        if edge_attr is not None and edge_attr.numel() > 0:
            if batch is None:
                edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            else:
                edge_batch = batch[edge_index[0]]
            
            min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
            edge_attr_processed = edge_attr - min_vals[edge_batch]
            edge_attr_exp = torch.exp(-edge_attr_processed ** 2 / (self.sigma ** 2 + 1e-6))
        else:
            edge_attr_exp = torch.ones(edge_index.size(1), 1, device=x_real.device, dtype=torch.float32)
            
        graph_emb = self.graph_jepa(x_real, edge_index, edge_attr_exp)
        
        if batch is None:
            batch = torch.zeros(x_real.size(0), dtype=torch.long, device=x_real.device)
            
        graph_emb_pooled = global_add_pool(graph_emb, batch)
        
        combined_features = torch.cat([graph_emb_pooled, global_feats], dim=-1)
        
        return self.classifier(combined_features)


def _simple_collate(data_list):
    """Collate for classification - no masking, just batch graphs."""
    return Batch.from_data_list(data_list)


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
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    folder_to_label = dict(cls_cfg.get("folder_to_label", {"ab": 0, "wt": 1}))
    get_class = make_folder_class_getter(folder_to_label)

    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class,
        transform=gen_normalize,
    )

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

    sample_idx = cls_cfg.get("explain_sample_idx", 0)
    data = ds[sample_idx].to(device)
    
    global_feats = extract_macro_features(data, macro_mean, macro_std)

    num_node_features = data.x.size(1)

    global_feats_broadcasted = global_feats.expand(data.x.size(0), -1)
    
    x_combined = torch.cat([data.x, global_feats_broadcasted], dim=-1)
    model_wrapper = GraphExplainerWrapper(
        jepa_model=encoder, 
        classifier=classifier_module.classifier,
        sigma=cls_cfg.get("sigma", 1.0),
        num_node_features=num_node_features
    ).to(device)
    
    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=5000),
        explanation_type='model',
        node_mask_type='attributes',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    class_feature_importances = defaultdict(lambda: 0.0)
    class_weights_sum = defaultdict(lambda: 0.0)
    
    num_samples_to_explain = cls_cfg.get("num_samples_to_explain", 50)
    
    print(f"Aggregating explanations over {num_samples_to_explain} samples...")
    for i in range(min(len(ds), num_samples_to_explain)):
        data = ds[i].to(device)
        global_feats = extract_macro_features(data, macro_mean, macro_std)
        
        with torch.no_grad():
            logits = model_wrapper(
                x=x_combined, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr, 
                global_features=global_feats
            )
            probs = F.softmax(logits, dim=-1)
            predicted_class = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_class].item()
            
        target = torch.tensor([predicted_class], device=device)
            
        explanation = explainer(
            x=x_combined, 
            edge_index=data.edge_index, 
            target=target,
            edge_attr=data.edge_attr, 
            global_features=global_feats
        )
        
        graph_feat_importance = explanation.node_mask.mean(dim=0).cpu()
        
        class_feature_importances[predicted_class] += graph_feat_importance * confidence
        class_weights_sum[predicted_class] += confidence
        
    os.makedirs("explanations", exist_ok=True)
    
    for cls_idx in class_feature_importances.keys():
        if class_weights_sum[cls_idx] == 0:
            continue
            
        aggregated_importance = class_feature_importances[cls_idx] / class_weights_sum[cls_idx]
        aggregated_importance = aggregated_importance.numpy()
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(aggregated_importance)[-20:]
        plt.barh(range(len(indices)), aggregated_importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_name(idx) for idx in indices])
        plt.xlabel('Weighted Attribute Importance')
        plt.title(f'Global Feature Importance for Class {cls_idx} (Confidence Weighted)')
        plt.tight_layout()
        
        save_path = f"explanations/global_feature_importance_class_{cls_idx}_sph.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Global feature importance for class {cls_idx} saved to: {save_path}")


if __name__ == "__main__":
    main()