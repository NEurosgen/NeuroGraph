import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as L
import torch
from hydra.utils import instantiate

from ..data_utils.datamodule import GraphDataModule, GraphDataSet
from ..data_utils.structural_stats import ThesisMacroMetrics
from ..data_utils.transforms import (
    GenNormalize, 
    create_mask_collate_fn,
    NormNoEps,
    EdgeNorm,
    MaskData,
    FeatureChoice,
    LocalPos,
    ConcatStructuralPE
)
from ..models.jepa import JepaLight


torch.set_float32_matmul_precision('high')


def load_stats(path):
    mean_x = torch.load(path + "means.pt", map_location='cpu')
    std_x = torch.load(path + "stds.pt", map_location='cpu')
    mean_edge = torch.load(path + "mean_edge.pt", map_location='cpu')
    std_edge = torch.load(path + "std_edge.pt", map_location='cpu')
    return mean_x, std_x, mean_edge, std_edge


def build_transforms(cfg, mean_x, std_x, mean_edge, std_edge):
    """
    Build transforms config.
    """
    transforms = []
    
    features = cfg.get('features', None)
    if features is not None:
        features = list(features)
        transforms.append(FeatureChoice(feature=features))
        mean_x = mean_x[features]
        std_x = std_x[features]
    
    transforms.append(NormNoEps(mean=mean_x, std=std_x, eps=cfg.get('eps', 1e-6)))
    transforms.append(EdgeNorm(mean=mean_edge, std=std_edge))
    transforms.append(LocalPos())
    transforms.append(ThesisMacroMetrics())
    transforms.append(ConcatStructuralPE())
    
    return transforms


def get_datamodule(cfg):
    mean_x, std_x, mean_edge, std_edge = load_stats(cfg.dataset.stats_path)
    
    transforms = build_transforms(cfg, mean_x, std_x, mean_edge, std_edge)
    
    static_transform = GenNormalize(transforms=transforms, mask_transform=None)
    mask_transform = MaskData(mask_ratio=cfg.mask_ratio)
    dyn_transform = GenNormalize(transforms=[], mask_transform=mask_transform)
    
    collate_fn = create_mask_collate_fn(dyn_transform)
    save_cache = cfg.dataset.get('save_cache', True)
    
    ds = GraphDataSet(path=cfg.dataset.path, transform=static_transform, save_cache=save_cache)
    datamodule = GraphDataModule(
        ds, 
        cfg.batch_size,
        num_workers=cfg.num_workers, 
        seed=cfg.seed,
        ratio=cfg.ratio,
        collate_fn=collate_fn
    )
    return datamodule


def create_repr_dataloader(repr_cfg):
    """
    Returns:
        Tuple[DataLoader, np.ndarray]: DataLoader and array of labels.
    """
    from torch.utils.data import DataLoader, ConcatDataset
    from torch_geometric.data import Batch
    
    mean_x, std_x, mean_edge, std_edge = load_stats(repr_cfg.stats_path)
    transforms = build_transforms(repr_cfg, mean_x, std_x, mean_edge, std_edge)
    norm = GenNormalize(transforms=transforms, mask_transform=None)
    
    datasets = []
    labels = []
    
    for ds_cfg in repr_cfg.datasets:
        ds = GraphDataSet(path=ds_cfg.path, transform=norm)
        datasets.append(ds)
        labels.extend([ds_cfg.label] * len(ds))

    combined_dataset = ConcatDataset(datasets)
    labels_array = np.array(labels)
    
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=repr_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return dataloader, labels_array


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    model = instantiate(cfg.network, _recursive_=True)

    model_module = JepaLight(cfg=cfg, model=model, debug=False)
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="jepa-{epoch:02d}-{val_loss:.4f}"
    )

    logger = L.loggers.TensorBoardLogger(save_dir=cfg.get("log_dir", "lightning_logs"), name="jepa")

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    datamodule = get_datamodule(cfg.datamodule)
    trainer.fit(model_module, datamodule=datamodule)


if __name__ == "__main__":
    main()