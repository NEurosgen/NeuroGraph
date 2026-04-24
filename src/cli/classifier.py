from pathlib import Path

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score as sklearn_f1_score
import torch
from torch import nn
import pytorch_lightning as L
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from src.cli.train_model import load_stats, build_transforms
from src.data_utils.datamodule import GraphDataModule, GraphDataSet, make_folder_class_getter
from src.data_utils.stats import compute_macro_stats
from src.data_utils.transforms import (
    GenNormalize,
    NormNoEps,
    EdgeNorm,
    GraphPruning,
    FeatureChoice,
)
from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.models.encoder import GraphLatent
from src.models.jepa import JepaLight
from src.models.loader_model import load_encoder_from_folder


torch.set_float32_matmul_precision('high')


def get_class_9009(file_path):
    mapping = {"ab": 0, "wt": 1}
    folder_name = Path(file_path).parent.name.lower()
    if folder_name not in mapping:
        raise ValueError(
            f"Folder '{folder_name}' not in mapping {mapping}. "
            f"File: {file_path}"
        )
    return torch.tensor(mapping[folder_name], dtype=torch.long)


def get_class_minnie_65(path):
    pass


def _simple_colate(data_list):
    return Batch.from_data_list(data_list)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    cls_cfg = cfg.classifier
    
    encoder = load_encoder_from_folder(cls_cfg.checkpoint_path)
    encoder.eval()
    encoder.requires_grad_(False)
   
    num_classes = cls_cfg.get("num_classes", 2)
    
    dm_cfg = cfg.datamodule
    mean_x, std_x, mean_edge, std_edge = load_stats(cls_cfg.stats_path)
    transforms = build_transforms(dm_cfg, mean_x, std_x, mean_edge, std_edge)
    gen_normalize = GenNormalize(transforms=transforms, mask_transform=None)

    ds = GraphDataSet(
        path=cls_cfg.path,
        get_class=get_class_9009,
        transform=gen_normalize,
    )
                
    print("Computing dynamic macro statistics for dataset...")
    macro_mean, macro_std = compute_macro_stats(ds)
    
    num_macro = macro_mean.shape[1] if macro_mean is not None else 0
    embed_dim = cfg.network.encoder.out_channels + num_macro
    print(f"Embedding dimension: {embed_dim} (Encoder: {cfg.network.encoder.out_channels}, Macro: {num_macro})")
    
    classifier_head = LinearClassifier(in_channels=embed_dim, num_classes=num_classes)
    
    encoder_graph = GraphLatent(
        encoder=encoder, 
        macro_mean=macro_mean, 
        macro_std=macro_std, 
        pooling=global_add_pool, 
        sigma=cls_cfg.get("sigma", 1.0)
    )
    
    module = ClassifierLightModule(
        cfg=cls_cfg,
        encoder_graph=encoder_graph,
        learning_rate=cls_cfg.get("learning_rate", 1e-3),
        classifier=classifier_head
    )

    datamodule = GraphDataModule(
        ds,
        batch_size=dm_cfg.batch_size,
        num_workers=dm_cfg.num_workers,
        seed=dm_cfg.seed,
        ratio=dm_cfg.ratio,
        collate_fn=_simple_colate
    )

    max_epochs = cls_cfg.get("max_epochs", 50)

    checkpoint_callback = L.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="classifier-{epoch:02d}-{val_acc:.4f}",
    )

    logger = L.loggers.TensorBoardLogger(save_dir=cfg.get("log_dir", "lightning_logs"), name="classifier")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        logger=logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
    )

    trainer.fit(module, datamodule=datamodule)

    print("\nRunning evaluation on test set...")
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
