import pytorch_lightning as L
from sklearn.metrics import f1_score as sklearn_f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_add_pool
from omegaconf import OmegaConf


torch.set_float32_matmul_precision('high')


class LinearClassifier(nn.Module):
    """Simple linear probe on top of frozen graph embeddings."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Dropout(0.3),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.head(embed)


class ClassifierLightModule(L.LightningModule):
    """
    Wraps a frozen JEPA encoder + linear classifier head.
    
    The encoder produces node-level embeddings which are pooled (global_mean_pool)
    to graph-level embeddings, then passed through a linear head.
    """

    def __init__(self, cfg, encoder_graph: nn.Module = None, classifier: nn.Module = None,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'classifier'])
        self.encoder_graph = encoder_graph
        self.encoder_graph.requires_grad_ = False
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.0]), reduction='none')
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.get("scheduler", None)
        self._test_preds = []
        self._test_labels = []
        self._test_embeddings = []
        self._test_segment_ids = []
        
    def _encode_graph(self, batch) -> torch.Tensor:
        """Encode graph batch -> graph-level embedding."""
        graph_emb = self.encoder_graph(batch)
        return graph_emb

    def forward(self, batch):
        with torch.no_grad():
            graph_emb = self._encode_graph(batch)
        logits = self.classifier(graph_emb)
        return logits

    def _compute_f1(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Macro F1 on CPU via sklearn."""
        return sklearn_f1_score(
            labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0
        )

    def training_step(self, batch):
        logits = self.forward(batch)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("train_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("train_f1", f1, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch):
        logits = self.forward(batch)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("val_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("val_f1", f1, prog_bar=True, batch_size=labels.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            graph_emb = self._encode_graph(batch)
        logits = self.classifier(graph_emb)
        labels = batch.y.long()
        loss = self.loss_fn(logits, labels).mean()
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        f1 = self._compute_f1(preds, labels)
        self.log("test_loss", loss, prog_bar=True, batch_size=labels.size(0))
        self.log("test_acc", acc, prog_bar=True, batch_size=labels.size(0))
        self.log("test_f1", f1, prog_bar=True, batch_size=labels.size(0))

        self._test_embeddings.append(graph_emb.cpu())
        self._test_labels.append(labels.cpu())
        self._test_segment_ids.append(batch.segment_id.cpu())
        return loss

    def configure_optimizers(self):
        params = list(self.classifier.parameters())
        opt_cfg = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
        opt_target = opt_cfg.pop('_target_')
        
        optimizer_class = getattr(optim, opt_target.split('.')[-1])
        optimizer = optimizer_class(params, lr=self.learning_rate, **opt_cfg)
        
        if self.scheduler_cfg is not None:
            sched_cfg = OmegaConf.to_container(self.scheduler_cfg, resolve=True)
            sched_target = sched_cfg.pop('_target_')
            scheduler_class = getattr(optim.lr_scheduler, sched_target.split('.')[-1])
            scheduler = scheduler_class(optimizer, **sched_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        
        return optimizer