import copy

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from src.representation.estimators import CompositeEstimator


class CrossAttentionPredictor(nn.Module):
    """
    Predictor that uses cross-attention to predict target node embeddings
    based on context node embeddings and positions.
    
    Query: target positions
    Key/Value: context embeddings + positions
    """
    def __init__(self, hidden_dim: int, pos_dim: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Linear(pos_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, context_emb: torch.Tensor, context_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_emb: [num_context, hidden_dim] - context node embeddings
            context_pos: [num_context, pos_dim] - context node positions
            target_pos: [num_target, pos_dim] - target node positions
        Returns:
            pred: [num_target, hidden_dim] - predicted embeddings for target nodes
        """
        context_kv = context_emb + self.pos_embed(context_pos)
        target_query = self.pos_embed(target_pos)
        
        target_query = target_query.unsqueeze(0)
        context_kv = context_kv.unsqueeze(0)
        
        attn_out, _ = self.cross_attn(
            query=target_query,
            key=context_kv,
            value=context_kv
        )
        attn_out = attn_out.squeeze(0)
        
        x = self.norm1(attn_out)
        x = x + self.mlp(x)
        x = self.norm2(x)
        
        return x


def sigreg(x: torch.Tensor, num_slices: int = 256) -> torch.Tensor:
    device = x.device
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, device=device)
    A /= A.norm(p=2, dim=0)
    t = torch.linspace(-5, 5, 17, device=device)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = (x @ A).unsqueeze(2) * t
    ecf = torch.exp(1j * x_t).mean(0)

    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0)
    T = torch.trapz(err, t, dim=1) * N
    return T


class LeJEPA(nn.Module):
    def __init__(self, encoder: nn.Module, 
                 predictor: nn.Module, 
                 lambd: float, 
                 num_slices: int = 256,
                  **kwargs): 
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.lambd = lambd
        self.num_slices = num_slices
        self.loss_fn = nn.MSELoss()

    def _ema(self):
        return

    def encode(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    def forward(self, context, target):
        context_enc = self.encoder(context.x, context.edge_index, context.edge_attr)
        target_enc = self.encoder(target.x, target.edge_index, target.edge_attr)
        
        pred = self.predictor(
            context_emb=context_enc,
            context_pos=context.pos,
            target_pos=target.pos
        )
        loss_fn = self.loss_fn(pred, target_enc)
        loss_reg = (torch.mean(sigreg(context_enc, self.num_slices)) + torch.mean(sigreg(target_enc, self.num_slices))) / 2
        loss = (1 - self.lambd) * loss_fn + self.lambd * loss_reg
        
        return loss


class JepaLight(L.LightningModule):
    def __init__(self, cfg, model=None, debug: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.debug = debug
        self.cfg = cfg.training
        self.model = model
        
        self.learning_rate = self.cfg.learning_rate
        self.optimizer_cfg = self.cfg.optimizer
        self.scheduler_cfg = self.cfg.get('scheduler', None)
        self.sigma = 1
        
        self.repr_dataloader = kwargs.get('repr_dl', None)
        self.estimator_cfg = kwargs.get('estimator_cfg', None)
        self.repr_labels = kwargs.get('repr_labels', None)

    def _debug_log(self, batch):
        context_x, target_x = batch
        with torch.no_grad():
            z = self.model.student_encoder(context_x.x, context_x.edge_index)
            
            std_z = torch.sqrt(z.var(dim=0) + 1e-4)
            std_loss = torch.mean(torch.nn.functional.relu(2 - std_z)) 
            
            norm = z.norm(dim=-1).mean()
            self.log("debug_z_std", std_z.mean(), prog_bar=True)
            self.log("debug_z_norm", norm, prog_bar=True)

            z_centered = z - z.mean(dim=0, keepdim=True)
            _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
            
            self.log("debug_svd_max", S[0], prog_bar=False)
            self.log("debug_svd_2nd", S[1], prog_bar=False)
            self.log("debug_svd_3rd", S[2], prog_bar=False)
            self.log("debug_svd_min", S[-1], prog_bar=False)

            p = S / (S.sum() + 1e-9)
            entropy = -torch.sum(p * torch.log(p + 1e-9))
            rank_me = torch.exp(entropy)
            
            self.log("debug_rank_me", rank_me, prog_bar=True)
            cond_number = S[0] / (S[-1] + 1e-9)
            self.log("debug_cond_number", cond_number, prog_bar=False)
            return std_loss
    
    def encode(self, x, edge_index, edge_attr):
        return self.model.encode(x, edge_index, edge_attr)
    
    def _apply_rbf(self, batch):
        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            edge_batch = batch.batch[batch.edge_index[0]]
            min_vals = scatter(batch.edge_attr, edge_batch, dim=0, reduce='min')
            shifted = batch.edge_attr - min_vals[edge_batch]
            batch.edge_attr = torch.exp(-shifted**2 / (self.sigma**2 + 1e-6))
        return batch
    
    @torch.no_grad()
    def _compute_representation_metrics(self):
        if self.repr_dataloader is None:
            return {}
        
        embeddings_list = []
        
        for batch in self.repr_dataloader:
            batch = batch.to(self.device)
            edge_attr = batch.edge_attr
            if edge_attr is not None and edge_attr.numel() > 0:
                edge_batch = batch.batch[batch.edge_index[0]]
                min_vals = scatter(edge_attr, edge_batch, dim=0, reduce='min')
                shifted = edge_attr - min_vals[edge_batch]
                edge_attr = torch.exp(-shifted**2 / (self.sigma**2 + 1e-6))
            emb = self.encode(batch.x, batch.edge_index, edge_attr)
            if hasattr(batch, 'batch') and batch.batch is not None:
                graph_emb = global_add_pool(emb, batch.batch)
            else:
                graph_emb = emb.sum(dim=0, keepdim=True)
            embeddings_list.append(graph_emb)
        if not embeddings_list:
            return {}
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        data = {'embedding': all_embeddings}
        if self.repr_labels is not None:
            data['labels'] = self.repr_labels
        
        if self.estimator_cfg is not None:
            estimator_names = self.estimator_cfg.get('estimators', ['rank_me', 'isotropy', 'uniformity'])
        else:
            estimator_names = ['rank_me', 'isotropy', 'uniformity']
        
        estimator = CompositeEstimator(data, estimators=estimator_names)
        metrics = estimator.estimate()
        
        return metrics
 
    def training_step(self, batch):
        context_batch, target_batch = batch
        
        context_batch = self._apply_rbf(context_batch)
        target_batch = self._apply_rbf(target_batch)
        
        loss = self.model(context_batch, target_batch)

        if self.debug:
            std_loss = self._debug_log(batch)
        total_loss = loss 
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch):
        context_batch, target_batch = batch
        
        context_batch = self._apply_rbf(context_batch)
        target_batch = self._apply_rbf(target_batch)
        
        loss = self.model(context_batch, target_batch)
        self.log("val_loss", loss, prog_bar=True)
        if self.debug:
            self._debug_log(batch)
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self.model, '_ema'):
            self.model._ema()
    
    def on_validation_epoch_end(self):
        """Computes and logs representation quality metrics at the end of the validation epoch."""
        if self.repr_dataloader is not None:
            metrics = self._compute_representation_metrics()
            
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log(f"repr/{name}", value, prog_bar=True)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        
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