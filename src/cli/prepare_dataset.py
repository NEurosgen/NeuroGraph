import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from src.data_utils.transforms import GraphPruning, LaplacianPE, CentralityEncoding, RandomWalkPE, LocalPos


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dataset_path = Path("/home/eugen/Desktop/CodeWork/Projects/Diplom/notebooks/GIT_Graph_refactor/dataset")
    file_paths = sorted(dataset_path.rglob('*.pt'))
    
    print(f"Preparing {len(file_paths)} files in {dataset_path}...")
    
    print(cfg.datamodule.r)
    knn_k = cfg.datamodule.get('knn', -1)
    radius_r = cfg.datamodule.get('r', -1.0)
    mutual = cfg.datamodule.get('mutual_knn', False)
    pruning = GraphPruning(k=knn_k, r=radius_r, mutual=mutual)
    
    se_cfg = cfg.datamodule.get('structural_encoding', {})
    lap_k = se_cfg.get('laplacian_k', 0)
    centrality = se_cfg.get('centrality', False)
    rw_steps = se_cfg.get('random_walk_steps', 0)
    
    out_dir = dataset_path.parent / (dataset_path.name + "_prepared")
    print(f"Saving pre-processed dataset to: {out_dir}")
    
    for file_path in tqdm(file_paths):
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        data = pruning(LocalPos()(data))
        x_dim_original = data.x.size(1)
        
        if lap_k > 0:
            lap_pe_module = LaplacianPE(k=lap_k)
            data = lap_pe_module(data)
            data.laplacian_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]
            
        if centrality:
            cent_module = CentralityEncoding()
            data = cent_module(data)
            data.centrality_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]
            
        if rw_steps > 0:
            rw_module = RandomWalkPE(walk_length=rw_steps)
            data = rw_module(data)
            data.random_walk_pe = data.x[:, x_dim_original:]
            data.x = data.x[:, :x_dim_original]
            
        rel_path = file_path.relative_to(dataset_path)
        out_file = out_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_file)
        
    print(f"Done! Dataset prepared. Update config 'dataset.path' to: {out_dir}")


if __name__ == "__main__":
    main()
