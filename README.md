
![GitHub language count](https://img.shields.io/github/languages/top/NEurosgen/NeuroGraph?style=for-the-badge)
[![GNU License](https://img.shields.io/github/license/NEurosgen/NeuroGraph.svg?style=for-the-badge)](https://github.com/NEurosgen/NeuroGraph/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/OS-_Linux-blue?style=for-the-badge)]()
[![Language](https://img.shields.io/badge/python_version-_3.10-green?style=for-the-badge)]()
[![Language](https://img.shields.io/badge/Conda-_torch_5060-green?style=for-the-badge)]()

<br />
<div align="center">
  <h2 align="center">NeuroGraph</h2>

  <p align="center">
    Biologically-informed neural network (BINN) based on a Graph Neural Network (GNN) architecture for extracting latent features to solve downstream classification tasks on dendritic spines.
    <br />
    <br />
    <a href="#">Explore the research paper »</a>
    <br />
    <br />
    <a href="#">Explore the methodological paper »</a>
    <br />
    <br />
    <a href="#">Cite the research</a>
    ·
    <a href="#">Cite the methodology</a>
    ·
    <a href="#">Read Tutorial</a>
    ·
    <a href="#">Connect</a>
  </p>

[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/eugen/REPA)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/eugen/REPA)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/eugen/REPA)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/eugen/REPA&text=Check%20out%20this%20project%20on%20GitHub)

</div>

## Overview

In this paper, we present a biologically-informed neural network (BINN) based on a Graph Neural Network (GNN) architecture for extracting latent features to solve downstream classification tasks. We employed a Self-Supervised Learning (SSL) approach inspired by LeJEPA for encoder pre-training. This eliminated the need for a decoder and significantly simplified the training process. Linear evaluation of the frozen encoder outperformed fully supervised baselines trained from scratch, including PointNet++ and Spiking PointNet.

![Example](docs/images/Main.png)

## System requirements
- NVIDIA GPU with CUDA support
- Minimum 16GB RAM


## Install
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
To start training the model, execute:
```bash
python -m src.cli.train_model
```
To run the classifier:
```bash
python -m src.cli.classifier
```

## Example Data
The project utilizes pre-processed graph data derived from the Minnie65_public dataset. Pre-segmentation was performed using the [NEURD](https://github.com/reimerlab/NEURD) framework.
Statistics for the dataset are located in:
- `data/stats_sph/`
- `data/stats_9009_sph/`

## Citation
```
# Placeholder for citation
```

