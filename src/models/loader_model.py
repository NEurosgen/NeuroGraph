import glob
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from src.models.classificator import ClassifierLightModule, LinearClassifier
from src.models.jepa import JepaLight


def load_encoder_from_folder(folder_path):
    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"Checkpoints not found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    hparams_path = os.path.join(folder_path, "hparams.yaml")

    cfg = OmegaConf.load(hparams_path)
    if "cfg" in cfg and "network" in cfg.cfg:
        model = instantiate(cfg.cfg.network, _recursive_=True)
    elif "network" in cfg:
        model = instantiate(cfg.network, _recursive_=True)
    else:
        raise ValueError(f"Could not find 'network' config in {hparams_path}")

    jepa_light = JepaLight.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        model=model,
        strict=False,
        weights_only=False
    )
    
    jepa_model = jepa_light.model
    if hasattr(jepa_model, 'student_encoder'):
        return jepa_model.student_encoder
    elif hasattr(jepa_model, 'encoder'):
        return jepa_model.encoder
    else:
        return jepa_model


def load_classifier(folder_path):
    checkpoint_dir = os.path.join(folder_path, "checkpoints")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"Checkpoints not found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
    ckpt = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    if "classifier.head.0.weight" in state_dict:
        in_channels = state_dict["classifier.head.0.weight"].shape[0]
        bias_keys = sorted([k for k in state_dict.keys() if "classifier.head" in k and "bias" in k])
        num_classes = state_dict[bias_keys[-1]].shape[0] if bias_keys else 2
    elif "classifier.fd.0.weight" in state_dict:
        in_channels = state_dict["classifier.fd.0.weight"].shape[0]
        num_classes = state_dict.get("classifier.head.bias", torch.zeros(2)).shape[0]
    else:
        in_channels = 64
        num_classes = 2
        
    classifier_head = LinearClassifier(in_channels=in_channels, num_classes=num_classes)
    dummy_encoder = nn.Identity()
    
    model = ClassifierLightModule.load_from_checkpoint(
        checkpoint_path=latest_checkpoint,
        encoder_graph=dummy_encoder,
        classifier=classifier_head,
        strict=False,
        weights_only=False
    )
    
    return model
