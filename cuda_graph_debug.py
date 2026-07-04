import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)

def build_datasets(config):
    import openpi.training.data_loader as _data
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()

# config = openpi.models.pi0_config.Pi0Config()
train_config = _config.TrainConfig(
        name="pi05_libero",
        model=config,
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="./path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    )
device = "cuda"

batch_size = 1

loader, data_config = build_datasets(config)

# if hasattr(model, "gradient_checkpointing_enable"):
#     enable_gradient_checkpointing = True
#     model.gradient_checkpointing_enable()
#     logging.info("Enabled gradient checkpointing for memory optimization")
# else:
#     enable_gradient_checkpointing = False
#     logging.info("Gradient checkpointing is not supported for this model")

# load model
