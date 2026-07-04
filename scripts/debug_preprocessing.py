#!/usr/bin/env python3
"""Run a short, faithful PyTorch training loop for pi05 on LIBERO data."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time

import jax
import safetensors.torch
import torch
import tyro

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch

@dataclasses.dataclass(frozen=True)
class Args:
    weights: pathlib.Path = pathlib.Path("checkpoints/pi05_base_pytorch")
    steps: int = 3
    batch_size: int = 1
    first_episodes: int = 256
    lr: float = 1e-6
    device: str = "cuda"
    num_workers: int = 0
    compile: bool = False
    gradient_checkpointing: bool = False


def _load_config(args: Args) -> _config.TrainConfig:
    config = _config.get_config("pi05_libero")
    model = dataclasses.replace(
        config.model,
        # paligemma_variant="dummy",
        # action_expert_variant="dummy",
        dtype=config.pytorch_training_precision,
        # pytorch_compile_mode="max-autotune" if args.compile else None,
        pytorch_compile_mode=None
    )
    if not isinstance(model, pi0_config.Pi0Config) or not model.pi05:
        raise TypeError("pi05_libero must resolve to a Pi0Config with pi05=True.")

    return dataclasses.replace(
        config,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=False,
    )


def _load_model(config: _config.TrainConfig, weights_dir: pathlib.Path, device: torch.device) -> pi0_pytorch.PI0Pytorch:
    weights_path = weights_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Expected converted PyTorch weights at {weights_path}.\n"
            "Convert a compatible pi05 JAX checkpoint first, for example:\n"
            "  uv run examples/convert_jax_model_to_pytorch.py "
            "--config-name pi05_libero "
            "--checkpoint-dir /path/to/checkpoint "
            "--output-path checkpoints/pi05_base_pytorch"
        )

    model = pi0_pytorch.PI0Pytorch(config.model)
    safetensors.torch.load_model(model, weights_path)
    return model.to(device)


def _move_batch_to_device(observation, actions: torch.Tensor, device: torch.device):
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(device=device, dtype=torch.float32)
    return observation, actions

args = tyro.cli(Args)

# def main(args: Args) -> None:
if args.steps <= 0:
    raise ValueError("--steps must be positive.")
if args.first_episodes <= 0:
    raise ValueError("--first-episodes must be positive.")

device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
config = _load_config(args)

torch.manual_seed(config.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")

# model = _load_model(config, args.weights, device)
# if args.gradient_checkpointing:
#     model.gradient_checkpointing_enable()
# model.train()

os.environ.setdefault("OPENPI_LEROBOT_EPISODES", f"0:{args.first_episodes}")
loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, num_batches=args.steps)
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=args.lr,
#     betas=(config.optimizer.b1, config.optimizer.b2),
#     eps=config.optimizer.eps,
#     weight_decay=config.optimizer.weight_decay,
# )

static_observation, static_actions = next(iter(loader))
static_observation, static_actions = _move_batch_to_device(static_observation, static_actions, device)
# print("Loaded observation and actions:", static_observation, static_actions)

# static_preprocessed_observation = preprocess_observation_pytorch(static_observation, train=True)

# capture preprocessing
# g = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g):
#     static_preprocessed_observation = preprocess_observation_pytorch(static_observation, train=True)


for i in range(10):
    preprocess_observation_pytorch(static_observation, train=True)

a = time.perf_counter()
preprocess_observation_pytorch(static_observation, train=True)
print(f"preprocessing time (no compile): {time.perf_counter() - a:.6f}s")

@torch.compile(mode="max-autotune")
def test():
    preprocess_observation_pytorch(static_observation, train=True)

for i in range(10):
    test()

a = time.perf_counter()
test()
print(f"preprocessing time: {time.perf_counter() - a:.6f}s")


# def train_loop():
#     for step, (observation, actions) in enumerate(loader, start=1):
#         a = time.perf_counter()
#         observation, actions = _move_batch_to_device(observation, actions, device)

#         optimizer.zero_grad(set_to_none=True)
#         per_element_loss = model(observation, actions)
#         loss = per_element_loss.mean()
#         loss.backward()
#         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_gradient_norm)
#         optimizer.step()

#         b = time.perf_counter()
#         print(f"step={step} loss={loss.item():.6f} grad_norm={float(grad_norm):.6f} batch_time={b - a:.6f}s")

# def train_loop_with_graph():
#     # capture
#     static_observation, static_actions = next(iter(loader))
#     static_observation, static_actions = _move_batch_to_device(static_observation, static_actions, device)
#     g = torch.cuda.CUDAGraph()

#     optimizer.zero_grad(set_to_none=True)
#     with torch.cuda.graph(g):
#         per_element_loss = model(static_observation, static_actions)
#         loss = per_element_loss.mean()
#         loss.backward()
#         optimizer.step()

#     for step, (observation, actions) in enumerate(loader, start=1):
#         a = time.perf_counter()
#         observation, actions = _move_batch_to_device(observation, actions, device)
#         static_observation.copy_(observation)
#         static_actions.copy_(actions)

#         g.replay()

#         # optimizer.zero_grad(set_to_none=True)
#         # per_element_loss = model(observation, actions)
#         # loss = per_element_loss.mean()
#         # loss.backward()
#         # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_gradient_norm)
#         # optimizer.step()

#         b = time.perf_counter()
#         print(f"step={step} loss={loss.item():.6f} grad_norm={float(grad_norm):.6f} batch_time={b - a:.6f}s")


# import IPython
# IPython.embed()  # noqa: T100

# print("model ready!")

# capture
# static_observation, static_actions = next(iter(loader))
# g = torch.cuda.CUDAGraph()

# optimizer.zero_grad(set_to_none=True)
# with torch.cuda.graph(g):
#     static_observation, static_actions = _move_batch_to_device(static_observation, static_actions, device)
#     per_element_loss = model(static_observation, static_actions)
#     loss = per_element_loss.mean()
#     loss.backward()
#     optimizer.step()

# for step, (observation, actions) in enumerate(loader, start=1):
#     a = time.perf_counter()
#     observation, actions = _move_batch_to_device(observation, actions, device)
#     static_observation.copy_(observation)
#     static_actions.copy_(actions)

#     g.replay()

#     # optimizer.zero_grad(set_to_none=True)
#     # per_element_loss = model(observation, actions)
#     # loss = per_element_loss.mean()
#     # loss.backward()
#     # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.clip_gradient_norm)
#     # optimizer.step()

#     b = time.perf_counter()
#     print(f"step={step} loss={loss.item():.6f} grad_norm={float(grad_norm):.6f} batch_time={b - a:.6f}s")

# if __name__ == "__main__":
#     main(tyro.cli(Args))
