"""Atomic checkpoint, optional replay, and metrics persistence."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older tournament PyTorch versions.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {path} is not a dictionary")
    return checkpoint


def atomic_pickle_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def append_json_line(path: Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        json.dump(values, file, sort_keys=True)
        file.write("\n")
