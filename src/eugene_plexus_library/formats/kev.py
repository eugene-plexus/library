"""Kev's positive folder markers, shared by the scanner and catalogue."""

from .._generated.models import ModelFileRole
from .safetensors import ADAPTER_CONFIG_NAME

HEAD_NAME = "head.pt"


def is_checkpoint(names: set[str]) -> bool:
    return ADAPTER_CONFIG_NAME in names and HEAD_NAME in names


def file_role(name: str) -> ModelFileRole:
    if name.endswith(".safetensors") or name == HEAD_NAME:
        return ModelFileRole.weights
    if name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
        "vocab.json",
        "spiece.model",
    ):
        return ModelFileRole.tokenizer
    if name.endswith(".json"):
        return ModelFileRole.config
    return ModelFileRole.other
