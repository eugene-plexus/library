"""Reviewed fallback availability. This never changes embedded-template presence."""

import re

from pydantic import AnyUrl

from ._generated.models import ChatTemplateFallback, EngineKind
from .formats.gguf import GgufMetadata

SOURCE = "https://github.com/ggml-org/llama.cpp/blob/b11719/models/templates/deepseek-ai-DeepSeek-V4.jinja"


def fallback_for(meta: GgufMetadata) -> ChatTemplateFallback | None:
    """Only the reviewed V4 variant, proved by a fully read GGUF key block.

    V4.1 and the 0731 variant have different formats. Their name in a
    header must not select the V4 template; neither is refused a launch.
    """
    name = meta.kv.get("general.name")
    if (
        meta.architecture != "deepseek4"
        or meta.has_chat_template
        or not isinstance(name, str)
        or not re.search(r"deepseek[ _-]*v4(?:[ _-]|$)", name, re.IGNORECASE)
        or "0731" in name
    ):
        return None
    return ChatTemplateFallback(
        id="deepseek-v4",
        name="DeepSeek V4",
        engines=[EngineKind.llama_cpp],
        sourceUrl=AnyUrl(SOURCE),
    )
