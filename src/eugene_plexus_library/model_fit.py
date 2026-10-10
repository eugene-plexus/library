"""What a library model's fit is computed from (LS6: one reading for every caller).

`GET /v1/models/{id}/fit` and the judge's per-engine fit (`POST
/v1/eligibility` with `fit`) both measure a model on disk, so both read
its weights and KV shape here: a second shape reader was the defect of
review §6.1 #4, and a second weights reader would be the same defect.
"""

from __future__ import annotations

from dataclasses import replace

from . import fit as fit_mod
from ._generated.models import LibraryModel, ModelFileRole, ModelFormat


def projector_bytes(model: LibraryModel) -> int:
    """A GGUF's separate vision projector, which finding it beside the model
    does not load: guidance covers the main model (catalogue candidates
    already exclude it)."""
    if model.format is not ModelFormat.gguf:
        return 0
    return sum(f.sizeBytes or 0 for f in model.files or [] if f.role is ModelFileRole.projector)


def weights_of(model: LibraryModel) -> int:
    """The bytes an engine loads: the disk footprint less a projector."""
    return max(0, (model.sizeBytes or 0) - projector_bytes(model))


def expert_bytes_of(model: LibraryModel) -> int | None:
    return model.gguf.expertBytes if model.gguf is not None else None


def shape_of(model: LibraryModel) -> fit_mod.ModelShape:
    """Recover the KV-cache terms from a stored library entry.

    **This delegates, and that is the whole of review §6.1 #4.** It used
    to read the stored KV dict itself, with a `by_suffix` helper that
    accepted only `int` and never built the per-layer form -- so when
    `attention.head_count_kv` came back as an array (which it is on a
    current mainstream 12B) it fell through to `head_count`, the
    pre-grouped-query assumption, ignored the sliding-window layers
    entirely, and answered tens of GiB of KV where the truth is under
    one. The starter set and the catalogue, which go through
    `preflight.shape_from_gguf`, answered correctly for the same file.
    Two numbers, one file, both labelled `basis: metadata`.

    `routes/guidance.py::_shape_for` had one commit, from M3, and the 43x fix of 2026-09-16
    landed in `preflight.py` and here in `fit.py` and not in it. **A
    second shape builder is the defect; there is one now.**

    The scan keeps selected raw KV pairs on `gguf.metadata` as an escape
    hatch for the long tail, so the material is all here -- what was
    missing was the reading. `GgufMetadata` is rebuilt from that dict
    (with `array_lengths` recovered from the synthetic `<key>.length`
    entries `public_kv` writes, which is how a stepped-over per-layer
    array stays visible as one) and handed to the one reader.

    A safetensors entry has none of this -- the shape lives in
    `config.json`, which the scan reads for architecture and context and
    not for head counts -- so its fit stays an estimate and says so.
    """
    from . import preflight
    from .formats import gguf as gguf_format

    if model.format is not ModelFormat.gguf or model.gguf is None:
        return fit_mod.ModelShape(context_length=model.contextLength, parameters=model.parameters)

    raw = model.gguf.metadata or {}
    kv: dict[str, object] = {}
    array_lengths: dict[str, int] = {}
    for key, value in raw.items():
        # `public_kv` folds a stepped-over array in as `<key>.length`.
        # Splitting it back out is what lets `per_layer_dropped` tell
        # "this file declares the simple form" from "this file has
        # per-layer attention and we did not keep it" -- and those get
        # different answers about whether the number is metadata.
        if key.endswith(".length") and isinstance(value, int):
            array_lengths[key[: -len(".length")]] = value
        else:
            kv[key] = value

    meta = gguf_format.GgufMetadata(
        version=model.gguf.ggufVersion or 0,
        tensor_count=0,
        kv_count=len(kv),
        header_bytes=0,
        kv=kv,
        array_lengths=array_lengths,
    )
    shape = preflight.shape_from_gguf(meta)
    # `context_length` and `parameters` are the library's own reading of
    # the entry rather than the file's -- the scan reconciles a
    # safetensors sidecar and a filename into them -- so they are kept
    # over what the KV block alone would say.
    return replace(
        shape,
        context_length=model.contextLength or shape.context_length,
        parameters=model.parameters,
    )
