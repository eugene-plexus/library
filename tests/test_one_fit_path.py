"""R1.3: the on-disk fit route reads the same shape as every other path.

Roadmap `docs/design/release-roadmap.md` §2.3, finding review §6.1 #4.

**Four call paths produce a fit verdict and only one was wrong — the one
the golden path uses.** Catalogue search and detail, preflight and the
starter set all build their shape through `preflight.shape_from_gguf`,
which calls `per_layer_kv` and `attention_layers`. `GET
/v1/models/{id}/fit` built its own in `routes/guidance.py::_shape_for`,
whose `by_suffix` accepted only `int` and never set `layers`. That route
has one commit in its history, from M3, and the 43x KV fix of 2026-09-16
landed everywhere but there.

So the same file was `fits` at 262,144 on one screen and `no` at 4,864
on another, both reporting `basis: metadata`, with nothing telling a
person which was the lie — and the wrong one is what one-click Run
writes into the `default` profile.

**The check is over a written GGUF, not a hand-built `ModelShape`.** A
`ModelShape` fixture would pass against `_shape_for` unchanged, because
the defect is in the reading and not in the arithmetic; the arithmetic
has had per-layer tests since S6. The file here declares what a real
current 12B declares: `head_count_kv` as a 48-element array and
`sliding_window_pattern` as a 48-element **bool** array, five layers in
every six sliding over a 1024-token window.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_library import fit as fit_mod
from eugene_plexus_library.formats import gguf
from eugene_plexus_library.routes import guidance

from .conftest import qwen_like_kv, write_gguf

GIB = 1024**3

# A real mainstream 12B's shape, and the numbers that matter: 48 blocks,
# `head_count_kv` an array (8 on the full-attention layer of each six, 1
# on the five sliding ones), `sliding_window_pattern` a bool array, and
# a 1024-token window with its own shorter key/value lengths.
BLOCKS = 48
PATTERN = [(index % 6) != 5 for index in range(BLOCKS)]
HEADS_KV = [1 if slides else 8 for slides in PATTERN]


def per_layer_kv(architecture: str = "llama", *, blocks: int = BLOCKS) -> dict[str, Any]:
    pattern = [(index % 6) != 5 for index in range(blocks)]
    return {
        f"{architecture}.attention.head_count_kv": [1 if s else 8 for s in pattern],
        f"{architecture}.attention.sliding_window_pattern": pattern,
        f"{architecture}.attention.sliding_window": 1024,
        f"{architecture}.attention.key_length": 128,
        f"{architecture}.attention.value_length": 128,
        f"{architecture}.attention.key_length_swa": 128,
        f"{architecture}.attention.value_length_swa": 128,
        f"{architecture}.attention.head_count": 32,
        f"{architecture}.block_count": blocks,
    }


def scanned(client: TestClient, models_dir: Path, kv: dict[str, Any]) -> str:
    write_gguf(models_dir / "m.gguf", kv)

    def wait_for_scan():
        deadline = time.perf_counter() + 5
        while client.get("/v1/scan").json()["state"] == "scanning":
            assert time.perf_counter() < deadline, "fixture scan did not complete"
            time.sleep(0.01)

    # Startup may already be scanning an empty folder. Wait for it before
    # requesting the scan that must include the fixture we just wrote.
    wait_for_scan()
    assert client.post("/v1/scan").status_code == 202
    wait_for_scan()
    models = client.get("/v1/models").json()["models"]
    assert models, "the fixture model was not scanned"
    return str(models[0]["id"])


# -- the finding ------------------------------------------------------------


def test_the_on_disk_route_reads_the_per_layer_cache(
    configured_client: TestClient, models_dir: Path
) -> None:
    """The roadmap's *Done when*, verbatim.

    Before this slice the array read as `None`, fell back to
    `head_count` (32, the pre-grouped-query assumption), ignored the
    sliding window entirely, and reported tens of GiB of KV at 16k for a
    model whose real cache is under a gigabyte.
    """
    kv = qwen_like_kv(vocab=64)
    kv.update(per_layer_kv())
    model_id = scanned(configured_client, models_dir, kv)

    body = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 16384}
    ).json()
    assert body["fit"]["basis"] == "metadata"
    assert body["fit"]["kvCacheBytes"] < GIB, (
        f"{body['fit']['kvCacheBytes'] / GIB:.2f} GiB of KV at 16k for a model whose "
        "five-in-six layers slide over 1024 tokens"
    )


def test_the_route_and_the_preflight_agree_on_the_same_file(
    configured_client: TestClient, models_dir: Path
) -> None:
    """Two screens, one file, one number.

    The starter panel and the Library detail printed 262,144 and 4,864
    for the same file on one install. Whatever the answer is, there is
    one of it — so this compares the route's shape against the shape
    every other path builds, from the same bytes.
    """
    kv = qwen_like_kv(vocab=64)
    kv.update(per_layer_kv())
    model_id = scanned(configured_client, models_dir, kv)

    meta = gguf.read_metadata(models_dir / "m.gguf")
    from eugene_plexus_library import preflight

    expected = preflight.shape_from_gguf(meta)
    model = configured_client.get(f"/v1/models/{model_id}").json()

    from eugene_plexus_library._generated.models import LibraryModel

    actual = guidance._shape_for(LibraryModel.model_validate(model))
    for context in (4096, 16384, 131072):
        assert actual.kv_bytes(context, fit_mod.KvCacheType.f16) == expected.kv_bytes(
            context, fit_mod.KvCacheType.f16
        ), f"the two paths disagree at {context:,} tokens"


def test_the_context_control_still_moves_the_answer(
    configured_client: TestClient, models_dir: Path
) -> None:
    """A sliding layer stops growing at its window, so the cache is
    affine in context rather than linear — but it must still MOVE.

    A fix that made every layer slide would produce a constant, which is
    the defect `context-control-was-inert` is about, one module over.
    """
    kv = qwen_like_kv(vocab=64)
    kv.update(per_layer_kv())
    model_id = scanned(configured_client, models_dir, kv)

    small = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 4096}
    ).json()["fit"]["kvCacheBytes"]
    large = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 131072}
    ).json()["fit"]["kvCacheBytes"]
    assert large > small


# -- the two forms of hybrid ------------------------------------------------


def test_layer_indices_is_honoured_as_well_as_the_interval(
    configured_client: TestClient, models_dir: Path
) -> None:
    """`_shape_for` handled only `full_attention_interval`.
    `attention.layer_indices` is the other convention in the wild, and
    `attention_layers()` has honoured both since M3 — the route just
    never called it."""
    kv = qwen_like_kv(vocab=64)
    kv["llama.block_count"] = 64
    kv["llama.attention.layer_indices"] = [index for index in range(64) if index % 4 == 0]
    kv["llama.attention.head_count_kv"] = 8
    kv["llama.attention.key_length"] = 128
    kv["llama.attention.value_length"] = 128
    model_id = scanned(configured_client, models_dir, kv)

    body = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 16384}
    ).json()
    assert body["fit"]["attentionLayers"] == 16


def test_the_interval_form_still_works(configured_client: TestClient, models_dir: Path) -> None:
    """The one form the old `_shape_for` did get right. Delegating must
    not lose it."""
    kv = qwen_like_kv(vocab=64)
    kv["llama.block_count"] = 64
    kv["llama.full_attention_interval"] = 4
    kv["llama.attention.head_count_kv"] = 8
    kv["llama.attention.key_length"] = 128
    kv["llama.attention.value_length"] = 128
    model_id = scanned(configured_client, models_dir, kv)

    body = configured_client.get(f"/v1/models/{model_id}/fit").json()
    assert body["fit"]["attentionLayers"] == 16


def test_an_ordinary_scalar_model_is_unchanged(
    configured_client: TestClient, models_dir: Path
) -> None:
    """Most files, and all the older ones, declare the simple form. The
    fix must not move their numbers."""
    kv = qwen_like_kv(vocab=64)
    kv["llama.attention.head_count_kv"] = 8
    kv["llama.attention.key_length"] = 128
    kv["llama.attention.value_length"] = 128
    model_id = scanned(configured_client, models_dir, kv)

    body = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 4096}
    ).json()["fit"]
    # 32 blocks x 4096 tokens x 8 heads x (128 + 128) x 2 bytes
    assert body["kvCacheBytes"] == 32 * 4096 * 8 * 256 * 2
    assert body["basis"] == "metadata"


def test_a_safetensors_model_still_estimates(
    configured_client: TestClient, models_dir: Path
) -> None:
    """The shape lives in `config.json`, which the scan reads for
    architecture and context and not for head counts. Its fit stays an
    estimate and says so — the early return must survive the delegation."""
    directory = models_dir / "st"
    directory.mkdir()
    (directory / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 4096}',
        encoding="utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    model_id = scanned(configured_client, models_dir, qwen_like_kv(vocab=64))
    del model_id

    models = configured_client.get("/v1/models").json()["models"]
    safetensors = [m for m in models if m["format"] == "safetensors"]
    if not safetensors:
        pytest.skip("the safetensors fixture did not scan; covered by test_scanner")
    body = configured_client.get(f"/v1/models/{safetensors[0]['id']}/fit").json()
    assert body["fit"]["basis"] == "estimate"


# -- the array that was silently dropped ------------------------------------


def test_a_per_layer_array_too_long_to_store_is_not_called_metadata(
    configured_client: TestClient, models_dir: Path
) -> None:
    """**Latent, with a margin of one.**

    `_INLINE_ARRAY_LIMIT` decided which arrays the scan keeps, and a
    per-layer array longer than the limit was stepped over and stored as
    a length. The shape then fell back to the scalars and the fit still
    said `basis: metadata` — the one word that tells a person the number
    is arithmetic rather than a guess. The shipped starter block counts
    are 32, 42, 48 and **65**.

    Two things had to change: the limit has to clear a real block count,
    and a fit whose per-layer terms were dropped has to stop claiming
    they were read.
    """
    kv = qwen_like_kv(vocab=64)
    kv.update(per_layer_kv(blocks=65))
    model_id = scanned(configured_client, models_dir, kv)

    body = configured_client.get(
        f"/v1/models/{model_id}/fit", params={"contextLength": 16384}
    ).json()
    assert body["fit"]["kvCacheBytes"] < GIB, (
        "a 65-layer per-layer array was dropped and the scalars answered instead"
    )
    assert body["fit"]["basis"] == "metadata"


def test_a_shape_that_lost_its_per_layer_terms_says_so() -> None:
    """The honest half, for a library scanned before this fix.

    Stored metadata persists, so an existing entry still has the array
    missing and its length recorded. That fit is a scalar guess in the
    direction that OOMs, and `basis: metadata` would be a lie about it.
    """
    shape = fit_mod.ModelShape(
        block_count=48,
        attention_layers=48,
        head_count_kv=8,
        key_length=128,
        value_length=128,
        per_layer_unavailable=True,
    )
    from eugene_plexus_library._generated.models import Basis, MemoryBudget, Source

    result = fit_mod.compute(
        weights_bytes=7 * GIB,
        budget=MemoryBudget(vramFreeBytes=32 * GIB, source=Source.override),
        context_length=16384,
        shape=shape,
    )
    assert result.basis == Basis.estimate
    assert any("per-layer" in note for note in result.notes)


# -- what the sabotage pass found missing -----------------------------------
#
# Three sabotages escaped the first pass, and each named a check that was
# not here rather than a guard that was not needed. They are below.


def _entry(metadata: dict[str, Any], **overrides: Any) -> Any:
    """A stored library entry, built directly.

    Some states are only reachable through history -- an entry scanned
    before a fix, whose stored metadata still has the gap -- and a
    fixture that can only be produced by scanning cannot express them.
    """
    from eugene_plexus_library._generated.models import LibraryModel

    body: dict[str, Any] = {
        "id": "abc123",
        "name": "stored",
        "path": "/models/stored.gguf",
        "format": "gguf",
        "sizeBytes": 7 * GIB,
        "fileCount": 1,
        "files": [],
        "status": "present",
        "gguf": {"ggufVersion": 3, "metadata": metadata},
    }
    body.update(overrides)
    return LibraryModel.model_validate(body)


def test_the_entrys_own_context_wins_over_the_kv_blocks() -> None:
    """The scan reconciles a filename, a sidecar and the KV block into
    one `contextLength`, and that reading is the library's answer. The
    shape builder reads the KV block alone, so the route puts the
    entry's value back — and `max_context_that_fits` uses it as the
    ceiling, so a disagreement is a number on a screen."""
    shape = guidance._shape_for(
        _entry(
            {
                "general.architecture": "llama",
                "llama.context_length": 262144,
                "llama.block_count": 32,
                "llama.attention.head_count_kv": 8,
                "llama.attention.key_length": 128,
                "llama.attention.value_length": 128,
            },
            contextLength=40960,
        )
    )
    assert shape.context_length == 40960


def test_a_library_scanned_before_this_fix_stops_claiming_metadata() -> None:
    """The state on every install that scanned under the old array
    limit: the per-layer array was stepped over, so the entry carries
    its LENGTH and not its values.

    Re-scanning fixes it; until then the scalars answer, and they
    over-estimate by up to 43x. `basis` is the one word that tells a
    person the number is arithmetic rather than a guess, so it must not
    say `metadata` here. Built as a stored entry because a scan with the
    current limit can no longer produce one.
    """
    shape = guidance._shape_for(
        _entry(
            {
                "general.architecture": "llama",
                "llama.context_length": 262144,
                "llama.block_count": 65,
                # What `public_kv` writes for an array it stepped over:
                # the length, and no value.
                "llama.attention.head_count_kv.length": 65,
                "llama.attention.head_count": 32,
                "llama.attention.key_length": 128,
                "llama.attention.value_length": 128,
            }
        )
    )
    assert shape.layers is None
    assert shape.per_layer_unavailable is True

    from eugene_plexus_library._generated.models import Basis, MemoryBudget, Source

    result = fit_mod.compute(
        weights_bytes=7 * GIB,
        budget=MemoryBudget(vramFreeBytes=32 * GIB, source=Source.override),
        context_length=16384,
        shape=shape,
    )
    assert result.basis == Basis.estimate


def test_a_file_that_simply_declares_the_scalars_is_not_accused(
    configured_client: TestClient, models_dir: Path
) -> None:
    """The other side of the same rule, and the one that decides whether
    it is usable: most files, and all the older ones, declare the simple
    form and nothing was dropped. Calling those an estimate would put a
    warning on almost every model in a library."""
    kv = qwen_like_kv(vocab=64)
    kv["llama.attention.head_count_kv"] = 8
    kv["llama.attention.key_length"] = 128
    kv["llama.attention.value_length"] = 128
    model_id = scanned(configured_client, models_dir, kv)
    assert configured_client.get(f"/v1/models/{model_id}/fit").json()["fit"]["basis"] == "metadata"


def test_the_fixture_writes_a_real_bool_array(models_dir: Path) -> None:
    """The reader tolerates an int array of 0/1 and answers the same, so
    nothing else here can tell the two apart — which means without this
    the suite would be asserting about a type the reader never meets in
    the wild. `bool` is an `int` in Python, so an encoder that checks
    `int` first writes every one of these as int32.

    The roadmap's *Done when* names a bool array specifically. This is
    what makes that sentence true rather than claimed.
    """
    kv = qwen_like_kv(vocab=64)
    kv.update(per_layer_kv())
    write_gguf(models_dir / "typed.gguf", kv)
    meta = gguf.read_metadata(models_dir / "typed.gguf")
    pattern = meta.arch_key("attention.sliding_window_pattern")
    assert isinstance(pattern, list)
    assert all(isinstance(value, bool) for value in pattern), (
        f"the pattern came back as {type(pattern[0]).__name__}, not bool"
    )
