"""Grouping an upstream repo into choices, and what the labels say.

The fixture is the real file list of `unsloth/Qwen3.8-27B-GGUF` as the
tree API returned it on 2026-09-08 — 30 `.gguf` entries of which 25 are
choices. A synthetic three-file repo would not have caught either of the
label bugs the real one did.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eugene_plexus_library import catalogue, fit
from eugene_plexus_library._generated.models import (
    Arch,
    Basis,
    FitVerdict,
    GateKind,
    Gpu,
    HostHardware,
    LibraryModel,
    ModelFileRole,
    ModelFormat,
    ModelStatus,
    Os,
    QuantSource,
    Vendor,
)
from eugene_plexus_library.hub import FileMetadata, RepoInfo
from eugene_plexus_library.store import StateStore

GIB = 1024**3
MIB = 1024**2

# (path, size) straight off the tree API. Sizes are the `lfs.size` field.
QWEN_FILES: list[tuple[str, int]] = [
    ("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", 49_986_159_616),
    ("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", 4_671_576_000),
    ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1_369_590_656),
    ("Qwen3.8-27B-Q4_0.gguf", 16_056_478_688),
    ("Qwen3.8-27B-Q4_1.gguf", 17_540_705_248),
    ("Qwen3.8-27B-Q8_0.gguf", 29_047_086_048),
    ("Qwen3.8-27B-UD-IQ1_M.gguf", 6_729_166_848),
    ("Qwen3.8-27B-UD-IQ1_S.gguf", 6_192_222_208),
    ("Qwen3.8-27B-UD-IQ2_S.gguf", 8_371_970_048),
    ("Qwen3.8-27B-UD-IQ2_XXS.gguf", 7_266_070_528),
    ("Qwen3.8-27B-UD-IQ3_S.gguf", 12_040_883_104),
    ("Qwen3.8-27B-UD-IQ3_XXS.gguf", 10_934_860_704),
    ("Qwen3.8-27B-UD-IQ4_XS.gguf", 14_252_845_984),
    ("Qwen3.8-27B-UD-Q2_K_XL.gguf", 9_828_981_664),
    ("Qwen3.8-27B-UD-Q3_K_XL.gguf", 13_146_393_504),
    ("Qwen3.8-27B-UD-Q4_K_M.gguf", 16_464_440_224),
    ("Qwen3.8-27B-UD-Q4_K_S.gguf", 15_358_213_024),
    ("Qwen3.8-27B-UD-Q4_K_XL.gguf", 17_559_178_144),
    ("Qwen3.8-27B-UD-Q5_K_M.gguf", 19_771_509_664),
    ("Qwen3.8-27B-UD-Q5_K_S.gguf", 18_665_753_504),
    ("Qwen3.8-27B-UD-Q5_K_XL.gguf", 20_876_938_144),
    ("Qwen3.8-27B-UD-Q6_K.gguf", 21_983_677_344),
    ("Qwen3.8-27B-UD-Q6_K_L.gguf", 24_193_919_904),
    ("Qwen3.8-27B-UD-Q6_K_M.gguf", 23_088_409_504),
    ("Qwen3.8-27B-UD-Q6_K_XL.gguf", 25_299_061_664),
    ("Qwen3.8-27B-UD-Q8_K_L.gguf", 28_045_695_904),
    ("Qwen3.8-27B-UD-Q8_K_XL.gguf", 31_457_991_680),
    ("imatrix_unsloth.gguf", 13_642_656),
    ("mmproj-BF16.gguf", 931_146_432),
    ("mmproj-F16.gguf", 927_607_488),
]


@pytest.fixture
def files() -> list[FileMetadata]:
    return [
        FileMetadata(path=path, size=size, sha256="a" * 64, lfs=True) for path, size in QWEN_FILES
    ]


@pytest.fixture
def info() -> RepoInfo:
    return RepoInfo(
        repo="unsloth/Qwen3.8-27B-GGUF",
        raw={
            "author": "unsloth",
            "sha": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
            "downloads": 10_675_683,
            "likes": 3708,
            "gated": False,
            "tags": ["gguf", "license:apache-2.0"],
            "gguf": {
                "total": 27_320_697_856,
                "architecture": "qwen35",
                "context_length": 262144,
                "chat_template": "{%- set x = 1 %}",
            },
        },
    )


@pytest.fixture
def budget():
    return fit.budget_from_hardware(
        HostHardware(
            hostname="dev",
            os=Os.windows,
            arch=Arch.x64,
            ramTotalBytes=int(93.56 * GIB),
            ramAvailableBytes=int(58.75 * GIB),
            gpus=[
                Gpu(
                    index=0,
                    name="RTX 5090",
                    vendor=Vendor.nvidia,
                    vramTotalBytes=32607 * MIB,
                    vramFreeBytes=29582 * MIB,
                )
            ],
        )
    )


@pytest.fixture
def store(tmp_path) -> StateStore:
    return StateStore(tmp_path / "state.json")


def build(info, files, store, budget, context=32768):
    return catalogue.build_model(
        info=info,
        files=files,
        revision="main",
        store=store,
        budget=budget,
        context_length=context,
    )


def test_thirty_gguf_files_are_twenty_five_choices(info, files, store, budget) -> None:
    """Listing the `.gguf` files would offer four things that cannot be
    launched and one that is half a model."""
    model = build(info, files, store, budget)
    assert len(model.candidates) == 25
    assert len(model.projectors or []) == 2
    other = {f.path for f in model.otherFiles or []}
    assert "imatrix_unsloth.gguf" in other
    assert "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" in other


def test_every_label_is_distinct(info, files, store, budget) -> None:
    """The bug the live run found: four files read as `UD-Q6_K` because
    only the `Q6_K` part matched llama.cpp's own ftype names, while
    others fell back to their whole filename. Four identical rows on a
    screen whose job is "pick one" is the worst possible outcome."""
    model = build(info, files, store, budget)
    labels = [c.label for c in model.candidates]
    assert len(set(labels)) == len(labels)
    assert "UD-Q6_K" in labels
    assert "UD-Q6_K_M" in labels
    assert "UD-Q6_K_L" in labels
    assert "UD-Q6_K_XL" in labels


def test_labels_carry_no_model_name(info, files, store, budget) -> None:
    """`Qwen3.8-27B-UD-Q4_K_XL` as a label is the model's name repeated
    25 times with the useful part at the end."""
    model = build(info, files, store, budget)
    assert all("Qwen" not in c.label for c in model.candidates)
    assert {"Q4_0", "Q4_1", "Q8_0", "BF16"} <= {c.label for c in model.candidates}


def test_a_tier_publishers_invented_is_still_a_quant(info, files, store, budget) -> None:
    """`Q4_K_XL` and `UD-Q8_K_XL` exist in no upstream enum. They are
    still quants, and reporting them as unlabelled would hide them."""
    model = build(info, files, store, budget)
    by_label = {c.label: c for c in model.candidates}
    assert by_label["UD-Q4_K_XL"].quantization == "UD-Q4_K_XL"
    assert by_label["UD-Q8_K_XL"].quantization == "UD-Q8_K_XL"
    assert all(c.quantSource is QuantSource.filename for c in model.candidates if c.quantization)


def test_a_split_candidate_is_summed_not_sampled(info, files, store, budget) -> None:
    """Shard 1 is 46.55 GiB and the model is 50.90 GiB. Scoring the file
    on the launch line understates it by every other shard."""
    model = build(info, files, store, budget)
    bf16 = next(c for c in model.candidates if c.label == "BF16")
    assert len(bf16.files) == 2
    assert bf16.sizeBytes == 49_986_159_616 + 4_671_576_000
    assert round(bf16.sizeBytes / GIB, 2) == 50.90
    assert [f.role for f in bf16.files] == [ModelFileRole.weights, ModelFileRole.shard]


def test_bits_per_weight_orders_the_whole_repo(info, files, store, budget) -> None:
    """Monotonic from 1.81 to exactly 16.00, which is the check that the
    hub's `gguf.total` really is a parameter count."""
    model = build(info, files, store, budget)
    series = [c.bitsPerWeight for c in model.candidates]
    assert all(b is not None for b in series)
    assert series == sorted(series)  # type: ignore[type-var]
    assert series[0] == 1.81
    assert series[-1] == 16.0


def test_the_recommendation_is_the_largest_that_fully_fits(info, files, store, budget) -> None:
    model = build(info, files, store, budget)
    assert model.recommended is not None
    fitting = [c for c in model.candidates if c.fit and c.fit.verdict is FitVerdict.fits]
    largest = max(fitting, key=lambda c: c.sizeBytes)
    assert model.recommended.label == largest.label
    # The reason names the arithmetic: differentiator #6 is "show why".
    assert "32,768 tokens" in model.recommended.reason
    assert "bits per weight" in model.recommended.reason


def test_partial_offload_is_never_recommended(info, files, store, budget) -> None:
    """`split` runs and is materially slower. That is a decision to make
    knowingly, not one to inherit from a recommendation."""
    model = build(info, files, store, budget)
    assert model.recommended is not None
    recommended = next(c for c in model.candidates if c.label == model.recommended.label)
    assert recommended.fit is not None
    assert recommended.fit.verdict is FitVerdict.fits


def test_a_tiny_budget_warns_instead_of_recommending_one_bit(info, files, store) -> None:
    """Recommending a 1.81-bpw quant without comment is how a first
    impression becomes "this thing is stupid"."""
    small = fit.budget_from_hardware(
        HostHardware(
            hostname="small",
            os=Os.linux,
            arch=Arch.x64,
            ramTotalBytes=16 * GIB,
            ramAvailableBytes=12 * GIB,
            gpus=[Gpu(index=0, name="RTX 3060", vramTotalBytes=12 * GIB, vramFreeBytes=11 * GIB)],
        )
    )
    model = build(info, files, store, small, context=4096)
    assert model.recommended is not None
    assert model.recommended.lowQualityWarning is not None
    assert "bits per weight" in model.recommended.lowQualityWarning
    assert any("bits per weight" in w for w in model.warnings or [])


def test_nothing_fits_says_so_and_names_the_closest(info, files, store) -> None:
    tiny = fit.budget_from_hardware(
        HostHardware(
            hostname="tiny",
            os=Os.linux,
            arch=Arch.x64,
            ramTotalBytes=4 * GIB,
            ramAvailableBytes=2 * GIB,
            gpus=[],
        )
    )
    model = build(info, files, store, tiny)
    assert model.recommended is None
    assert model.warnings
    assert any("smaller model" in w for w in model.warnings)


def test_a_multimodal_repo_says_the_projector_is_separate(info, files, store, budget) -> None:
    """ "This model can see" is a fact the browser should show, and the
    projector is a second download the operator has to choose."""
    model = build(info, files, store, budget)
    assert any("projector" in w for w in model.warnings or [])
    assert all(p.role is ModelFileRole.projector for p in model.projectors or [])


def test_a_gated_repo_warns_before_the_download(info, files, store, budget) -> None:
    """Metadata, sizes and digests are public; only the bytes are not. So
    the warning has to arrive on the detail screen rather than as a 403
    after a quant has been chosen."""
    info.raw["gated"] = "manual"
    model = build(info, files, store, budget)
    assert model.gated is GateKind.manual
    assert model.warnings
    assert "manual approval" in model.warnings[0]
    assert all(c.gated for c in model.candidates)

    info.raw["gated"] = "auto"
    auto = build(info, files, store, budget)
    assert auto.gated is GateKind.auto
    assert "licence" in auto.warnings[0]


def test_a_catalogue_fit_is_an_estimate_until_preflighted(info, files, store, budget) -> None:
    """The hub reports no layer or head counts, so the KV term cannot be
    real yet — and saying so is the whole point of the field."""
    model = build(info, files, store, budget)
    assert all(c.fit and c.fit.basis is Basis.estimate for c in model.candidates)


def test_a_file_already_on_disk_is_flagged(info, files, store, budget, tmp_path) -> None:
    """The join only this component can make, and what stops a 16 GB
    re-download of something the operator already has."""
    owned = tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf"
    owned.write_bytes(b"x")
    store.replace_models(
        [
            LibraryModel(
                id="abc123",
                path=str(owned),
                format=ModelFormat.gguf,
                name="Qwen3.8-27B-UD-Q4_K_M",
                status=ModelStatus.present,
                sizeBytes=16_464_440_224,
            )
        ],
        scanned_at=datetime.now(tz=UTC),
    )
    model = build(info, files, store, budget)
    flagged = [c for c in model.candidates if c.alreadyOwned]
    assert len(flagged) == 1
    assert flagged[0].label == "UD-Q4_K_M"
    assert flagged[0].alreadyOwned is not None
    assert flagged[0].alreadyOwned.modelId == "abc123"


def test_a_same_named_file_of_a_different_size_is_not_the_same_file(
    info, files, store, budget, tmp_path
) -> None:
    """A requantized re-upload keeps its filename. Claiming the operator
    owns it would talk them out of a download they need."""
    owned = tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf"
    owned.write_bytes(b"x")
    store.replace_models(
        [
            LibraryModel(
                id="abc123",
                path=str(owned),
                format=ModelFormat.gguf,
                name="Qwen3.8-27B-UD-Q4_K_M",
                status=ModelStatus.present,
                sizeBytes=999,
            )
        ],
        scanned_at=datetime.now(tz=UTC),
    )
    model = build(info, files, store, budget)
    assert not any(c.alreadyOwned for c in model.candidates)


def test_a_safetensors_repo_keeps_its_sidecars() -> None:
    """`config.json` and the tokenizer files are as necessary as the
    weights: a download that fetched only the `.safetensors` files
    produces a directory nothing can load."""
    repo_files = [
        FileMetadata(path=".gitattributes", size=1000),
        FileMetadata(path="README.md", size=20_000),
        FileMetadata(path="config.json", size=728),
        FileMetadata(path="generation_config.json", size=200),
        FileMetadata(path="merges.txt", size=1_670_000),
        FileMetadata(path="model-00001-of-00002.safetensors", size=4_000_000_000, lfs=True),
        FileMetadata(path="model-00002-of-00002.safetensors", size=1_200_000_000, lfs=True),
        FileMetadata(path="model.safetensors.index.json", size=30_000),
        FileMetadata(path="tokenizer.json", size=11_420_000, lfs=True),
        FileMetadata(path="vocab.json", size=2_780_000),
    ]
    groups, other = catalogue._safetensors_groups(repo_files, repo="Qwen/Qwen3-8B")
    assert len(groups) == 1
    paths = {f.path for f in groups[0].files}
    assert "config.json" in paths
    assert "tokenizer.json" in paths
    assert "model.safetensors.index.json" in paths
    assert "README.md" not in paths  # documentation, not the model
    assert groups[0].format is ModelFormat.safetensors
    assert {f.path for f in other} == {"README.md", ".gitattributes"}


def test_search_rows_carry_no_sizes() -> None:
    """Upstream's search response has filenames without sizes, so a fit
    verdict per row would cost a call per row."""
    row = catalogue.build_search_result(
        {
            "id": "unsloth/Qwen3.8-27B-GGUF",
            "author": "unsloth",
            "downloads": 10_675_683,
            "likes": 3708,
            "trendingScore": 26,
            "gated": False,
            "tags": ["gguf", "license:apache-2.0"],
            "library_name": "gguf",
            "siblings": [{"rfilename": "Qwen3.8-27B-UD-Q4_K_M.gguf"}],
        }
    )
    assert row.repo == "unsloth/Qwen3.8-27B-GGUF"
    assert row.owner == "unsloth"
    assert row.license == "apache-2.0"
    assert row.formats == [ModelFormat.gguf]
    assert row.gated is GateKind.open
    assert not hasattr(row, "sizeBytes")


def test_the_candidate_size_helper_finds_the_whole_set(files) -> None:
    assert (
        catalogue.candidate_size(files, "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf")
        == 49_986_159_616 + 4_671_576_000
    )
    assert catalogue.candidate_size(files, "mmproj-F16.gguf") is None


def test_a_single_candidate_repo_keeps_its_filename_as_a_label() -> None:
    """With one candidate there is no shared prefix to strip, and an
    empty label would be worse than a long one."""
    one = [FileMetadata(path="model.gguf", size=1000)]
    groups, _, _ = catalogue._gguf_groups(one)
    assert len(groups) == 1
    assert groups[0].label == "model"
