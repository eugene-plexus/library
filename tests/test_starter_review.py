"""The review's ranking, drops and verdicts.

Every fixture here is shaped from a real listing row measured against
the live hub, because four of the design's assumptions about that shape
were wrong and fixtures invented from the assumptions would have locked
all four in.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from eugene_plexus_library import starter_review as review

from .test_starter import cpu_proof


def row(
    repo: str,
    *,
    downloads: int,
    base: Any = None,
    total: int | None = 9_000_000_000,
    architecture: str = "qwen35",
    chat_template: str | None = "{{ messages }}",
    gated: Any = False,
    license: str | None = "apache-2.0",
) -> dict[str, Any]:
    gguf: dict[str, Any] = {"architecture": architecture, "context_length": 262144}
    if total is not None:
        gguf["total"] = total
    if chat_template is not None:
        gguf["chat_template"] = chat_template
    card: dict[str, Any] = {}
    if base is not None:
        card["base_model"] = base
    if license is not None:
        card["license"] = license
    return {
        "id": repo,
        "downloads": downloads,
        "gguf": gguf,
        "cardData": card,
        "gated": gated,
        "tags": [],
    }


# --- what the listing actually looks like ------------------------------


def test_mirrors_of_one_model_are_one_candidate() -> None:
    """Five publishers mirroring one model are one candidate, not five.
    Without this the ranking is a popularity contest between quantizers."""
    candidates, _ = review.rank(
        [
            row("unsloth/M-GGUF", downloads=100, base="Vendor/M"),
            row("bartowski/Vendor_M-GGUF", downloads=60, base="Vendor/M"),
            row("ggml-org/M-GGUF", downloads=40, base="Vendor/M"),
        ]
    )
    assert len(candidates) == 1
    only = next(iter(candidates.values()))
    assert only.downloads == 200
    assert only.name == "Vendor/M"


def test_capitalisation_does_not_split_a_family() -> None:
    """Measured: `google/gemma-4-E4B-it` and `google/gemma-4-e4b-it` are
    one model spelled two ways by two publishers. Unnormalised they rank
    as two candidates with half the downloads each, and neither wins."""
    candidates, _ = review.rank(
        [
            row("a/x-GGUF", downloads=2_528_565, base="google/gemma-4-E4B-it"),
            row("b/x-GGUF", downloads=2_133_271, base="google/gemma-4-e4b-it"),
        ]
    )
    assert len(candidates) == 1
    assert next(iter(candidates.values())).downloads == 4_661_836


def test_a_repo_with_no_chat_template_is_dropped() -> None:
    """The discriminator that works. `pipeline_tag` does not: adding
    `filter=text-generation` to the listing call dropped the second
    most-downloaded GGUF repo on the hub, because the field is simply
    absent on many repos. ASR, TTS, embedding and projector repos all
    rank in the top twenty and all lack a chat template."""
    candidates, dropped = review.rank(
        [
            row("audio-cpp/audio.cpp-gguf", downloads=3_800_350, chat_template=None),
            row("unsloth/M-GGUF", downloads=10, base="Vendor/M"),
        ]
    )
    assert list(candidates) == ["vendor/m"]
    assert dropped[0]["repo"] == "audio-cpp/audio.cpp-gguf"
    assert "chat template" in dropped[0]["why"]


def test_a_gated_repo_is_dropped_with_its_reason() -> None:
    _, dropped = review.rank([row("meta-llama/M-GGUF", downloads=9, gated="manual")])
    assert dropped and "gated" in dropped[0]["why"]


def test_sixty_base_models_is_a_bundle_not_a_derivative() -> None:
    """Measured: one top-twenty repo lists sixty base models. Merging it
    into sixty families would move its downloads into every one."""
    candidates, _ = review.rank(
        [row("audio-cpp/bundle", downloads=500, base=[f"v/m{i}" for i in range(60)])]
    )
    only = next(iter(candidates.values()))
    assert only.key == "audio-cpp/bundle"
    assert any("many base models" in flag for flag in only.flags)


def test_a_repo_with_no_base_model_counts_as_itself_and_is_flagged() -> None:
    """Measured in the top three: a repo declaring no `base_model` at
    all. Counting it as itself is right; ranking it unflagged is not,
    because nothing connects it to a model anyone can name."""
    candidates, _ = review.rank([row("ornith-ai/Ornith-1.5-9B-GGUF", downloads=5_320_513)])
    only = next(iter(candidates.values()))
    assert any("no base_model" in flag for flag in only.flags)


def test_the_parameter_count_is_the_mode_of_contributors() -> None:
    """Measured: the hub's repo-level `gguf.total` is read off one file
    and is sometimes the wrong one -- a 27B repo reported 0.5B because
    the projector was read. Any single reading can be wrong; agreement
    is the only signal available without downloading something."""
    candidates, _ = review.rank(
        [
            row("a/M-GGUF", downloads=30, base="V/M", total=27_320_697_856),
            row("b/M-GGUF", downloads=20, base="V/M", total=27_320_697_856),
            row("c/M-GGUF", downloads=10, base="V/M", total=500_000_000),
        ]
    )
    assert next(iter(candidates.values())).parameter_count == 27_320_697_856


# --- flags, and what they do -------------------------------------------


def test_a_specialised_model_is_flagged_out_of_the_ranking() -> None:
    """A coder model is a good model and the wrong answer to "I have
    never run one of these before"."""
    candidates, _ = review.rank(
        [row("unsloth/C-GGUF", downloads=12_866_779, base="Qwen/Qwen3-Coder-30B-A3B-Instruct")]
    )
    assert review.eligible(candidates)["30B"] == []


def test_a_de_aligned_derivative_is_never_a_default() -> None:
    candidates, _ = review.rank(
        [row("x/Y-GGUF", downloads=9_000_000, base="DavidAU/Qwen3.5-9B-Uncensored-Heretic")]
    )
    only = next(iter(candidates.values()))
    assert any("de-aligned" in flag for flag in only.flags)
    assert review.eligible(candidates)["8B"] == []


def test_a_model_only_one_repo_mirrors_is_thin_evidence() -> None:
    """One repo is a publisher, not a consensus, and the ranking is a
    consensus measure. Flagged into the report for a human rather than
    proposed."""
    candidates, _ = review.rank([row("bartowski/Solo-GGUF", downloads=241_351, base="V/Solo")])
    only = next(iter(candidates.values()))
    assert any("only one repo" in flag for flag in only.flags)


def test_an_unknown_publisher_at_the_top_surfaces_rather_than_vanishing() -> None:
    """The known-publisher list is itself a curation that ages, which is
    why an unknown publisher is a flag and not a silent drop."""
    candidates, _ = review.rank(
        [
            row("nobody/M-GGUF", downloads=100, base="V/M"),
            row("alsonobody/M-GGUF", downloads=90, base="V/M"),
        ]
    )
    only = next(iter(candidates.values()))
    assert any("known list" in flag for flag in only.flags)
    assert review.eligible(candidates)["8B"] == []


def test_a_licence_off_the_allowlist_is_a_flag_not_a_rejection() -> None:
    candidates, _ = review.rank(
        [
            row("unsloth/M-GGUF", downloads=100, base="V/M", license="openmdw-1.1"),
            row("bartowski/M-GGUF", downloads=90, base="V/M", license="openmdw-1.1"),
        ]
    )
    only = next(iter(candidates.values()))
    assert any("allowlist" in flag for flag in only.flags)


def test_size_classes_bucket_by_total_parameters() -> None:
    """Total, not active: every expert of a mixture-of-experts model is
    resident, so 30B-A3B occupies a 30B's worth of memory, which is the
    only thing this bucket is used for."""
    candidates, _ = review.rank(
        [
            row("unsloth/A", downloads=10, base="V/A", total=4_205_751_296),
            row("bartowski/A", downloads=9, base="V/A", total=4_205_751_296),
            row("unsloth/B", downloads=10, base="V/B", total=30_532_122_624),
            row("bartowski/B", downloads=9, base="V/B", total=30_532_122_624),
        ]
    )
    by_class = review.eligible(candidates)
    assert [c.name for c in by_class["4B"]] == ["V/A"]
    assert [c.name for c in by_class["30B"]] == ["V/B"]


# --- verdicts ----------------------------------------------------------


def candidate(name: str, downloads: int, *, architecture: str = "qwen35") -> review.Candidate:
    c = review.Candidate(key=name.lower())
    c.names[name] = 1
    c.downloads = downloads
    c.architectures[architecture] = 1
    c.repos = [("unsloth/x", downloads)]
    return c


ARCHES = {"qwen35", "gemma4"}


def test_the_current_leader_is_kept() -> None:
    v = review.verdict_for(
        "8B",
        current={"baseModel": "V/M", "class": "8B"},
        ranked=[candidate("V/M", 100)],
        architectures=ARCHES,
        streak={},
    )
    assert v.verdict == "KEEP"


def test_a_new_leader_needs_two_reviews_before_it_is_proposed() -> None:
    """A launch-week spike is not a trend, and hysteresis is the only
    defence against one. It is also why the review runs monthly rather
    than on release day: on release day there is no second month."""
    first = review.verdict_for(
        "8B",
        current={"baseModel": "V/Old", "class": "8B"},
        ranked=[candidate("V/New", 100), candidate("V/Old", 95)],
        architectures=ARCHES,
        streak={},
    )
    assert first.verdict == "REVIEW"
    assert "1 of 2" in first.detail

    second = review.verdict_for(
        "8B",
        current={"baseModel": "V/Old", "class": "8B"},
        ranked=[candidate("V/New", 100), candidate("V/Old", 95)],
        architectures=ARCHES,
        streak={"v/new": 1},
    )
    assert second.verdict == "REPLACE"


def test_falling_out_of_the_top_three_replaces_immediately() -> None:
    v = review.verdict_for(
        "8B",
        current={"baseModel": "V/Old", "class": "8B"},
        ranked=[candidate("V/A", 100), candidate("V/B", 90), candidate("V/C", 80)],
        architectures=ARCHES,
        streak={},
    )
    assert v.verdict == "REPLACE"


def test_an_architecture_the_pinned_engine_cannot_load_is_never_proposed() -> None:
    """A model the engine we install cannot load is the one
    recommendation that would be worse than none."""
    v = review.verdict_for(
        "8B",
        current={"baseModel": "V/Old", "class": "8B"},
        ranked=[candidate("V/New", 10_000, architecture="somethingnew")],
        architectures=ARCHES,
        streak={"v/new": 5},
    )
    assert v.verdict == "REVIEW"
    assert "somethingnew" in v.detail


def test_a_failed_engine_lookup_is_review_never_a_silent_keep() -> None:
    v = review.verdict_for(
        "8B",
        current={"baseModel": "V/M", "class": "8B"},
        ranked=[candidate("V/M", 100)],
        architectures=None,
        streak={},
    )
    assert v.verdict == "REVIEW"
    assert v.check.startswith("none")


def test_the_top_two_being_close_is_said_out_loud() -> None:
    v = review.verdict_for(
        "8B",
        current={"baseModel": "V/Old", "class": "8B"},
        ranked=[candidate("V/New", 100), candidate("V/Old", 95)],
        architectures=ARCHES,
        streak={},
    )
    assert "within 20%" in v.detail


# --- picking the file --------------------------------------------------


def test_k_quants_beat_i_quants_and_legacy_layouts_at_the_same_width() -> None:
    """The first run of this review chose `IQ4_XS` for one class and
    `Q4_0` for another, both within 0.1 bits of `Q4_K_M` and neither the
    file a stranger should be handed. An I-quant needs an importance
    matrix and is materially slower on CPU and several GPU backends;
    `Q4_0` is a legacy layout kept for compatibility."""
    assert review.quant_rank("Q4_K_M") == 0
    assert review.quant_rank("UD-Q4_K_XL") == 0
    assert review.quant_rank("Q5_K_S") == 1
    assert review.quant_rank("Q6_K") == 1
    assert review.quant_rank("IQ4_XS") == 2
    assert review.quant_rank("Q4_0") == 2


class _Group:
    def __init__(self, label: str, size: int) -> None:
        self.label = label
        self.size = size


def test_the_chosen_quant_is_the_preferred_family_nearest_the_target() -> None:
    chosen = review._choose_quant(
        [
            _Group("Q8_0", 8_000_000_000),
            _Group("IQ4_XS", 4_100_000_000),
            _Group("Q4_K_M", 4_800_000_000),
            _Group("Q2_K", 2_000_000_000),
        ],
        parameters=8_000_000_000,
    )
    assert chosen is not None and chosen.label == "Q4_K_M"


def test_without_a_parameter_count_the_family_still_decides() -> None:
    chosen = review._choose_quant(
        [_Group("Q4_0", 4_000_000_000), _Group("Q4_K_M", 4_800_000_000)], parameters=None
    )
    assert chosen is not None and chosen.label == "Q4_K_M"


# --- the engine's own architecture list --------------------------------


def test_the_architecture_list_is_parsed_from_the_engine_source() -> None:
    source = """
    static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
        { LLM_ARCH_LLAMA,           "llama"        },
        { LLM_ARCH_QWEN3,           "qwen3"        },
        { LLM_ARCH_GEMMA4,          "gemma4"       },
        { LLM_ARCH_UNKNOWN,         "(unknown)"    },
    };
    """
    assert review.parse_architectures(source) >= {"llama", "qwen3", "gemma4"}


def recorded_entry(size=5 * 1024**3):
    return {
        "class": "8B",
        "baseModel": "V/Current",
        "repo": "unsloth/current-GGUF",
        "recommended": {"file": "current-Q4_K_M.gguf", "sizeBytes": size},
        "cpuProof": cpu_proof(),
        "evidence": {"consecutiveReviewsAtTop": 2},
    }


def review_command(
    monkeypatch,
    tmp_path,
    entries,
    *,
    engine="b10948",
    enabled=True,
    previous=None,
    challenger=False,
    cli=False,
):
    answers = [
        row("unsloth/current-GGUF", downloads=100, base="V/Current"),
        row("bartowski/current-GGUF", downloads=90, base="V/Current"),
    ]
    if challenger:
        answers.extend(
            [
                row("unsloth/new-GGUF", downloads=300, base="V/New"),
                row("bartowski/new-GGUF", downloads=290, base="V/New"),
            ]
        )

    async def listing(self, **kwargs):
        return answers

    async def architectures(tag):
        return {"qwen35"}

    async def build(client, candidate, *, size_class):
        result = recorded_entry()
        result.update(baseModel=candidate.name, repo="unsloth/new-GGUF")
        result["recommended"]["file"] = "new-Q4_K_M.gguf"
        result.pop("cpuProof")
        return result

    monkeypatch.setattr(review.HubClient, "list_models", listing)
    monkeypatch.setattr(review, "engine_architectures", architectures)
    monkeypatch.setattr(review, "build_entry", build)
    path = tmp_path / "current.yaml"
    path.write_text(
        yaml.safe_dump(
            previous
            or {
                "reviewed": "2026-09-21",
                "engine": "llama_cpp b10948",
                "classes": entries,
            }
        )
    )
    if cli:
        report = tmp_path / "report.md"
        proposal = tmp_path / "proposed.yaml"
        review.main(
            [
                "--starter-file",
                str(path),
                "--report",
                str(report),
                "--proposed",
                str(proposal),
                "--no-cpu-proof",
                "--pages",
                "1",
                "--limit",
                "100",
                "--base-url",
                "https://hub.invalid",
            ]
        )
        return report.read_text(encoding="utf-8"), yaml.safe_load(proposal.read_text()), []
    return asyncio.run(
        review.review(
            starter_file=path,
            engine_tag=engine,
            pages=1,
            limit=100,
            base_url="https://hub.invalid",
            token=None,
            cpu_proof_enabled=enabled,
        )
    )


def test_proof_kept_when_file_and_engine_unchanged(monkeypatch, tmp_path):
    entry = recorded_entry()
    text, proposed, verdicts = review_command(monkeypatch, tmp_path, [entry])
    assert proposed["classes"][0]["cpuProof"] == entry["cpuProof"]
    assert next(v for v in verdicts if v.size_class == "8B").check == "cpu"
    assert "13.85 tok/s" in text and cpu_proof()["machine"] in text


@pytest.mark.parametrize("changed", ["repo", "file", "size"])
def test_large_proof_invalidated_when_inputs_change(changed):
    entry = recorded_entry(9 * 1024**3)
    # The recorded proof belongs to the current row, whereas the replacement
    # names a changed file. Exercise the carry check used for KEEP and REPLACE.
    original = copy.deepcopy(entry)
    if changed == "repo":
        entry["repo"] += "-changed"
    elif changed == "file":
        entry["recommended"]["file"] = "changed.gguf"
    else:
        entry["recommended"]["sizeBytes"] += 1
    assert review.kept_cpu_proof(entry, original, "b10948") is None


def test_a_large_files_hand_proof_survives_a_new_engine_tag(monkeypatch, tmp_path):
    entry = recorded_entry(9 * 1024**3)
    entry["cpuProof"]["engine"] = "llama_cpp old"
    text, proposed, verdicts = review_command(monkeypatch, tmp_path, [entry])
    assert proposed["classes"][0]["cpuProof"] == entry["cpuProof"]
    assert "hand measurement is needed" not in text
    assert next(v for v in verdicts if v.size_class == "8B").check == "cpu"


def test_a_small_files_proof_is_measured_again_under_a_new_engine_tag():
    entry = recorded_entry()
    entry["cpuProof"]["engine"] = "llama_cpp old"
    assert review.kept_cpu_proof(entry, copy.deepcopy(entry), "b10948") is None


def test_large_hand_proof_kept(monkeypatch, tmp_path):
    entry = recorded_entry(22_134_528_992)
    _, proposed, _ = review_command(monkeypatch, tmp_path, [entry])
    assert proposed["classes"][0]["cpuProof"] == entry["cpuProof"]


def test_small_files_measured_one_at_a_time_and_deleted(monkeypatch, tmp_path):
    from eugene_plexus_library import cpu_proof as cpu

    paths = []
    calls = []

    async def engine(tag, directory):
        calls.append(tag)
        return Path(directory) / "llama-bench"

    async def download(client, entry, path):
        assert not any(p.exists() for p in paths)
        path.write_bytes(b"weights")
        paths.append(path)

    def measure(bench, model, tag, **kwargs):
        assert model.exists()
        return cpu.CpuProof.model_validate(cpu_proof(11))

    monkeypatch.setattr(cpu, "download_engine", engine)
    monkeypatch.setattr(review, "download_cpu_model", download)
    monkeypatch.setattr(cpu, "measure", measure)
    first = recorded_entry(8 * 1024**3)
    first.pop("cpuProof")
    second = copy.deepcopy(first)
    second["class"] = "4B"
    second["recommended"]["sizeBytes"] = 2 * 1024**3
    _, proposed, verdicts = review_command(monkeypatch, tmp_path, [first, second])
    assert calls == ["b10948"]
    assert len(paths) == 2 and not any(p.exists() for p in paths)
    assert all(e["cpuProof"]["decodeTokensPerSecond"] == 11 for e in proposed["classes"])
    assert next(v for v in verdicts if v.size_class == "8B").check == "cpu"


@pytest.mark.parametrize("failure", ["engine", "download", "bench"])
def test_cpu_failure_makes_review_with_cause(monkeypatch, tmp_path, failure):
    from eugene_plexus_library import cpu_proof as cpu

    async def engine(tag, directory):
        if failure == "engine":
            raise cpu.CpuProofError("HTTP 503: build unavailable")
        return Path(directory) / "llama-bench"

    async def download(client, entry, path):
        if failure == "download":
            raise review.HubError("HTTP 403: model download denied")
        path.write_bytes(b"weights")

    def measure(*args, **kwargs):
        raise cpu.CpuProofError("unsupported model architecture: qwen35")

    monkeypatch.setattr(cpu, "download_engine", engine)
    monkeypatch.setattr(review, "download_cpu_model", download)
    monkeypatch.setattr(cpu, "measure", measure)
    entry = recorded_entry()
    entry.pop("cpuProof")
    text, proposed, verdicts = review_command(monkeypatch, tmp_path, [entry])
    verdict = next(v for v in verdicts if v.size_class == "8B")
    assert verdict.verdict == "REVIEW" and verdict.check == "architecture"
    assert proposed["classes"][0]["cpuProof"] is None
    expected = {
        "engine": "HTTP 503",
        "download": "HTTP 403",
        "bench": "unsupported model architecture",
    }
    assert expected[failure] in text
    assert "b10948" in text


def test_no_cpu_proof_flag_skips_measurement(monkeypatch, tmp_path):
    entry = recorded_entry()
    entry.pop("cpuProof")
    _, proposed, _ = review_command(monkeypatch, tmp_path, [entry], enabled=False)
    assert proposed["classes"][0]["cpuProof"] is None


def test_two_reviews_fire_replace(monkeypatch, tmp_path):
    text, first, _ = review_command(
        monkeypatch,
        tmp_path,
        [recorded_entry()],
        challenger=True,
        cli=True,
    )
    assert "| 8B | **REVIEW**" in text
    assert first["classes"][0]["baseModel"] == "V/Current"
    text, second, _ = review_command(
        monkeypatch,
        tmp_path,
        [],
        previous=first,
        challenger=True,
        cli=True,
    )
    assert "| 8B | **REPLACE**" in text
    entry = second["classes"][0]
    assert entry["baseModel"] == "V/New"
    assert entry["evidence"]["consecutiveReviewsAtTop"] == 2
