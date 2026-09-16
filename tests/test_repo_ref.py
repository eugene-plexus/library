"""Reading a pasted repo reference out of a search box.

The distinction the whole module turns on: a URL names one repo, so a
miss is a 404; a bare `owner/name` is a guess, so a miss falls through to
an ordinary search.
"""

from __future__ import annotations

import pytest

from eugene_plexus_library import repo_ref


@pytest.mark.parametrize(
    ("text", "repo", "revision"),
    [
        ("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF", "unsloth/Qwen3.8-27B-GGUF", None),
        ("https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/", "unsloth/Qwen3.8-27B-GGUF", None),
        (
            "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main",
            "unsloth/Qwen3.8-27B-GGUF",
            "main",
        ),
        (
            # What the address bar actually holds while someone is
            # looking at the file they want, which is when they copy it.
            "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_0.gguf",
            "ggml-org/gemma-4-E4B-it-GGUF",
            "main",
        ),
        (
            "https://huggingface.co/a/b/resolve/9fb0a04/model.gguf?download=true",
            "a/b",
            "9fb0a04",
        ),
        ("http://huggingface.co/a/b", "a/b", None),
        # Any host: `catalogueBaseUrl` is a config field and mirrors exist.
        ("https://hf-mirror.example.com/unsloth/x-GGUF", "unsloth/x-GGUF", None),
        ("www.huggingface.co/a/b", "a/b", None),
        ("  https://huggingface.co/a/b  ", "a/b", None),
    ],
)
def test_a_url_is_a_certain_reference(text: str, repo: str, revision: str | None) -> None:
    parsed = repo_ref.parse(text)
    assert parsed is not None
    assert (parsed.repo, parsed.revision) == (repo, revision)
    assert parsed.certain is True


@pytest.mark.parametrize(
    "text",
    [
        "https://huggingface.co/datasets/owner/name",
        "https://huggingface.co/spaces/owner/name",
        "https://huggingface.co/collections/owner/name",
        "https://huggingface.co/models?search=qwen",
        "https://huggingface.co/",
        "https://huggingface.co/gpt2",
    ],
)
def test_a_url_that_is_not_a_model_repo_is_not_a_reference(text: str) -> None:
    """A dataset link and a search page are not models, and a
    single-segment path is not two segments. Reading any of them as a
    repo would produce a confident 404 about a repo nobody named."""
    assert repo_ref.parse(text) is None


def test_a_bare_owner_slash_name_is_a_guess() -> None:
    parsed = repo_ref.parse("unsloth/Qwen3.5-4B-GGUF")
    assert parsed is not None
    assert parsed.repo == "unsloth/Qwen3.5-4B-GGUF"
    assert parsed.certain is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "qwen",
        "qwen coder gguf",
        "a/b/c",
        "unsloth / qwen",
        "best model for 8gb",
        "what/is this",
        "models/qwen",
    ],
)
def test_ordinary_queries_are_left_alone(text: str) -> None:
    """One word is a search. Three segments is not a repo id. A sentence
    with a slash in it is a sentence -- and a single segment is never a
    reference even though single-segment repos exist, because breaking
    the common case to serve the rare one is the wrong trade."""
    assert repo_ref.parse(text) is None


def test_none_and_empty_are_not_references() -> None:
    assert repo_ref.parse(None) is None
