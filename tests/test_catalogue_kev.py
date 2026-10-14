"""Published Kev checkpoints must survive Discover -> download -> scan."""

import pytest

from eugene_plexus_library import catalogue
from eugene_plexus_library._generated.models import ModelFileRole, ModelFormat
from eugene_plexus_library.hub import FileMetadata, RepoInfo

# hf models ls jaredpalmer/kev-0.8b --recursive, 2026-10-14.
# Revision bf75a6a8848ea6960ff2ed108d9ed44c2941174f; only the filenames/sizes.
KEV_FILES = {
    ".gitattributes": 1570,
    "README.md": 24767,
    "adapter_config.json": 1273,
    "adapter_model.safetensors": 43338624,
    "head.pt": 2103999,
    "provenance.json": 4221,
    "result.json": 80760,
    "tokenizer.json": 19989325,
    "tokenizer_config.json": 1128,
    "train.log": 18427,
    "training_config.json": 1767,
    "training_metrics.json": 326,
}


@pytest.mark.parametrize("prefix", ["", "checkpoint/"])
def test_a_kev_candidate_keeps_the_entire_checkpoint(prefix):
    files = [FileMetadata(path=prefix + path, size=size) for path, size in KEV_FILES.items()]
    groups, _ = catalogue._safetensors_groups(files, repo="jaredpalmer/kev-0.8b")
    assert len(groups) == 1
    group = groups[0]
    assert group.format is ModelFormat.kev_checkpoint
    assert {f.path for f in group.files} == {prefix + name for name in KEV_FILES}
    assert group.size == sum(KEV_FILES.values())
    facts = catalogue.candidate_facts(
        group, repo="jaredpalmer/kev-0.8b", info=RepoInfo(repo="jaredpalmer/kev-0.8b"), configs={}
    )
    assert facts.format is ModelFormat.kev_checkpoint
    assert catalogue._role_for(prefix + "head.pt", group.format, index=2) is ModelFileRole.weights


@pytest.mark.parametrize("missing", ["adapter_config.json", "head.pt"])
def test_neither_marker_alone_turns_an_adapter_into_kev(missing):
    files = [
        FileMetadata(path=name, size=size) for name, size in KEV_FILES.items() if name != missing
    ]
    groups, _ = catalogue._safetensors_groups(files, repo="fixture/adapter")
    assert groups[0].format is ModelFormat.safetensors


def test_search_facts_judge_a_known_checkpoint_as_kev():
    row = catalogue.build_search_result(
        {
            "id": "jaredpalmer/kev-0.8b",
            "tags": ["safetensors"],
            "siblings": [{"rfilename": name} for name in KEV_FILES],
        }
    )
    assert row.formats == [ModelFormat.kev_checkpoint]
    assert row.facts[0].format is ModelFormat.kev_checkpoint
