# Eugene Plexus — `library`

[![CI](https://github.com/eugene-plexus/library/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/library/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

The model library for the [Eugene Plexus](https://github.com/eugene-plexus) control plane. Point it at directories you already keep models in; it scans them, reads what each model actually is, and remembers the launch settings that worked.

**The rule it exists to honour:** your model files stay yours. Your directories, your filenames, your layout. This service reads. It never relocates, renames, hash-addresses, or takes custody of anything. Delete Eugene Plexus and every model is still where it was, correctly named.

## What it holds

**Scanned models, both formats.** GGUF — one quantized file, possibly sharded, possibly with a vision projector beside it — and HF safetensors, a directory of weights plus `config.json`. Format is a dimension of the data model, not an assumption.

**Named per-model launch profiles.** N per model, one default. This is the answer to *"tired of tweaking llama.cpp settings for every different model"*: the settings that worked are saved against the model instead of retyped into a runtime declaration every time. Plural because a long-context profile and a fast one are different flags on the same file — and because two replicas of one model across two GPUs is the same profile twice with a different `CUDA_VISIBLE_DEVICES`.

Profiles are the **only** thing this service persists about a model. Everything else is a projection of what is on disk, which is why a model can be listed as `missing`: the file is gone but the profile you spent an afternoon tuning is not.

## What it isn't

- **Not a store.** No managed directory, no content addressing, no blob cache. Model roots are paths you choose, in this component's own config.
- **Not in the request path.** No inference traffic passes through here. The gateway never calls it.
- **Not the launcher.** Launching is composed by the caller: read a profile here, then `POST /v1/runtimes` on the [`watchdog`](https://github.com/eugene-plexus/watchdog) with the model's `path` and the profile's `flags`. A profile's field names are `RuntimeSpec`'s field names precisely so that composition is a copy rather than a translation. This service holds no engine knowledge and never calls the watchdog.
- **Not a validator of engine flags.** It stores `flags` as given. Validation belongs where the curated flag surface lives — the watchdog's engine adapter, when a runtime is actually created. Two validators would mean two copies of engine knowledge and one of them going stale.

## Status

**v0.1 (milestone M2), working.** Local scan, both formats, metadata, profiles, and the config trio are wired end to end. Catalogue search, downloads and hardware-fit quant guidance are M3 and are not here yet.

## Wire contract

Defined in [`specs/openapi/library.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/library.yaml). Port **8082**.

| | |
|---|---|
| `GET /v1/models` | Every model. `?path=` for reverse lookup from an absolute path |
| `GET /v1/models/{id}` | One model, with full metadata |
| `DELETE /v1/models/{id}` | Forget a **missing** entry and its profiles. Never touches a file |
| `GET`/`POST /v1/models/{id}/profiles` | List / create launch profiles |
| `GET`/`PUT`/`DELETE /v1/models/{id}/profiles/{pid}` | Read / replace / delete one |
| `GET`/`POST`/`DELETE /v1/scan` | State of / start / cancel the walk |
| `GET`/`PATCH /v1/config`, `GET /v1/config/schema`, `POST /v1/config/test` | The standard config trio |
| `POST /v1/admin/restart`, `GET /healthz` | Meta |

## Reading a model is not free

Measured, not assumed. The KV block of a 248k-vocab GGUF is **~10.9 MB**, because the tokenizer lives in it, and the fields worth having can sit on the far side of that — `general.file_type`, the quant tier, is the *last* key in one of the files this was checked against. You cannot skip a string array in one seek either: each string's length is inline, so stepping 248,320 of them is 248,320 reads, and that loop rather than the I/O is the cost. Roughly **50–60 ms per large-vocab model, warm**.

Two consequences shape the service:

- **Scanning is asynchronous** — `POST /v1/scan` returns immediately and you poll `GET /v1/scan`. Seconds of disk work does not belong on a request the UI is waiting on.
- **Rescans are incremental.** A file whose `(path, size, mtime)` is unchanged keeps its cached metadata and is never opened. `{"full": true}` forces a re-read, for when the metadata reader has changed rather than the files.

## What is *not* a model

Most of the difficulty. A directory of GGUFs is not a list of models:

| On disk | Why it isn't an entry |
|---|---|
| `mmproj-*.gguf` | A vision projector belonging to the model beside it. 931 MB in one real case, 1.8 GB in another. Detected by `general.type: mmproj`, not by filename |
| `…-00002-of-00003.gguf` | Shard 2..N of one split model. Only the first shard goes on a launch line |
| `adapter_model.safetensors` + `adapter_config.json` | A LoRA. `--lora` in `extraArgs` covers anyone who needs one now |
| `models--org--name/snapshots/<older-rev>/` | A HuggingFace cache keeps one directory *per revision*. `refs/main` names the current one |
| `blobs/`, `refs/`, `.no_exist/` | HF cache infrastructure. `.no_exist/` is a *negative* cache full of zero-byte files with real-looking names, so a naive "is there an `adapter_config.json`?" probe finds one |

Every one of these is reported on `GET /v1/scan` under `skipped[]`, with a reason. A scanner that silently drops what it did not understand is indistinguishable from a broken one, and *"why isn't my model showing up"* with no answer is how a tool gets uninstalled.

## Identity

A model's identity is its **path**. The `id` is a URL-safe handle derived from it — the first 16 hex characters of the SHA-256 of the normalized absolute path — because Windows paths do not go in URLs. Normalized means absolute, canonical separators, case-folded on Windows, and **symlinks left unresolved**: resolving them turns a HuggingFace snapshot into `blobs/<sha>` and smuggles content addressing in through the back door.

It is derived rather than random so it survives restarts, since profiles are keyed to it. Treat it as **opaque** — use `GET /v1/models?path=` to go the other way rather than reproducing the normalization. And it is not content addressing: no byte of the model is read to produce it.

**Move a model and it reads as a new model.** The old entry goes `missing` and keeps its profiles so you can copy them across. Nothing guesses that a file which vanished from one root and appeared in another is the same file, because a wrong guess silently applies one model's tuning to another.

## Running

### From source

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate
pip install -e ".[dev]"
python -m eugene_plexus_library
```

Binds `127.0.0.1:8082` by default. Under the watchdog the port comes from the topology via `EUGENE_PLEXUS_LIBRARY_BIND_PORT`.

### Configuration

Runtime config lives in `config.yaml` (path via `EUGENE_PLEXUS_LIBRARY_CONFIG_FILE`) and is editable over HTTP through the standard trio. The field that matters is `modelRoots`, a `path_list` — the directories to scan. `POST /v1/config/test` stats them and tells you which are unreadable without walking them, which is what you want next to a directory picker.

Scanned metadata and saved profiles live in a separate machine-owned state file (`EUGENE_PLEXUS_LIBRARY_STATE_FILE`, default `library-state.json`). JSON rather than YAML deliberately: it is written on every scan, holds one record per model, and nobody hand-edits it.

#### Degraded mode

Bad config never crashes the component. A **configured** root that cannot be read reports `degraded` on `/healthz` — you asked for something that isn't working, and an unplugged drive or a dead network share is exactly what that is for. Having **no** roots configured is `ok`: that's a fresh install waiting for the wizard, not a fault.

## Codegen

Pydantic models for the wire contract are generated from the pinned commit of `eugene-plexus/specs` recorded in [`SPECS_REF`](SPECS_REF):

```bash
python scripts/codegen.py
```

The script downloads the specs at the pinned SHA, runs `datamodel-code-generator` against them, and writes Pydantic v2 models to `src/eugene_plexus_library/_generated/`. **The generated files are committed** so builds are reproducible without network access. CI re-runs codegen and fails the build if the working tree differs.

To bump to a newer specs commit:

```bash
echo "<new-sha>" > SPECS_REF
python scripts/codegen.py
```

…then commit `SPECS_REF` and the regenerated `_generated/` directory together.

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check .
ruff format --check .

# Type-check
mypy src/

# Test
pytest

# Codegen freshness
python scripts/codegen.py && git diff --exit-code src/eugene_plexus_library/_generated/
```

The suite builds real GGUF and safetensors files byte by byte in `tmp_path` rather than mocking the readers — the format details are the part most likely to be wrong, so they are the part tested against actual bytes.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
