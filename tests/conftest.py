"""Fixtures, and the builders that write real model files.

The format readers are the part of this component most likely to be
wrong, so the suite writes **actual GGUF and safetensors bytes** into
`tmp_path` and reads them back rather than mocking the readers out. A
mocked reader agrees with whatever the code believes; a real file
disagrees when the code is wrong, which is the entire point.

The builders below encode the layouts independently of the readers —
`struct.pack` against the documented format, not a call back into
`formats.gguf` — so a bug in the reader cannot be papered over by the
same bug in the fixture.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_library import tokens
from eugene_plexus_library.app import create_app
from eugene_plexus_library.auth_state import AuthState, load_auth_state
from eugene_plexus_library.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no `EUGENE_PLEXUS_*` in the environment.

    `Settings` reads the process environment, and a developer's machine is
    exactly where one is set: a Windows service install leaves
    `EUGENE_PLEXUS_LIBRARY_DEFAULT_MODEL_ROOTS` in the machine environment,
    and with it seven tests about "no roots configured" failed here while
    passing in CI -- asserting about the developer's install rather than
    the code. The agent's suite has cleared the prefix the same way since
    it met this; a test that wants a variable sets it itself.
    """
    for key in [k for k in os.environ if k.startswith("EUGENE_PLEXUS_")]:
        monkeypatch.delenv(key, raising=False)


def make_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Windows symlink creation needs Developer Mode or symlink privilege")
        raise


@pytest.fixture(params=["symlink", "junction"])
def directory_link(request: pytest.FixtureRequest) -> Callable[[Path, Path], None]:
    if request.param == "junction":
        if os.name != "nt":
            pytest.skip("directory junctions are Windows-only")

        def junction(link: Path, target: Path) -> None:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True,
                capture_output=True,
            )
            assert link.is_junction()

        return junction

    def symlink(link: Path, target: Path) -> None:
        make_symlink(link, target, directory=True)

    return symlink


# GGUF value type tags, restated here rather than imported: a fixture
# that shares constants with the code under test can agree with it about
# something wrong.
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(value: Any) -> bytes:
    """Encode one KV value, tag included."""
    if isinstance(value, str):
        return struct.pack("<I", T_STRING) + _gguf_string(value)
    if isinstance(value, bool):
        return struct.pack("<I", T_BOOL) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", T_UINT32) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", T_FLOAT32) + struct.pack("<f", value)
    if isinstance(value, list):
        if not value:
            return struct.pack("<I", T_ARRAY) + struct.pack("<I", T_INT32) + struct.pack("<Q", 0)
        if isinstance(value[0], str):
            body = b"".join(_gguf_string(v) for v in value)
            element = T_STRING
        elif isinstance(value[0], bool):
            # **Checked before `int`, because `bool` IS an `int` in
            # Python and every one of these would otherwise be written
            # as an int32 array.** A real file's
            # `attention.sliding_window_pattern` is a BOOL array, and a
            # fixture that writes 0/1 int32s exercises a type the reader
            # never meets -- the recurring shape here being a fixture
            # that cannot produce the case it is named for.
            body = b"".join(struct.pack("<?", v) for v in value)
            element = T_BOOL
        else:
            body = b"".join(struct.pack("<i", v) for v in value)
            element = T_INT32
        head = (
            struct.pack("<I", T_ARRAY) + struct.pack("<I", element) + struct.pack("<Q", len(value))
        )
        return head + body
    raise TypeError(f"no GGUF encoding for {type(value).__name__}")


def write_gguf(
    path: Path,
    kv: dict[str, Any],
    *,
    version: int = 3,
    tensor_count: int = 0,
    payload: bytes = b"",
    magic: bytes = b"GGUF",
) -> Path:
    """Write a GGUF file with the given KV block.

    `magic`, `version` and a truncated `payload` are parameters so the
    malformed-input tests can build genuinely malformed files instead of
    asserting against a mock that raises on request.
    """
    body = bytearray()
    body += magic
    body += struct.pack("<I", version)
    body += struct.pack("<Q", tensor_count)
    body += struct.pack("<Q", len(kv))
    for key, value in kv.items():
        body += _gguf_string(key)
        body += _gguf_value(value)
    body += payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return path


def qwen_like_kv(
    *,
    name: str = "Test 7B",
    architecture: str = "llama",
    file_type: int = 15,
    context_length: int = 4096,
    vocab: int = 128,
) -> dict[str, Any]:
    """A KV block shaped like a real model's.

    Ordered so `general.file_type` lands **after** the token array, which
    is where a real 27B puts it — the reason the reader cannot stop
    early. A fixture with the quant helpfully near the front would let a
    short-circuiting reader pass.
    """
    return {
        "general.architecture": architecture,
        "general.type": "model",
        "general.name": name,
        "general.size_label": "7B",
        "general.sampling.temp": 1.0,
        "general.sampling.top_k": 20,
        f"{architecture}.block_count": 32,
        f"{architecture}.context_length": context_length,
        f"{architecture}.embedding_length": 4096,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(vocab)],
        "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "general.quantization_version": 2,
        "general.file_type": file_type,
    }


def projector_kv(*, name: str = "Test 7B") -> dict[str, Any]:
    """A vision projector: `general.type: mmproj`, `clip` architecture,
    and the same `general.name` as the model it belongs to — which is
    how a real pair looks."""
    return {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "general.name": name,
        "general.file_type": 32,
        "clip.has_vision_encoder": True,
    }


def embedding_kv(*, name: str = "Test Embed") -> dict[str, Any]:
    """An embedding model: a pooling type, non-causal attention, and
    **no `general.type`** — verified absent on a real one."""
    return {
        "general.architecture": "nomic-bert",
        "general.name": name,
        "nomic-bert.block_count": 12,
        "nomic-bert.context_length": 2048,
        "nomic-bert.pooling_type": 1,
        "nomic-bert.attention.causal": False,
        "general.file_type": 15,
    }


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a safetensors file whose header declares `tensors`.

    `{name: (dtype, shape)}`. Data offsets are computed so the header is
    self-consistent; the tensor bytes themselves are zeros, because
    nothing here ever reads past the header.
    """
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U8": 1}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        length = count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    if metadata is not None:
        header["__metadata__"] = metadata

    raw = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def write_hf_model(
    directory: Path,
    *,
    architectures: list[str] | None = None,
    max_position_embeddings: int = 2048,
    parameters: int = 1024,
) -> Path:
    """A minimal HuggingFace model directory: config, weights, tokenizer."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": architectures or ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": max_position_embeddings,
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(directory / "model.safetensors", {"weight": ("F32", [parameters])})
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


# --------------------------------------------------------------------- #
# App fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    return root


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        config_file=tmp_path / "config.yaml",
        state_file=tmp_path / "library-state.json",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def configured_client(settings: Settings, models_dir: Path) -> Iterator[TestClient]:
    """A client whose config already points at `models_dir`.

    Written to the config file before startup rather than PATCHed
    afterwards, so the startup-scan path is the one under test.
    """
    settings.config_file.write_text(
        f"modelRoots:\n  - {models_dir.as_posix()}\nscanOnStartup: true\n",
        encoding="utf-8",
    )
    with TestClient(create_app(settings=settings)) as c:
        yield c


# --------------------------------------------------------------------------- #
# Trust (per-node token keys, 2026-09-25)
# --------------------------------------------------------------------------- #


@dataclass
class FakeInstall:
    """A trust bundle as this component's agent would keep it on disk.

    The component runs on node `gw`. The root's token key signs sessions
    and client keys; `gw`'s own key signs the tokens its agent hands its
    children; `far` is another machine of the install.
    """

    directory: Path
    name: str = "gw"
    grants: tuple[str, ...] = ()
    far_grants: tuple[str, ...] = ()
    identity: Ed25519PrivateKey = field(default_factory=tokens.generate_private_key)
    root: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    )
    node: tokens.Signer | None = None
    far: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="node:far")
    )
    version: int = 0

    def __post_init__(self) -> None:
        if self.node is None:
            self.node = tokens.Signer(key=tokens.generate_private_key(), issuer=self.recipient)
        self.publish()

    @property
    def recipient(self) -> str:
        return tokens.node_recipient(self.name)

    @property
    def authority(self) -> str:
        return tokens.public_b64(self.identity)

    @property
    def bundle_path(self) -> Path:
        return self.directory / "trust_bundle.json"

    def publish(self, *, revoked: tuple[tuple[str, int], ...] = ()) -> tokens.TrustBundle:
        assert self.node is not None
        self.version += 1
        bundle = tokens.build_bundle(
            authority=self.identity,
            version=self.version,
            epoch=1,
            keys=[
                self.root.trust_key(["authority"]),
                self.node.trust_key(["node", *self.grants]),
                self.far.trust_key(["node", *self.far_grants]),
            ],
            revoked_sessions=revoked,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        tokens.write_bundle_file(self.bundle_path, bundle)
        return bundle

    def auth_state(self, *, master_key_b64: str | None = None) -> AuthState:
        return load_auth_state(
            trust_bundle_file=str(self.bundle_path),
            trust_authority=self.authority,
            auth_recipient=self.recipient,
            service_token=self.service("gateway", ttl=365 * 24 * 3600),
            master_key_b64=master_key_b64,
        )

    def session(
        self,
        *,
        sub: str = "operator",
        ttl: int = 3600,
        aud: list[str] | None = None,
        now: int | None = None,
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_SESSION,
            sub=sub,
            aud=aud or [self.recipient, "control"],
            ttl_seconds=ttl,
            now=now,
        )
        return token

    def client_key(
        self, *, name: str = "app", jti: str = "key-1", ttl: int = 3600, now: int | None = None
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_CLIENT, sub=name, aud=["gateway"], ttl_seconds=ttl, now=now, jti=jti
        )
        return token

    def service(self, sub: str = "gateway", *, ttl: int = 3600) -> str:
        """A token this machine's agent minted for one of its children."""
        assert self.node is not None
        token, _ = self.node.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def foreign_service(self, sub: str = "agent", *, ttl: int = 600) -> str:
        """Another machine's token, addressed here and correctly signed."""
        token, _ = self.far.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def raw(self, signer: tokens.Signer, typ: str, **claims: Any) -> str:
        """A token with exactly these claims, for the shapes `mint` will not make."""
        now = int(time.time())
        body: dict[str, Any] = {"iss": signer.issuer, "iat": now, "exp": now + 60, **claims}
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(
            body, signer.key, algorithm="EdDSA", headers={"typ": typ, "kid": signer.kid}
        )


@pytest.fixture
def install(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path / "node")
