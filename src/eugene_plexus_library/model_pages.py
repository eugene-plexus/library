"""Keyset pages with authenticated, self-contained continuation tokens."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import secrets

from ._generated.models import LibraryModel


def sort_key(model: LibraryModel) -> tuple[str, str]:
    # A scan's traversal order can change. Equal names still need one stable
    # order, including two files with the same name in different folders.
    return model.name.lower(), model.id


class ModelPages:
    """One Library process's cursor issuer; rescans do not replace it.

    Authentication rejects invented keys without keeping an ever-growing
    registry of cursors or looking up a model a rescan may have removed.
    A restart invalidates old cursors, with a reason rather than a guess.
    """

    def __init__(self) -> None:
        self._secret = secrets.token_bytes(32)

    def _encode(self, key: tuple[str, str]) -> str:
        payload = base64.urlsafe_b64encode(json.dumps(key).encode()).decode().rstrip("=")
        signature = hmac.digest(self._secret, payload.encode(), "sha256").hex()
        return f"v1.{payload}.{signature}"

    def _decode(self, cursor: str) -> tuple[str, str]:
        parts = cursor.split(".")
        if len(parts) != 3:
            raise ValueError(
                "The cursor is malformed: expected a version, sort key, and signature."
            )
        version, payload, signature = parts
        if version != "v1":
            raise ValueError("The cursor version is unknown; request the first page again.")
        try:
            raw = base64.b64decode(
                payload + "=" * (-len(payload) % 4), altchars=b"-_", validate=True
            )
            key = json.loads(raw)
            supplied = bytes.fromhex(signature)
        except (ValueError, binascii.Error, UnicodeError, RecursionError) as exc:
            raise ValueError(
                "The cursor is malformed: its sort key or signature cannot be read."
            ) from exc
        if (
            not isinstance(key, list)
            or len(key) != 2
            or not all(isinstance(part, str) for part in key)
            or not key[1]
            or key[0] != key[0].lower()
            or len(supplied) != 32
        ):
            raise ValueError("The cursor is malformed: expected a model sort key and signature.")
        expected = hmac.digest(self._secret, payload.encode(), "sha256")
        if not hmac.compare_digest(supplied, expected):
            raise ValueError(
                "The cursor is unknown to this Library process; request the first page again."
            )
        return key[0], key[1]

    def page(
        self, models: list[LibraryModel], limit: int | None, cursor: str | None
    ) -> tuple[list[LibraryModel], str | None]:
        after = self._decode(cursor) if cursor is not None else None
        ordered = sorted(models, key=sort_key)
        if after is not None:
            ordered = [model for model in ordered if sort_key(model) > after]
        if limit is None or len(ordered) <= limit:
            return ordered, None
        page = ordered[:limit]
        return page, self._encode(sort_key(page[-1]))
