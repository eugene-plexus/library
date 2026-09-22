"""Reading a pasted repo reference out of a search box.

The commonest way a person arrives with a model in mind is that someone
linked them one. Before this, pasting that link into the search box put
the whole URL through upstream's full-text index and returned nothing --
a dead end whose only clue was an empty list, at the exact moment the
person was closest to succeeding.

## What counts as a reference, and what does not

A **URL** on the catalogue's host is unambiguous: it names one repo and
nothing else, so a URL that does not resolve is a 404 rather than an
empty search. Everything after the two path segments is discarded, which
is what makes a pasted deep link work -- people paste the address bar,
and the address bar is usually sitting on `/tree/main` or on a file.

A bare **`owner/name`** is a guess. It is the shape of a repo id, and it
is also a thing a person might type meaning to search ("qwen/coder").
So it is looked up, and a miss falls through to an ordinary search
rather than becoming an error.

A **single segment** is never a reference, even though single-segment
repos exist (`gpt2`): one word in a search box is a search, and reading
it as a repo id would break the ordinary case to serve the rare one.

## Host

Any host, not just `huggingface.co` -- `catalogueBaseUrl` is a config
field, mirrors exist, and a person on a mirror pastes a mirror's link.
The host is discarded once the path is read; what comes back is a repo
id, which is host-relative anyway.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

__all__ = ["RepoReference", "parse"]

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_NON_REPO_PREFIXES = frozenset(
    {
        # Hub paths that are two segments deep and are not models.
        "datasets",
        "spaces",
        "collections",
        "organizations",
        "settings",
        "docs",
        "blog",
        "papers",
        "posts",
        "join",
        "login",
        "pricing",
        "models",  # /models?search=... is the browse page, not a repo
    }
)


class RepoReference:
    """A parsed reference. `certain` is what decides a 404 from a search."""

    __slots__ = ("certain", "repo", "revision")

    def __init__(self, repo: str, *, revision: str | None, certain: bool) -> None:
        self.repo = repo
        self.revision = revision
        self.certain = certain
        """True when it came from a URL. A URL names one repo, so a miss
        is an error; a bare `owner/name` is a guess, so a miss is a
        search."""

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"RepoReference({self.repo!r}, revision={self.revision!r}, certain={self.certain})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RepoReference) and (self.repo, self.revision, self.certain) == (
            other.repo,
            other.revision,
            other.certain,
        )


def parse(query: str | None) -> RepoReference | None:
    """`None` when this is an ordinary search, which is most of the time."""
    if not query:
        return None
    text = query.strip()
    if not text:
        return None

    if "://" in text or text.lower().startswith("www."):
        return _from_url(text)

    # A bare `owner/name`. Whitespace anywhere means it is a sentence.
    if any(c.isspace() for c in text):
        return None
    parts = text.strip("/").split("/")
    if len(parts) != 2 or not all(_SEGMENT.match(p) for p in parts):
        return None
    if parts[0].lower() in _NON_REPO_PREFIXES:
        return None
    return RepoReference(f"{parts[0]}/{parts[1]}", revision=None, certain=False)


def _from_url(text: str) -> RepoReference | None:
    candidate = text if "://" in text else f"https://{text}"
    try:
        split = urlsplit(candidate)
    except ValueError:
        return None
    if not split.netloc:
        return None

    segments = [unquote(s) for s in split.path.split("/") if s]
    if len(segments) < 2:
        return None
    if segments[0].lower() in _NON_REPO_PREFIXES:
        return None
    if not _SEGMENT.match(segments[0]) or not _SEGMENT.match(segments[1]):
        return None

    repo = f"{segments[0]}/{segments[1]}"
    revision = None
    # `/tree/<rev>/...` and `/blob/<rev>/<file>` are what the address bar
    # holds while someone is looking at the files -- which is exactly
    # when they copy it.
    if len(segments) >= 4 and segments[2] in {"tree", "blob", "resolve"}:
        revision = segments[3]
    return RepoReference(repo, revision=revision, certain=True)
