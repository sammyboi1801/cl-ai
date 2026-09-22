"""Download the tldr corpus, so a fresh install has something to suggest.

WHY THIS HAS TO EXIST
`default_root()` looks in the standard tldr client caches and returns None
when it finds nothing, which on a machine without a tldr client installed is
always. Without a corpus there is no catalog, and Tab degrades to completing
names off PATH -- correct behaviour, and indistinguishable to the user from
the product not working. Every run of this system so far has depended on
CL_AI_TLDR_ROOT being set by hand.

EXPLICIT, NEVER IMPLICIT
Only ever run because someone typed `cl-ai fetch`. Nothing downloads on
import, on daemon start, or on a cache miss. A tool that reaches the network
without being asked is a tool people cannot run on a locked-down machine,
and a shell integration is exactly the wrong place for a surprise HTTP
request.

The destination is one of the paths `default_root()` already searches, so a
fetch is picked up with no configuration afterwards.

LICENCE
tldr-pages is CC-BY-4.0. The attribution is printed on fetch rather than
buried, because the user is acquiring the content at that moment.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_URL", "FetchError", "FetchResult", "default_target", "fetch"]

#: The archive published with every tldr-pages release. This is the same
#: asset the official clients download.
DEFAULT_URL = "https://github.com/tldr-pages/tldr/releases/latest/download/tldr.zip"

LICENCE = (
    "tldr-pages content is CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/), "
    "(C) the tldr-pages contributors: https://github.com/tldr-pages/tldr"
)

#: Refuse an implausible download. The real archive is a few megabytes; this
#: is a sanity bound, not a security boundary, and exists so a redirect to
#: something enormous cannot fill the user's disk.
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024


class FetchError(RuntimeError):
    """The corpus could not be fetched. Always actionable in the message."""


@dataclass(frozen=True)
class FetchResult:
    path: Path
    pages: int
    bytes_downloaded: int


def default_target() -> Path:
    """Where to put the corpus so `default_root()` finds it unaided.

    Deliberately one of the locations already searched, rather than a path of
    our own: a cache nobody else knows about would mean a user who later
    installs a real tldr client has two copies.
    """
    override = os.environ.get("CL_AI_TLDR_ROOT")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
        return Path(base) / "tldr"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "tldr"
    try:
        return Path.home() / ".cache" / "tldr"
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "tldr"


def _download(url: str, into: Path) -> int:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_ARCHIVE_BYTES:
                raise FetchError(
                    f"refusing a {int(declared) // (1024 * 1024)}MB download from {url}"
                )
            written = 0
            with into.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_ARCHIVE_BYTES:
                        raise FetchError(f"download from {url} exceeded the size limit")
                    handle.write(chunk)
    except FetchError:
        raise
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"could not reach {url}: {exc.reason}") from exc
    except OSError as exc:
        raise FetchError(f"could not write the download: {exc}") from exc
    if written == 0:
        raise FetchError(f"{url} returned an empty response")
    return written


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Members that are safe to write, with a total-size bound.

    CPython's extract() already strips leading separators and `..` segments,
    so this is belt-and-braces rather than the only defence -- but a zip from
    the network is exactly the input where an explicit check is worth its
    lines, and the uncompressed-size bound is not something extract() does at
    all (a zip bomb is a few KB on the wire).
    """
    members: list[zipfile.ZipInfo] = []
    total = 0
    for info in archive.infolist():
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise FetchError(f"archive contains an unsafe path: {name!r}")
        if ":" in name[:3] and os.name == "nt":
            raise FetchError(f"archive contains a drive-qualified path: {name!r}")
        total += info.file_size
        if total > MAX_UNPACKED_BYTES:
            raise FetchError("archive expands to more than the unpacked size limit")
        members.append(info)
    if not members:
        raise FetchError("archive is empty")
    return members


def _corpus_root(extracted: Path) -> Path:
    """Find the directory holding `pages*`, however the archive is shaped.

    The release asset unpacks `pages/` at the top level, but a source
    tarball nests everything under `tldr-main/`. Accepting both means a user
    who points --url at a branch zip is not silently left with nothing.
    """
    from .extract.tldr import ENGLISH_DIRS

    if any((extracted / name).is_dir() for name in ENGLISH_DIRS):
        return extracted
    children = [p for p in extracted.iterdir() if p.is_dir()]
    for child in children:
        if any((child / name).is_dir() for name in ENGLISH_DIRS):
            return child
    raise FetchError(
        "the archive does not look like a tldr corpus: no pages/ directory"
    )


def fetch(
    url: str = DEFAULT_URL,
    target: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
) -> FetchResult:
    """Download and install the corpus. Returns where it landed.

    Replaces any existing corpus atomically-ish: the new tree is built in a
    sibling directory and swapped in, so an interrupted fetch leaves the old
    one intact rather than a half-extracted mixture of two releases.
    """
    from .extract.tldr import TldrSource

    destination = Path(target) if target is not None else default_target()
    if destination.exists() and not force:
        existing = TldrSource(destination)
        if existing.available():
            raise FetchError(
                f"{destination} already holds a corpus; pass force=True to replace it"
            )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FetchError(f"cannot create {destination.parent}: {exc}") from exc

    # Staged beside the destination, not in the system temp: the final move
    # is only cheap and only atomic within one filesystem.
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".cl-ai-tldr-"))
    try:
        archive_path = staging / "tldr.zip"
        downloaded = _download(url, archive_path)

        unpacked = staging / "unpacked"
        unpacked.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(unpacked, members=_safe_members(archive))
        except zipfile.BadZipFile as exc:
            raise FetchError(f"{url} did not return a valid zip archive") from exc

        root = _corpus_root(unpacked)
        staged = TldrSource(root)
        if not staged.available():
            raise FetchError("the downloaded corpus has no English pages")
        pages = sum(1 for _ in root.glob("pages*/**/*.md"))

        # Swap: move the old tree aside, put the new one in place, then drop
        # the old one. Ordered so that a failure never leaves nothing there.
        backup = None
        if destination.exists():
            backup = destination.with_name(destination.name + ".old")
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(destination, backup)
        try:
            os.replace(root, destination)
        except OSError:
            if backup is not None:
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)

        return FetchResult(
            path=destination, pages=pages, bytes_downloaded=downloaded
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
