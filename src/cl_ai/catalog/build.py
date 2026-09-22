"""The catalog build pipeline: discover -> extract -> normalize -> lint -> store.

One function, so there is exactly one order these stages run in. Composing
them at each call site would let the order drift, and the order is load-bearing:
lint needs the raw findings (placeholder leaks are only detectable there), so
they have to outlive normalisation.

Never raises for data reasons. A tier that is unavailable, a page that will not
parse, a tool whose schema is unusable -- all of these are recorded and the
build continues. The only thing that should fail a build is being unable to
write the artifact, because then there is nothing to show for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .discovery import Discovered
from .extract.base import RawTool, Source
from .lint import LintReport, check_findings, lint
from .normalize import Catalog, normalize
from .store import cache_key, save

__all__ = ["BuildResult", "build", "default_sources"]


@dataclass(frozen=True)
class BuildResult:
    catalog: Catalog
    report: LintReport
    findings: int = 0
    #: Tiers that ran, and tiers that were skipped because they had nothing to
    #: read. Reported separately because an absent tier and an empty one look
    #: identical in the output but mean very different things.
    used: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    key: str = ""
    path: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether the catalog is free of entries that could render wrongly."""
        return not self.report.errors

    def summary(self) -> str:
        counts = self.report.counts()
        return (
            f"{len(self.catalog)} tools from {self.findings} findings "
            f"({', '.join(self.used) or 'no tiers'}"
            + (f"; skipped {', '.join(self.skipped)}" if self.skipped else "")
            + ") "
            f"errors={counts.get('error', 0)} "
            f"warnings={counts.get('warning', 0)} "
            f"conflicts={len(self.catalog.conflicts)}"
        )


def default_sources() -> tuple[Source, ...]:
    """Every tier that is implemented today, best first.

    Only tier 4 exists so far. Returned as a tuple rather than hardcoded at the
    call site so adding a tier is a one-line change here, and so a caller can
    substitute its own for testing.
    """
    from .extract.tldr import TldrSource

    return (TldrSource(),)


def build(
    sources: Sequence[Source] | None = None,
    *,
    binaries: Iterable[str] | None = None,
    discovered: Iterable[Discovered] | None = None,
    path: str | Path | None = None,
    write: bool = True,
) -> BuildResult:
    """Build a catalog, optionally restricted to `binaries` and persisted.

    `binaries` is how this stays cheap on a real machine: there is no point
    schematising 7,000 documented commands when only the ~1,200 actually on
    PATH can ever be suggested. Passing None means "everything the tiers know",
    which is what a test or a full offline build wants.
    """
    tiers = tuple(sources) if sources is not None else default_sources()

    findings: list[RawTool] = []
    used: list[str] = []
    skipped: list[str] = []
    for source in tiers:
        name = getattr(source, "name", type(source).__name__)
        try:
            if not source.available():
                skipped.append(name)
                continue
            harvested = list(source.harvest(binaries))
        except Exception:  # noqa: BLE001 - a broken tier must not fail the build
            # Deliberately broad. A third-party or future tier that raises for
            # its own reasons costs us that tier, not the whole catalog, and
            # there is no useful distinction to draw between the ways it might
            # fail from out here.
            skipped.append(name)
            continue
        used.append(name)
        findings.extend(harvested)

    catalog = normalize(findings)
    report = lint(catalog, extra=check_findings(findings))
    key = cache_key(discovered, extra="+".join(used)) if discovered is not None else ""

    written: Path | None = None
    if write:
        written = save(catalog, path, key=key, built_with=used)

    return BuildResult(
        catalog=catalog,
        report=report,
        findings=len(findings),
        used=tuple(used),
        skipped=tuple(skipped),
        key=key,
        path=written,
    )
