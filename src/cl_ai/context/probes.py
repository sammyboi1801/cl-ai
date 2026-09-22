"""cwd markers (package.json, Dockerfile, go.mod, .git, *.tf), installed binaries,
os/shell pair.

$PATH availability is a HARD GATE, not a score: never suggest kubectl on a
machine that does not have it.
"""
