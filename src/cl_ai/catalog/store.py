"""Artifact IO + cache keys.

Cache key is binary IDENTITY (path + size + mtime + content hash), never
`--version` output -- version detection would require execution, and identity
also catches a rebuilt binary at an unchanged version number.

Must also distinguish variants: macOS BSD `ls` and GNU `ls` are the same name
with different schemas.
"""
