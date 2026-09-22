"""Stage 3 -- raw findings -> canonical schemas.

Merge rule: HIGHEST TIER WINS. Losing values are not discarded; they are kept
in provenance with a conflict flag. Silent merging is how catalogs rot.
"""
