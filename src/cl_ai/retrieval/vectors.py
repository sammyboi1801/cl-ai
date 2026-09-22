"""Vector half.

MANDATORY: mean-center every query and document. Raw Needle embeddings are
anisotropic -- measured, everything lands at ~0.93 cosine and `docker run`
outranked `git reset` for a git query. Centering fixes the ranking completely.
Skip it and retrieval returns noise at high confidence.
"""
