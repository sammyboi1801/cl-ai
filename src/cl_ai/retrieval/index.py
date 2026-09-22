"""Two-stage index.

Stage 1 picks the binary (~dozens, well separated) -- this is where the context
engine is strongest. Stage 2 picks the leaf within it (aws_s3_cp vs aws_s3_mv),
a much easier ranking problem once the binary is fixed.

Fine-grained leaves cost the model nothing (only the top 5 are ever declared),
but they move the discrimination burden onto retrieval. This two-stage split is
what keeps that tractable.

Owns the score floor that yields an honest "no match" instead of the
nearest-neighbour substitution that turns a missing tool into a wrong command.
"""
